"""
Full definition of a GPT language model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# @dataclass自动生成常见的特殊方法
@dataclass
class GPTConfig:
    block_size: int = 1024 # 模型能够处理的最长序列
    vocab_size: int = 50304 # GPT的vocab_size 为50257， padding到一个64的倍数
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    """
    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        # 这里关于self.weight 为什么需要权重是因为如果不存在权重则只会是 \frac{\alpha + \sigma}{X - X_{mean}}, 局限在了0均值单位方差的情况
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zero_(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        # 这个F.layer_norm 本身是不带学习参数的， 最终结果为0-1正则化后的值乘以缩放因子后加上bias的值
        return F.layer_norm(input, self.weight.shape, self.weight
                            , self.bias, self.eps)



# 这里是把所有的attention的部分合并，最后显示出一个因果自注意机制
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 这里是因为后续会做多头处理
        assert config.n_embed % config.n_head == 0

        # key, query, value projection for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        # flash attention make GPU go brrrr
        # hasattr本质上就是检查这个模块里面有没有这个函数
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print('Flash Attention requires Pytorch >= 2.0')
            # 引用父函数把他注册到模型的buffers属性里面（）并且命名为bias，这表示bias是一个持久态或者说常数，不会跟着一起更新
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).
             view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size()

        # Calculate query , key, values for all heads in batch and move head forward to the batch dim
        # 注意这里是输入了X了, 所以第一次相乘后的维度为（B, T, embed * 3）
        q, k, v = self.c_attn.split(self.n_embed, dim=2)

        # 分为多个头最终产生最后的效果，多头的使用不是每个K，V，Q负责所有模块而是负责一部分
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention: Self-attend: (B, nh, T, hs) x (B, nh, ns, T) -> (B, nh, T, T)
        if self.flash:
            # causal self-attention using Flash Attention CUDA kernels
            # 这个is_causal能够自动产生mask矩阵
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1])))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, -float('inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) X (B , nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection--一个残差连接
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        """
        def gelu(x):
            return 0.5 * x * (1 + nn.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        """
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config. n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config


        #
        self.transformers = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            # 正好和之前的max_block_embedding对应
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed, bias=config.bias),
        ))

        # 这个lm_head可以理解为选择的自回归为下流任务
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # with weight trying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in the future version."
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # 自回归属于单独的一个模块

        self.transformers.wte.weight = self.lm_head.weight

        # init all weights--在huggingface的库里面也会产生这种问题，也就是扩大在fine-tuning的时候需要初始化，原因是他的过程是先
        # 初始化再选择pretrain的覆盖，这样可以避免部分参数没有成功初始化
        # apply是遍历模型的所有权重
        self.apply(self._init_weights)

        # apply special scaled init to the residual projection, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))


    def get_num_params(self, non_embedding=True):
        """
        Return the number of the parameters in the model.
        Fro non-embedding count(default), the position embedddings get substracted.
        The token embeddings would too, except due to the parameter sharing these params
        are actually used as weights in the final layer, so we include them.
        """
        # numel返回张量的总数
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformers.wpe.weight.numel()
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这个可以尝试xavier  https://zhuanlan.zhihu.com/p/648576849
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding)        :
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, target=None):
        device = idx.device
        # 这里传入的是tokenizer后的序列
        b, t = idx.shape
        assert t<= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        token_embed = self.transformers.wte(idx)
        pos_embed = self.transformers.wpe(pos)

        x = self.transformers.drop(token_embed+pos_embed)
        for block in self.transformers.h:
            x = block(x)
        x = self.transformers.ln_f(x)

        if target is None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # 这个ignore index处理的是类似于padding之类的，这里是padding为-1
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1),ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last postion
            # 这里只是用最后一个token进行预测，上一个token已经通过MLA获取到了前面所有token的信息，是一个具像化的表现
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g.we may load the GPT2 pretrained model checkpoint (block_size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.block_size
        self.config.block_size = block_size
        self.transformers.wpe.weight = self.transformers.wpe.weight[:block_size]

        for block in self.transformers.h:
            if hasattr(block.attn, 'bias'):
                # 见上述：self.register_buffer    ？这个作用
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    # 这个表示这个方法是和类绑定的而不是和实体类绑定的--可以通过这个方法直接修改到class的内部属性
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert  model_type in {'gpt', 'gpt2-medium', 'gpt2-large', 'gpt-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" %model_type)


        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f'overriding dropout rate to {override_args['dropout']}')
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized miniGpt model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()


        # copy white ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
