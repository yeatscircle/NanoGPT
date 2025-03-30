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
    block_size: int = 1024
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
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # with weight trying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in the future version."
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformers.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._in)




    #
    # def get_num_params(self, non_embedding=True):
    #     """
    #
    #     :param non_embedding:
    #     :return:
    #     """
    #     n_params = sum(p)
