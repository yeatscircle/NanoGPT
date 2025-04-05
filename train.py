"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import torch
import os

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText

# I/O
# 所有模型的日志以及模型的checkpoint
out_dir = 'out'
# 每过eval_interval就进行一次训练集的评估
eval_interval = 2000
# 每 1 个 iteration 打印一次训练日志（loss、lr、速度等）
log_interval = 1
# 每次 evaluation 使用 200 个 batch 来估计 eval loss（更稳定）
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# 这里分别是从头开始， 加载节点以及加载预训练模型
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


# wandb logging
# 一旦启用该项目则会传递到仪表盘上去
wandb_log = False # disabled by default
wandb_project = 'owt'
# 这个名字是静态的，如果想要每次上传的名字不唯一，则使用可以使用 wandb_run_name = 'gpt2' + str(time.time())
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes--这个是用与显存不足的情况，即每40个minibatch更新一次gradient
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size--这里是一个大batch， 即一个大的batch为5 * 8 * 12
block_size = 1024 # 每次train最大的序列参数

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer--adam使用不了L2正则
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# 梯度裁剪，防止爆炸梯度
# 若某步梯度范数 > 1，则进行缩放，使其不超过该值
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
# 随机分布的，以来上猛药容易直接让模型崩塌
warmup_iters = 2000 # how many steps to warm up for-- 前2000步逐渐线性升高学习率（warmup） ，然后逐步降低到min_lr
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
# 单卡用不了通讯，所以这个选项没用
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
