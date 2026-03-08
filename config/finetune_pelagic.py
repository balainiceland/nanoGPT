"""
Fine-tune GPT-2 124M on blue economy corpus for PelagicGPT.

Usage:
  python train.py config/finetune_pelagic.py

Estimated: ~30-60 min on MacBook MPS, ~500MB checkpoint.
"""

import time

out_dir = 'out-pelagic'
eval_interval = 10
eval_iters = 20
log_interval = 1
wandb_log = False
wandb_project = 'pelagic-gpt'
wandb_run_name = 'finetune-' + str(int(time.time()))

# Data
dataset = 'pelagic'
init_from = 'gpt2'  # GPT-2 124M base

# Only save when val loss improves
always_save_checkpoint = False

# Batch sizing:
# 1 batch_size * 32 grad_accum * 1024 block_size = 32,768 tokens/iter
# With ~2M tokens corpus, 1 epoch ≈ 61 iters
# 300 iters ≈ ~5 epochs
batch_size = 1
gradient_accumulation_steps = 32
block_size = 1024
max_iters = 300

# Fine-tuning LR (low to preserve pretrained weights)
learning_rate = 3e-5
decay_lr = False  # Constant LR for fine-tuning

# Regularization
dropout = 0.1

# Apple Silicon MPS
device = 'mps'
compile = False       # MPS doesn't support torch.compile
dtype = 'float32'     # MPS requires float32
