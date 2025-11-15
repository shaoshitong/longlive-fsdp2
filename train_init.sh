# #!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_init.yaml
LOGDIR=logs_fsdp2
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR