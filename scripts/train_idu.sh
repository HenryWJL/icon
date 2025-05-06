#!/bin/bash

# noisy mask
python scripts/train.py task=open_door algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
python scripts/train.py task=stack_cube algo=crossway_diffusion_unet train.seed=100 train.device=cuda:7 train.epoch=100 train.batch_size=16 train.val.ckpt_manager.val_freq=5 train.val.ckpt_manager.topk=20