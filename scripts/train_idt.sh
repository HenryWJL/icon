#!/bin/bash

# approximate mask
python scripts/train.py task=open_door algo=icon_diffusion_unet train.seed=0 train.device=cuda:6
python scripts/train.py task=stack_cube algo=icon_diffusion_unet train.seed=100 train.device=cuda:6 train.epoch=100 train.batch_size=16 train.val.ckpt_manager.val_freq=5 train.val.ckpt_manager.topk=20