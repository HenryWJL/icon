#!/bin/bash

python scripts/train.py task=close_microwave algo=diffusion_transformer train.seed=100 train.device=cuda:5 train.lr_scheduler.num_warmup_steps=500
python scripts/train.py task=close_microwave algo=icon_diffusion_transformer train.seed=100 train.device=cuda:5 train.lr_scheduler.num_warmup_steps=500