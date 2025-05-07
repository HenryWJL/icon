#!/bin/bash

python scripts/train.py task=close_microwave algo=diffusion_unet train.device=cuda:4
python scripts/train.py task=close_microwave algo=diffusion_unet train.seed=0 train.device=cuda:4
python scripts/train.py task=close_microwave algo=diffusion_unet train.seed=100 train.device=cuda:4
