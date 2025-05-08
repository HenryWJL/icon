#!/bin/bash

python scripts/train.py task=play_jenga algo=crossway_diffusion_unet train.device=cuda:7
python scripts/train.py task=play_jenga algo=crossway_diffusion_unet train.seed=0 train.device=cuda:7
python scripts/train.py task=play_jenga algo=crossway_diffusion_unet train.seed=100 train.device=cuda:7