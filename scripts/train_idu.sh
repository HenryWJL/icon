#!/bin/bash

python scripts/train.py task=close_microwave algo=icon_diffusion_unet
python scripts/train.py task=close_microwave algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=close_microwave algo=icon_diffusion_unet train.seed=100