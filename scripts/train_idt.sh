#!/bin/bash

python scripts/train.py task=close_microwave algo=icon_diffusion_transformer
python scripts/train.py task=close_microwave algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=close_microwave algo=icon_diffusion_transformer train.seed=100