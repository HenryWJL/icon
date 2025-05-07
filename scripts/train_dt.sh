#!/bin/bash

python scripts/train.py task=close_microwave algo=diffusion_transformer train.device=cuda:5
python scripts/train.py task=close_microwave algo=diffusion_transformer train.seed=0 train.device=cuda:5
python scripts/train.py task=close_microwave algo=diffusion_transformer train.seed=100 train.device=cuda:5