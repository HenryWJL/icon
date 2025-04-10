#!/bin/bash

python scripts/train.py task=put_rubbish_in_bin algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=put_rubbish_in_bin algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=put_rubbish_in_bin algo=diffusion_transformer train.device=cuda:6
python scripts/train.py task=put_rubbish_in_bin algo=diffusion_transformer train.seed=0 train.device=cuda:6
python scripts/train.py task=put_rubbish_in_bin algo=diffusion_transformer train.seed=100 train.device=cuda:6
