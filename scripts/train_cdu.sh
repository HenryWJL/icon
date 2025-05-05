#!/bin/bash

python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.device=cuda:5
python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.seed=0 train.device=cuda:5
python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.seed=100 train.device=cuda:5