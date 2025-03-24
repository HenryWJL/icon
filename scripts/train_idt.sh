#!/bin/bash

python scripts/train.py task=open_box algo=icon_diffusion_transformer train.seed=0 train.lr_scheduler.num_warmup_steps=1000
python scripts/train.py task=close_drawer algo=icon_diffusion_transformer train.seed=0 train.lr_scheduler.num_warmup_steps=1000
python scripts/train.py task=open_box algo=icon_diffusion_transformer train.seed=100 train.lr_scheduler.num_warmup_steps=1000
python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=close_drawer algo=icon_diffusion_transformer train.seed=100 train.lr_scheduler.num_warmup_steps=1000
# python scripts/train.py task=play_jenga algo=icon_diffusion_transformer train.seed=100
# python scripts/train.py task=open_microwave algo=icon_diffusion_transformer train.seed=100
# python scripts/train.py task=put_rubbish_in_bin algo=icon_diffusion_transformer train.seed=100
