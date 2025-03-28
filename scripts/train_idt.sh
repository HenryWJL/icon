#!/bin/bash

python scripts/train.py task=open_box algo=icon_diffusion_transformer
python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_transformer
python scripts/train.py task=close_drawer algo=icon_diffusion_transformer
python scripts/train.py task=open_box algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=close_drawer algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=open_box algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=close_drawer algo=icon_diffusion_transformer train.seed=100
