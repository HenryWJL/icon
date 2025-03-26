#!/bin/bash

python scripts/train.py task=open_door algo=icon_diffusion_transformer
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_transformer
python scripts/train.py task=stack_cube algo=icon_diffusion_transformer
python scripts/train.py task=open_door algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=stack_cube algo=icon_diffusion_transformer train.seed=0
python scripts/train.py task=open_door algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_transformer train.seed=100
python scripts/train.py task=stack_cube algo=icon_diffusion_transformer train.seed=100
