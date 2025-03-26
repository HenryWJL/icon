#!/bin/bash

python scripts/train.py task=pick_place_cereal algo=diffusion_unet
python scripts/train.py task=stack_cube algo=diffusion_unet
python scripts/train.py task=lift_cube algo=diffusion_unet train.seed=0
python scripts/train.py task=open_door algo=diffusion_unet train.seed=0
python scripts/train.py task=pick_place_cereal algo=diffusion_unet train.seed=0
python scripts/train.py task=stack_cube algo=diffusion_unet train.seed=0
python scripts/train.py task=lift_cube algo=diffusion_unet train.seed=100
python scripts/train.py task=open_door algo=diffusion_unet train.seed=100
python scripts/train.py task=pick_place_cereal algo=diffusion_unet train.seed=100
python scripts/train.py task=stack_cube algo=diffusion_unet train.seed=100