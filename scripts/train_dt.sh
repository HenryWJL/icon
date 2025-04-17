#!/bin/bash

python scripts/train.py task=open_door algo=crossway_diffusion_unet train.seed=0 train.device=cuda:7
python scripts/train.py task=stack_cube algo=crossway_diffusion_unet train.seed=0 train.device=cuda:7
python scripts/train.py task=lift_cube algo=crossway_diffusion_unet train.seed=100 train.device=cuda:7
python scripts/train.py task=open_door algo=crossway_diffusion_unet train.seed=100 train.device=cuda:7
python scripts/train.py task=stack_cube algo=crossway_diffusion_unet train.seed=100 train.device=cuda:7
# python scripts/train.py task=pick_place_cereal algo=diffusion_transformer
# python scripts/train.py task=open_door algo=diffusion_transformer train.seed=0
# python scripts/train.py task=pick_place_cereal algo=diffusion_transformer train.seed=0
# python scripts/train.py task=open_door algo=diffusion_transformer train.seed=100
# python scripts/train.py task=pick_place_cereal algo=diffusion_transformer train.seed=100
# python scripts/train.py task=stack_cube algo=diffusion_transformer
# python scripts/train.py task=stack_cube algo=diffusion_transformer train.seed=0
# python scripts/train.py task=stack_cube algo=diffusion_transformer train.seed=100
# python scripts/train.py task=lift_cube algo=diffusion_transformer train.seed=0
# python scripts/train.py task=lift_cube algo=diffusion_transformer train.seed=100