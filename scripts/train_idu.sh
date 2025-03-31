#!/bin/bash

python scripts/train.py task=lift_cube algo=icon_diffusion_unet
python scripts/train.py task=open_door algo=icon_diffusion_unet
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_unet
python scripts/train.py task=stack_cube algo=icon_diffusion_unet
python scripts/train.py task=lift_cube algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=open_door algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=stack_cube algo=icon_diffusion_unet train.seed=0
python scripts/train.py task=lift_cube algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=open_door algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=pick_place_cereal algo=icon_diffusion_unet train.seed=100
python scripts/train.py task=stack_cube algo=icon_diffusion_unet train.seed=100

# python scripts/train.py task=open_box algo=icon_diffusion_unet
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=0
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=0
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=0
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=100
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=100
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=100