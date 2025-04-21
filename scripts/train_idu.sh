#!/bin/bash

python scripts/train.py task=open_door algo=diffusion_unet train.seed=0 train.device=cuda:5 train.num_epochs=1000 train.val.ckpt_manager.topk=20
python scripts/train.py task=open_door algo=icon_diffusion_unet train.seed=0 train.device=cuda:5 train.num_epochs=1000 train.val.ckpt_manager.topk=20
python scripts/train.py task=open_door algo=crossway_diffusion_unet train.seed=0 train.device=cuda:5 train.num_epochs=1000 train.val.ckpt_manager.topk=20

# python scripts/train.py task=open_box algo=icon_diffusion_unet
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.device=cuda:7
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=100 train.device=cuda:7
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=100 train.device=cuda:7
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=100 train.device=cuda:7