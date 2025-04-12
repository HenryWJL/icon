#!/bin/bash

python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_fps=false train.seed=0
python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_weighted_loss=false train.seed=0
python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_fps=false algo.policy.obs_encoder.image_encoder.backbone.enable_weighted_loss=false train.seed=0
python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_fps=false train.seed=100
python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_weighted_loss=false train.seed=100
python scripts/train.py task=open_box algo=icon_diffusion_unet algo.policy.obs_encoder.image_encoder.backbone.enable_fps=false algo.policy.obs_encoder.image_encoder.backbone.enable_weighted_loss=false train.seed=100

# python scripts/train.py task=open_box algo=icon_diffusion_unet
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.device=cuda:7
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=0 train.device=cuda:7
# python scripts/train.py task=open_box algo=icon_diffusion_unet train.seed=100 train.device=cuda:7
# python scripts/train.py task=take_lid_off_saucepan algo=icon_diffusion_unet train.seed=100 train.device=cuda:7
# python scripts/train.py task=close_drawer algo=icon_diffusion_unet train.seed=100 train.device=cuda:7