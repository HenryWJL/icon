#!/bin/bash

python scripts/train.py task=lift_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.checkpoints="outputs/lift_cube/diffusion_unet/2025-03-25/12-25-26/checkpoints/600_2.06996.pth"
python scripts/train.py task=stack_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.checkpoints="outputs/stack_cube/diffusion_unet/2025-03-27/23-26-22/checkpoints/600_0.2923.pth"
python scripts/train.py task=lift_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.seed=0 train.checkpoints="outputs/lift_cube/diffusion_unet/2025-03-28/07-47-18/checkpoints/600_2.0699.pth"
python scripts/train.py task=stack_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.seed=0 train.checkpoints="outputs/stack_cube/diffusion_unet/2025-03-31/05-16-07/checkpoints/600_0.34582.pth"
python scripts/train.py task=lift_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.seed=100 train.checkpoints="outputs/lift_cube/diffusion_unet/2025-03-29/22-45-53/checkpoints/600_1.79152.pth"
python scripts/train.py task=stack_cube algo=diffusion_unet train.device=cuda:4 train.num_epochs=300 train.seed=100 train.checkpoints="outputs/stack_cube/diffusion_unet/2025-03-31/05-16-07/checkpoints/600_0.34582.pth"
python scripts/train.py task=open_door algo=diffusion_unet train.device=cuda:4 train.num_epochs=600 train.checkpoints="outputs/open_door/diffusion_unet/2025-03-25/21-43-58/checkpoints/600_0.57843.pth"
python scripts/train.py task=open_door algo=diffusion_unet train.device=cuda:4 train.num_epochs=600 train.seed=0 train.checkpoints="outputs/open_door/diffusion_unet/2025-03-28/14-11-07/checkpoints/600_0.55087.pth"
python scripts/train.py task=open_door algo=diffusion_unet train.device=cuda:4 train.num_epochs=600 train.seed=100 train.checkpoints="outputs/open_door/diffusion_unet/2025-03-30/05-17-29/checkpoints/600_0.59035.pth"
