#!/bin/bash

python scripts/train.py task=lift_cube algo=icon_diffusion_unet train.device=cuda:3 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/lift_cube/icon_diffusion_unet/2025-04-01/10-46-19/checkpoints/600_2.04608.pth"
python scripts/train.py task=open_door algo=icon_diffusion_unet train.device=cuda:3 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/open_door/icon_diffusion_unet/2025-04-04/11-39-10/checkpoints/600_0.75956.pth"
python scripts/train.py task=stack_cube algo=icon_diffusion_unet train.device=cuda:3 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/stack_cube/icon_diffusion_unet/2025-04-05/22-55-40/checkpoints/600_0.39937.pth"