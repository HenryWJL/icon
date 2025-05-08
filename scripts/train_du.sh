#!/bin/bash

python scripts/train.py task=lift_cube algo=diffusion_unet train.device=cuda:2 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/lift_cube/diffusion_unet/2025-03-25/12-25-26/checkpoints/600_2.06996.pth"
python scripts/train.py task=open_door algo=diffusion_unet train.device=cuda:2 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/open_door/diffusion_unet/2025-03-25/21-43-58/checkpoints/600_0.57843.pth"
python scripts/train.py task=stack_cube algo=diffusion_unet train.device=cuda:2 train.num_epochs=300 train.optimizer.learning_rate=0.00001 train.checkpoints="outputs/stack_cube/diffusion_unet/2025-03-27/23-26-22/checkpoints/600_0.2923.pth"