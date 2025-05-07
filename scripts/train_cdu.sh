#!/bin/bash

python scripts/train.py task=stack_cube algo=diffusion_unet train.seed=100 train.device=cuda:3 train.num_epochs=100 train.batch_size=16 train.val.ckpt_manager.val_freq=5 train.val.ckpt_manager.topk=20
python scripts/train.py task=stack_cube algo=icon_diffusion_unet train.seed=100 train.device=cuda:3 train.num_epochs=100 train.batch_size=16 train.val.ckpt_manager.val_freq=5 train.val.ckpt_manager.topk=20
python scripts/train.py task=stack_cube algo=crossway_diffusion_unet train.seed=100 train.device=cuda:3 train.num_epochs=100 train.batch_size=16 train.val.ckpt_manager.val_freq=5 train.val.ckpt_manager.topk=20
# python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.device=cuda:5
# python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.seed=0 train.device=cuda:5
# python scripts/train.py task=put_rubbish_in_bin algo=crossway_diffusion_unet train.seed=100 train.device=cuda:5