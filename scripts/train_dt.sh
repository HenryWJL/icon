#!/bin/bash

python scripts/train.py task=open_box algo=diffusion_transformer
python scripts/train.py task=take_lid_off_saucepan algo=diffusion_transformer
python scripts/train.py task=close_drawer algo=diffusion_transformer
python scripts/train.py task=play_jenga algo=diffusion_transformer
python scripts/train.py task=open_microwave algo=diffusion_transformer
python scripts/train.py task=put_rubbish_in_bin algo=diffusion_transformer

