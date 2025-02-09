#!/bin/bash

python scripts/train.py task=open_microwave algo=raw
python scripts/train.py task=put_rubbish_in_bin algo=raw
python scripts/train.py task=open_box algo=raw
python scripts/train.py task=take_lid_off_saucepan algo=raw
python scripts/train.py task=close_drawer algo=raw
python scripts/train.py task=close_microwave algo=raw
python scripts/train.py task=play_jenga algo=raw