#!/bin/bash

python scripts/train.py task=open_microwave algo=sear
python scripts/train.py task=put_rubbish_in_bin algo=sear
python scripts/train.py task=open_box algo=sear
python scripts/train.py task=take_lid_off_saucepan algo=sear
python scripts/train.py task=close_drawer algo=sear
python scripts/train.py task=close_microwave algo=sear
python scripts/train.py task=play_jenga algo=sear