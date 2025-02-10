#!/bin/bash

python scripts/train.py task=open_microwave algo=ae
python scripts/train.py task=put_rubbish_in_bin algo=ae
python scripts/train.py task=open_box algo=ae
python scripts/train.py task=take_lid_off_saucepan algo=ae
python scripts/train.py task=close_drawer algo=ae
python scripts/train.py task=close_microwave algo=ae
python scripts/train.py task=play_jenga algo=ae