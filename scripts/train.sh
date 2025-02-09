#!/bin/bash

# python scripts/train.py task=take_lid_off_saucepan
# python scripts/train.py task=open_microwave
# python scripts/train.py task=put_rubbish_in_bin
# python scripts/train.py task=open_box

python scripts/train.py algo=r3m task=take_lid_off_saucepan
python scripts/train.py algo=r3m task=open_microwave
python scripts/train.py algo=r3m task=put_rubbish_in_bin
python scripts/train.py algo=r3m task=open_box

# python scripts/train.py algo=icon task=close_drawer
# python scripts/train.py algo=icon task=close_microwave
# python scripts/train.py algo=icon task=take_lid_off_saucepan
# python scripts/train.py algo=icon task=put_rubbish_in_bin
# python scripts/train.py algo=icon task=open_microwave
# python scripts/train.py algo=icon task=open_box