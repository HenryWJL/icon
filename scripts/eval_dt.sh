#!/bin/bash

python scripts/eval_sim_robot.py -t stack_cube -a icon_diffusion_unet -nt 50
python scripts/eval_sim_robot.py -t stack_cube -a diffusion_unet -nt 50