import numpy as np
import zarr
from pathlib import Path

task = "open_box"
source_dir = Path(f"../cross_embodiment/data/{task}").absolute()
target_dir = Path(f"data/{task}")
target_dir.joinpath("train").mkdir(parents=True, exist_ok=True)
target_dir.joinpath("val").mkdir(parents=True, exist_ok=True)

for episode in list(source_dir.glob("**/*.zarr")):
    with zarr.open(str(episode), 'r') as f:
        img_front = f['/images/front_camera'][()]
        img_wrist = f['/images/wrist_camera'][()]
        if 'masks' in dict(f).keys():
            mask = f['/masks/front_camera'][()]
        else:
            mask = None
        qpos = f['/joint_properties/local/joint_positions'][()]
        pose = f['/joint_properties/global/ee_pose'][()]
        gripper = f['/joint_properties/global/gripper_open'][()]
        proprio = np.concatenate([qpos, pose, gripper], axis=1)
        actions = f['/actions'][()]
    with zarr.open(str(episode).replace("cross_embodiment", "icon"), 'w') as f:
        f['/images/front_camera'] = img_front
        f['/images/wrist_camera'] = img_wrist
        if mask is not None:
            f['/masks/front_camera'] = mask
        f['/proprios'] = proprio
        f['/actions'] = actions
    print(f"{str(episode)} is done!")
