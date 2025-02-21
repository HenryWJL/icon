import torch
import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation as R
from icon.env_runners.base_runner import EnvRunner


class RLBenchRunner(EnvRunner):
    
    def _process_obs(self, obs: Dict) -> Dict:
        obs_dict = dict()
        # RGB observations
        images = dict()
        for camera in self.cameras:
            image = obs[camera.replace('camera', 'rgb')]
            image = torch.from_numpy(image).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            images[camera] = image
        obs_dict['images'] = images
        # Low-dimentional observations
        qpos = obs['joint_positions']
        ee_pose = obs['gripper_pose']
        gripper_state = obs['gripper_open']
        ee_pose = np.concatenate([
            ee_pose[:, :3],
            R.from_quat(ee_pose[:, 3:]).as_euler('xyz')
        ], axis=1)
        low_dims = np.concatenate([qpos, ee_pose, gripper_state], axis=1)
        low_dims = torch.from_numpy(low_dims).float().unsqueeze(0)
        obs_dict['low_dims'] = low_dims
        return obs_dict