import torch
import numpy as np
import rlbench.tasks as tasks
import rlbench.action_modes.arm_action_modes as aam
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.observation import Observation
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Union, Tuple, List
from icon.envs.base import Env


def get_task(name: str):
    if name == "close_drawer":
        return getattr(tasks, "CloseDrawer")
    elif name == "close_microwave":
        return getattr(tasks, "CloseMicrowave")
    elif name == "put_rubbish_in_bin":
        return getattr(tasks, "PutRubbishInBin")
    elif name == "play_jenga":
        return getattr(tasks, "PlayJenga")
    elif name == "take_lid_off_saucepan":
        return getattr(tasks, "TakeLidOffSaucepan")
    elif name == "open_box":
        return getattr(tasks, "OpenBox")
    elif name == "open_microwave":
        return getattr(tasks, "OpenMicrowave")


def get_arm_controller(type: str):
    if type == "joint_pos":
        return getattr(aam, "JointPosition")(True)
    elif type == "joint_vel":
        return getattr(aam, "JointVelocity")(True)
    elif type == "ee_pose":
        return getattr(aam, "EndEffectorPoseViaIK")(True)
    elif type == "ee_delta_pose":
        return getattr(aam, "EndEffectorPoseViaIK")(False)


class RLBenchEnv(Env):

    def __init__(
        self,
        task: str,
        cameras: List,
        image_size: Optional[int] = 224,
        robot: Optional[str] = 'panda',
        arm_controller: Optional[str] = 'ee_delta_pose',
        headless: Optional[bool] = True
    ) -> None:
        """
        Args:
            cameras (list): camera name(s).
                - 'left_shoulder_camera'
                - 'right_shoulder_camera'
                - 'overhead_camera'
                - 'wrist_camera'
                - 'front_camera'
            robot (str, optional): robot type.
                - 'panda'
                - 'jaco'
                - 'mico'
                - 'sawyer'
                - 'ur5'
            arm_controller (str): robotic arm controller type.
                - 'joint_pos'
                - 'joint_vel'
                - 'ee_pose'
                - 'ee_delta_pose'
        """
        super().__init__()
        all_cameras = {
            'left_shoulder_camera',
            'right_shoulder_camera',
            'overhead_camera',
            'wrist_camera',
            'front_camera'
        }
        cameras_del = list(all_cameras - set(cameras))
        camera_cfgs = {
            camera: CameraConfig(
                rgb=True,
                depth=False,
                point_cloud=False,
                mask=False,
                image_size=(image_size, image_size)
            )
            for camera in cameras
        }
        camera_del_cfgs = {
            camera: CameraConfig(
                rgb=False,
                depth=False,
                point_cloud=False,
                mask=False
            )
            for camera in cameras_del
        }
        obs_cfgs = ObservationConfig(
            **camera_cfgs,
            **camera_del_cfgs,
            joint_velocities=False,
            joint_forces=False
        )
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=get_arm_controller(arm_controller),
                gripper_action_mode=Discrete()
            ),
            obs_config=obs_cfgs,
            headless=headless,
            robot_setup=robot
        )
        self.cameras = cameras
        self.arm_controller = arm_controller
        self.task = self.env.get_task(get_task(task))

    def process_obs(self, obs: Observation) -> Dict:
        obs_dict = dict()
        # Image observations
        images = dict()
        for camera in self.cameras:
            image = getattr(obs, camera.replace('camera', 'rgb'))
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            images[camera] = image
        obs_dict['images'] = images
        # Proprioception observations
        qpos = obs.joint_positions
        ee_pose = np.concatenate([
            obs.gripper_pose[:3],
            R.from_quat(obs.gripper_pose[3:]).as_euler('xyz')
        ])
        gripper_state = np.array([obs.gripper_open], dtype=np.float32)
        proprios = np.concatenate([qpos, ee_pose, gripper_state])
        proprios = torch.from_numpy(proprios).float().unsqueeze(0)
        obs_dict['proprios'] = proprios
        return obs_dict

    def reset(self) -> Dict:
        self.task.sample_variation()
        _, obs = self.task.reset()
        obs = self.process_obs(obs)
        return obs
    
    def step(self, actions: Union[torch.Tensor, np.ndarray]) -> Tuple:
        """
        Returns:
            obs (Dict): next observations.
            rewards (float): action rewards.
            dones (bool): if True, the episode is done.
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if self.arm_controller.startswith('ee'):
            translation = actions[:3]
            rotation = R.from_euler('xyz', actions[3: 6]).as_quat()
            gripper_state = actions[6][np.newaxis]
            actions = np.concatenate([translation, rotation, gripper_state])
        obs, rewards, dones = self.task.step(actions)
        obs = self.process_obs(obs)
        return obs, rewards, dones

    def terminate(self) -> None:
        self.env.shutdown()