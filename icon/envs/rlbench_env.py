import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity, EndEffectorPoseViaIK
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.utils import name_to_task_class
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Union, Tuple, Literal


def name_to_action_mode(name: str):
    absolute_mode = True
    if name.startswith("delta"):
        absolute_mode = False
    if name.endswith("joint_pos"):
        return JointPosition(absolute_mode)
    elif name.endswith("joint_vel"):
        return JointVelocity(absolute_mode)
    elif name.endswith("ee_pose"):
        return EndEffectorPoseViaIK(absolute_mode)

# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/gym.py
class RLBenchEnv(gym.Env):
    metadata = {
        'robots': ['panda', 'jaco', 'mico', 'sawyer', 'ur5'],
        'action_modes': [
            'joint_pos',
            'joint_vel',
            'ee_pose',
            'delta_joint_pos',
            'delta_joint_vel',
            'delta_ee_pose'
        ],
        'render_modes': ['human', 'rgb_array']
    }

    def __init__(
        self, 
        task: str,
        image_size: Optional[int] = 224,
        robot: Literal['panda', 'jaco', 'mico', 'sawyer', 'ur5'] = 'panda',
        action_mode: Literal['joint_pos', 'joint_vel', 'ee_pose', 'delta_joint_pos', 'delta_joint_vel', 'delta_ee_pose'] = 'delta_ee_pose',
        render_mode: Literal['human', 'rgb_array', None] = None,
    ) -> None:
        assert robot in self.metadata['robots']
        assert action_mode in self.metadata['action_modes']

        camera_config = CameraConfig(
            rgb=True,
            depth=False,
            point_cloud=False,
            mask=False,
            image_size=(image_size, image_size)
        )
        obs_config = ObservationConfig(
            left_shoulder_camera=copy.copy(camera_config),
            right_shoulder_camera=copy.copy(camera_config),
            overhead_camera=copy.copy(camera_config),
            wrist_camera=copy.copy(camera_config),
            front_camera=copy.copy(camera_config)
        )
        obs_config.set_all(True)

        self.action_mode = action_mode
        action_mode = MoveArmThenGripper(
            arm_action_mode=name_to_action_mode(action_mode),
            gripper_action_mode=Discrete()
        )

        self.rlbench_env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True,
            robot_setup=robot
        )
        self.rlbench_env.launch()
        self.rlbench_task_env = self.rlbench_env.get_task(name_to_task_class(task))
        
        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"]
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([512, 512])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)
        self.render_mode = render_mode

        _, obs = self.rlbench_task_env.reset()
        gym_obs = self._extract_obs(obs)
        self.observation_space = {}
        for key, value in gym_obs.items():
            if "rgb" in key:
                self.observation_space[key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=value.shape, 
                    dtype=value.dtype
                )
            else:
                self.observation_space[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=value.shape,
                    dtype=np.float32
                )
        self.observation_space = spaces.Dict(self.observation_space)
        
        # We use euler angles instead of quaternions to represent rotations
        action_shape = (7,) if self.action_mode.endswith("ee_pose") else self.rlbench_env.action_shape
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32
        )

    def _extract_obs(self, rlbench_obs):
        gym_obs = {} 
        for state_name in ["joint_velocities", "joint_positions", "joint_forces", "gripper_open", "gripper_pose", "gripper_joint_positions", "gripper_touch_forces", "task_low_dim_state"]:
            state_data = getattr(rlbench_obs, state_name)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs[state_name] = state_data
                
        gym_obs.update({
            "left_shoulder_rgb": rlbench_obs.left_shoulder_rgb,
            "right_shoulder_rgb": rlbench_obs.right_shoulder_rgb,
            "wrist_rgb": rlbench_obs.wrist_rgb,
            "front_rgb": rlbench_obs.front_rgb,
            "overhead_rgb": rlbench_obs.overhead_rgb
        })
        return gym_obs

    def render(self):
        if self.render_mode == 'rgb_array':
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self, seed: Union[int, None] = None, options: Union[Dict, None] = None) -> Dict:
        super().reset(seed=seed, options=options)
        np.random.seed(seed=seed)
        _, obs = self.rlbench_task_env.reset()
        return self._extract_obs(obs)

    def step(self, action: np.ndarray) -> Tuple:
        if self.action_mode.endswith("ee_pose"):
            translation = action[:3]
            rotation = R.from_euler('xyz', action[3: 6]).as_quat()
            gripper_state = action[6][np.newaxis]
            action = np.concatenate([translation, rotation, gripper_state])
        obs, reward, terminated = self.rlbench_task_env.step(action)
        return self._extract_obs(obs), reward, terminated, {}

    def close(self) -> None:
        self.rlbench_env.shutdown()