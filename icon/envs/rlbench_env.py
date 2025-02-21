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
from typing import Dict, Optional, Union, Tuple, List, Literal


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


class RLBenchEnv(gym.Env):
    metadata = {
        "cameras": [
            'left_shoulder_camera',
            'right_shoulder_camera',
            'overhead_camera',
            'wrist_camera',
            'front_camera'
        ],
        "robots": ['panda', 'jaco', 'mico', 'sawyer', 'ur5'],
        "action_modes": [
            'joint_pos',
            'joint_vel',
            'ee_pose',
            'delta_joint_pos',
            'delta_joint_vel',
            'delta_ee_pose'
        ],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4
    }

    def __init__(
        self, 
        task: str,
        cameras: List,
        image_size: Optional[int] = 224,
        robot: Literal['panda', 'jaco', 'mico', 'sawyer', 'ur5'] = 'panda',
        action_mode: Literal['joint_pos', 'joint_vel', 'ee_pose', 'delta_joint_pos', 'delta_joint_vel', 'delta_ee_pose'] = 'delta_ee_pose',
        render_mode: Literal['human', 'rgb_array', None] = None,
    ) -> None:
        assert all(camera in self.metadata['cameras'] for camera in cameras)
        assert robot in self.metadata['robots']
        assert action_mode in self.metadata['action_modes']
        # During rollout, we only launch the cameras we need
        self.cameras = cameras
        cameras_del = list(set(self.metadata['cameras']) - set(cameras))
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
        obs_config = ObservationConfig(**camera_cfgs, **camera_del_cfgs)
        
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
                    low=0 if str(value.dtype).startswith("uint") else -np.inf,
                    high=1 if str(value.dtype).startswith("uint") else -np.inf,
                    shape=value.shape,
                    dtype=value.dtype
                )
        self.observation_space = spaces.Dict(self.observation_space)

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.rlbench_env.action_shape,
            dtype=np.float32
        )

        self._seed = None

    def _extract_obs(self, rlbench_obs) -> Dict:
        gym_obs = {} 
        for state_name in ["joint_velocities", "joint_positions", "joint_forces", "gripper_open", "gripper_pose", "gripper_joint_positions", "gripper_touch_forces", "task_low_dim_state"]:
            state_data = getattr(rlbench_obs, state_name)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs[state_name] = state_data
        gym_obs.update({
            camera.replace('camera', 'rgb'): getattr(rlbench_obs, camera.replace('camera', 'rgb')) for camera in self.cameras
        })
        return gym_obs

    def render(self):
        if self.render_mode == 'rgb_array':
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self, seed: Union[int, None] = None, options: Union[Dict, None] = None) -> Tuple:
        super().reset(seed=seed)
        np.random.seed(seed=seed)
        _, obs = self.rlbench_task_env.reset()
        obs = self._extract_obs(obs)
        return obs

    def step(self, action: np.ndarray) -> Tuple:
        obs, reward, done = self.rlbench_task_env.step(action)
        obs = self._extract_obs(obs)
        return obs, reward, done, None

    def close(self) -> None:
        self.rlbench_env.shutdown()
        