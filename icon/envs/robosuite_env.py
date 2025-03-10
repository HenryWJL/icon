import robosuite
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Union, Tuple, Literal, List


def task_to_env_name(task: str) -> str:
    if task == 'stack_cube':
        return 'Stack'
    elif task == 'open_door':
        return 'Door'
    elif task == 'assemble_square_nut':
        return 'NutAssemblySquare'
    elif task == 'pick_place_can':
        return 'PickPlaceCan'


class RobosuiteEnv(gym.Env):

    def __init__(
        self,
        task: str,
        cameras: List,
        image_size: Optional[int] = 224,
        robot: Literal['Panda', 'Sawyer', 'UR5e', 'Kinova3', 'Jaco', 'IIWA'] = 'Panda',
        controller: Literal['OSC_POSE', 'JOINT_POSITION', 'JOINT_VELOCITY'] = 'OSC_POSE',
        render_mode: Literal['human', 'rgb_array', None] = None,
        render_camera: Union[str, None] = None,
        gpu_id: Union[int, None] = None
    ) -> None:
        self.env = robosuite.make(
            env_name=task_to_env_name(task),
            camera_heights=image_size,
            camera_widths=image_size,
            robots=robot,
            controller_configs=robosuite.load_controller_config(default_controller=controller),
            has_renderer=(render_mode == 'human'),
            has_offscreen_renderer=True,
            render_camera=render_camera,
            camera_names=['frontview', 'agentview', 'robot0_eye_in_hand'],
            use_object_obs=False,
            use_camera_obs=True,
            camera_depths=False,
            ignore_done=True,   
            control_freq=20,
            render_gpu_device_id=gpu_id
        )

        action_shape = (8,) if controller.startswith('JOINT') else (7,)
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32
        )

        obs = self.env.reset()
        observation_space = dict()
        for key, value in obs.items():
            if 'image' in key:
                observation_space[key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=value.shape, 
                    dtype=np.uint8
                )
            else:
                observation_space[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=value.shape,
                    dtype=np.float32
                )
        self.observation_space = spaces.Dict(observation_space)
        
        self.render_cache = None

    def _extract_obs()