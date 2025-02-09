import zarr
import imageio
import torch
import numpy as np
from pathlib import Path
import rlbench.tasks as tasks
import rlbench.action_modes.arm_action_modes as aam
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.observation import Observation
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion as Q
from typing import Dict, Optional, Union, Tuple, List
from cross_embodiment.envs.base import Env
from cross_embodiment.utils.file_utils import str2path, mkdir


def permute(
    x: Union[List, np.ndarray],
    order: Union[Tuple, List]
) -> np.ndarray:
    """
    Permute the elements of a 1D array-like object.
    """
    if isinstance(x, List):
        assert len(x) == len(order)
        y = list()
        for o in order:
            y.append(x[o])
    elif isinstance(x, np.ndarray):
        assert len(order) == x.shape[0]
        y = np.zeros_like(x, dtype=x.dtype)
        for i, o in enumerate(order):
            y[i] = x[o]
    return y


def get_task(task_name: str):
    """
    Return a RLBench Task object. For a complete list of tasks, see
    https://github.com/stepjam/RLBench/tree/master/rlbench/tasks.
    """
    if task_name == "close_drawer":
        return getattr(tasks, "CloseDrawer")
    elif task_name == "open_door":
        return getattr(tasks, "OpenDoor")
    elif task_name == "close_microwave":
        return getattr(tasks, "CloseMicrowave")
    elif task_name == "open_grill":
        return getattr(tasks, "OpenGrill")
    elif task_name == "put_rubbish_in_bin":
        return getattr(tasks, "PutRubbishInBin")
    elif task_name == "slide_block_to_target":
        return getattr(tasks, "SlideBlockToTarget")
    elif task_name == "open_window":
        return getattr(tasks, "OpenWindow")
    elif task_name == "play_jenga":
        return getattr(tasks, "PlayJenga")
    elif task_name == "take_lid_off_saucepan":
        return getattr(tasks, "TakeLidOffSaucepan")
    elif task_name == "put_umbrella_in_umbrella_stand":
        return getattr(tasks, "PutUmbrellaInUmbrellaStand")
    elif task_name == "sweep_to_dustpan":
        return getattr(tasks, "SweepToDustpan")
    elif task_name == "open_box":
        return getattr(tasks, "OpenBox")
    elif task_name == "open_microwave":
        return getattr(tasks, "OpenMicrowave")


def get_arm_controller(controller_type: str):
    if controller_type == "joint_pos":
        return getattr(aam, "JointPosition")(True)
    elif controller_type == "joint_vel":
        return getattr(aam, "JointVelocity")(True)
    elif controller_type == "ee_pose":
        return getattr(aam, "EndEffectorPoseViaIK")(True)
    elif controller_type == "ee_delta_pose":
        return getattr(aam, "EndEffectorPoseViaIK")(False)


class RLBenchEnv(Env):

    def __init__(
        self,
        task: str,
        robot: Optional[str] = 'panda',
        image_size: Union[int, Tuple[int, int]] = 1024,
        cameras: List[str] = ['overhead_camera'],
        joint_properties: Dict = {'global': [], 'local': []},
        max_joint_dof: Optional[int] = 7,
        arm_controller: Optional[str] = 'ee_delta_pose',
        headless: Optional[bool] = True
    ) -> None:
        """
        Args:
            task (str): task name.
            robot (str): robot name.
                - 'panda'
                - 'jaco'
                - 'mico'
                - 'sawyer'
                - 'ur5'
            image_size (int or tuple): size of RGB image.
            cameras (List): camera name(s).
                - 'left_shoulder_camera'
                - 'right_shoulder_camera'
                - 'overhead_camera'
                - 'wrist_camera'
                - 'front_camera'
            joint_properties (Dict): global and local joint properties.

                Example:
                >>> joint_properties = {
                    'global': ['ee_pose', 'gripper_open'],
                    'local': ['joint_positions', 'joint_velocities', 'joint_forces']
                }
            
            arm_controller (str): controller of robot arm.
                - 'joint_pos'
                - 'joint_vel'
                - 'ee_pose'
                - 'ee_delta_pose'
            headless (bool): if True, running headlessly.
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        all_cameras = {
            'left_shoulder_camera',
            'right_shoulder_camera',
            'overhead_camera',
            'wrist_camera',
            'front_camera'
        }
        del_cameras = list(all_cameras - set(cameras))
        camera_configs = {
            camera: CameraConfig(
                rgb=True,
                depth=False,
                point_cloud=False,
                mask=False,
                image_size=image_size
            )
            for camera in cameras
        }
        del_camera_configs = {
            camera: CameraConfig(
                rgb=False,
                depth=False,
                point_cloud=False,
                mask=False
            )
            for camera in del_cameras
        }
        obs_config = ObservationConfig(**camera_configs, **del_camera_configs)
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=get_arm_controller(arm_controller),
                gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=headless,
            robot_setup=robot
        )
        self.cameras = cameras
        self.joint_properties = joint_properties
        self.max_joint_dof = max_joint_dof
        self.arm_controller = arm_controller
        self.task = self.env.get_task(get_task(task))

    def _process_obs(self, obs: Observation) -> Dict:
        """
        Convert raw observations into dictionaries for downstream processing. 

        Returns:
            obs_dict (Dict): a dictionary of observations.

            Example:
            >>> obs = {
                    'images': {
                        'wrist_camera': tensor (1, 3, height, width),
                        'front_camera': tensor (1, 3, height, width),
                        ...
                    },
                    'joint_properties': tensor (1, global_dim + local_dim * max_dof),
                    'joint_padding_masks': tensor (1, max_dof)
                }
            >>> obs = {
                    'images' (only one camera): tensor (1, 3, height, width),
                    'joint_properties': tensor (1, global_dim + local_dim * max_dof),
                    'joint_padding_masks': tensor (1, max_dof)
                }
        """
        obs_dict = dict()
        # RGB observations
        obs_dict['images'] = {
            camera: torch.from_numpy(
                getattr(obs, camera.replace("camera", "rgb"))
            ).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            for camera in self.cameras
        }
        if len(self.cameras) == 1:
            obs_dict['images'] = obs_dict['images'][self.cameras[0]]
        # Proprioception observations
        joint_properties = list()
        if 'ee_pose' in self.joint_properties['global']:
            ee_pose = np.concatenate([
                obs.gripper_pose[:3],
                R.from_quat(obs.gripper_pose[3:]).as_euler('xyz')
            ])
            joint_properties.append(ee_pose)
        if 'gripper_open' in self.joint_properties['global']:
            gripper_open = np.array([obs.gripper_open], dtype=np.float32)
            joint_properties.append(gripper_open)
        if len(self.joint_properties['local']) > 0:
            local_joint_properties = [
                getattr(obs, joint_property)
                for joint_property in self.joint_properties['local']
            ]
            local_joint_properties = np.stack(local_joint_properties, axis=-1)
            joint_dof, local_property_dim = local_joint_properties.shape
            if joint_dof < self.max_joint_dof:
                local_joint_properties = np.concatenate([
                    local_joint_properties,
                    np.full(
                        (self.max_joint_dof - joint_dof, local_property_dim),
                        -1,
                        dtype=np.float32
                    )
                ])
            joint_padding_masks = (torch.arange(self.max_joint_dof) >= joint_dof).unsqueeze(0)
            joint_properties.append(local_joint_properties.reshape(-1))
        else:
            joint_padding_masks = None
        if len(joint_properties) > 0:
            joint_properties = torch.from_numpy(
                np.concatenate(joint_properties)
            ).float().unsqueeze(0)
        else:
            joint_properties = None
        obs_dict['joint_properties'] = joint_properties
        if joint_padding_masks is not None:
            obs_dict['joint_padding_masks'] = joint_padding_masks
        return obs_dict

    def reset(self) -> Dict:
        self.task.sample_variation()
        _, obs = self.task.reset()
        return self._process_obs(obs)
    
    def step(self, actions: Union[torch.Tensor, np.ndarray]) -> Tuple:
        """
        Args:
            actions (torch.Tensor or np.ndarray): actions to perform (action_dim,).

        Returns:
            obs (Dict): next observations.
            rewards (float): rewards obtained by stepping @actions.
            dones (bool): a flag indicating whether the rollout is terminated.
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if self.arm_controller == "ee_pose" or self.arm_controller == "ee_delta_pose":
            translation = actions[:3]
            rotation = R.from_euler('xyz', actions[3: 6]).as_quat()
            gripper_open = actions[6][np.newaxis]
            actions = np.concatenate([translation, rotation, gripper_open])
        obs, rewards, dones = self.task.step(actions)
        return self._process_obs(obs), rewards, dones

    def get_demo(self, save_dir: Union[str, Path]) -> None:
        """
        Sample one episode of @task and store it in @save_dir.

        Episode storing structure:
            @save_dir (folder):
                videos (folder):
                    ${camera_name}.mp4
                masks (folder):
                    (add segmented masks here)
                states.zarr (file):
                    'joint_properties':
                        'global':
                            'ee_pose': array,
                            ...
                        'local':
                            'joint_positions': array,
                            'joint_velocities': array,
                            ...,
                    'actions': array 
        """
        save_dir = str2path(save_dir)
        self.reset()
        demo = self.task.get_demos(1, live_demos=True)[0]
        # Videos
        videos_save_dir = mkdir(save_dir.joinpath("videos"))
        mkdir(save_dir.joinpath("masks"))
        for camera in self.cameras:
            image_frames = np.stack([
                getattr(obs, camera.replace("camera", "rgb")) for obs in demo
            ])
            # Note that 24 is the maximum video fps supported by SAM2
            writer = imageio.get_writer(str(videos_save_dir.joinpath(f"{camera}.mp4")), fps=24)
            for i in range(image_frames.shape[0]):
                writer.append_data(image_frames[i])
            writer.close()
        # Robot joint properties and actions
        with zarr.open(str(save_dir.joinpath("states.zarr")), 'w') as f:
            ee_pose = np.stack([obs.gripper_pose for obs in demo])  # x, y, z, qx, qy, qz, qw
            ee_pose_euler = np.concatenate([
                ee_pose[:, :3],
                R.from_quat(ee_pose[:, 3:]).as_euler('xyz')
            ], axis=1)
            gripper_open = np.stack([obs.gripper_open for obs in demo])[..., np.newaxis]
            
            if 'ee_pose' in self.joint_properties['global']:
                f[f"/joint_properties/global/ee_pose"] = ee_pose_euler
            if 'gripper_open' in self.joint_properties['global']:
                f[f"/joint_properties/global/gripper_open"] = gripper_open
            if len(self.joint_properties['local']) > 0:
                for joint_property in self.joint_properties['local']:
                    f[f"/joint_properties/local/{joint_property}"] = np.stack(
                        [getattr(obs, joint_property) for obs in demo]
                    )
            
            if self.arm_controller == "joint_pos":
                actions = np.stack([obs.joint_positions for obs in demo])
            elif self.arm_controller == "joint_vel":
                actions = np.stack([obs.joint_velocities for obs in demo])
            elif self.arm_controller == "ee_pose":
                actions = ee_pose_euler
            elif self.arm_controller == "ee_delta_pose":
                delta_ee_pose = list()
                for i in range(ee_pose.shape[0]):
                    if i < ee_pose.shape[0] - 1:
                        # Note that scipy.spatial.transform.Rotation.from_quat() receives [qx, qy, qz, qw],
                        # while pyquaternion.Quaternion receives [qw, qx, qy, qz].
                        current_pose = permute(ee_pose[i], (0, 1, 2, 6, 3, 4, 5))  # x, y, z, qw, qx, qy, qz
                        next_pose = permute(ee_pose[i + 1], (0, 1, 2, 6, 3, 4, 5))  # x, y, z, qw, qx, qy, qz
                        delta_translation = next_pose[:3] - current_pose[:3]
                        delta_rotation = list(Q(next_pose[3:]) / Q(current_pose[3:]))  # qw, qx, qy, qz
                        delta_rotation = permute(delta_rotation, (1, 2, 3, 0))  # qx, qy, qz, qw
                        delta_pose = np.concatenate(
                            [delta_translation, R.from_quat(delta_rotation).as_euler('xyz')]
                        )
                        delta_ee_pose.append(delta_pose)
                    else:
                        delta_ee_pose.append(np.zeros((6,), dtype=np.float32))
                actions = np.stack(delta_ee_pose)

            actions = np.concatenate([actions, gripper_open], axis=1)
            f["/actions"] = actions

    def terminate(self) -> None:
        self.env.shutdown()