import tqdm
import torch
import numpy as np
import gymnasium as gym
from typing import Optional, Union, Dict, List
from icon.utils.gym_utils.multistep_wrapper import MultiStepWrapper
from icon.utils.gym_utils.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from icon.utils.file_utils import str2path
from icon.utils.pytorch_utils import to


class EnvRunner:

    def __init__(
        self,
        env: gym.Env,
        obs_horizon: int,
        action_horizon: int,
        cameras: List,
        max_episode_steps: Optional[int] = 200,
        num_trials: Optional[int] = 50,
        initial_seed: Optional[int] = 10000,
        video_save_dir: Union[str, None] = None
    ) -> None:
        if video_save_dir is None:
            video_save_dir = str2path("videos")
        else:
            video_save_dir = str2path(video_save_dir)
    
        self.env = MultiStepWrapper(
            env=env,
            # VideoRecordingWrapper(
            #     env=env,
            #     video_recoder=VideoRecorder.create_h264(
            #         fps=fps,
            #         codec='h264',
            #         input_pix_fmt='rgb24',
            #         crf=22,
            #         thread_type='FRAME',
            #         thread_count=1
            #     ),
            #     file_path="video.mp4",
            #     steps_per_render=max(10 // fps, 1)
            # ),
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_episode_steps=max_episode_steps,
            enable_temporal_ensemble=True
        )
        
        self.cameras = cameras
        self.max_episode_steps = max_episode_steps
        self.num_trials = num_trials
        self.initial_seed = initial_seed
    
    def _process_obs(self, obs: Dict) -> Dict:
        """
        Process raw observations from gym environments such that
        their formats satisfy the requirements of policies.
        """
        raise NotImplementedError
    
    def run(self, policy, device: torch.device) -> None:
        for t in range(self.num_trials):
            seed = self.initial_seed + t
            obs = self.env.reset(seed=seed)
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Trial {t + 1}/{self.num_trials}", 
                leave=False,
                mininterval=5.0
            )
            done = False
            while not done:
                obs_dict = self._process_obs(obs)
                to(obs_dict, device)
                with torch.no_grad():
                    action = policy.predict_action(obs_dict)['actions']
                action = action.detach().to('cpu').squeeze(0).numpy()
                # # Due to precision loss in the process of converting absolute actions
                # # to delta actions, we need to add an extra term to the predicted actions.
                # drift = 5e-4
                # action = action + drift
                obs, reward, done, _ = self.env.step(action)
                self.env.render()
                done = np.all(done)
                if self.env.enable_temporal_ensemble:
                    pbar.update(1)
                else:
                    pbar.update(action.shape[1])
            pbar.close()
        # Clear out video buffer
        self.env.reset()