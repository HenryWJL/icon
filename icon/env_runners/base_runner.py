import tqdm
import torch
import numpy as np
import gymnasium as gym
from typing import Optional, Union, Dict, List
from icon.utils.gym_utils.multistep_wrapper import MultiStepWrapper
from icon.utils.gym_utils.video_recording_wrapper import VideoRecordingWrapper
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
        env = VideoRecordingWrapper(
            env=env,
            video_save_dir=video_save_dir
        )
        env = MultiStepWrapper(
            env=env,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            max_episode_steps=max_episode_steps,
            enable_temporal_ensemble=True
        )
        self.env = env
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
                action = action.detach().to('cpu').squeeze(0).numpy().astype(np.float64)
                # # Due to precision loss in the process of converting absolute actions
                # # to delta actions, we need to add an extra term to the predicted actions.
                # drift = 1e-4
                # action += drift
                # action[action >= 0] += drift
                # action[action < 0] -= drift
                # print("original: ", action[0])
                # action[:, 2] += (action[:, 2] > 0).astype(np.float64) * 5e-3
                # print("new: ", action[0])
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