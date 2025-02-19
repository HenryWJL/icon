import zarr
import torch
import numpy as np
from torch import nn
from copy import copy
from torch.utils.data import Dataset
from torchvision import transforms as tfs
from typing import Optional, Tuple, Dict, List
from icon.utils.normalizer import Normalizer
from icon.utils.replay_buffer import ReplayBuffer
from icon.utils.file_utils import str2path
from icon.utils.sampler import SequenceSampler


class EpisodicDataset(Dataset):

    def __init__(
        self,
        zarr_path: str,
        cameras: List,
        prediction_horizon: int,
        obs_horizon: int, 
        action_horizon: int,
        image_mask_keys: Optional[List] = list()
    ) -> None:
        super().__init__()
        image_keys = [f'{camera}_images' for camera in cameras]
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['low_dims', 'actions'] + image_keys + image_mask_keys
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=prediction_horizon,
            pad_before=obs_horizon - 1, 
            pad_after=action_horizon - 1,
        )
        self.cameras = cameras
        self.obs_horizon = obs_horizon

    def get_normalizer(self) -> Normalizer:
        data = dict(
            low_dims=torch.from_numpy(self.replay_buffer['low_dims']).float(),
            actions=torch.from_numpy(self.replay_buffer['actions']).float()
        )
        mode = dict(
            low_dims='max_min',
            actions='max_min'
        )
        normalizer = Normalizer()
        normalizer.fit(data, mode)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.sampler.sample_sequence(idx)
        # Raw observations are (prediction_horizon, ...)
        low_dims = torch.from_numpy(sample['low_dims'][:self.obs_horizon]).float()
        actions = torch.from_numpy(sample['actions']).float()
        images = dict()
        image_masks = dict()
        for camera in self.cameras:
            images[camera] = torch.from_numpy(sample[f'{camera}_images'][:self.obs_horizon]).permute(0, 3, 1, 2) / 255.0
            if f'{camera}_masks' in sample.keys():
                image_masks[camera] = torch.from_numpy(sample[f'{camera}_masks'][:self.obs_horizon]).float()

        data = dict(
            obs=dict(
                images=images,
                low_dims=low_dims
            ),
            actions=actions
        )
        if any(image_masks):
            data['image_masks'] = image_masks
        return data
    

# class EpisodicDataset(Dataset):
    
#     def __init__(
#         self,
#         episode_dir: str,
#         cameras: List,
#         action_horizon: List,
#         norm_modes: Dict,
#         transform_cfg: Optional[Dict] = dict(),
#     ) -> None:
#         """
#         Args:
#             episode_dir (str): directory where episodes (zarr files) are stored.
                
#                 >>> Format:
#                 {
#                     'images': {
#                         'front_camera': array (episode_len, height, width, 3),
#                         'wrist_camera': array (episode_len, height, width, 3)
#                     },
#                     'masks': {
#                         'front_camera': array (episode_len, height, width)
#                     },
#                     'proprios': array (episode_len, proprio_dim),
#                     'actions': array (episode_len, action_dim)
#                 }

#             norm_modes (dict): normalization modes.
#             transform_cfg (dict): configuration of image transformations.
#         """
#         super().__init__()
#         self.episode_paths = list(str2path(episode_dir).glob("**/*.zarr"))
#         assert len(self.episode_paths) != 0, f"No episodes found in directory {episode_dir}!"
#         self.cameras = cameras
#         self.action_horizon = action_horizon
#         self.norm_modes = norm_modes
#         self.transform_cfg = transform_cfg
#         self.episode_lens = list()
#         self.cumulative_episode_lens = list()
#         self.transforms = nn.Identity()
#         self.normalizer = Normalizer()
#         self.fit()
#         self.configure_transforms()

#     def fit(self) -> None:
#         """
#         Fit a normalizer and extract episodes' metadata.
#         """
#         proprios = list()
#         actions = list()
#         episode_lens = list()
#         for episode_path in self.episode_paths:
#             with zarr.open(episode_path, 'r') as f:
#                 proprio = f['/proprios'][()]
#                 action = f['/actions'][()]
#                 proprios.append(proprio)
#                 actions.append(action)
#                 episode_lens.append(action.shape[0])
#         proprios = torch.from_numpy(np.concatenate(proprios)).float()
#         actions = torch.from_numpy(np.concatenate(actions)).float()
#         data = dict(
#             proprios=proprios,
#             actions=actions
#         )
#         self.normalizer.fit(data, self.norm_modes)    
#         self.episode_lens = episode_lens
#         self.cumulative_episode_lens = np.cumsum(episode_lens)

#     def configure_transforms(self) -> None:
#         resize_shape = self.transform_cfg.get('resize_shape')
#         crop_shape = self.transform_cfg.get('crop_shape')
#         random_crop = self.transform_cfg.get('random_crop', False)
#         enable_imagenet_norm = self.transform_cfg.get('enable_imagenet_norm', False)
#         transforms = list()
#         if resize_shape is not None:
#             transforms.append(tfs.Resize(resize_shape, antialias=True))
#         if crop_shape is not None:
#             if random_crop:
#                 transforms.append(tfs.RandomCrop((crop_shape, crop_shape)))
#             else:
#                 transforms.append(tfs.CenterCrop((crop_shape, crop_shape)))
#         if enable_imagenet_norm:
#             transforms.append(
#                 tfs.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             )
#         self.transforms = nn.Sequential(*transforms)

#     def _locate_transition(self, index: int) -> Tuple:
#         """
#         Locate current episode and time step. This is necessary when
#         more than one zarr files are stored in the directory, since
#         we need to know the relative position of current index w.r.t
#         the whole episodes.
#         """
#         assert index < self.cumulative_episode_lens[-1]
#         episode_index = np.argmax(self.cumulative_episode_lens > index)
#         current_timestep = index - (self.cumulative_episode_lens[episode_index] \
#                             - self.episode_lens[episode_index])
#         return episode_index, current_timestep

#     def __getitem__(self, index: int) -> Tuple:
#         episode_index, timestep = self._locate_transition(index)
#         episode_path = self.episode_paths[episode_index]
#         with zarr.open(episode_path, 'r') as f:
#             # Images
#             images = dict()
#             for camera in self.cameras:
#                 image = f[f'/images/{camera}'][()][timestep]
#                 image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
#                 image = self.transforms(image)
#                 images[camera] = image
#             # Proprioception
#             proprios = torch.from_numpy(f['/proprios'][()][timestep]).float()
#             # Actions
#             actions = torch.from_numpy(f['/actions'][()]).float()
#             H = self.action_horizon
#             actions_padded = torch.cat([actions, actions[-1:].repeat(H - 1, 1)])
#             actions = actions_padded[timestep: (timestep + H)]
            
#             items = dict(
#                 images=images,
#                 proprios=proprios,
#                 actions=actions
#             )
#             items = self.normalizer.normalize(items)
#             return items
               
#     def __len__(self) -> int:
#         return self.cumulative_episode_lens[-1]
    
#     def set_normalizer(self, normalizer: Normalizer) -> None:
#         self.normalizer = normalizer

#     def get_normalizer(self) -> Normalizer:
#         return self.normalizer
    

# class EpisodicDatasetWithMask(EpisodicDataset):

#     def __getitem__(self, index: int) -> Tuple:
#         """
#         obs:
#             images:
#             low_dims:
#         image_masks:
#         actions:
#         """
#         episode_index, timestep = self._locate_transition(index)
#         episode_path = self.episode_paths[episode_index]
#         with zarr.open(episode_path, 'r') as f:
#             # Image masks
#             masks = dict()
#             for camera in self.cameras:
#                 if camera in dict(f['/masks']).keys():
#                     mask = f[f'/masks/{camera}'][()][timestep]
#                     mask = torch.from_numpy(mask).float()
#                     masks[camera] = mask
#             # Images
#             images = dict()
#             for camera in self.cameras:
#                 image = f[f'/images/{camera}'][()][timestep]
#                 image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
#                 # Image transformations. Note that image masks also require the same transformation.
#                 mask = masks.get(camera)
#                 if mask is None:
#                     image = self.transforms(image)
#                     images[camera] = image
#                 else:
#                     mask = mask.unsqueeze(0).repeat(3, 1, 1)
#                     image_mask_stack = torch.stack([image, mask])
#                     image, mask = self.transforms(image_mask_stack).chunk(2)
#                     image, mask = image.squeeze(0), (mask.squeeze(0)[0] > 0.5).float()
#                     images[camera] = image
#                     masks[camera] = mask
#             # Proprioception
#             proprios = torch.from_numpy(f['/proprios'][()][timestep]).float()
#             # Actions
#             actions = torch.from_numpy(f['/actions'][()]).float()
#             H = self.action_horizon
#             actions_padded = torch.cat([actions, actions[-1:].repeat(H - 1, 1)])
#             actions = actions_padded[timestep: (timestep + H)]
            
#             items = dict(
#                 images=images,
#                 masks=masks,
#                 proprios=proprios,
#                 actions=actions
#             )
#             items = self.normalizer.normalize(items)
#             return items
               
#     def __len__(self) -> int:
#         return self.cumulative_episode_lens[-1]
    
#     def set_normalizer(self, normalizer: Normalizer) -> None:
#         self.normalizer = normalizer

#     def get_normalizer(self) -> Normalizer:
#         return self.normalizer