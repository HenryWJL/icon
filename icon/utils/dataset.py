import zarr
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as tfs
from typing import Optional, Tuple, Dict, List
from icon.utils.normalizer import Normalizer
from icon.utils.file_utils import str2path


class EpisodicDataset(Dataset):
    
    def __init__(
        self,
        episode_dir: str,
        cameras: List,
        action_horizon: List,
        norm_modes: Dict,
        load_masks: Optional[bool] = True,
        transform_cfg: Optional[Dict] = dict(),
    ) -> None:
        """
        Args:
            episode_dir (str): directory where episodes (zarr files) are stored.
                
                >>> Format:
                {
                    'images': {
                        'front_camera': array (episode_len, height, width, 3),
                        'wrist_camera': array (episode_len, height, width, 3)
                    },
                    'masks': {
                        'front_camera': array (episode_len, height, width)
                    },
                    'proprios': array (episode_len, proprio_dim),
                    'actions': array (episode_len, action_dim)
                }

            norm_modes (dict): normalization modes.
            load_masks (bool, optional): if True, load image masks.
            transform_cfg (dict): configuration of image transformations.
        """
        super().__init__()
        self.episode_paths = list(str2path(episode_dir).glob("**/*.zarr"))
        assert len(self.episode_paths) != 0, f"No episodes found in directory {episode_dir}!"
        self.cameras = cameras
        self.action_horizon = action_horizon
        self.norm_modes = norm_modes
        self.load_masks = load_masks
        self.transform_cfg = transform_cfg
        self.episode_lens = list()
        self.cumulative_episode_lens = list()
        self.transforms = nn.Identity()
        self.normalizer = Normalizer()
        self.fit()
        self.configure_transforms()

    def fit(self) -> None:
        """
        Fit a normalizer and extract episodes' metadata.
        """
        proprios = list()
        actions = list()
        episode_lens = list()
        for episode_path in self.episode_paths:
            with zarr.open(episode_path, 'r') as f:
                proprio = f['/proprios'][()]
                action = f['/actions'][()]
                proprios.append(proprio)
                actions.append(action)
                episode_lens.append(action.shape[0])
        proprios = torch.from_numpy(np.concatenate(proprios)).float()
        actions = torch.from_numpy(np.concatenate(actions)).float()
        data = dict(
            proprios=proprios,
            actions=actions
        )
        self.normalizer.fit(data, self.norm_modes)    
        self.episode_lens = episode_lens
        self.cumulative_episode_lens = np.cumsum(episode_lens)

    def configure_transforms(self) -> None:
        resize_shape = self.transform_cfg.get('resize_shape')
        crop_shape = self.transform_cfg.get('crop_shape')
        random_crop = self.transform_cfg.get('random_crop', False)
        enable_imagenet_norm = self.transform_cfg.get('enable_imagenet_norm', False)
        transforms = list()
        if resize_shape is not None:
            transforms.append(tfs.Resize(resize_shape, antialias=True))
        if crop_shape is not None:
            if random_crop:
                transforms.append(tfs.RandomCrop((crop_shape, crop_shape)))
            else:
                transforms.append(tfs.CenterCrop((crop_shape, crop_shape)))
        if enable_imagenet_norm:
            transforms.append(
                tfs.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        self.transforms = nn.Sequential(*transforms)

    def _locate_transition(self, index: int) -> Tuple:
        """
        Locate current episode and time step. This is necessary when
        more than one zarr files are stored in the directory, since
        we need to know the relative position of current index w.r.t
        the whole episodes.
        """
        assert index < self.cumulative_episode_lens[-1]
        episode_index = np.argmax(self.cumulative_episode_lens > index)
        current_timestep = index - (self.cumulative_episode_lens[episode_index] \
                            - self.episode_lens[episode_index])
        return episode_index, current_timestep

    def __getitem__(self, index: int) -> Tuple:
        episode_index, timestep = self._locate_transition(index)
        episode_path = self.episode_paths[episode_index]
        with zarr.open(episode_path, 'r') as f:
            # Image masks
            if self.load_masks:
                masks = dict()
                for camera in self.cameras:
                    if camera in dict(f['/masks']).keys():
                        mask = f[f'/masks/{camera}'][()][timestep]
                        mask = torch.from_numpy(mask).float()
                        masks[camera] = mask
            else:
                masks = None
            # Images
            images = dict()
            for camera in self.cameras:
                image = f[f'/images/{camera}'][()][timestep]
                image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
                # Image transformations. Note that image masks also require the same transformation.
                mask = masks.get(camera) if masks is not None else None
                if mask is not None:
                    mask = mask.unsqueeze(0).repeat(3, 1, 1)
                    image_mask_stack = torch.stack([image, mask])
                    image, mask = self.transforms(image_mask_stack).chunk(2)
                    image, mask = image.squeeze(0), (mask.squeeze(0)[0] > 0.5).float()
                    images[camera] = image
                    masks[camera] = mask
                else:
                    image = self.transforms(image)
                    images[camera] = image
            # Proprioception
            proprios = torch.from_numpy(f['/proprios'][()][timestep]).float()
            # Actions
            actions = torch.from_numpy(f['/actions'][()]).float()
            H = self.action_horizon
            actions_padded = torch.cat([actions, actions[-1:].repeat(H - 1, 1)])
            actions = actions_padded[timestep: (timestep + H)]
            
            items = dict(
                images=images,
                proprios=proprios,
                actions=actions
            )
            if masks is not None:
                items['masks'] = masks
            items = self.normalizer.normalize(items)
            return items
               
    def __len__(self) -> int:
        return self.cumulative_episode_lens[-1]
    
    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def get_normalizer(self) -> Normalizer:
        return self.normalizer