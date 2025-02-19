import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
from icon.policies.diffusion_transformer_policy import DiffusionTransformerPolicy


class IConDiffusionTransformerPolicy(DiffusionTransformerPolicy):

    def __init__(
        self,
        contrast_loss_coef: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contrast_loss_coef = contrast_loss_coef

    def compute_loss(self, batch: Dict) -> Tensor:
        batch = self.normalizer.normalize(batch)
        x = batch['actions']
        obs_cond, contrast_loss = self.obs_encoder(batch['obs'], batch['image_masks'])

        batch_size = x.shape[0]
        with torch.device(x.device):
            noise = torch.randn(x.shape)
            timesteps = torch.randint(
                low=0, 
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                dtype=torch.long
            )
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        pred = self.noise_predictor(noisy_x, timesteps, obs_cond)
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = x
        diffusion_loss = F.mse_loss(pred, target)
        loss = diffusion_loss + self.contrast_loss_coef * contrast_loss
        loss_dict = dict(
            diffusion_loss=diffusion_loss,
            contrast_loss=contrast_loss,
            loss=loss
        )
        return loss_dict