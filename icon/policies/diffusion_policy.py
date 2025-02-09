import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from typing import Optional, Tuple, Union, Dict, List
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from timm.models.vision_transformer import Attention
from icon.models.obs_encoder import ObsEncoder
from icon.models.diffusion import TransformerForDiffusion
from icon.utils.train_utils import get_optim_groups


class DiffusionPolicy(nn.Module):
    
    def __init__(
        self,
        obs_encoder: ObsEncoder,
        noise_predictor: TransformerForDiffusion,
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
        action_shape: List,
        num_inference_timesteps: int
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.noise_predictor = noise_predictor
        self.noise_scheduler = noise_scheduler
        self.action_shape = action_shape
        self.num_inference_timesteps = num_inference_timesteps

    def compute_losses(self, batch: Dict) -> Dict:
        loss_dict = dict()
        actions = batch['actions']
        batch_size = actions.shape[0]
        with torch.device(actions.device):
            noise = torch.randn(actions.shape)
            timesteps = torch.randint(
                low=0, 
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                dtype=torch.long
            )
        actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        obs_cond, auxiliary_loss = self.obs_encoder.forward_train(batch)
        pred = self.noise_predictor(actions, timesteps, obs_cond)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = actions
        action_loss = F.mse_loss(pred, target)
        loss = action_loss + auxiliary_loss  
        loss_dict['action_loss'] = action_loss
        loss_dict['auxiliary_loss'] = auxiliary_loss
        loss_dict['loss'] = loss
        return loss_dict
        
    @torch.no_grad()    
    def sample(self, obs: Dict) -> Tensor:
        obs_cond = self.obs_encoder(obs)
        batch_size = obs_cond.shape[0]
        device = obs_cond.device
        actions = torch.randn(
            batch_size,
            *self.action_shape,
            device=device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=device)
        for timestep in self.noise_scheduler.timesteps:
            pred = self.noise_predictor(actions, timestep.repeat(batch_size).long(), obs_cond)
            actions = self.noise_scheduler.step(pred, timestep, actions).prev_sample
        return actions

    def configure_optimizer(
        self,
        lr: float,
        obs_encoder_weight_decay: Optional[float] = 1e-6,
        noise_predictor_weight_decay: Optional[float] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        optim_groups = list()
        optim_groups.extend(
            get_optim_groups(
                self.obs_encoder,
                obs_encoder_weight_decay,
                (nn.Linear, Attention, nn.Conv2d),
                (nn.LayerNorm, nn.Embedding)
            )
        )
        optim_groups.extend(
            get_optim_groups(
                self.noise_predictor,
                noise_predictor_weight_decay,
                (nn.Linear, nn.MultiheadAttention),
                (nn.LayerNorm, nn.Embedding)
            )
        )
        optimizer = AdamW(
            params=optim_groups,
            lr=lr,
            betas=betas
        )
        return optimizer