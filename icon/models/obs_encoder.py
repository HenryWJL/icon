import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from einops import rearrange
from typing import Optional, Union, Dict, Tuple, List
from timm.models.vision_transformer import VisionTransformer
from icon.utils.sample_utils import random_sample, farthest_point_sample
from icon.utils.loss_utils import info_nce_loss


def patchify(x: Tensor, patch_size: int) -> Tensor:
    height, width = x.shape[-2:]
    assert height == width and height % patch_size == 0
    x = rearrange(x, 'b c (h p) (w q) -> b (h w) (p q c)', p=patch_size, q=patch_size)
    return x


class ViT(VisionTransformer):

    def forward_train(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(x), 0.0
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        return x


class DecoderViT(ViT):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        embed_dim: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            *args,
            **kwargs
        )
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(in_channels, embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * out_channels)
        nn.init.normal_(self.decoder_embed.weight, std=0.02)
        nn.init.zeros_(self.decoder_embed.bias)
        nn.init.normal_(self.decoder_pred.weight, std=0.02)
        nn.init.zeros_(self.decoder_pred.bias)
        del self.patch_embed
        del self.cls_token

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:]
        return x


class AutoencoderViT(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        num_encoder_heads: int,
        num_decoder_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int
    ) -> None:
        super().__init__()
        self.encoder = ViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            depth=num_encoder_layers,
            num_heads=num_encoder_heads
        )
        self.decoder = DecoderViT(
            in_channels=encoder_embed_dim,
            out_channels=3,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=num_decoder_layers,
            num_heads=num_decoder_heads
        )
        self.patch_size = patch_size
        self.num_features = encoder_embed_dim

    def forward_train(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder.forward_features(x)
        pred = self.decoder(latent)
        target = patchify(x, self.patch_size)
        eps = 1e-6
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + eps) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1).mean()
        return latent[:, 0], loss
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    

class SEARViT(AutoencoderViT):

    def __init__(
        self,
        mask_loss_coef: float,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mask_loss_coef = mask_loss_coef
        self.mask_decoder = DecoderViT(
            in_channels=kwargs['encoder_embed_dim'],
            out_channels=1,
            patch_size=kwargs['patch_size'],
            embed_dim=kwargs['decoder_embed_dim'],
            depth=kwargs['num_decoder_layers'],
            num_heads=kwargs['num_decoder_heads']
        )

    def forward_train(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor]:
        if mask is None:
            return self.forward(x), 0.0
        else:
            latent = self.encoder.forward_features(x)
            image_pred = self.decoder(latent)
            mask_pred = self.mask_decoder(latent)
            image_target = patchify(x, self.patch_size)
            mask_target = patchify(mask.unsqueeze(1), self.patch_size)
            # Image reconstruction loss
            eps = 1e-6
            mean = image_target.mean(dim=-1, keepdim=True)
            var = image_target.var(dim=-1, keepdim=True)
            image_target = (image_target - mean) / (var + eps) ** 0.5
            image_loss = ((image_pred - image_target) ** 2).mean(dim=-1).mean()
            # Mask reconstruction loss
            mean = mask_target.mean(dim=-1, keepdim=True)
            var = mask_target.var(dim=-1, keepdim=True)
            mask_target = (mask_target - mean) / (var + eps) ** 0.5
            mask_loss = ((mask_pred - mask_target) ** 2).mean(dim=-1).mean()
            loss = image_loss + self.mask_loss_coef * mask_loss
            return latent[:, 0], loss


class IConViT(ViT):

    def __init__(
        self,
        num_samples_mask: Optional[int] = 10,
        num_samples_unmask: Optional[int] = 50,
        temperature: Optional[float] = 0.3,
        enable_fps: Optional[bool] = True,
        enable_weighted_sum: Optional[bool] = True,
        gamma: Optional[float] = 0.1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_samples_mask = num_samples_mask
        self.num_samples_unmask = num_samples_unmask
        self.temperature = temperature
        self.enable_fps = enable_fps
        self.enable_weighted_sum = enable_weighted_sum
        self.gamma = gamma
    
    def forward_loss(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): token sequences with cls tokens (batch_size, 1+seq_len, embed_dim).
            mask (torch.Tensor): binary masks (batch_size, seq_len), where 1 for masked regions
                and 0 for unmasked regions.

        Returns:
            loss (torch.Tensor): contrastive loss.
        """
        tokens = x[:, 1:]
        # Count masked and unmasked tokens
        count_mask = mask.sum(dim=1).unsqueeze(1)
        count_unmask = (1.0 - mask).sum(dim=1).unsqueeze(1)
        # Obtain queries corresponding to masked and unmasked regions.
        eps = 1e-6
        query_mask = (tokens * mask.unsqueeze(-1)).sum(dim=1) / (count_mask + eps)
        query_unmask = (tokens * (1.0 - mask.unsqueeze(-1))).sum(dim=1) / (count_unmask + eps)
        if self.enable_fps:
            key_unmask = farthest_point_sample(tokens, num_samples=self.num_samples_unmask, masks=1.0 - mask)
            key_mask = farthest_point_sample(tokens, num_samples=self.num_samples_mask, masks=mask)
        else:
            key_unmask, key_mask = random_sample(tokens, mask, self.num_samples_mask, self.num_samples_unmask)
        # Compute contrastive losses.
        loss_unmask = info_nce_loss(
            query=query_unmask,
            pos_key=key_unmask,
            neg_key=key_mask,
            temp=self.temperature
        )
        loss_mask = info_nce_loss(
            query=query_mask,
            pos_key=key_mask,
            neg_key=key_unmask,
            temp=self.temperature
        )
        # Losses are computed on batches with enough samples.
        flag = torch.logical_and(count_mask >= self.num_samples_mask, count_unmask >= self.num_samples_unmask)
        loss = ((loss_unmask + loss_mask) * flag.float()).mean()
        return loss
        
    def forward_train(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor]:
        if mask is None:
            return super().forward_train(x)
        else:
            # For each mask patch, it is regarded as fully masked if the square of mask
            # region is larger than half of the total patch square.
            mask = patchify(mask.unsqueeze(1), self.patch_embed.patch_size[0])
            mask = (mask.sum(dim=-1) > self.patch_embed.patch_size[0] ** 2 / 2).float()
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.pos_embed
            if self.enable_weighted_sum:
                loss = 0.0
                weights = torch.exp(self.gamma * torch.arange(len(self.blocks)))
                weights = weights / weights.sum()
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    loss += weights[i] * self.forward_loss(x, mask)
            else:
                x = self.blocks(x)
                loss = self.forward_loss(x, mask)
            x = self.norm(x)
            x = x[:, 0]
            return x, loss
        

class ObsEncoder(nn.Module):
    
    def __init__(
        self,
        cameras: List,
        proprio_dim: int,
        embed_dim: int,
        image_encoder: ViT,
    ) -> None:
        super().__init__()
        self.proprio_embed = nn.Linear(proprio_dim, embed_dim)
        if len(cameras) > 0:
            self.image_encoders = nn.ModuleDict({
                camera: deepcopy(image_encoder) for camera in cameras
            })
        else:
            raise ValueError(f"Expected at least one camera, but got 0!")
        # If the dimension of image feature does not match @embed_dim, a linear projection
        # would map the image feature to a latent embedding.
        image_feature_dim = image_encoder.num_features
        if image_feature_dim != embed_dim:
            self.image_embed = nn.Linear(image_feature_dim, embed_dim)
        else:
            self.image_embed = nn.Identity()

        nn.init.normal_(self.proprio_embed.weight, std=0.02)
        nn.init.zeros_(self.proprio_embed.bias)

    def forward_train(self, obs: Dict) -> Tuple[Tensor, Tensor]:
        """
        Args:
            obs (dict): image and proprioception observations.
        
        Returns:
            obs_embed (torch.Tensor): observation embeddings (batch_size, obs_cond_horizon, embed_dim).
            losses (torch.Tensor): auxiliary losses. 
        """
        proprios = obs['proprios']
        images = obs['images']
        masks = obs.get('masks')
        # Process proprioception observations
        proprio_embed = self.proprio_embed(proprios)
        # Process image observations
        image_features = dict()
        losses = 0.0
        for key in self.image_encoders.keys():
            inputs = [images[key]]
            if masks is not None:
                inputs.append(masks.get(key))
            image_feature, loss = self.image_encoders[key].forward_train(*inputs)
            image_features[key] = image_feature
            losses += loss
        image_features = torch.stack(list(image_features.values()), dim=1)
        image_embed = self.image_embed(image_features)
        losses = losses / len(self.image_encoders.keys())
        obs_embed = torch.cat([image_embed, proprio_embed.unsqueeze(1)], dim=1)
        return obs_embed, losses
    
    def forward(self, obs: Dict) -> Tensor:
        proprios = obs['proprios']
        images = obs['images']
        proprio_embed = self.proprio_embed(proprios)
        image_features = {
            key: self.image_encoders[key](images[key])
            for key in self.image_encoders.keys()
        }
        image_features = torch.stack(list(image_features.values()), dim=1)
        image_embed = self.image_embed(image_features)
        obs_embed = torch.cat([image_embed, proprio_embed.unsqueeze(1)], dim=1)
        return obs_embed