import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from einops import rearrange
from typing import Optional, Union, Dict, Tuple, List
from timm.models.vision_transformer import VisionTransformer
from icon.utils.sample_utils import random_sample, farthest_point_sample
from icon.utils.loss_utils import info_nce_loss


class ViT(VisionTransformer):

    def forward_train(self, x: Tensor) -> Tensor:
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
        super().__init__(*args, **kwargs)
        self.decoder_embed = nn.Linear(in_channels, embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * out_channels)

    def unpatchify(self, x: Tensor) -> Tensor:
        H = int(x.shape[1] ** 0.5)
        P = self.patch_embed.patch_size[0]
        x = rearrange(x, 'b (h w) (p q c) -> b c (h p) (w q)', h=H, p=P, q=P)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:]
        x = self.unpatchify(x)
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
        num_decoder_layers: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__()


class IntraContrastViT(ViT):

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

    def patchify(self, x: Tensor) -> Tensor:
        H, W = x.shape[1:]
        P = self.patch_embed.patch_size[0]
        assert H == W and H % P == 0
        x = rearrange(x, 'b (h p) (w q) -> b (h w) (p q)', p=P, q=P)
        return x
    
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
        count_mask = torch.count_nonzero(mask, dim=1).unsqueeze(1)
        count_unmask = tokens.shape[1] - count_mask
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
            mask = self.patchify(mask)
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