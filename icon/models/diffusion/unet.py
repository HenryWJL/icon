import torch
from torch import nn
from torch import Tensor
from typing import List
from einops import rearrange
from icon.models.diffusion.positional_embedding import SinusoidalPosEmb

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L7
class Downsample1d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L15
class Upsample1d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py#L23
class Conv1dBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L14
class ConditionalResidualBlock1D(nn.Module):

    def __init__(
        self,  
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        residuals = self.residual_conv(x)
        cond = self.cond_encoder(cond)
        cond = rearrange(cond, 'b (d t) -> b d t', t=2)
        scale, bias = cond.chunk(2, dim=-1)
        x = self.blocks[0](x)
        x = x * scale + bias
        x = self.blocks[1](x)
        x = x + residuals
        return x

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py#L69
class ConditionalUnet1D(nn.Module):

    def __init__(
        self, 
        input_dim: int,
        obs_cond_dim: int,
        timestep_embed_dim: int,
        down_dims: List,
        kernel_size: int,
        n_groups: int
    ) -> None:
        super().__init__()
        self.timestep_embed = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        cond_dim = timestep_embed_dim + obs_cond_dim
        all_dims = [input_dim] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))
        common_kwargs = dict(
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups
        )

        self.down_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(in_out_dims):
            is_last = idx >= (len(in_out_dims) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(in_dim, out_dim, **common_kwargs),
                    ConditionalResidualBlock1D(out_dim, out_dim, **common_kwargs),
                    Downsample1d(out_dim) if not is_last else nn.Identity()
                ])
            )
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], **common_kwargs),
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], **common_kwargs)
        ])
        self.up_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(reversed(in_out_dims[1:])):
            is_last = idx >= (len(in_out_dims) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(out_dim * 2, in_dim, **common_kwargs),
                    ConditionalResidualBlock1D(in_dim, in_dim, **common_kwargs),
                    Upsample1d(in_dim) if not is_last else nn.Identity()
                ])
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        obs_cond: Tensor
    ) -> Tensor:
        """
        Args:
            timesteps (torch.Tensor): diffusion timesteps (batch_size,).
            obs_cond (torch.Tensor): observation conditionings (batch_size, obs_cond_dim).
        """
        timestep_embed = self.timestep_embed(timesteps)
        cond = torch.cat([timestep_embed, obs_cond], dim=1)
        x = x.permute(0, 2, 1)
        residuals = list()
        for block1, block2, downsample in self.down_modules:
            x = block1(x, cond)
            x = block2(x, cond)
            residuals.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, cond)
        for block1, block2, upsample in self.up_modules:
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x