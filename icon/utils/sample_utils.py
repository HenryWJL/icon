import math
import torch
from torch import Tensor
from typing import Optional, Union


def random_sample(
    x: Tensor, 
    masks: Tensor,
    num_samples_mask: int,
    num_samples_unmask: int,
) -> Tensor:
    """
    Args:
        x (torch.Tensor): token sequences (batch_size, seq_len, dim).
        masks (torch.Tensor): binary masks (batch_size, seq_len).
        num_samples_mask (int): numbers of samples in masked regions.
        num_samples_unmask (int): numbers of samples in unmasked regions.

    Returns:
        samples_unmask (torch.Tensor): tokens sampled in unmasked regions.
        samples_mask (torch.Tensor): tokens sampled in masked regions.
    """
    seq_len, dim = x.shape[1:]
    ids_shuffle = torch.randperm(seq_len)
    x_shuffle = x[:, ids_shuffle]
    masks_shuffle = masks[:, ids_shuffle]
    ids_sort = torch.argsort(masks_shuffle, dim=1)
    ids_unmask = ids_sort[:, :num_samples_unmask] 
    ids_mask = ids_sort[:, -num_samples_mask:]
    samples_unmask = torch.gather(x_shuffle, 1, ids_unmask.unsqueeze(-1).repeat(1, 1, dim))
    samples_mask = torch.gather(x_shuffle, 1, ids_mask.unsqueeze(-1).repeat(1, 1, dim))
    return samples_unmask, samples_mask


def farthest_point_sample(
    x: Tensor,
    num_samples: int,
    p: Optional[int] = 1,
    masks: Union[Tensor, None] = None
) -> Tensor:
    """
    Args:
        x (torch.Tensor): flattened 2D feature maps (batch_size, height * width, dim).
        num_samples (int): number of samples to generate.
        p (int, optional): p-norm for distance function.
        mask (torch.Tensor, optional): binary masks (batch_size, height * width).
            If provided, sampling would be conducted in masked regions (where masks == 1).

    Returns:
        samples (torch.Tensor): sampled points (batch_size, num_samples, dim)
    """
    device = x.device
    batch_size, seq_len = x.shape[:2]
    height = width = int(math.sqrt(seq_len))
    if masks is None:
        masks = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    else:
        assert masks.shape == x.shape[:2]
    # Obtain x-y coordinates (batch_size, height * width, 2)
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device)
        ),
        dim=-1
    ).float().reshape(-1, 2).unsqueeze(0).repeat(batch_size, 1, 1)
    # Initialize
    sample_ids = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    dists = torch.ones(batch_size, height * width, device=device) * 1e3
    batch_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    # Randomly select initial points
    new_ids = torch.randperm(seq_len) 
    x = x[:, new_ids]
    coordinates = coordinates[:, new_ids]
    masks = masks[:, new_ids]
    ids_sort = torch.argsort(masks, dim=1, descending=True)
    farthest_ids = ids_sort[:, 0]  # Initial points' ids
    # Iterate
    for i in range(num_samples):
        sample_ids[:, i] = farthest_ids
        sample_coordinates = coordinates[batch_ids, farthest_ids].unsqueeze(1)
        dist = torch.cdist(sample_coordinates, coordinates, p=p).squeeze(1)
        dists[dist < dists] = dist[dist < dists]
        farthest_ids = torch.max(dists * masks, dim=-1)[1]
    batch_ids = batch_ids.unsqueeze(1).repeat(1, num_samples)
    samples = x[batch_ids, sample_ids]
    return samples