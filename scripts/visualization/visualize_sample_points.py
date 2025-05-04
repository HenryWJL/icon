import torch
import av
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
from einops import rearrange
from icon.utils.sampler import random_sample, farthest_point_sample


def patchify(x, patch_size):
    height, width = x.shape[-2:]
    assert height == width and height % patch_size == 0
    x = rearrange(x, 'b c (h p) (w q) -> b (h w) (p q c)', p=patch_size, q=patch_size)
    return x


task = "open_box"
camera = "front_camera"
patch_size = 16
num_samples = 10

# Image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])
# Load image
frame_id = 90
video_path = f"/home/wangjl/project/cross_embodiment/data/ee_pose/{task}/train/episode_003/masks/{camera}.mp4"
mask = None
with av.open(str(video_path)) as container:
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_id:
            mask = frame.to_ndarray(format='gray')
            break
mask = Image.fromarray(mask)
mask = (transform(mask) > 0.5).float()
# Sample some points inside the mask
indices = torch.stack(
    torch.meshgrid(
        torch.arange(mask.shape[1] / patch_size), 
        torch.arange(mask.shape[2] / patch_size)
    ),
    dim=-1
).float().reshape(-1, 2).unsqueeze(0)
mask_1d = patchify(mask.unsqueeze(1), patch_size)
mask_1d = (mask_1d.sum(dim=-1) > patch_size ** 2 / 2).float()

_, random_sample_indices = random_sample(indices, mask_1d, 10, 50)
fps_sample_indices = farthest_point_sample(indices, num_samples, masks=mask_1d)
random_sample_indices = random_sample_indices.squeeze(0).numpy().astype(np.uint8) * patch_size
fps_sample_indices = fps_sample_indices.squeeze(0).numpy().astype(np.uint8) * patch_size

mask = mask.squeeze(0).numpy().astype(np.uint8)

# Visualize
plt.imshow(mask)
plt.scatter(random_sample_indices[:, 1], random_sample_indices[:, 0], c='cyan', s=30)
plt.axis('off')
plt.savefig("random_sample.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()
