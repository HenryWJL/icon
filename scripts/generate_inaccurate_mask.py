import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from PIL import Image


def random_flip(mask_batch, flip_prob=0.1):
    """
    Randomly flip pixel values in the mask region (mask == 1) with given probability.
    
    Parameters:
        mask_batch (np.ndarray): Binary masks of shape (B, H, W), values 0 or 1.
        flip_prob (float): Probability of flipping a pixel in the foreground (mask == 1).
    
    Returns:
        np.ndarray: Noised mask batch of the same shape.
    """
    mask_batch = mask_batch.copy()
    flip_mask = (mask_batch == 1) & (np.random.rand(*mask_batch.shape) < flip_prob)
    mask_batch[flip_mask] = 0  # flip 1 -> 0
    return mask_batch


def erode(mask, kernel_size=3, iterations=3):
    B, H, W = mask.shape
    # Create a kernel for erosion (a square of ones)
    structure = np.ones((kernel_size, kernel_size), dtype=bool)

    eroded_masks = []
    for i in range(B):
        # Erode each mask individually
        eroded_mask = binary_erosion(mask[i], structure=structure, iterations=iterations)
        eroded_masks.append(eroded_mask)
    
    # Stack back the results into a single batch
    return np.stack(eroded_masks, axis=0).astype(np.uint8)


def resize(mask_batch, target_size=(256, 256), intermediate_size=(64, 64)):
    """
    Resizes a batch of binary masks by first downscaling and then upscaling using PIL.
    mask_batch: (B, H, W) numpy array of binary masks
    target_size: Final output size (height, width)
    intermediate_size: Intermediate resize size (height, width)
    """
    B, H, W = mask_batch.shape
    resized_batch = []

    for i in range(B):
        # Convert the mask to a PIL image
        mask_pil = Image.fromarray(mask_batch[i])

        # First resize down
        mask_down = mask_pil.resize(intermediate_size, Image.NEAREST)
        
        # Then resize back up
        mask_up = mask_down.resize(target_size, Image.NEAREST)
        
        # Convert back to numpy array and append
        resized_batch.append(np.array(mask_up))

    return np.stack(resized_batch, axis=0)


with zarr.open("data/train_data.zarr", 'r') as f:
    meta = f['/meta/episode_ends'][()]
    low_dims = f['/data/low_dims'][()]
    actions = f['/data/actions'][()]
    image1 = f['/data/agentview_images'][()]
    image2 = f['/data/robot0_eye_in_hand_images'][()]
    mask = f['/data/agentview_masks'][()]
    mask = (mask).astype(np.uint8)
    corrupted_mask = erode(mask)
    approximate_mask = resize(mask)
    noisy_mask = random_flip(mask)

names = ['corrupted', 'approximate', 'noisy']
masks = [corrupted_mask, approximate_mask, noisy_mask]
for i in range(len(names)):
    with zarr.open(f"data/train_data_{names[i]}.zarr", 'w') as f:
        f['/meta/episode_ends'] = meta
        f['/data/low_dims'] = low_dims
        f['/data/actions'] = actions
        f['/data/agentview_images'] = image1
        f['/data/robot0_eye_in_hand_images'] = image2
        f['/data/agentview_masks'] = masks[i]


# idx = 0
# plt.imshow(mask[idx])
# plt.axis('off')
# plt.savefig("original_mask.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
# plt.show()

# plt.imshow(corrupted_mask[idx])
# plt.axis('off')
# plt.savefig("corrupted_mask.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
# plt.show()

# plt.imshow(approximate_mask[idx])
# plt.axis('off')
# plt.savefig("approximate_mask.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
# plt.show()

# plt.imshow(noisy_mask[idx])
# plt.axis('off')
# plt.savefig("noisy_mask.svg", bbox_inches='tight', pad_inches=0.0, dpi=300)
# plt.show()
