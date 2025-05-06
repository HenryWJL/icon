# import numpy as np
# import zarr
# from pathlib import Path

# task = "play_jenga"
# source_dir = Path(f"../cross_embodiment/data/ee_delta_pose/{task}").absolute()

# for episode in list(source_dir.glob("**/*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         img_front = f['/images/front_camera'][()]
#         img_wrist = f['/images/wrist_camera'][()]
#         if 'masks' in dict(f).keys():
#             mask = f['/masks/front_camera'][()]
#         else:
#             mask = None
#         qpos = f['/joint_properties/local/joint_positions'][()]
#         pose = f['/joint_properties/global/ee_pose'][()]
#         gripper = f['/joint_properties/global/gripper_open'][()]
#         proprio = np.concatenate([qpos, pose, gripper], axis=1)
#         actions = f['/actions'][()]
#     with zarr.open(str(episode).replace("cross_embodiment", "icon"), 'w') as f:
#         f['/images/front_camera'] = img_front
#         f['/images/wrist_camera'] = img_wrist
#         if mask is not None:
#             f['/masks/front_camera'] = mask
#         f['/proprios'] = proprio
#         f['/actions'] = actions
#     print(f"{str(episode)} is done!")



# import zarr
# import numpy as np
# from pathlib import Path

# dir = Path("data/play_jenga")
# # Train
# episode_lens = []
# front_images = []
# wrist_images = []
# front_masks = []
# low_dims = []
# actions = []
# for episode in list(dir.joinpath("train").glob("*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         front_images.append(f['/images/front_camera'][()])
#         wrist_images.append(f['/images/wrist_camera'][()])
#         front_masks.append(f['/masks/front_camera'][()])
#         low_dims.append(f['/proprios'][()])
#         actions.append(f['/actions'][()])
#         episode_len = f['/actions'][()].shape[0]
#         episode_lens.append(episode_len)

# front_images = np.concatenate(front_images)
# wrist_images = np.concatenate(wrist_images)
# front_masks = np.concatenate(front_masks)
# low_dims = np.concatenate(low_dims)
# actions = np.concatenate(actions)    
# cumulative_episode_lens = np.cumsum(episode_lens)
# with zarr.open(str(dir).replace("data", "data_copy") + "/train_data.zarr", 'w') as f:
#     f['/data/low_dims'] = low_dims
#     f['/data/actions'] = actions
#     f['/data/front_camera_images'] = front_images
#     f['/data/wrist_camera_images'] = wrist_images
#     f['/data/front_camera_masks'] = front_masks
#     f['/meta/episode_ends'] = cumulative_episode_lens

# ### Val
# episode_lens = []
# front_images = []
# wrist_images = []
# low_dims = []
# actions = []
# for episode in list(dir.joinpath("val").glob("*.zarr")):
#     with zarr.open(str(episode), 'r') as f:
#         front_images.append(f['/images/front_camera'][()])
#         wrist_images.append(f['/images/wrist_camera'][()])
#         low_dims.append(f['/proprios'][()])
#         actions.append(f['/actions'][()])
#         episode_len = f['/actions'][()].shape[0]
#         episode_lens.append(episode_len)

# front_images = np.concatenate(front_images)
# wrist_images = np.concatenate(wrist_images)
# low_dims = np.concatenate(low_dims)
# actions = np.concatenate(actions)    
# cumulative_episode_lens = np.cumsum(episode_lens)
# with zarr.open(str(dir).replace("data", "data_copy") + "/val_data.zarr", 'w') as f:
#     f['/data/low_dims'] = low_dims
#     f['/data/actions'] = actions
#     f['/data/front_camera_images'] = front_images
#     f['/data/wrist_camera_images'] = wrist_images
#     f['/meta/episode_ends'] = cumulative_episode_lens


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
