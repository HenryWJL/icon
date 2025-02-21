# import click
# import hydra
# import torch
# import imageio
# import numpy as np
# from pathlib import Path
# from torchvision.transforms import Resize
# from icon.envs.base import Env
# from icon.policies.diffusion_policy import DiffusionPolicy
# from icon.utils.file_utils import str2path, mkdir
# from icon.utils.train_utils import to_device


# @click.command(help="Evaluate policies in simulation.")
# @click.option("-t", "--task", type=str, required=True, help="Task name.")
# @click.option("-a", "--algo", type=str, required=True, help="Algorithm name.")
# @click.option("-c", "--checkpoint", type=str, default="", help="Pretrained checkpoint.")
# @click.option("-r", "--robot", type=str, default="panda", help="Robot type.")
# @click.option("-d", "--device", type=str, default="cuda", help="Device type.")
# @click.option("-ns", "--num_steps", type=int, default=200, help="Number of rollout steps.")
# @click.option("-nt", "--num_trials", type=int, default=5, help="Number of trials.")
# @click.option("-h", "--headless", type=bool, default=False, help="Headless setting.")
# def main(task, algo, checkpoint, robot, device, num_steps, num_trials, headless):
#     with hydra.initialize_config_dir(
#         config_dir=str(Path(__file__).parent.parent.joinpath("icon/configs")),
#         version_base="1.2" 
#     ):
#         overrides = [
#             f'task={task}',
#             f'algo={algo}',
#             f'task.env.robot={robot}',
#             f'task.env.headless={headless}',
#             f'task.env.image_size=1024'
#         ]
#         cfg = hydra.compose(config_name="config", overrides=overrides)
#         env: Env = hydra.utils.instantiate(cfg.task.env)
#         device = torch.device(device)
#         policy: DiffusionPolicy = hydra.utils.instantiate(cfg.algo.policy)
#         policy.to(device)

#         checkpoint = str2path(checkpoint)
#         normalizer = None
#         if checkpoint.is_file():
#             state_dict = torch.load(str(checkpoint), map_location=device)
#             policy.load_state_dict(state_dict['model'])
#             normalizer = state_dict['normalizer']
#         else:
#             checkpoint = f"checkpoints/{task}/{algo}.pth"
#             state_dict = torch.load(checkpoint, map_location=device)
#             policy.load_state_dict(state_dict['model'])
#             normalizer = state_dict['normalizer']

#         video_save_dir = str2path("videos").joinpath(task).joinpath(algo)
#         mkdir(video_save_dir, parents=True, exist_ok=True)

#         T = num_steps
#         H = cfg.task.action_horizon
#         D = cfg.task.action_dim
#         num_success = 0
#         with torch.inference_mode():
#             policy.eval()
#             for i in range(num_trials):
#                 obs = env.reset()
#                 all_time_actions = torch.zeros([T, T + H, D]).to(device)
#                 video_writer = imageio.get_writer(video_save_dir.joinpath(f"trial_{str(i + 1).zfill(3)}.mp4"), fps=24)
#                 for t in range(T):
#                     obs['images']['front_camera'] = Resize(224)(obs['images']['front_camera'])
#                     obs['images']['wrist_camera'] = Resize(224)(obs['images']['wrist_camera'])
#                     to_device(obs, device)
#                     if normalizer is not None:
#                         obs = normalizer.normalize(obs)
#                     actions = policy.sample(obs)
#                     if normalizer is not None:
#                         actions = normalizer.unnormalize(actions, key='actions')
#                     # Due to precision loss in the process of converting absolute actions
#                     # to delta actions, we need to add an extra term to the predicted actions.
#                     drift = 5e-4
#                     actions = actions + drift
#                     all_time_actions[[t], t: t + H] = actions
#                     curr_actions = all_time_actions[:, t]
#                     actions_populated = torch.all(curr_actions != 0, dim=1)
#                     curr_actions = curr_actions[actions_populated]
#                     exp_weights = np.exp(-0.01 * np.arange(len(curr_actions)))
#                     exp_weights = exp_weights / exp_weights.sum()
#                     exp_weights = torch.from_numpy(exp_weights)[..., None].to(device)
#                     actions = (curr_actions * exp_weights).sum(dim=0)
#                     obs, rewards, dones = env.step(actions)
                    
#                     image = obs['images']['front_camera']
#                     image = (image.squeeze(0).permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
#                     video_writer.append_data(image)
#                     if rewards and dones:
#                         num_success += 1
#                         video_writer.close()
#                         break
                
#         print(f"Success Rate: {num_success}/{num_trials}.")
#         env.terminate()
    

# if __name__ == "__main__":
#     main()



"""
The following code does not contain video recording. Use it in the official release.
"""
import click
import hydra
import torch
import numpy as np
from pathlib import Path
from icon.envs.base import Env
from icon.policies.icon_diffusion_transformer_policy import DiffusionPolicy
from icon.utils.file_utils import str2path
from icon.utils.pytorch_utils import to
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command(help="Evaluate policies in simulation.")
@click.option("-t", "--task", type=str, required=True, help="Task name.")
@click.option("-a", "--algo", type=str, required=True, help="Algorithm name.")
@click.option("-c", "--checkpoint", type=str, default="", help="Pretrained checkpoint.")
@click.option("-r", "--robot", type=str, default="panda", help="Robot type.")
@click.option("-d", "--device", type=str, default="cuda", help="Device type.")
@click.option("-ns", "--num_steps", type=int, default=200, help="Number of rollout steps.")
@click.option("-nt", "--num_trials", type=int, default=5, help="Number of trials.")
@click.option("-h", "--headless", type=bool, default=False, help="Headless setting.")
def main(task, algo, checkpoint, robot, device, num_steps, num_trials, headless):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("icon/configs")),
        version_base="1.2" 
    ):
        overrides = [
            f'task={task}',
            f'algo={algo}',
            f'task.env.robot={robot}',
            f'task.env.headless={headless}'
        ]
        cfg = hydra.compose(config_name="config", overrides=overrides)
        env: Env = hydra.utils.instantiate(cfg.task.env)
        device = torch.device(device)
        policy: DiffusionPolicy = hydra.utils.instantiate(cfg.algo.policy)
        policy.to(device)

        checkpoint = str2path(checkpoint)
        normalizer = None
        if checkpoint.is_file():
            state_dict = torch.load(str(checkpoint), map_location=device)
            policy.load_state_dict(state_dict['model'])
            normalizer = state_dict['normalizer']
        else:
            checkpoint = f"checkpoints/{task}/{algo}.pth"
            state_dict = torch.load(checkpoint, map_location=device)
            policy.load_state_dict(state_dict['model'])
            normalizer = state_dict['normalizer']

        T = num_steps
        H = cfg.task.action_horizon
        D = cfg.task.action_dim
        num_success = 0
        with torch.inference_mode():
            policy.eval()
            for i in range(num_trials):
                obs = env.reset()
                all_time_actions = torch.zeros([T, T + H, D]).to(device)
                for t in range(T):
                    to(obs, device)
                    if normalizer is not None:
                        obs = normalizer.normalize(obs)
                    actions = policy.sample(obs)
                    if normalizer is not None:
                        actions = normalizer.unnormalize(actions, key='actions')
                    # Due to precision loss in the process of converting absolute actions
                    # to delta actions, we need to add an extra term to the predicted actions.
                    drift = 5e-4
                    actions = actions + drift
                    all_time_actions[[t], t: t + H] = actions
                    curr_actions = all_time_actions[:, t]
                    actions_populated = torch.all(curr_actions != 0, dim=1)
                    curr_actions = curr_actions[actions_populated]
                    exp_weights = np.exp(-0.01 * np.arange(len(curr_actions)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights)[..., None].to(device)
                    actions = (curr_actions * exp_weights).sum(dim=0)
                    obs, rewards, dones = env.step(actions)
                    if rewards and dones:
                        num_success += 1
                        break
                
        print(f"Success Rate: {num_success}/{num_trials}.")
        env.terminate()
    

if __name__ == "__main__":
    main()