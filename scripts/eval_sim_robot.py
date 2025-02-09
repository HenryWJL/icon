"""
Notice: do not use workspace to predict actions, as it may cause
the failure of environment resetting (no variations as a result).
"""
import click
import hydra
import torch
import numpy as np
from pathlib import Path
from cross_embodiment.utils.file_utils import str2path


@click.command(help="Evaluating robots in simulation.")
@click.option("-t", "--task", type=str, required=True, help="The manipulation task.")
@click.option("-a", "--algo", type=str, default='baseline', help="The algorithm.")
@click.option("-r", "--robot", type=str, default='panda', help="The robot type.")
@click.option("-c", "--checkpoint", type=str, default='', help="The pre-trained checkpoint.")
@click.option("-d", "--device", type=str, default='cuda', help="The name of device.")
@click.option("-ns", "--num_steps", type=int, default=200, help="The number of running steps.")
@click.option("-nt", "--num_trials", type=int, default=5, help="The number of trials.")
@click.option("-h", "--headless", type=bool, default=False, help="If True, running headlessly.")
def main(task, algo, robot, checkpoint, device, num_steps, num_trials, headless):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("cross_embodiment/configs")),
        version_base="1.2" 
    ):
        overrides = [
            f'task={task}',
            f'algo={algo}',
            f'env.robot={robot}',
            f'env.headless={headless}'
        ]
        cfg = hydra.compose(config_name="config", overrides=overrides)
        env = hydra.utils.instantiate(cfg.env)
        device = torch.device(device)
        policy = hydra.utils.instantiate(cfg.algo.policy)
        policy.to(device)

        checkpoint = str2path(checkpoint)
        normalizer = None
        if checkpoint.is_file():
            state_dicts = torch.load(str(checkpoint), map_location=device)
            policy.load_state_dict(state_dicts['model'])
            normalizer = state_dicts['normalizer']
        else:
            checkpoint = f"data/checkpoints/{cfg.task.arm_controller}/{task}/{algo}.pth"
            state_dicts = torch.load(checkpoint, map_location=device)
            policy.load_state_dict(state_dicts['model'])
            normalizer = state_dicts['normalizer']

        T = num_steps
        H = cfg.task.action_horizon
        D = cfg.task.action_dim
        num_success = 0
        with torch.inference_mode():
            policy.eval()
            for _ in range(num_trials):
                obs = env.reset()
                all_time_actions = torch.zeros([T, T + H, D]).to(device)
                for t in range(T):
                    for k, v in obs.items():
                        if isinstance(v, dict):
                            for p, q in v.items():
                                obs[k][p] = q.to(device)
                        else:
                            obs[k] = v.to(device)
                    if normalizer is not None:
                        obs = normalizer.normalize(obs)
                    actions = policy.sample(obs)
                    if normalizer is not None:
                        actions = normalizer.unnormalize(actions, key='actions')
                    all_time_actions[[t], t: t + H] = actions
                    curr_actions = all_time_actions[:, t]
                    actions_populated = torch.all(curr_actions != 0, dim=1)
                    curr_actions = curr_actions[actions_populated]
                    exp_weights = np.exp(-0.01 * np.arange(len(curr_actions)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights)[..., None].to(device)
                    actions = (curr_actions * exp_weights).sum(dim=0)
                    obs, rewards, dones = env.step(actions)
                    if rewards == 1.0:
                        num_success += 1
                        # print(f"Successfully completed {cfg.task_name} task.")
                        break
        print(f"Success Rate: {num_success}/{num_trials}.")
        env.terminate()
    

if __name__ == "__main__":
    main()