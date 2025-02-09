import wandb
import hydra
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Dict
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
from icon.utils.file_utils import str2path, create_logger


class WorkSpace:

    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        
        seed = self.cfg.train.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device(self.cfg.train.device)
        self.policy = hydra.utils.instantiate(self.cfg.algo.policy)
        self.policy.to(self.device)
        self.logger = create_logger()
        self.global_step = 0
        self.normalizer = None

    def configure_dataloader(self) -> None:
        dataset_dir = str2path(self.cfg.train.dataset_dir)
        if not dataset_dir.joinpath("train").is_dir():
            self.cfg.dataloader.train_loader.dataset.episode_dir = str(dataset_dir)
        self.train_dataloader = hydra.utils.instantiate(self.cfg.dataloader.train_loader)
        self.normalizer = self.train_dataloader.dataset.get_normalizer()
        if not dataset_dir.joinpath("val").is_dir():
            self.cfg.train.val.enable = False
            self.logger.warning("Validation is disabled as no validation set is provided.")
        else:
            self.val_dataloader = hydra.utils.instantiate(self.cfg.dataloader.val_loader)
        
    def train(self) -> None:
        self.configure_dataloader()
        ckpt_manager = hydra.utils.instantiate(self.cfg.train.val.ckpt_manager)
        optimizer = self.policy.configure_optimizer(**self.cfg.train.optimizer)
        num_training_steps = self.cfg.train.num_epochs * len(self.train_dataloader)
        lr_scheduler = hydra.utils.instantiate(
            self.cfg.train.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            last_epoch=-1
        )
        enable_ema = self.cfg.train.ema.enable
        if enable_ema:
            self.ema_model = deepcopy(self.policy)
            ema = hydra.utils.instantiate(
                self.cfg.train.ema.runner,
                model=self.ema_model
            )
        enable_wandb = self.cfg.train.wandb.enable
        if enable_wandb:
            wandb.init(
                dir=str(Path.cwd()),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **self.cfg.train.wandb.logging
            )
        
        num_epochs = self.cfg.train.num_epochs
        for epoch in tqdm(range(num_epochs), desc="Policy Training"):
            # Training
            train_losses = dict(
                action_loss=list(),
                auxiliary_loss=list(),
                loss=list()
            )
            self.policy.train()
            for _, batch in enumerate(self.train_dataloader):
                for k, v in batch.items():
                    if isinstance(v, dict):
                        for p, q in v.items():
                            batch[k][p] = q.to(self.device)
                    else:
                        batch[k] = v.to(self.device)
                loss_dict = self.policy.compute_losses(batch)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                if enable_ema:
                    ema.step(self.policy)
                for k, v in loss_dict.items():
                    train_losses[k].append(v)
            
            train_losses_avg = dict()
            for k, v in train_losses.items():
                if len(v) > 0:
                    train_losses_avg[k] = round(torch.mean(torch.tensor(v)).item(), 3)
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}], training losses: {train_losses_avg}")
            if enable_wandb:
                wandb.log(train_losses_avg)
                
            # Validation
            if self.cfg.train.val.enable:
                if (epoch + 1) % self.cfg.train.val.ckpt_manager.val_freq == 0:
                    self.policy.eval()
                    val_loss = list()
                    with torch.no_grad():
                        for _, batch in enumerate(self.val_dataloader):
                            for k, v in batch.items():
                                if isinstance(v, dict):
                                    for p, q in v.items():
                                        batch[k][p] = q.to(self.device)
                                else:
                                    batch[k] = v.to(self.device)
                            actions_pred = self.policy.sample(batch)
                            loss = F.mse_loss(actions_pred, batch['actions'])
                            val_loss.append(loss.detach())

                    if len(val_loss) > 0:
                        val_loss_avg = torch.mean(torch.tensor(val_loss)).item()
                        self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}], validation loss: {round(val_loss_avg, 3)}")
                        if enable_wandb:
                            wandb.log({'val_loss': val_loss_avg})
                        ckpt_manager.update(val_loss_avg, self.state_dict())

        if self.cfg.train.val.enable:
            ckpt_manager.save_topk()
        else:
            ckpt_manager.save(self.state_dict())
        self.logger.info("Checkpoints saved.")

    def predict_actions(self, obs: Dict) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        for k, v in obs.items():
            if isinstance(v, dict):
                for p, q in v.items():
                    obs[k][p] = q.to(self.device)
            else:
                obs[k] = v.to(self.device)
        actions = self.policy.sample(obs)
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize(actions)
        return actions

    def load_checkpoint(self, checkpoint: Union[str, Path]) -> None:
        checkpoint = str2path(checkpoint)
        if checkpoint.is_file():
            state_dict = torch.load(str(checkpoint), map_location=self.device)
            self.policy.load_state_dict(state_dict['model'])
            self.normalizer = state_dict['normalizer']
        else:
            raise FileExistsError("Checkpoint does not exist!")

    def state_dict(self) -> Dict:
        return dict(
            model=self.ema_model.state_dict() \
                if self.cfg.train.ema.enable \
                else self.policy.state_dict(),
            normalizer=self.normalizer
        )