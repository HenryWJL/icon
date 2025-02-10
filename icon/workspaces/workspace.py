import wandb
import hydra
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
from icon.policies.diffusion_policy import DiffusionPolicy
from icon.utils.datasets import EpisodicDataset
from icon.utils.train_utils import EMA, to_device
from icon.utils.file_utils import str2path, create_logger, CheckpointManager


class WorkSpace:

    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        
        seed = self.cfg.train.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device(self.cfg.train.device)
        self.policy: DiffusionPolicy = hydra.utils.instantiate(self.cfg.algo.policy)
        self.policy.to(self.device)
        self.logger = create_logger()
        self.global_step = 0
        self.normalizer = None

    def configure_dataloader(self) -> None:
        dataset_root_dir = str2path(self.cfg.train.dataset_root_dir)
        train_dataset_dir = dataset_root_dir.joinpath("train")
        val_dataset_dir = dataset_root_dir.joinpath("val")
        if not val_dataset_dir.is_dir():
            self.cfg.train.val.enable = False
            train_dataset_dir = dataset_root_dir
            self.logger.warning("Validation is disabled as no validation set is provided.")
        
        train_dataset: EpisodicDataset = hydra.utils.instantiate(
            self.cfg.train.dataset.train,
            episode_dir=train_dataset_dir
        )
        self.train_dataloader = hydra.utils.instantiate(
            self.cfg.train.dataloader.train,
            dataset=train_dataset
        )
        self.normalizer = train_dataset.get_normalizer()
        
        if self.cfg.train.val.enable:
            val_dataset: EpisodicDataset = hydra.utils.instantiate(
                self.cfg.train.dataset.val,
                episode_dir=val_dataset_dir
            )
            val_dataset.set_normalizer(self.normalizer)
            self.val_dataloader = hydra.utils.instantiate(
                self.cfg.train.dataloader.val,
                dataset=val_dataset
            )
            
    def train(self) -> None:
        self.configure_dataloader()
        ckpt_manager: CheckpointManager = hydra.utils.instantiate(self.cfg.train.val.ckpt_manager)
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
            ema: EMA = hydra.utils.instantiate(
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
        # Training
        num_epochs = self.cfg.train.num_epochs
        for epoch in tqdm(range(num_epochs), desc="Policy Training"):
            train_losses = dict(
                action_loss=list(),
                auxiliary_loss=list(),
                loss=list()
            )
            self.policy.train()
            for _, batch in enumerate(self.train_dataloader):
                to_device(batch, self.device)
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
                    policy = self.policy
                    if enable_ema:
                        policy = self.ema_model
                    policy.eval()
                    val_loss = list()
                    with torch.no_grad():
                        for _, batch in enumerate(self.val_dataloader):
                            for k, v in batch.items():
                                to_device(batch, self.device)
                            actions_pred = policy.sample(batch)
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
            self.logger.info("Checkpoints saved.")

    def predict_actions(self, obs: Dict) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        to_device(obs, self.device)
        actions = self.policy.sample(obs)
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize(actions)
        return actions

    def load_checkpoint(self, checkpoint: str) -> None:
        if checkpoint.is_file():
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.policy.load_state_dict(state_dict['model'])
            self.normalizer = state_dict['normalizer']
        else:
            raise FileExistsError("Checkpoint does not exist!")

    def state_dict(self, *args, **kwargs) -> Dict:
        return dict(
            model=self.ema_model.state_dict(*args, **kwargs) \
                if self.cfg.train.ema.enable \
                else self.policy.state_dict(*args, **kwargs),
            normalizer=self.normalizer
        )