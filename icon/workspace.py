import os
import wandb
import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from icon.policies.base_policy import BasePolicy
from icon.utils.pytorch_utils import to
from icon.utils.train_utils import set_seed, EMA
from icon.utils.file_utils import create_logger, CheckpointManager


class Workspace:

    def __init__(self, cfg: OmegaConf) -> None:
        set_seed(cfg.train.seed)

        # ------------------------------
        # DDP initialization
        # ------------------------------
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_distributed = True
        else:
            self.local_rank = 0
            self.device = torch.device(cfg.train.device)
            self.is_distributed = False

        # Logger (rank 0 only)
        self.logger = create_logger() if self.local_rank == 0 else None

        # Checkpoint Manager
        self.ckpt_manager: CheckpointManager = hydra.utils.instantiate(cfg.train.val.ckpt_manager)
        self.enable_val = cfg.train.val.enable
        if self.enable_val:
            self.val_freq = cfg.train.val.ckpt_manager.val_freq

        # ------------------------------
        # Policy setup
        # ------------------------------
        self.policy: BasePolicy = hydra.utils.instantiate(cfg.algo.policy)
        self.policy.to(self.device)

        # Load checkpoint if provided
        self.start_epoch = 0
        optimizer_state_dict = None
        lr_scheduler_state_dict = None
        if Path(cfg.train.checkpoints).is_file():
            state_dicts = torch.load(cfg.train.checkpoints, map_location=self.device)
            self.policy.load_state_dicts(state_dicts)
            self.start_epoch = state_dicts.get('epoch', 0)
            optimizer_state_dict = state_dicts.get('optimizer')
            lr_scheduler_state_dict = state_dicts.get('lr_scheduler')

        # Wrap model with DDP
        if self.is_distributed:
            self.policy = DDP(
                self.policy,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

        # ------------------------------
        # Dataset and Dataloaders
        # ------------------------------
        train_dataset: Dataset = hydra.utils.instantiate(
            cfg.train.dataset,
            zarr_path=f"data/{cfg.task_name}/train_data.zarr"
        )
        normalizer = train_dataset.get_normalizer()

        # Use DistributedSampler if distributed
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None

        self.train_dataloader: DataLoader = hydra.utils.instantiate(
            cfg.dataloader.train,
            dataset=train_dataset,
            sampler=train_sampler,
            shuffle=(train_sampler is None)
        )

        if self.enable_val:
            val_dataset: Dataset = hydra.utils.instantiate(
                cfg.train.dataset,
                zarr_path=f"data/{cfg.task_name}/val_data.zarr",
                image_mask_keys=list()
            )
            self.val_dataloader: DataLoader = hydra.utils.instantiate(
                cfg.dataloader.val,
                dataset=val_dataset
            )

        self.policy.module.set_normalizer(normalizer) if self.is_distributed else self.policy.set_normalizer(normalizer)

        # ------------------------------
        # Optimizer, LR scheduler, EMA
        # ------------------------------
        self.optimizer = self.policy.module.get_optimizer(**cfg.train.optimizer) if self.is_distributed \
            else self.policy.get_optimizer(**cfg.train.optimizer)
        to(self.optimizer, self.device)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.num_epochs = cfg.train.num_epochs
        self.lr_scheduler = hydra.utils.instantiate(
            cfg.train.lr_scheduler,
            optimizer=self.optimizer,
            num_training_steps=self.num_epochs * len(self.train_dataloader),
            last_epoch=self.start_epoch * len(self.train_dataloader) -1
        )
        if lr_scheduler_state_dict is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        self.enable_ema = cfg.train.ema.enable
        if self.enable_ema:
            model_ref = self.policy.module if self.is_distributed else self.policy
            self.ema_policy = deepcopy(model_ref)
            self.ema: EMA = hydra.utils.instantiate(
                cfg.train.ema.runner,
                model=self.ema_policy
            )

        # ------------------------------
        # WandB (rank 0 only)
        # ------------------------------
        self.enable_wandb = cfg.train.wandb.enable and self.local_rank == 0
        if self.enable_wandb:
            wandb.init(
                dir=str(Path.cwd()),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.train.wandb.logging
            )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self) -> None:
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc="Policy Training", disable=(self.local_rank != 0)):
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            train_losses = dict(
                diffusion_loss=list(),
                recons_loss=list(),
                contrast_loss=list(),
                loss=list()
            )

            model_ref = self.policy.module if self.is_distributed else self.policy
            model_ref.train()

            for _, batch in enumerate(self.train_dataloader):
                to(batch, self.device)
                loss_dict = model_ref.compute_loss(batch)
                loss = loss_dict['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if self.enable_ema:
                    self.ema.step(model_ref)

                for k, v in loss_dict.items():
                    train_losses[k].append(v.item())

            # ------------------------------
            # Logging (only rank 0)
            # ------------------------------
            if self.local_rank == 0:
                train_losses_mean = {k: round(torch.tensor(v).mean().item(), 5)
                                     for k, v in train_losses.items() if len(v) > 0}
                self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], training losses: {train_losses_mean}")
                if self.enable_wandb:
                    wandb.log(train_losses_mean)

            # ------------------------------
            # Validation (rank 0 only)
            # ------------------------------
            if self.enable_val and self.local_rank == 0 and (epoch + 1) % self.val_freq == 0:
                policy_eval = self.ema_policy if self.enable_ema else model_ref
                policy_eval.eval()
                val_loss = []
                with torch.no_grad():
                    for _, batch in enumerate(self.val_dataloader):
                        to(batch, self.device)
                        actions_pred = policy_eval.predict_action(batch['obs'])['actions_pred']
                        loss = F.mse_loss(actions_pred, batch['actions']).item()
                        val_loss.append(loss)

                if len(val_loss) > 0:
                    val_loss = torch.tensor(val_loss).mean().item()
                    self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], validation loss: {round(val_loss, 5)}")
                    if self.enable_wandb:
                        wandb.log({'val_loss': val_loss})
                    state_dicts = policy_eval.state_dicts()
                    state_dicts['epoch'] = epoch
                    state_dicts['optimizer'] = self.optimizer.state_dict()
                    state_dicts['lr_scheduler'] = self.lr_scheduler.state_dict()
                    torch.save(state_dicts, str(self.ckpt_manager.save_dir.joinpath(f"{epoch + 1}.pth")))

        # ------------------------------
        # Save checkpoint (rank 0 only)
        # ------------------------------
        if self.local_rank == 0:
            if self.enable_val:
                pass
            else:
                final_model = self.ema_policy if self.enable_ema else (
                    self.policy.module if self.is_distributed else self.policy
                )
                self.ckpt_manager.save(final_model.state_dicts())
            self.logger.info("Training Finished.")



# import wandb
# import hydra
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from pathlib import Path
# from copy import deepcopy
# from omegaconf import OmegaConf
# from torch.utils.data import Dataset, DataLoader
# from icon.policies.base_policy import BasePolicy
# from icon.utils.pytorch_utils import to
# from icon.utils.train_utils import set_seed, EMA
# from icon.utils.file_utils import create_logger, CheckpointManager


# class Workspace:

#     def __init__(self, cfg: OmegaConf) -> None:
#         set_seed(cfg.train.seed)
#         self.device = torch.device(cfg.train.device)
#         # Logger
#         self.logger = create_logger()
#         # Checkpoint Manager
#         self.ckpt_manager: CheckpointManager = hydra.utils.instantiate(cfg.train.val.ckpt_manager)
#         self.enable_val = cfg.train.val.enable
#         if self.enable_val:
#             self.val_freq = cfg.train.val.ckpt_manager.val_freq
#         # Policy
#         self.policy: BasePolicy = hydra.utils.instantiate(cfg.algo.policy)
#         self.policy.to(self.device)
#         if Path(cfg.train.checkpoints).is_file():
#             state_dicts = torch.load(cfg.train.checkpoints, map_location=self.device)
#             self.policy.load_state_dicts(state_dicts)
#         # Dataloader
#         train_dataset: Dataset = hydra.utils.instantiate(
#             cfg.train.dataset,
#             zarr_path=f"data/{cfg.task_name}/train_data.zarr"
#         )
#         normalizer = train_dataset.get_normalizer()
#         self.train_dataloader: DataLoader = hydra.utils.instantiate(
#             cfg.dataloader.train,
#             dataset=train_dataset
#         )
#         if self.enable_val:
#             val_dataset: Dataset = hydra.utils.instantiate(
#                 cfg.train.dataset,
#                 zarr_path=f"data/{cfg.task_name}/val_data.zarr",
#                 image_mask_keys=list()
#             )
#             self.val_dataloader: DataLoader = hydra.utils.instantiate(
#                 cfg.dataloader.val,
#                 dataset=val_dataset
#             )
#         self.policy.set_normalizer(normalizer)
#         # Optimizer
#         self.optimizer = self.policy.get_optimizer(**cfg.train.optimizer)
#         to(self.optimizer, self.device)
#         # Learning rate scheduler
#         self.num_epochs = cfg.train.num_epochs
#         self.lr_scheduler = hydra.utils.instantiate(
#             cfg.train.lr_scheduler,
#             optimizer=self.optimizer,
#             num_training_steps=self.num_epochs * len(self.train_dataloader),
#             last_epoch=-1
#         )
#         # Exponential Moving Average (EMA)
#         self.enable_ema = cfg.train.ema.enable
#         if self.enable_ema:
#             self.ema_policy = deepcopy(self.policy)
#             self.ema: EMA = hydra.utils.instantiate(
#                 cfg.train.ema.runner,
#                 model=self.ema_policy
#             )
#         # Weights & Biases
#         self.enable_wandb = cfg.train.wandb.enable
#         if self.enable_wandb:
#             wandb.init(
#                 dir=str(Path.cwd()),
#                 config=OmegaConf.to_container(cfg, resolve=True),
#                 **cfg.train.wandb.logging
#             )
            
#     def train(self) -> None:
#         for epoch in tqdm(range(self.num_epochs), desc="Policy Training"):
#             train_losses = dict(
#                 diffusion_loss=list(),
#                 recons_loss=list(),
#                 contrast_loss=list(),
#                 loss=list()
#             )
#             self.policy.train()
#             for _, batch in enumerate(self.train_dataloader):
#                 to(batch, self.device)
#                 loss_dict = self.policy.compute_loss(batch)
#                 loss = loss_dict['loss']
#                 loss.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#                 self.lr_scheduler.step()
#                 if self.enable_ema:
#                     self.ema.step(self.policy)
#                 for k, v in loss_dict.items():
#                     train_losses[k].append(v.item())
            
#             train_losses_mean = dict()
#             for k, v in train_losses.items():
#                 if len(v) > 0:
#                     train_losses_mean[k] = round(torch.tensor(v).mean().item(), 5)
#             self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], training losses: {train_losses_mean}")
#             if self.enable_wandb:
#                 wandb.log(train_losses_mean)
                
#             if self.enable_val:
#                 if (epoch + 1) % self.val_freq == 0:
#                     policy = self.policy
#                     if self.enable_ema:
#                         policy = self.ema_policy
#                     policy.eval()
#                     val_loss = list()
#                     with torch.no_grad():
#                         for _, batch in enumerate(self.val_dataloader):
#                             to(batch, self.device)
#                             actions_pred = policy.predict_action(batch['obs'])['actions_pred']
#                             loss = F.mse_loss(actions_pred, batch['actions']).item()
#                             val_loss.append(loss)

#                     if len(val_loss) > 0:
#                         val_loss = torch.tensor(val_loss).mean().item()
#                         self.logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], validation loss: {round(val_loss, 5)}")
#                         if self.enable_wandb:
#                             wandb.log({'val_loss': val_loss})
#                         self.ckpt_manager.update(val_loss, policy.state_dicts())

#         if self.enable_val:
#             self.ckpt_manager.save_topk()
#         else:
#             policy = self.ema_policy if self.enable_ema else self.policy
#             self.ckpt_manager.save(policy.state_dicts())
#         self.logger.info("Training Finished.")
