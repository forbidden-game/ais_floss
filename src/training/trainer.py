# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import json

from ..models.floss_ais import FLOSSAIS, FLOSSAISLoss
from ..data.collision_generator import AISCollisionDataset
from ..utils.metrics import compute_metrics


class Trainer:
    """FLOSS-AIS 训练器"""

    def __init__(self,
                 model: FLOSSAIS,
                 train_dataset: AISCollisionDataset,
                 val_dataset: AISCollisionDataset = None,

                 # 训练参数
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 grad_clip: float = 1.0,

                 # 损失权重
                 flow_weight: float = 1.0,
                 reconstruction_weight: float = 0.1,
                 consistency_weight: float = 0.1,

                 # 其他
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 use_wandb: bool = True,
                 project_name: str = 'floss-ais'):

        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb

        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=AISCollisionDataset.collate_fn
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=AISCollisionDataset.collate_fn
            )
        else:
            self.val_loader = None

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate / 100
        )

        # 损失函数
        self.criterion = FLOSSAISLoss(
            flow_weight=flow_weight,
            reconstruction_weight=reconstruction_weight,
            consistency_weight=consistency_weight
        )

        # WandB
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name, config={
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'model_config': str(model)
                })
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling wandb logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        self.best_val_loss = float('inf')
        self.global_step = 0

    def train(self):
        """完整训练循环"""
        for epoch in range(self.num_epochs):
            # 训练
            train_metrics = self._train_epoch(epoch)

            # 验证
            if self.val_loader:
                val_metrics = self._validate(epoch)
            else:
                val_metrics = {}

            # 学习率调度
            self.scheduler.step()

            # 保存检查点
            self._save_checkpoint(epoch, val_metrics.get('total_loss', train_metrics['total_loss']))

            # 日志
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'lr': self.scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })

            print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
                  f"val_loss={val_metrics.get('total_loss', 'N/A')}")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_metrics = {}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # 移动到设备
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)

            # 前向传播
            output = self.model(mixture, sources)

            # 计算损失
            targets = {'sources': sources, 'mixture': mixture}
            losses = self.criterion(output, targets)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # 记录
            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item()

            pbar.set_postfix({'loss': losses['total_loss'].item()})
            self.global_step += 1

        # 平均
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_metrics = {}

        for batch in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)

            output = self.model(mixture, sources)
            targets = {'sources': sources, 'mixture': mixture}
            losses = self.criterion(output, targets)

            # 计算额外指标 (SNR等)
            separated = output['separated_signals']
            extra_metrics = compute_metrics(separated, sources)

            for k, v in {**losses, **extra_metrics}.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item() if torch.is_tensor(v) else v

        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint with model config for proper loading"""
        # Extract model config from the model
        model_config = {
            'encoder_type': 'standard' if self.model.encoder.__class__.__name__ == 'IQEncoder' else 'multiscale',
            'encoder_dim': self.model.encoder.encoder[-1].num_features,  # Get dim from last BatchNorm
            'num_sources': self.model.num_sources,
            'separator_layers': len(self.model.separator.vector_field.layers),
            'separator_heads': self.model.separator.vector_field.layers[0].attn.num_heads,
            'num_sampling_steps': self.model.num_sampling_steps,
            'decode_bits': self.model.decode_bits,
        }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': model_config,
        }

        # 保存最新
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # 保存最佳
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"New best model saved with val_loss={val_loss:.4f}")
