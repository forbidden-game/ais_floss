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
import time
from datetime import datetime, timedelta

from ..models.floss_ais import FLOSSAIS, FLOSSAISLoss
from ..data.collision_generator import AISCollisionDataset
from ..utils.metrics import compute_metrics


class Trainer:
    """FLOSS-AIS Trainer with comprehensive logging"""

    def __init__(self,
                 model: FLOSSAIS,
                 train_dataset: AISCollisionDataset,
                 val_dataset: AISCollisionDataset = None,

                 # Training parameters
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 grad_clip: float = 1.0,

                 # Loss weights
                 flow_weight: float = 1.0,
                 reconstruction_weight: float = 0.1,
                 consistency_weight: float = 0.1,

                 # Other
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 use_wandb: bool = True,
                 project_name: str = 'floss-ais',
                 log_interval: int = 50):

        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Data loaders
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

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate / 100
        )

        # Loss function
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
        self.best_val_si_snr = float('-inf')
        self.global_step = 0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_si_snr': [],
            'val_snr': [],
            'learning_rate': [],
            'epoch_time': [],
        }

        # Print training config
        self._print_config(train_dataset, val_dataset)

    def _print_config(self, train_dataset, val_dataset):
        """Print training configuration"""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "="*70)
        print("FLOSS-AIS TRAINING")
        print("="*70)
        print(f"  Device:            {self.device}")
        print(f"  Model parameters:  {num_params:,} ({num_trainable:,} trainable)")
        print(f"  Train samples:     {len(train_dataset):,}")
        print(f"  Val samples:       {len(val_dataset) if val_dataset else 'N/A':,}")
        print(f"  Batch size:        {self.batch_size}")
        print(f"  Num epochs:        {self.num_epochs}")
        print(f"  Learning rate:     {self.learning_rate}")
        print(f"  Batches/epoch:     {len(self.train_loader)}")
        print("="*70 + "\n")

    def train(self):
        """Full training loop with comprehensive logging"""
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Print epoch header
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{self.num_epochs}")
            print(f"{'='*70}")

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            if self.val_loader:
                val_metrics = self._validate(epoch)
            else:
                val_metrics = {}

            # Learning rate scheduler step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            remaining_epochs = self.num_epochs - epoch - 1
            eta = timedelta(seconds=int(remaining_epochs * epoch_time))

            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics.get('total_loss', 0))
            self.history['val_si_snr'].append(val_metrics.get('si_snr', 0))
            self.history['val_snr'].append(val_metrics.get('snr', 0))
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # Save checkpoint
            val_loss = val_metrics.get('total_loss', train_metrics['total_loss'])
            self._save_checkpoint(epoch, val_loss, val_metrics)

            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics, current_lr, epoch_time, eta)

            # WandB logging
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'lr': current_lr,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })

        # Training complete
        total_time = time.time() - start_time
        self._print_training_summary(total_time)

        # Save training history
        self._save_history()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_metrics = {}
        batch_times = []

        pbar = tqdm(self.train_loader, desc=f'Training', ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()

            # Move to device
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)

            # Forward pass
            output = self.model(mixture, sources)

            # Compute loss
            targets = {'sources': sources, 'mixture': mixture}
            losses = self.criterion(output, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            if self.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                grad_norm = 0

            self.optimizer.step()

            # Record metrics
            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item()

            batch_times.append(time.time() - batch_start)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses["total_loss"].item():.4f}',
                'flow': f'{losses.get("flow_loss", torch.tensor(0)).item():.4f}',
            })

            self.global_step += 1

        # Average metrics
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validation with comprehensive metrics"""
        self.model.eval()
        total_metrics = {}
        all_si_snrs = []
        all_snrs = []

        pbar = tqdm(self.val_loader, desc='Validating', ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for batch in pbar:
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)

            output = self.model(mixture, sources)
            targets = {'sources': sources, 'mixture': mixture}
            losses = self.criterion(output, targets)

            # Compute separation quality metrics
            separated = output['separated_signals']
            extra_metrics = compute_metrics(separated, sources)

            all_si_snrs.append(extra_metrics['si_snr'])
            all_snrs.append(extra_metrics['snr'])

            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item() if torch.is_tensor(v) else v

        num_batches = len(self.val_loader)
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        # Add separation quality metrics
        avg_metrics['si_snr'] = float(np.mean(all_si_snrs))
        avg_metrics['snr'] = float(np.mean(all_snrs))

        return avg_metrics

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, lr, epoch_time, eta):
        """Print formatted epoch summary"""
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"{'─'*70}")

        # Training metrics
        print(f"  [Train]")
        print(f"    Total Loss:      {train_metrics['total_loss']:.6f}")
        if 'flow_loss' in train_metrics:
            print(f"    Flow Loss:       {train_metrics['flow_loss']:.6f}")
        if 'reconstruction_loss' in train_metrics:
            print(f"    Recon Loss:      {train_metrics['reconstruction_loss']:.6f}")
        if 'consistency_loss' in train_metrics:
            print(f"    Consist Loss:    {train_metrics['consistency_loss']:.6f}")

        # Validation metrics
        if val_metrics:
            print(f"  [Validation]")
            print(f"    Total Loss:      {val_metrics.get('total_loss', 0):.6f}")
            print(f"    SI-SNR:          {val_metrics.get('si_snr', 0):.2f} dB")
            print(f"    SNR:             {val_metrics.get('snr', 0):.2f} dB")

        # Other info
        print(f"  [Info]")
        print(f"    Learning Rate:   {lr:.2e}")
        print(f"    Epoch Time:      {epoch_time:.1f}s")
        print(f"    ETA:             {eta}")
        print(f"    Best Val Loss:   {self.best_val_loss:.6f}")

        if val_metrics.get('si_snr', float('-inf')) > self.best_val_si_snr:
            self.best_val_si_snr = val_metrics.get('si_snr', float('-inf'))
            print(f"    *** New best SI-SNR: {self.best_val_si_snr:.2f} dB ***")

    def _print_training_summary(self, total_time):
        """Print final training summary"""
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"  Total Time:        {timedelta(seconds=int(total_time))}")
        print(f"  Best Val Loss:     {self.best_val_loss:.6f}")
        print(f"  Best Val SI-SNR:   {self.best_val_si_snr:.2f} dB")
        print(f"  Final Train Loss:  {self.history['train_loss'][-1]:.6f}")
        print(f"  Final Val Loss:    {self.history['val_loss'][-1]:.6f}")
        print(f"  Checkpoints:       {self.checkpoint_dir}")
        print("="*70 + "\n")

    def _save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict = None):
        """Save checkpoint with model config"""
        # Extract model config
        model_config = {
            'encoder_type': 'standard' if self.model.encoder.__class__.__name__ == 'IQEncoder' else 'multiscale',
            'encoder_dim': self.model.encoder.out_dim,  # Use out_dim attribute
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
            'val_metrics': val_metrics,
            'model_config': model_config,
            'history': self.history,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  >>> New best model saved (val_loss={val_loss:.6f})")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch+1}.pt')

    def _save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


# Import numpy for metrics
import numpy as np
