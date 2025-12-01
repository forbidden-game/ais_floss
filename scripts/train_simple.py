#!/usr/bin/env python
"""
Simple training script without hydra dependency.
Usage: python scripts/train_simple.py --train_size 1000 --num_epochs 5
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.ais_generator import AISConfig
from src.data.channel_model import ChannelConfig
from src.data.collision_generator import CollisionConfig, AISCollisionDataset
from src.models.floss_ais import FLOSSAIS
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train FLOSS-AIS model')

    # Data parameters
    parser.add_argument('--train_size', type=int, default=1000, help='Training dataset size')
    parser.add_argument('--val_size', type=int, default=200, help='Validation dataset size')
    parser.add_argument('--pregenerate', action='store_true', help='Pregenerate all data')

    # Model parameters
    parser.add_argument('--encoder_type', type=str, default='standard', choices=['standard', 'multiscale'])
    parser.add_argument('--encoder_dim', type=int, default=256, help='Encoder dimension')
    parser.add_argument('--separator_layers', type=int, default=4, help='Number of separator layers')
    parser.add_argument('--separator_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_sources', type=int, default=2, help='Number of sources')
    parser.add_argument('--num_sampling_steps', type=int, default=10, help='Sampling steps')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    # Loss weights
    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--reconstruction_weight', type=float, default=0.1)
    parser.add_argument('--consistency_weight', type=float, default=0.1)

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--project_name', type=str, default='floss-ais')

    args = parser.parse_args()

    print("=" * 60)
    print("FLOSS-AIS Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Train size: {args.train_size}")
    print(f"Val size: {args.val_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 60)

    # 配置
    ais_config = AISConfig()
    channel_config = ChannelConfig(
        sample_rate=ais_config.sample_rate,
        max_doppler_hz=3800.0,
        max_delay_samples=100,
        snr_range_db=(-5.0, 20.0),
        amplitude_range=(0.5, 1.5)
    )
    collision_config = CollisionConfig(
        num_sources=args.num_sources,
        overlap_ratio_range=(0.3, 1.0),
        sir_range_db=(-10.0, 10.0),
        allow_partial_overlap=True
    )

    # 数据集
    print("Creating datasets...")
    train_dataset = AISCollisionDataset(
        size=args.train_size,
        ais_config=ais_config,
        channel_config=channel_config,
        collision_config=collision_config,
        pregenerate=args.pregenerate
    )

    val_dataset = AISCollisionDataset(
        size=args.val_size,
        ais_config=ais_config,
        channel_config=channel_config,
        collision_config=collision_config,
        pregenerate=True  # 验证集预生成
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # 模型
    print("Creating model...")
    model = FLOSSAIS(
        encoder_type=args.encoder_type,
        encoder_dim=args.encoder_dim,
        num_sources=args.num_sources,
        separator_layers=args.separator_layers,
        separator_heads=args.separator_heads,
        num_sampling_steps=args.num_sampling_steps,
        decode_bits=False
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        flow_weight=args.flow_weight,
        reconstruction_weight=args.reconstruction_weight,
        consistency_weight=args.consistency_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        project_name=args.project_name
    )

    # 开始训练
    print("\nStarting training...")
    trainer.train()

    print("\nTraining completed!")
    print(f"Best model saved at: {args.checkpoint_dir}/best.pt")
    print(f"Latest model saved at: {args.checkpoint_dir}/latest.pt")


if __name__ == '__main__':
    main()
