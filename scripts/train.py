#!/usr/bin/env python
# scripts/train.py

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.data.ais_generator import AISConfig
from src.data.channel_model import ChannelConfig
from src.data.collision_generator import CollisionConfig, AISCollisionDataset
from src.models.floss_ais import FLOSSAIS
from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 配置
    ais_config = AISConfig(**cfg.ais)
    channel_config = ChannelConfig(
        sample_rate=cfg.ais.sample_rate,
        max_doppler_hz=cfg.channel.max_doppler_hz,
        max_delay_samples=cfg.channel.max_delay_samples,
        snr_range_db=tuple(cfg.channel.snr_range_db),
        amplitude_range=tuple(cfg.channel.amplitude_range)
    )
    collision_config = CollisionConfig(
        num_sources=cfg.collision.num_sources,
        overlap_ratio_range=tuple(cfg.collision.overlap_ratio_range),
        sir_range_db=tuple(cfg.collision.sir_range_db),
        allow_partial_overlap=cfg.collision.allow_partial_overlap
    )

    # 数据集
    print("Creating datasets...")
    train_dataset = AISCollisionDataset(
        size=cfg.data.train_size,
        ais_config=ais_config,
        channel_config=channel_config,
        collision_config=collision_config,
        pregenerate=cfg.data.pregenerate
    )

    val_dataset = AISCollisionDataset(
        size=cfg.data.val_size,
        ais_config=ais_config,
        channel_config=channel_config,
        collision_config=collision_config,
        pregenerate=True  # 验证集预生成
    )

    # 模型
    print("Creating model...")
    model = FLOSSAIS(
        encoder_type=cfg.model.encoder_type,
        encoder_dim=cfg.model.encoder_dim,
        num_sources=cfg.model.num_sources,
        separator_layers=cfg.model.separator_layers,
        separator_heads=cfg.model.separator_heads,
        num_sampling_steps=cfg.model.num_sampling_steps,
        decode_bits=cfg.model.decode_bits
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        flow_weight=cfg.training.flow_weight,
        reconstruction_weight=cfg.training.reconstruction_weight,
        consistency_weight=cfg.training.consistency_weight,
        device=cfg.device,
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
        project_name=cfg.project_name
    )

    # 开始训练
    print("Starting training...")
    trainer.train()


if __name__ == '__main__':
    main()
