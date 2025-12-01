# src/models/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class IQEncoder(nn.Module):
    """
    IQ信号编码器
    将实数IQ信号 [B, 2, T] 编码为特征表示 [B, D, T']
    """

    def __init__(self,
                 in_channels: int = 2,
                 hidden_dim: int = 256,
                 out_dim: int = 512,
                 num_layers: int = 4,
                 kernel_size: int = 7,
                 stride: int = 2,
                 use_batch_norm: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim

        # 构建编码层
        layers = []
        current_dim = in_channels

        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else out_dim

            layers.append(
                nn.Conv1d(
                    current_dim,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i < num_layers - 1 else 1,
                    padding=kernel_size // 2
                )
            )

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))

            layers.append(nn.GELU())
            current_dim = out_channels

        self.encoder = nn.Sequential(*layers)

        # 计算下采样因子
        self.downsample_factor = stride ** (num_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, T] IQ信号
        Returns:
            [B, D, T'] 编码特征
        """
        return self.encoder(x)

    def get_output_length(self, input_length: int) -> int:
        """计算输出长度"""
        return input_length // self.downsample_factor


class MultiScaleIQEncoder(nn.Module):
    """
    多尺度IQ编码器
    同时捕获不同时间尺度的特征
    """

    def __init__(self,
                 in_channels: int = 2,
                 hidden_dim: int = 128,
                 out_dim: int = 512,
                 scales: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()

        self.scales = scales
        dim_per_scale = out_dim // len(scales)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3 * s, stride=s, padding=s),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, dim_per_scale, kernel_size=3, padding=1),
                nn.BatchNorm1d(dim_per_scale),
                nn.GELU()
            )
            for s in scales
        ])

        self.downsample_factor = max(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度编码并上采样到统一长度"""
        features = []
        target_length = x.shape[-1] // self.downsample_factor

        for encoder in self.encoders:
            feat = encoder(x)
            # 上/下采样到统一长度
            if feat.shape[-1] != target_length:
                feat = F.interpolate(feat, size=target_length, mode='linear', align_corners=False)
            features.append(feat)

        return torch.cat(features, dim=1)
