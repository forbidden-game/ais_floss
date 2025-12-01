# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQDecoder(nn.Module):
    """
    IQ信号解码器
    将特征表示 [B, K, D, T'] 解码为IQ信号 [B, K, 2, T]
    """

    def __init__(self,
                 in_dim: int = 512,
                 hidden_dim: int = 256,
                 out_channels: int = 2,
                 num_layers: int = 4,
                 upsample_factor: int = 8):
        super().__init__()

        self.upsample_factor = upsample_factor

        # 构建解码层
        layers = []
        current_dim = in_dim
        remaining_upsample = upsample_factor

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else out_channels

            # 确定本层上采样因子
            if remaining_upsample >= 2 and i < num_layers - 1:
                layer_upsample = 2
                remaining_upsample //= 2
            else:
                layer_upsample = 1

            if layer_upsample > 1:
                layers.append(nn.Upsample(scale_factor=layer_upsample, mode='linear', align_corners=False))

            layers.append(
                nn.Conv1d(current_dim, out_dim, kernel_size=7, padding=3)
            )

            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.GELU())

            current_dim = out_dim

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, D, T'] or [B, D, T'] 特征
        Returns:
            [B, K, 2, T] or [B, 2, T] IQ信号
        """
        if x.dim() == 4:
            B, K, D, T = x.shape
            x = x.view(B * K, D, T)
            out = self.decoder(x)
            return out.view(B, K, 2, -1)
        else:
            return self.decoder(x)


class BitDecoder(nn.Module):
    """
    比特解码器 (可选的端到端解码)
    将IQ信号直接解码为比特
    """

    def __init__(self,
                 in_channels: int = 2,
                 hidden_dim: int = 128,
                 num_bits: int = 256,  # 总比特数
                 samples_per_bit: int = 10):
        super().__init__()

        self.num_bits = num_bits
        self.samples_per_bit = samples_per_bit

        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # 按比特聚合
        self.bit_predictor = nn.Sequential(
            nn.Linear(hidden_dim * samples_per_bit, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, T] IQ信号
        Returns:
            [B, num_bits] 比特概率
        """
        features = self.feature_extractor(x)  # [B, D, T]

        B, D, T = features.shape
        expected_length = self.num_bits * self.samples_per_bit

        # 调整长度
        if T != expected_length:
            features = F.interpolate(features, size=expected_length, mode='linear', align_corners=False)

        # 重塑为 [B, num_bits, D * samples_per_bit]
        features = features.view(B, D, self.num_bits, self.samples_per_bit)
        features = features.permute(0, 2, 1, 3).reshape(B, self.num_bits, -1)

        # 预测比特
        bits = self.bit_predictor(features).squeeze(-1)

        return bits
