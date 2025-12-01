# src/utils/metrics.py

import torch
import numpy as np
from typing import Dict


def compute_si_snr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算Scale-Invariant SNR (SI-SNR)

    Args:
        pred: [B, 2, T] 预测信号
        target: [B, 2, T] 目标信号

    Returns:
        SI-SNR in dB
    """
    # 展平IQ为单一向量
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    # 零均值
    pred_flat = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    target_flat = target_flat - target_flat.mean(dim=-1, keepdim=True)

    # 计算投影
    dot = (pred_flat * target_flat).sum(dim=-1, keepdim=True)
    s_target = dot * target_flat / (target_flat ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # 噪声
    e_noise = pred_flat - s_target

    # SI-SNR
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / (e_noise ** 2).sum(dim=-1).clamp(min=1e-8)
    )

    return si_snr.mean()


def compute_snr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算SNR"""
    noise = pred - target
    signal_power = (target ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10 * torch.log10(signal_power / noise_power.clamp(min=1e-8))
    return snr


def compute_metrics(separated: torch.Tensor,
                    sources: torch.Tensor,
                    compute_pit: bool = True) -> Dict[str, float]:
    """
    计算分离质量指标

    Args:
        separated: [B, K, 2, T] 分离信号
        sources: [B, K, 2, T] 真实源信号

    Returns:
        指标字典
    """
    B, K, C, T = separated.shape

    if compute_pit and K == 2:
        # 尝试两种置换，取最好的
        snr1 = (compute_snr(separated[:, 0], sources[:, 0]) +
                compute_snr(separated[:, 1], sources[:, 1])) / 2
        snr2 = (compute_snr(separated[:, 0], sources[:, 1]) +
                compute_snr(separated[:, 1], sources[:, 0])) / 2

        si_snr1 = (compute_si_snr(separated[:, 0], sources[:, 0]) +
                   compute_si_snr(separated[:, 1], sources[:, 1])) / 2
        si_snr2 = (compute_si_snr(separated[:, 0], sources[:, 1]) +
                   compute_si_snr(separated[:, 1], sources[:, 0])) / 2

        snr = max(snr1.item(), snr2.item())
        si_snr = max(si_snr1.item(), si_snr2.item())
    else:
        snr = sum(compute_snr(separated[:, k], sources[:, k]).item() for k in range(K)) / K
        si_snr = sum(compute_si_snr(separated[:, k], sources[:, k]).item() for k in range(K)) / K

    return {
        'snr': snr,
        'si_snr': si_snr
    }
