# src/data/channel_model.py

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ChannelConfig:
    """信道配置"""
    sample_rate: int = 96000

    # 多普勒参数
    max_doppler_hz: float = 3800.0    # 最大多普勒频移
    doppler_spread_hz: float = 100.0  # 多普勒扩展 (可选)

    # 时延参数
    max_delay_samples: int = 100      # 最大时延 (采样点)

    # 幅度参数
    amplitude_range: Tuple[float, float] = (0.5, 1.5)

    # 噪声参数
    snr_range_db: Tuple[float, float] = (-5.0, 20.0)  # SNR范围


class SatelliteChannel:
    """星载AIS信道模型"""

    def __init__(self, config: ChannelConfig = None):
        self.config = config or ChannelConfig()

    def apply(self,
              signal: np.ndarray,
              doppler_hz: float = None,
              delay_samples: int = None,
              amplitude: float = None,
              snr_db: float = None,
              add_noise: bool = True) -> Tuple[np.ndarray, dict]:
        """
        应用信道效应

        Args:
            signal: 输入复基带信号 [T] or [2, T]
            doppler_hz: 多普勒频移 (Hz), None则随机
            delay_samples: 时延 (采样点), None则随机
            amplitude: 幅度因子, None则随机
            snr_db: 信噪比 (dB), None则随机
            add_noise: 是否添加噪声

        Returns:
            (output_signal, channel_params)
        """
        # 处理输入格式
        if signal.ndim == 2:
            # [2, T] -> complex
            signal = signal[0] + 1j * signal[1]

        # 随机生成参数
        if doppler_hz is None:
            doppler_hz = np.random.uniform(
                -self.config.max_doppler_hz,
                self.config.max_doppler_hz
            )

        if delay_samples is None:
            delay_samples = np.random.randint(0, self.config.max_delay_samples + 1)

        if amplitude is None:
            amplitude = np.random.uniform(*self.config.amplitude_range)

        if snr_db is None:
            snr_db = np.random.uniform(*self.config.snr_range_db)

        # 应用时延
        output = self._apply_delay(signal, delay_samples)

        # 应用多普勒频移
        output = self._apply_doppler(output, doppler_hz)

        # 应用幅度衰减
        output = amplitude * output

        # 添加噪声
        if add_noise:
            output = self._add_awgn(output, snr_db)

        # 记录参数
        params = {
            'doppler_hz': doppler_hz,
            'delay_samples': delay_samples,
            'amplitude': amplitude,
            'snr_db': snr_db
        }

        return output, params

    def _apply_delay(self, signal: np.ndarray, delay_samples: int) -> np.ndarray:
        """应用整数采样点时延"""
        if delay_samples == 0:
            return signal.copy()

        output = np.zeros_like(signal)
        output[delay_samples:] = signal[:-delay_samples]
        return output

    def _apply_doppler(self, signal: np.ndarray, doppler_hz: float) -> np.ndarray:
        """应用多普勒频移"""
        t = np.arange(len(signal)) / self.config.sample_rate
        phase_shift = np.exp(1j * 2 * np.pi * doppler_hz * t)
        return signal * phase_shift

    def _add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """添加AWGN噪声"""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )

        return signal + noise


class TorchSatelliteChannel(nn.Module):
    """PyTorch版本的信道模型 (用于端到端训练)"""

    def __init__(self, config: ChannelConfig = None):
        super().__init__()
        self.config = config or ChannelConfig()

    def forward(self,
                signal: torch.Tensor,
                doppler_hz: torch.Tensor = None,
                delay_samples: torch.Tensor = None,
                amplitude: torch.Tensor = None,
                snr_db: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        批量应用信道效应

        Args:
            signal: [B, 2, T] 实数IQ信号
            其他参数: [B] 或标量

        Returns:
            output: [B, 2, T]
        """
        B, _, T = signal.shape
        device = signal.device

        # 随机生成参数
        if doppler_hz is None:
            doppler_hz = torch.empty(B, device=device).uniform_(
                -self.config.max_doppler_hz,
                self.config.max_doppler_hz
            )

        if amplitude is None:
            amplitude = torch.empty(B, device=device).uniform_(
                *self.config.amplitude_range
            )

        if snr_db is None:
            snr_db = torch.empty(B, device=device).uniform_(
                *self.config.snr_range_db
            )

        # 转换为复数
        signal_complex = torch.complex(signal[:, 0], signal[:, 1])

        # 应用多普勒
        t = torch.arange(T, device=device).float() / self.config.sample_rate
        phase = 2 * np.pi * doppler_hz.unsqueeze(1) * t.unsqueeze(0)  # [B, T]
        doppler_shift = torch.exp(1j * phase)
        output = signal_complex * doppler_shift

        # 应用幅度
        output = amplitude.unsqueeze(1) * output

        # 添加噪声
        signal_power = torch.mean(torch.abs(output) ** 2, dim=1, keepdim=True)
        noise_power = signal_power / (10 ** (snr_db.unsqueeze(1) / 10))
        noise = torch.sqrt(noise_power / 2) * torch.complex(
            torch.randn_like(output.real),
            torch.randn_like(output.imag)
        )
        output = output + noise

        # 转回实数表示
        output_real = torch.stack([output.real, output.imag], dim=1)

        params = {
            'doppler_hz': doppler_hz,
            'amplitude': amplitude,
            'snr_db': snr_db
        }

        return output_real, params
