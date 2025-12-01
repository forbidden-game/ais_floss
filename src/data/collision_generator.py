# src/data/collision_generator.py

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .ais_generator import AISSignalGenerator, AISConfig, BatchAISGenerator
from .channel_model import SatelliteChannel, ChannelConfig, TorchSatelliteChannel


@dataclass
class CollisionConfig:
    """碰撞配置"""
    num_sources: int = 2              # 碰撞信号数量
    overlap_ratio_range: Tuple[float, float] = (0.3, 1.0)  # 重叠比例范围
    sir_range_db: Tuple[float, float] = (-10.0, 10.0)      # 信干比范围
    allow_partial_overlap: bool = True  # 是否允许部分重叠


class CollisionGenerator:
    """AIS信号碰撞生成器"""

    def __init__(self,
                 ais_config: AISConfig = None,
                 channel_config: ChannelConfig = None,
                 collision_config: CollisionConfig = None):
        self.ais_config = ais_config or AISConfig()
        self.channel_config = channel_config or ChannelConfig()
        self.collision_config = collision_config or CollisionConfig()

        self.ais_generator = AISSignalGenerator(self.ais_config)
        self.channel = SatelliteChannel(self.channel_config)

    def generate_collision(self,
                           num_sources: int = None,
                           snr_db: float = None) -> dict:
        """
        生成碰撞信号

        Returns:
            {
                'mixture': np.ndarray [2, T],           # 混合信号 (IQ)
                'sources': np.ndarray [K, 2, T],        # 各源信号 (干净)
                'sources_channel': np.ndarray [K, 2, T], # 经过信道的源信号
                'bits': List[np.ndarray],               # 各源的比特序列
                'channel_params': List[dict],           # 各源的信道参数
                'metadata': dict                        # 元数据
            }
        """
        if num_sources is None:
            num_sources = self.collision_config.num_sources

        # 计算信号长度
        signal_length = self.ais_config.total_bits * self.ais_config.samples_per_bit

        # 考虑部分重叠，需要更长的输出
        if self.collision_config.allow_partial_overlap:
            output_length = int(signal_length * 1.5)
        else:
            output_length = signal_length

        # 生成各源信号
        sources = []
        sources_channel = []
        bits_list = []
        channel_params_list = []
        offsets = []

        for k in range(num_sources):
            # 生成AIS信号
            signal, bits = self.ais_generator.generate()

            # 确定时间偏移
            if k == 0:
                offset = 0
            else:
                if self.collision_config.allow_partial_overlap:
                    # 随机重叠比例
                    overlap_ratio = np.random.uniform(
                        *self.collision_config.overlap_ratio_range
                    )
                    max_offset = int(signal_length * (1 - overlap_ratio))
                    offset = np.random.randint(0, max_offset + 1)
                else:
                    offset = 0

            offsets.append(offset)

            # 确定信干比 (相对于第一个信号)
            if k == 0:
                amplitude_factor = 1.0
            else:
                sir_db = np.random.uniform(*self.collision_config.sir_range_db)
                amplitude_factor = 10 ** (-sir_db / 20)

            # 应用信道
            signal_channel, params = self.channel.apply(
                signal,
                amplitude=amplitude_factor,
                snr_db=None,  # 最后统一加噪声
                add_noise=False
            )

            # 放置到输出位置
            source_padded = np.zeros(output_length, dtype=np.complex64)
            end_idx = min(offset + len(signal), output_length)
            source_padded[offset:end_idx] = signal[:end_idx - offset]

            source_channel_padded = np.zeros(output_length, dtype=np.complex64)
            source_channel_padded[offset:end_idx] = signal_channel[:end_idx - offset]

            sources.append(source_padded)
            sources_channel.append(source_channel_padded)
            bits_list.append(bits)
            params['offset'] = offset
            channel_params_list.append(params)

        # 生成混合信号
        mixture = np.sum(sources_channel, axis=0)

        # 添加噪声到混合信号
        if snr_db is None:
            snr_db = np.random.uniform(*self.channel_config.snr_range_db)

        signal_power = np.mean(np.abs(mixture) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(output_length) + 1j * np.random.randn(output_length)
        )
        mixture = mixture + noise

        # 转换为实数表示 [2, T]
        def complex_to_real(x):
            return np.stack([x.real, x.imag], axis=0).astype(np.float32)

        return {
            'mixture': complex_to_real(mixture),
            'sources': np.stack([complex_to_real(s) for s in sources], axis=0),
            'sources_channel': np.stack([complex_to_real(s) for s in sources_channel], axis=0),
            'bits': bits_list,
            'channel_params': channel_params_list,
            'metadata': {
                'num_sources': num_sources,
                'snr_db': snr_db,
                'offsets': offsets,
                'signal_length': signal_length,
                'output_length': output_length
            }
        }

    def generate_batch(self, batch_size: int, device: str = 'cpu') -> dict:
        """批量生成碰撞数据"""
        mixtures = []
        sources = []
        bits = []
        params = []

        for _ in range(batch_size):
            sample = self.generate_collision()
            mixtures.append(sample['mixture'])
            sources.append(sample['sources'])
            bits.append(sample['bits'])
            params.append(sample['channel_params'])

        return {
            'mixture': torch.tensor(np.stack(mixtures), device=device),
            'sources': torch.tensor(np.stack(sources), device=device),
            'bits': bits,
            'channel_params': params
        }


class AISCollisionDataset(torch.utils.data.Dataset):
    """AIS碰撞数据集"""

    def __init__(self,
                 size: int = 10000,
                 ais_config: AISConfig = None,
                 channel_config: ChannelConfig = None,
                 collision_config: CollisionConfig = None,
                 pregenerate: bool = False):
        """
        Args:
            size: 数据集大小
            pregenerate: 是否预生成所有数据 (内存换速度)
        """
        self.size = size
        self.generator = CollisionGenerator(ais_config, channel_config, collision_config)
        self.pregenerate = pregenerate

        if pregenerate:
            print(f"Pregenerating {size} samples...")
            self.data = [self.generator.generate_collision() for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.pregenerate:
            sample = self.data[idx]
        else:
            sample = self.generator.generate_collision()

        return {
            'mixture': torch.tensor(sample['mixture']),
            'sources': torch.tensor(sample['sources']),
            # bits和channel_params在collate_fn中处理
        }

    @staticmethod
    def collate_fn(batch):
        """自定义collate函数"""
        return {
            'mixture': torch.stack([b['mixture'] for b in batch]),
            'sources': torch.stack([b['sources'] for b in batch]),
        }
