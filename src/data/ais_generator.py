# src/data/ais_generator.py

import numpy as np
from scipy.signal import lfilter
from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class AISConfig:
    """AIS信号配置"""
    bit_rate: int = 9600           # bps
    sample_rate: int = 96000       # 10x过采样
    bt_product: float = 0.4        # 高斯滤波器BT积
    modulation_index: float = 0.5  # GMSK调制指数
    preamble_bits: int = 24
    start_flag: int = 0x7E
    data_bits: int = 168
    crc_bits: int = 16
    end_flag: int = 0x7E
    buffer_bits: int = 32

    @property
    def samples_per_bit(self) -> int:
        return self.sample_rate // self.bit_rate

    @property
    def total_bits(self) -> int:
        return self.preamble_bits + 8 + self.data_bits + self.crc_bits + 8 + self.buffer_bits


class AISSignalGenerator:
    """AIS信号生成器"""

    def __init__(self, config: AISConfig = None):
        self.config = config or AISConfig()
        self._init_gaussian_filter()

    def _init_gaussian_filter(self):
        """初始化高斯滤波器用于GMSK"""
        # 高斯滤波器长度 (通常取3-4个符号周期)
        L = 4  # 符号周期数
        sps = self.config.samples_per_bit
        BT = self.config.bt_product

        # 时间轴
        t = np.arange(-L * sps // 2, L * sps // 2 + 1) / sps

        # 高斯滤波器冲激响应
        alpha = np.sqrt(np.log(2) / 2) / BT
        self.gaussian_filter = np.sqrt(np.pi) / alpha * np.exp(-(np.pi * t / alpha) ** 2)
        self.gaussian_filter /= np.sum(self.gaussian_filter)  # 归一化

    def generate_ais_bits(self, message_id: int = 1, mmsi: int = None) -> np.ndarray:
        """
        生成AIS比特序列

        Args:
            message_id: AIS消息类型 (1-27)
            mmsi: 船舶识别码 (9位数字)

        Returns:
            完整的AIS比特序列
        """
        # 前导序列
        preamble = np.array([0, 1] * (self.config.preamble_bits // 2))

        # 起始标志
        start_flag = self._byte_to_bits(self.config.start_flag)

        # 数据域 (生成随机或指定消息)
        if mmsi is None:
            mmsi = np.random.randint(100000000, 999999999)
        data_bits = self._generate_ais_message(message_id, mmsi)

        # CRC-16
        crc = self._compute_crc16(data_bits)
        crc_bits = self._int_to_bits(crc, 16)

        # 结束标志
        end_flag = self._byte_to_bits(self.config.end_flag)

        # 缓冲
        buffer = np.zeros(self.config.buffer_bits, dtype=np.int8)

        # NRZI编码数据和CRC
        payload = np.concatenate([data_bits, crc_bits])
        payload_nrzi = self._nrzi_encode(payload)

        # 组装完整帧
        frame = np.concatenate([preamble, start_flag, payload_nrzi, end_flag, buffer])

        return frame.astype(np.float32)

    def _nrzi_encode(self, bits: np.ndarray) -> np.ndarray:
        """NRZI编码: 0保持电平，1翻转电平"""
        encoded = np.zeros_like(bits)
        state = 1
        for i, bit in enumerate(bits):
            if bit == 0:
                state = 1 - state  # 翻转
            encoded[i] = state
        return encoded

    def _generate_ais_message(self, msg_id: int, mmsi: int) -> np.ndarray:
        """生成AIS消息数据域"""
        # 简化实现: 生成随机数据但保持正确的消息ID和MMSI
        bits = np.zeros(self.config.data_bits, dtype=np.int8)

        # 消息ID (6 bits)
        bits[0:6] = self._int_to_bits(msg_id, 6)

        # 重复指示 (2 bits)
        bits[6:8] = [0, 0]

        # MMSI (30 bits)
        bits[8:38] = self._int_to_bits(mmsi, 30)

        # 其余随机填充
        bits[38:] = np.random.randint(0, 2, self.config.data_bits - 38)

        return bits

    def _compute_crc16(self, data: np.ndarray) -> int:
        """计算CRC-16 (ITU标准)"""
        # 多项式: x^16 + x^12 + x^5 + 1 (0x1021)
        crc = 0xFFFF
        poly = 0x1021

        for bit in data:
            crc ^= (int(bit) << 15)
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

        return crc ^ 0xFFFF

    def modulate_gmsk(self, bits: np.ndarray) -> np.ndarray:
        """
        GMSK调制

        Args:
            bits: 比特序列 (0/1)

        Returns:
            复基带IQ信号
        """
        sps = self.config.samples_per_bit
        h = self.config.modulation_index

        # NRZ编码: 0 -> -1, 1 -> +1
        nrz = 2 * bits.astype(np.float32) - 1

        # 上采样
        upsampled = np.zeros(len(nrz) * sps)
        upsampled[::sps] = nrz

        # 高斯滤波
        filtered = np.convolve(upsampled, self.gaussian_filter, mode='same')

        # 频率脉冲积分得到相位
        phase = np.cumsum(filtered) * (np.pi * h / sps)

        # 生成复基带信号
        signal = np.exp(1j * phase)

        return signal.astype(np.complex64)

    def generate(self, message_id: int = 1, mmsi: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整的AIS信号

        Returns:
            (iq_signal, bits): 复基带信号和原始比特
        """
        bits = self.generate_ais_bits(message_id, mmsi)
        signal = self.modulate_gmsk(bits)
        return signal, bits

    @staticmethod
    def _byte_to_bits(byte: int) -> np.ndarray:
        return np.array([(byte >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)

    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> np.ndarray:
        return np.array([(value >> i) & 1 for i in range(n_bits - 1, -1, -1)], dtype=np.int8)


class BatchAISGenerator:
    """批量AIS信号生成器 (用于训练数据生成)"""

    def __init__(self, config: AISConfig = None):
        self.generator = AISSignalGenerator(config)
        self.config = self.generator.config

    def generate_batch(self, batch_size: int, device: str = 'cpu') -> dict:
        """
        生成一批AIS信号

        Returns:
            {
                'signals': [B, 2, T] 实数表示的IQ信号,
                'bits': [B, N] 原始比特序列
            }
        """
        signals = []
        bits_list = []

        for _ in range(batch_size):
            signal, bits = self.generator.generate()
            # 转换为实数表示 [2, T]
            signal_real = np.stack([signal.real, signal.imag], axis=0)
            signals.append(signal_real)
            bits_list.append(bits)

        return {
            'signals': torch.tensor(np.stack(signals), dtype=torch.float32, device=device),
            'bits': torch.tensor(np.stack(bits_list), dtype=torch.float32, device=device)
        }
