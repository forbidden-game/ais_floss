# src/models/floss_ais.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoder import IQEncoder, MultiScaleIQEncoder
from .flow_matching import FlowMatchingSeparator
from .decoder import IQDecoder, BitDecoder


class FLOSSAIS(nn.Module):
    """
    FLOSS-AIS: 完整的信号分离模型
    """

    def __init__(self,
                 # 编码器参数
                 encoder_type: str = 'standard',  # 'standard' or 'multiscale'
                 encoder_dim: int = 512,

                 # 分离器参数
                 num_sources: int = 2,
                 separator_layers: int = 6,
                 separator_heads: int = 8,

                 # 解码器参数
                 decode_bits: bool = False,
                 num_bits: int = 256,
                 samples_per_bit: int = 10,

                 # 采样参数
                 num_sampling_steps: int = 10):
        super().__init__()

        self.num_sources = num_sources
        self.num_sampling_steps = num_sampling_steps
        self.decode_bits = decode_bits

        # 编码器
        if encoder_type == 'standard':
            self.encoder = IQEncoder(
                in_channels=2,
                out_dim=encoder_dim
            )
        else:
            self.encoder = MultiScaleIQEncoder(
                in_channels=2,
                out_dim=encoder_dim
            )

        # Flow Matching分离器
        self.separator = FlowMatchingSeparator(
            dim=encoder_dim,
            num_sources=num_sources,
            num_layers=separator_layers,
            num_heads=separator_heads
        )

        # IQ解码器
        self.iq_decoder = IQDecoder(
            in_dim=encoder_dim,
            upsample_factor=self.encoder.downsample_factor
        )

        # 可选的比特解码器
        if decode_bits:
            self.bit_decoder = BitDecoder(
                num_bits=num_bits,
                samples_per_bit=samples_per_bit
            )

    def forward(self,
                mixture: torch.Tensor,
                sources: torch.Tensor = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            mixture: [B, 2, T] 混合信号
            sources: [B, K, 2, T] 真实源信号 (训练时)
            return_features: 是否返回中间特征

        Returns:
            dict containing:
                - 'separated_signals': [B, K, 2, T]
                - 'loss': scalar (if sources provided)
                - 'bits': [B, K, num_bits] (if decode_bits)
        """
        B = mixture.shape[0]
        output = {}

        # 编码混合信号
        mixture_features = self.encoder(mixture)  # [B, D, T']
        mixture_features = mixture_features.permute(0, 2, 1)  # [B, T', D]

        if sources is not None:
            # 训练模式: 计算损失
            # 编码源信号
            K = sources.shape[1]
            sources_flat = sources.view(B * K, 2, -1)
            sources_features = self.encoder(sources_flat)  # [B*K, D, T']
            sources_features = sources_features.view(B, K, -1, sources_features.shape[-1])
            sources_features = sources_features.permute(0, 1, 3, 2)  # [B, K, T', D]

            # 计算Flow Matching损失
            loss = self.separator.compute_loss(sources_features, mixture_features)
            output['loss'] = loss

        # 分离 (采样)
        separated_features = self.separator(
            mixture_features,
            num_steps=self.num_sampling_steps
        )  # [B, K, T', D]

        # 解码IQ信号
        separated_features = separated_features.permute(0, 1, 3, 2)  # [B, K, D, T']
        separated_signals = self.iq_decoder(separated_features)  # [B, K, 2, T]
        output['separated_signals'] = separated_signals

        # 可选: 解码比特
        if self.decode_bits:
            bits = []
            for k in range(self.num_sources):
                bits_k = self.bit_decoder(separated_signals[:, k])
                bits.append(bits_k)
            output['bits'] = torch.stack(bits, dim=1)  # [B, K, num_bits]

        if return_features:
            output['mixture_features'] = mixture_features
            output['separated_features'] = separated_features

        return output

    def separate(self,
                 mixture: torch.Tensor,
                 num_steps: int = None) -> torch.Tensor:
        """
        推理模式: 分离混合信号

        Args:
            mixture: [B, 2, T] 混合信号
            num_steps: 采样步数 (默认使用初始化时的值)

        Returns:
            separated: [B, K, 2, T] 分离信号
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps

        # 临时修改采样步数
        old_steps = self.num_sampling_steps
        self.num_sampling_steps = num_steps

        with torch.no_grad():
            output = self.forward(mixture)

        self.num_sampling_steps = old_steps

        return output['separated_signals']


class FLOSSAISLoss(nn.Module):
    """
    FLOSS-AIS 综合损失函数
    """

    def __init__(self,
                 flow_weight: float = 1.0,
                 reconstruction_weight: float = 0.1,
                 consistency_weight: float = 0.1,
                 bit_weight: float = 0.5):
        super().__init__()

        self.flow_weight = flow_weight
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.bit_weight = bit_weight

    def forward(self,
                model_output: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算综合损失

        Args:
            model_output: 模型输出
            targets: {
                'sources': [B, K, 2, T],
                'mixture': [B, 2, T],
                'bits': [B, K, num_bits] (optional)
            }
        """
        losses = {}

        # Flow Matching损失 (主损失)
        if 'loss' in model_output:
            losses['flow_loss'] = model_output['loss'] * self.flow_weight

        # 重构损失
        if 'separated_signals' in model_output and 'sources' in targets:
            separated = model_output['separated_signals']
            sources = targets['sources']

            # 处理置换模糊 (PIT)
            recon_loss = self._pit_loss(separated, sources)
            losses['reconstruction_loss'] = recon_loss * self.reconstruction_weight

        # 混合一致性损失
        if 'separated_signals' in model_output and 'mixture' in targets:
            separated = model_output['separated_signals']
            mixture = targets['mixture']

            reconstructed_mixture = separated.sum(dim=1)  # [B, 2, T]
            consistency_loss = F.mse_loss(reconstructed_mixture, mixture)
            losses['consistency_loss'] = consistency_loss * self.consistency_weight

        # 比特损失
        if 'bits' in model_output and 'bits' in targets:
            pred_bits = model_output['bits']
            target_bits = targets['bits'].float()
            bit_loss = F.binary_cross_entropy(pred_bits, target_bits)
            losses['bit_loss'] = bit_loss * self.bit_weight

        # 总损失
        losses['total_loss'] = sum(losses.values())

        return losses

    def _pit_loss(self,
                  pred: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        """
        Permutation Invariant Training (PIT) 损失
        处理输出源与真实源的置换模糊
        """
        B, K, C, T = pred.shape

        if K == 2:
            # K=2时直接枚举
            loss1 = F.mse_loss(pred[:, 0], target[:, 0]) + F.mse_loss(pred[:, 1], target[:, 1])
            loss2 = F.mse_loss(pred[:, 0], target[:, 1]) + F.mse_loss(pred[:, 1], target[:, 0])
            return torch.min(loss1, loss2)
        else:
            # K>2时使用匈牙利算法 (简化版: 贪心)
            # 实际应用中建议使用scipy.optimize.linear_sum_assignment
            total_loss = 0
            for b in range(B):
                # 计算成本矩阵
                cost = torch.zeros(K, K, device=pred.device)
                for i in range(K):
                    for j in range(K):
                        cost[i, j] = F.mse_loss(pred[b, i], target[b, j])

                # 贪心匹配
                used = set()
                batch_loss = 0
                for i in range(K):
                    min_j = -1
                    min_cost = float('inf')
                    for j in range(K):
                        if j not in used and cost[i, j] < min_cost:
                            min_cost = cost[i, j]
                            min_j = j
                    used.add(min_j)
                    batch_loss += min_cost

                total_loss += batch_loss

            return total_loss / B
