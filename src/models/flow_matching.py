# src/models/flow_matching.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码 (用于时间步嵌入)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionalLayerNorm(nn.Module):
    """条件Layer Normalization"""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return gamma * x + beta


class FlowMatchingBlock(nn.Module):
    """
    Flow Matching Transformer Block
    用于学习向量场 v(t, x_t, condition)
    """

    def __init__(self,
                 dim: int = 512,
                 num_heads: int = 8,
                 ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.dim = dim

        # Self-attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention with condition
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self,
                x: torch.Tensor,
                condition: torch.Tensor,
                time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 当前状态
            condition: [B, T, D] 条件 (混合信号编码)
            time_emb: [B, D] 时间步嵌入
        """
        # Time modulation
        time_mod = self.time_mlp(time_emb).unsqueeze(1)

        # Self-attention
        x_norm = self.attn_norm(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]

        # Cross-attention with condition
        x_norm = self.cross_norm(x)
        x = x + self.cross_attn(x_norm, condition, condition)[0]

        # Feed-forward with time modulation
        x_norm = self.ff_norm(x)
        x = x + self.ff(x_norm + time_mod)

        return x


class VectorFieldNetwork(nn.Module):
    """
    向量场网络
    估计 v(t, x_t, condition)
    """

    def __init__(self,
                 dim: int = 512,
                 num_sources: int = 2,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_sources = num_sources

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # 输入投影 (包含source index)
        self.input_proj = nn.Linear(dim + num_sources, dim)

        # Transformer层
        self.layers = nn.ModuleList([
            FlowMatchingBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor,
                source_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, K, T, D] K个源的当前状态
            t: [B] 时间步
            condition: [B, T, D] 条件 (混合信号)
            source_idx: [B, K] source索引的one-hot

        Returns:
            v: [B, K, T, D] 向量场
        """
        B, K, T, D = x.shape

        # 时间嵌入
        time_emb = self.time_embed(t)  # [B, D]

        # 处理每个源
        outputs = []
        for k in range(K):
            x_k = x[:, k]  # [B, T, D]

            # 添加source index
            if source_idx is not None:
                src_emb = source_idx[:, k].unsqueeze(1).expand(-1, T, -1)  # [B, T, K]
                x_k = torch.cat([x_k, src_emb], dim=-1)
            else:
                # 默认one-hot
                src_emb = F.one_hot(torch.tensor(k), K).float().to(x.device)
                src_emb = src_emb.view(1, 1, K).expand(B, T, -1)
                x_k = torch.cat([x_k, src_emb], dim=-1)

            x_k = self.input_proj(x_k)

            # Transformer层
            for layer in self.layers:
                x_k = layer(x_k, condition, time_emb)

            outputs.append(self.output_proj(x_k))

        return torch.stack(outputs, dim=1)  # [B, K, T, D]


class FlowMatchingSeparator(nn.Module):
    """
    基于Flow Matching的信号分离器
    实现FLOSS的核心算法
    """

    def __init__(self,
                 dim: int = 512,
                 num_sources: int = 2,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()

        self.dim = dim
        self.num_sources = num_sources

        # 向量场网络
        self.vector_field = VectorFieldNetwork(
            dim=dim,
            num_sources=num_sources,
            num_layers=num_layers,
            num_heads=num_heads
        )

        # 混合信号均值投影 (用于初始化)
        self.mean_proj = nn.Linear(dim, dim)

    def forward(self,
                condition: torch.Tensor,
                num_steps: int = 10,
                method: str = 'euler') -> torch.Tensor:
        """
        采样过程: 从噪声分布生成分离信号

        Args:
            condition: [B, T, D] 混合信号的编码
            num_steps: 采样步数
            method: 'euler' or 'midpoint'

        Returns:
            separated: [B, K, T, D] 分离后的信号特征
        """
        B, T, D = condition.shape
        K = self.num_sources
        device = condition.device

        # 计算混合信号均值
        s_bar = self.mean_proj(condition)  # [B, T, D]

        # 初始化: x_0 = s_bar + noise (混合一致性初始化)
        # 参考FLOSS: x_0 = P·s_bar + P_perp·z
        noise = torch.randn(B, K, T, D, device=device)
        x_t = s_bar.unsqueeze(1).expand(-1, K, -1, -1) + noise * 0.5

        # Euler采样
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.ones(B, device=device) * (step * dt)

            if method == 'euler':
                v = self.vector_field(x_t, t, condition)
                x_t = x_t + v * dt
            elif method == 'midpoint':
                v1 = self.vector_field(x_t, t, condition)
                x_mid = x_t + v1 * (dt / 2)
                t_mid = t + dt / 2
                v2 = self.vector_field(x_mid, t_mid, condition)
                x_t = x_t + v2 * dt

            # 可选: 混合一致性约束投影
            # x_t_mean = x_t.mean(dim=1, keepdim=True)
            # x_t = x_t - x_t_mean + s_bar.unsqueeze(1)

        return x_t

    def compute_loss(self,
                     sources: torch.Tensor,
                     condition: torch.Tensor,
                     sigma_min: float = 0.001) -> torch.Tensor:
        """
        计算Flow Matching训练损失

        Args:
            sources: [B, K, T, D] 真实源信号特征
            condition: [B, T, D] 混合信号特征

        Returns:
            loss: scalar
        """
        B, K, T, D = sources.shape
        device = sources.device

        # 采样时间 t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # 采样噪声
        noise = torch.randn_like(sources)

        # 计算 x_t = (1-t) * noise + t * sources (线性插值)
        t_expanded = t.view(B, 1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * sources

        # 目标向量场: v* = sources - noise
        target_v = sources - noise

        # 预测向量场
        pred_v = self.vector_field(x_t, t, condition)

        # MSE损失
        loss = F.mse_loss(pred_v, target_v)

        return loss
