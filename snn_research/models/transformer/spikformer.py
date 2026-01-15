# snn_research/models/transformer/spikformer.py
# Title: Spikformer (Phase 2 Optimized)
# Description:
#   T=1 高速推論パスの実装と、SpikingJelly依存度の低減による高速化。
#   [Fix] TransformerToMambaAdapter を確実に定義・エクスポート

import torch
import torch.nn as nn
from typing import List, Optional, Union
from spikingjelly.activation_based import functional as SJ_F
from spikingjelly.activation_based import layer

from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
from snn_research.core.base import BaseModel


class SpikingSelfAttention(nn.Module):
    """
    Softmax-less SSA. T=1 対応版。
    """

    def __init__(self, d_model: int, num_heads: int, tau_m: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.125

        self.q_linear = layer.Linear(d_model, d_model)
        self.q_bn = nn.BatchNorm1d(d_model)
        self.q_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.k_linear = layer.Linear(d_model, d_model)
        self.k_bn = nn.BatchNorm1d(d_model)
        self.k_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.v_linear = layer.Linear(d_model, d_model)
        self.v_bn = nn.BatchNorm1d(d_model)
        self.v_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.attn_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(d_model, d_model)
        self.proj_bn = nn.BatchNorm1d(d_model)
        self.proj_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape

        # Q, K, V
        q = self.q_lif(self.q_bn(self.q_linear(
            x).transpose(1, 2)).transpose(1, 2))
        k = self.k_lif(self.k_bn(self.k_linear(
            x).transpose(1, 2)).transpose(1, 2))
        v = self.v_lif(self.v_bn(self.v_linear(
            x).transpose(1, 2)).transpose(1, 2))

        # Split Heads
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention (SSA)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        # Output
        x_attn = self.attn_lif(x_attn)
        x_attn = self.proj_lif(self.proj_bn(
            self.proj_linear(x_attn).transpose(1, 2)).transpose(1, 2))

        return x_attn


class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, tau_m: float = 2.0):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.fc1 = layer.Linear(d_model, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.fc2 = layer.Linear(hidden_dim, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.lif2 = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        x = self.lif1(self.bn1(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        x = self.lif2(self.bn2(self.fc2(x).transpose(1, 2)).transpose(1, 2))
        return x


class SpikformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = SpikingSelfAttention(d_model, num_heads)
        self.mlp = SpikingMLP(d_model, mlp_ratio)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(BaseModel):
    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        T: int = 4,
        num_classes: int = 1000
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Components
        self.patch_embed = layer.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.bn_embed = nn.BatchNorm2d(embed_dim)
        self.lif_embed = DualAdaptiveLIFNode(detach_reset=True)

        num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

        self.head = layer.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.lif_embed(self.bn_embed(self.patch_embed(x)))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor):
        # Optimized forward pass
        # T=1 の場合は高速パスを使用

        if self.T == 1:
            # Reset only if stateful (Single step inference usually resets before or doesn't keep state)
            # SJ_F.reset_net(self) # Avoid heavy reset for T=1 if managed externally

            if x.dim() == 5:  # (B, T, C, H, W)
                x = x.squeeze(1)

            feat = self.forward_features(x)  # (B, N, D)
            x_gap = feat.mean(dim=1)
            return self.head(x_gap)

        else:
            # Temporal Processing
            if x.dim() == 4:
                x_seq = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
            else:
                x_seq = x

            SJ_F.reset_net(self)  # Necessary for multi-step

            outputs_list = []
            for t in range(self.T):
                outputs_list.append(self.forward_features(x_seq[:, t]))

            outputs = torch.stack(outputs_list, dim=1)
            x_mean = outputs.mean(dim=1)
            x_gap = x_mean.mean(dim=1)
            return self.head(x_gap)


class TransformerToMambaAdapter(nn.Module):
    def __init__(self, vis_dim: int, model_dim: int, seq_len: Optional[int] = None):
        super().__init__()
        self.proj = nn.Linear(vis_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x_vis: torch.Tensor) -> torch.Tensor:
        if x_vis.dim() == 4:
            x_vis = x_vis.mean(dim=1)
        return self.ln(self.proj(x_vis))
