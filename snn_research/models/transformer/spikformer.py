# ファイルパス: snn_research/models/transformer/spikformer.py
# Title: Spikformer (Spiking Transformer with Softmax-less SSA)
# Description:
#   ROADMAP Phase 3 Step 3 実装。
#   - Spiking Self-Attention (SSA): Softmaxを使用せず、Q, K, V のスパイク積でAttentionを計算。
#   - Spiking MLP Block: 活性化関数としてLIFニューロンを使用。
#   - TransformerToMambaAdapter: 視覚野(Spikformer)と前頭前野(Mamba)を接続するアダプター。

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, cast
from spikingjelly.activation_based import functional as SJ_F
from spikingjelly.activation_based import layer

# 既存モジュールからインポート (パスはプロジェクト構造に準拠)
from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
from snn_research.core.base import SNNLayerNorm

class SpikingSelfAttention(nn.Module):
    """
    Softmax-less Spiking Self-Attention (SSA).
    論文 "Spikformer: A Spiking Transformer" に基づく実装。
    Q, K, V をスパイク化し、Attention Map = (Q @ K^T) * scale で計算する。
    Softmaxを使わないため、ハードウェア効率が高く、スパイクの疎性を維持できる。
    """
    def __init__(self, d_model: int, num_heads: int, tau_m: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.125 # Scaling factor suggested in some SNN papers or standard -0.5

        # 線形層 (BN付きを使用することでスパイク発火を安定させる)
        self.q_linear = layer.Linear(d_model, d_model)
        self.q_bn = nn.BatchNorm1d(d_model)
        self.q_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.k_linear = layer.Linear(d_model, d_model)
        self.k_bn = nn.BatchNorm1d(d_model)
        self.k_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.v_linear = layer.Linear(d_model, d_model)
        self.v_bn = nn.BatchNorm1d(d_model)
        self.v_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.attn_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(d_model, d_model)
        self.proj_bn = nn.BatchNorm1d(d_model)
        self.proj_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        # x shape: (T, B, N, D) or (B, N, D) depending on usage.
        # SpikingJelly standard is often (T, B, ...). Assume x is single step (B, N, D) for modularity,
        # or handle time dimension externally.
        # ここではループ内で呼ばれることを想定し、(B, N, D) を処理する形にするが、
        # BN等のためにTime軸との整合性に注意が必要。
        # BN1dは (B, C, L) を期待するため、Reshapeが必要。
        
        B, N, D = x.shape
        
        # Q, K, V Generation
        # Linear -> BN -> LIF
        q = self.q_linear(x) # (B, N, D)
        q = self.q_bn(q.transpose(1, 2)).transpose(1, 2)
        q = self.q_lif(q)

        k = self.k_linear(x)
        k = self.k_bn(k.transpose(1, 2)).transpose(1, 2)
        k = self.k_lif(k)

        v = self.v_linear(x)
        v = self.v_bn(v.transpose(1, 2)).transpose(1, 2)
        v = self.v_lif(v)

        # Multi-head Split
        # (B, N, H, D_h) -> (B, H, N, D_h)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Softmax-less Attention Calculation
        # Attn = (Q @ K^T) * scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Attn @ V
        x_attn = attn @ v # (B, H, N, D_h)
        
        # Combine Heads
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
        
        # Output Projection -> LIF
        x_attn = self.attn_lif(x_attn)
        x_attn = self.proj_linear(x_attn)
        x_attn = self.proj_bn(x_attn.transpose(1, 2)).transpose(1, 2)
        x_attn = self.proj_lif(x_attn)

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
        B, N, D = x.shape
        
        x = self.fc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.lif1(x)
        
        x = self.fc2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.lif2(x)
        
        return x

class SpikformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = SpikingSelfAttention(d_model, num_heads)
        self.mlp = SpikingMLP(d_model, mlp_ratio)
        
    def forward(self, x: torch.Tensor):
        # Residual Connection is handled inherently if spikes are additive, 
        # but typically in SNNs: x = x + block(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Spikformer(nn.Module):
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
        T: int = 4
    ):
        super().__init__()
        self.T = T
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch Embedding (Conv2d)
        self.patch_embed = layer.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.bn_embed = nn.BatchNorm2d(embed_dim)
        self.lif_embed = DualAdaptiveLIFNode(detach_reset=True)

        num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Encoder Blocks
        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Global Average Pooling equivalent (Mean over patches) is usually done later
        
        # Classification Head (Optional, for pre-training)
        self.head = layer.Linear(embed_dim, 1000) # Assuming ImageNet

    def forward_features(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.patch_embed(x) # (B, D, H', W')
        x = self.bn_embed(x)
        x = self.lif_embed(x)
        
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        return x

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (B, T, C, H, W)
        # Time-loop execution
        SJ_F.reset_net(self)
        
        outputs = []
        for t in range(self.T):
            x = x_seq[:, t]
            feat = self.forward_features(x)
            outputs.append(feat)
        
        return torch.stack(outputs, dim=1) # (B, T, N, D)


class TransformerToMambaAdapter(nn.Module):
    """
    Adapter to connect Spikformer (Vision/Space) to BitSpikeMamba (Reasoning/Time).
    Spikformerの出力 (B, T, N_patches, D_vis) を Mambaの入力 (B, L, D_model) に変換する。
    """
    def __init__(self, vis_dim: int, model_dim: int, seq_len: int):
        super().__init__()
        self.proj = nn.Linear(vis_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim)
        # N_patches を維持するか、Global Average Poolingするかはタスク依存。
        # ここではパッチ系列をそのまま時系列の「初期文脈」として扱うケースを想定。

    def forward(self, x_vis: torch.Tensor) -> torch.Tensor:
        # x_vis: (B, T, N, D)
        # 平均レートに変換してMambaに渡す（System 1 -> System 2 へのハンドオーバー）
        # または、T次元をBatchに畳み込むなど。ここでは時間平均をとって静的なEmbeddingにする。
        
        x_mean = x_vis.mean(dim=1) # (B, N, D) - Rate coding interpretation
        x_proj = self.proj(x_mean)
        x_out = self.ln(x_proj)
        
        return x_out # (B, N, D_model) -> Mamba input