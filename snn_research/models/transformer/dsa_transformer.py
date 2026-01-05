# ファイルパス: snn_research/models/transformer/dsa_transformer.py
# Title: Spiking DSA Transformer (BitNet Scaled & Causal)
# 機能: DSAを用いた軽量かつ高性能なSNNトランスフォーマー。
# 修正: mypy型エラー (戻り値型、Embedding型、代入型) の修正

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Type
from snn_research.core.neurons import AdaptiveLIFNeuron

# 最適化されたDSA層とBitNet層をインポート
from snn_research.core.layers.dsa import DSALayer

# 型チェック回避のための動的インポート処理
BitSpikeLinear: Type[nn.Module]
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear  # type: ignore


class SpikingDSABlock(nn.Module):
    """
    1つのTransformerブロック:
    DSA Attention -> LayerNorm -> FeedForward -> LayerNorm
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1, use_bitnet: bool = True, is_causal: bool = True):
        super().__init__()

        # Attention Layer (Optimized SNN-DSA with Causal Masking)
        self.attn = DSALayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_bitnet=use_bitnet,
            is_causal=is_causal
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward Network (FFN)
        Linear = BitSpikeLinear if use_bitnet else nn.Linear

        self.fc1 = Linear(d_model, dim_feedforward)
        self.fc2 = Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN内部の活性化関数としてスパイキングニューロンを使用
        self.act = AdaptiveLIFNeuron(features=dim_feedforward)

    def forward(self, x: torch.Tensor, return_spikes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # x: (Batch, Time, d_model)

        collected_spikes: List[torch.Tensor] = []

        # 1. Attention Block
        attn_out, attn_spikes = self.attn(x)

        if return_spikes and attn_spikes is not None:
            collected_spikes.append(attn_spikes)

        x = self.norm1(x + self.dropout(attn_out))

        # 2. Feed Forward Block
        ff_out = self.fc1(x)

        if hasattr(self.act, 'stateful') and not self.act.stateful:
            self.act.reset()

        # 時系列データとして処理
        spike_out_list = []
        for t in range(ff_out.shape[1]):
            s, _ = self.act(ff_out[:, t, :])
            spike_out_list.append(s)
        ff_out_spikes = torch.stack(spike_out_list, dim=1)

        if return_spikes:
            collected_spikes.append(ff_out_spikes)

        ff_out = self.fc2(ff_out_spikes)

        # 残差接続 + Norm
        x = self.norm2(x + self.dropout(ff_out))

        if return_spikes:
            return x, collected_spikes
        else:
            return x


class SpikingDSATransformer(nn.Module):
    """
    SNN-DSA Transformer Model (Main Architecture)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        time_window: int = 16,
        use_bitnet: bool = True,
        num_classes: int = 10,
        vocab_size: Optional[int] = None,
        is_causal: bool = True
    ):
        super().__init__()
        input_dim = int(input_dim)
        d_model = int(d_model)

        self.d_model = d_model
        self.time_window = time_window
        self.vocab_size = vocab_size

        Linear = BitSpikeLinear if use_bitnet else nn.Linear

        # Embedding / Projection Layer
        self.embedding: Optional[nn.Embedding]
        if vocab_size is not None and vocab_size > 0:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.input_proj = None
        else:
            self.embedding = None
            self.input_proj = Linear(input_dim, d_model)

        # Positional Embedding (Learnable)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, time_window, d_model) * 0.02)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            SpikingDSABlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                use_bitnet=use_bitnet,
                is_causal=is_causal
            )
            for _ in range(num_layers)
        ])

        # Output Head
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, return_spikes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # 入力が静止画(2D)の場合、時間軸へ拡張
        if x.dim() == 2:
            if self.embedding is not None and (x.dtype == torch.long or x.dtype == torch.int):
                pass
            else:
                x = x.unsqueeze(1).expand(-1, self.time_window, -1)

        # Embedding or Projection
        if self.embedding is not None and (x.dtype == torch.long or x.dtype == torch.int):
            x = self.embedding(x)
        elif self.input_proj is not None:
            x = self.input_proj(x)

        B, T, _ = x.shape

        # Add Positional Embedding
        max_len = self.pos_embedding.shape[1]
        if T <= max_len:
            x = x + self.pos_embedding[:, :T, :]
        else:
            repeat_count = (T // max_len) + 1
            pe = self.pos_embedding.repeat(1, repeat_count, 1)
            x = x + pe[:, :T, :]

        all_spikes: List[torch.Tensor] = []

        # Apply Blocks
        for layer in self.layers:
            if return_spikes:
                x, layer_spikes = layer(x, return_spikes=True)
                if isinstance(layer_spikes, list):
                    all_spikes.extend(layer_spikes)
            else:
                x = layer(x)

        # Classification (Global Average Pooling)
        x_mean = x.mean(dim=1)

        logits = self.classifier(x_mean)

        if return_spikes:
            return logits, all_spikes

        return logits


# Registryとの互換性のためのエイリアス
DSASpikingTransformer = SpikingDSATransformer
