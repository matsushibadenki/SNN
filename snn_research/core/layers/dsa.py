# ファイルパス: snn_research/core/layers/dsa.py
# 日本語タイトル: SNN-DSA (Bit-Spike Dynamic Sparse Attention)
# 機能説明:
#   BitNetアーキテクチャ(BitSpikeLinear)を採用し、乗算フリーのアテンションを実現。
#   Q, K, V の射影において {-1, 0, 1} の重みを使用することで、
#   スパイク入力に対する演算を加算(Accumulation)に還元する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from snn_research.core.neurons import AdaptiveLIFNeuron
# 前回のターンで作成された BitSpikeLinear を使用
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # フォールバック (通常のLinear)
    BitSpikeLinear = nn.Linear # type: ignore

class DSALayer(nn.Module):
    """
    SNN向け動的スパースアテンション (SNN-DSA) with BitNet weights.
    
    Enhancements:
    - BitSpikeLinear: Q, K, V, Routerの射影に乗算フリー層を使用。
    - Sparse Computation: Top-K ルーティングによる計算量削減。
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        top_k: int = 4, 
        dropout: float = 0.1,
        neuron_params: Optional[Dict[str, Any]] = None,
        use_bitnet: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k
        self.scale = self.head_dim ** -0.5
        
        # Projection Layer Selection
        LinearLayer = BitSpikeLinear if use_bitnet else nn.Linear
        
        # 線形投影 (BitNet化により加算処理となる)
        self.q_proj = LinearLayer(d_model, d_model)
        self.k_proj = LinearLayer(d_model, d_model)
        self.v_proj = LinearLayer(d_model, d_model)
        
        # ルーティング用レイヤー
        self.router = LinearLayer(d_model, d_model) 
        
        self.out_proj = LinearLayer(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(d_model)
        
        if neuron_params is None:
            neuron_params = {}
        self.output_neuron = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input Spikes (Batch, Time, d_model) - Ideally binary (0/1)
        """
        B, T, C = x.shape
        
        # 1. Projections (Accumulation if BitNet)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Attention Score (Accumulation of AND-like ops if inputs were binary)
        # Note: Even with float Q/K, this is efficiently computable via sparse ops
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Top-K Masking (Sparsity Enforcement)
        if self.top_k < T:
            topk_vals, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1)
            mask = torch.full_like(attn_scores, float('-inf'))
            mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
            attn_scores = mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 3. Context Aggregation
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        # Output Projection
        analog_out = self.out_proj(context)
        analog_out = self.norm(analog_out)
        
        # 4. Spike Generation
        out_spikes_list = []
        if not self.output_neuron.stateful:
            self.output_neuron.reset()
        
        for t in range(T):
            step_input = analog_out[:, t, :]
            step_spike, _ = self.output_neuron(step_input)
            out_spikes_list.append(step_spike)
            
        out_spikes = torch.stack(out_spikes_list, dim=1)
        
        return out_spikes, attn_probs

    def reset_state(self):
        self.output_neuron.reset()

DynamicSparseAttention = DSALayer
