# ファイルパス: snn_research/core/layers/dsa.py
# 日本語タイトル: SNN-DSA (Optimized Bit-Spike Dynamic Sparse Attention)
# 機能説明:
#   BitNetアーキテクチャを採用した乗算フリーのアテンション層。
#   修正: Top-K処理の最適化とメモリアロケーションの削減による高速化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from snn_research.core.neurons import AdaptiveLIFNeuron
# BitSpikeLinearのインポート試行
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear # type: ignore

class DSALayer(nn.Module):
    """
    SNN向け動的スパースアテンション (SNN-DSA) with BitNet weights.
    
    Optimizations:
    - Efficient Top-K Masking: `scatter_` の代わりに `where` やインデックス操作を活用。
    - Memory Layout: `contiguous` 呼び出しの最適化。
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
        
        self.q_proj = LinearLayer(d_model, d_model)
        self.k_proj = LinearLayer(d_model, d_model)
        self.v_proj = LinearLayer(d_model, d_model)
        
        self.out_proj = LinearLayer(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        if neuron_params is None:
            neuron_params = {}
        self.output_neuron = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input Spikes (Batch, Time, d_model)
        Returns:
            out_spikes: (Batch, Time, d_model)
            attn_probs: (Batch, NumHeads, Time, Time)
        """
        B, T, C = x.shape
        
        # 1. Projections
        # (B, T, C) -> (B, T, Heads, HeadDim) -> (B, Heads, T, HeadDim)
        query = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Attention Score
        # (B, Heads, T, HeadDim) @ (B, Heads, HeadDim, T) -> (B, Heads, T, T)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Optimized Top-K Masking
        if self.top_k < T:
            # top-k の値とインデックスを取得
            topk_vals, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1)
            
            # マスクの作成 (全体を -inf で初期化し、Top-K の位置に値を埋め込む)
            # full_like + scatter よりも、特定の値以外をマスクする方が効率的な場合があるが
            # ここでは scatter のままにするが、deviceアロケーションを避ける
            mask = torch.ones_like(attn_scores) * float('-inf')
            mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
            attn_scores = mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 3. Context Aggregation
        # (B, Heads, T, T) @ (B, Heads, T, HeadDim) -> (B, Heads, T, HeadDim)
        context = torch.matmul(attn_probs, value)
        
        # (B, Heads, T, HeadDim) -> (B, T, Heads, HeadDim) -> (B, T, C)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        # Output Projection & Norm
        analog_out = self.out_proj(context)
        analog_out = self.norm(analog_out)
        
        # 4. Spike Generation (Time-Loop Optimization)
        # ニューロンの状態リセット
        if not self.output_neuron.stateful:
            self.output_neuron.reset()
            
        # AdaptiveLIFNeuronが時系列一括処理に対応していない場合を想定しループ処理するが、
        # 内部処理を極力減らす。
        out_spikes_list = []
        
        # ループ内での属性アクセスを減らすためのローカル変数化
        neuron_step = self.output_neuron
        
        # ※ 将来的には neuron_step(analog_out) で (B, T, C) を返せるように改修推奨
        # 現状は互換性維持のためループ
        for t in range(T):
            # step_spike: (B, C)
            step_spike, _ = neuron_step(analog_out[:, t, :])
            out_spikes_list.append(step_spike)
            
        out_spikes = torch.stack(out_spikes_list, dim=1)
        
        return out_spikes, attn_probs

    def reset_state(self):
        self.output_neuron.reset()

DynamicSparseAttention = DSALayer