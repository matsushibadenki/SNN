# ファイルパス: snn_research/core/layers/dsa.py
# 日本語タイトル: SNN-DSA (Dynamic Sparse Attention) レイヤー [Fixed Class Name]
# 目的・内容:
#   ロードマップ Phase 8-2 に基づく、動的スパース注意機構の実装。
#   修正: クラス名を 'DSALayer' に変更し、外部からの import エラーを解消。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from snn_research.core.neurons import AdaptiveLIFNeuron

class DSALayer(nn.Module):
    """
    SNN向け動的スパースアテンション (SNN-DSA)。
    
    仕組み:
    1. 入力スパイクをQ, K, V, および Router に投影。
    2. Routerがクエリに基づいて「注目すべきK/Vのインデックス」をTop-K個選択。
    3. 選択されたK, Vのみを用いてAttention Scoreを計算 (Sparse Operation)。
    4. 結果を統合・正規化し、出力ニューロン(LIF)を通してスパイクとして出力。
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        top_k: int = 4, # Default value added
        dropout: float = 0.1,
        neuron_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k
        self.scale = self.head_dim ** -0.5
        
        # 線形投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # ルーティング用レイヤー
        self.router = nn.Linear(d_model, d_model) 
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 正規化層 (発火安定化のため追加)
        self.norm = nn.LayerNorm(d_model)
        
        # 出力スパイク生成用ニューロン
        if neuron_params is None:
            neuron_params = {}
        self.output_neuron = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input Spikes (Batch, Time, d_model)
            
        Returns:
            out_spikes: Output Spikes (Batch, Time, d_model)
            attention_maps: Sparse Attention Maps
        """
        B, T, C = x.shape
        
        # 1. Q, K, V, Router Scoreの計算
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        # router_scores = self.router(x) # 将来的なGatherルーティングで使用予定
        
        # Multi-head reshape: (B, Num_Heads, T, Head_Dim)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Dynamic Routing (Top-K Selection)
        # Attention Score: (B, Num_Heads, T, T)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Top-K Masking
        if self.top_k < T:
            topk_vals, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1)
            
            # マスク作成: Top-K 以外を -inf にする
            mask = torch.full_like(attn_scores, float('-inf'))
            mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
            attn_scores = mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 3. Aggregation
        context = torch.matmul(attn_probs, value)
        
        # Reshape back: (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        # Output Projection & Normalization
        analog_out = self.out_proj(context)
        analog_out = self.norm(analog_out) # 正規化して膜電位を適切なレンジにする
        
        # 4. Spike Generation
        out_spikes_list = []
        
        # ニューロンの状態リセット（非ステートフルモード時）
        if not self.output_neuron.stateful:
            self.output_neuron.reset()
        
        for t in range(T):
            step_input = analog_out[:, t, :]
            step_spike, _ = self.output_neuron(step_input)
            out_spikes_list.append(step_spike)
            
        out_spikes = torch.stack(out_spikes_list, dim=1) # (B, T, C)
        
        return out_spikes, attn_probs

    def reset_state(self):
        """ニューロン状態のリセット"""
        self.output_neuron.reset()

# 互換性のためのエイリアス（もし旧名で参照されている場合）
DynamicSparseAttention = DSALayer