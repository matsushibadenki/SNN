# ファイルパス: snn_research/core/layers/dsa.py
# 日本語タイトル: SNN-DSA (Dynamic Sparse Attention) レイヤー
# 目的・内容:
#   ロードマップ Phase 8-2 に基づく、動的スパース注意機構の実装。
#   従来の全結合Attentionとは異なり、Routerを用いて重要なトークン/チャネルのみを
#   Top-Kで選択し、計算コストを削減する。
#   入力: (Batch, Time, Dim) のスパイク列
#   出力: (Batch, Time, Dim) のスパイク列

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from snn_research.core.neurons import AdaptiveLIFNeuron

class DynamicSparseAttention(nn.Module):
    """
    SNN向け動的スパースアテンション (SNN-DSA)。
    
    仕組み:
    1. 入力スパイクをQ, K, V, および Router に投影。
    2. Routerがクエリに基づいて「注目すべきK/Vのインデックス」をTop-K個選択。
    3. 選択されたK, Vのみを用いてAttention Scoreを計算 (Sparse Operation)。
    4. 結果を統合し、出力ニューロン(LIF)を通してスパイクとして出力。
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        top_k: int, 
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
        
        # 線形投影 (Biasなしでスパイクのスパース性を維持する設計も可能だが、ここでは標準的にBiasあり)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # ルーティング用レイヤー: 各トークン/時間ステップにおいてどのKeyを見るべきかを予測
        # ここでは自己注意(Self-Attention)を想定し、入力から重要度スコアを算出
        self.router = nn.Linear(d_model, d_model) 
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
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
            attention_maps: Sparse Attention Maps (Debug/Visualization purposes)
        """
        B, T, C = x.shape
        
        # 1. Q, K, V, Router Scoreの計算
        # SNNでは入力xは0/1のスパイクだが、重みを掛けて膜電位(アナログ値)空間へ投影する
        # (Batch, Time, d_model)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        router_scores = self.router(x) # (B, T, d_model) -> (B, T, Target_T) への簡易マッピングと仮定
        
        # Multi-head reshape: (B, T, Num_Heads, Head_Dim) -> (B, Num_Heads, T, Head_Dim)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Dynamic Routing (Top-K Selection)
        # 本来は Router Score に基づいて Key/Value を gather するが、
        # ここでは簡易化のため、Attention Score自体をスパース化するアプローチをとる。
        # (DSAの完全な実装は KV-Cache からの gather を含むが、ここでは行列演算レベルでシミュレート)
        
        # Attention Score: (B, Num_Heads, T, T)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Top-K Masking (Routing Simulation)
        # 各Queryステップ(T)において、スコアが高い Top-K 個の Key だけを残す
        if self.top_k < T:
            topk_vals, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1)
            
            # マスク作成: Top-K 以外を -inf にする
            mask = torch.full_like(attn_scores, float('-inf'))
            mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
            attn_scores = mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 3. Aggregation
        # (B, Num_Heads, T, T) x (B, Num_Heads, T, Head_Dim) -> (B, Num_Heads, T, Head_Dim)
        context = torch.matmul(attn_probs, value)
        
        # Reshape back: (B, T, Num_Heads * Head_Dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        # Output Projection
        analog_out = self.out_proj(context)
        
        # 4. Spike Generation
        # 時間軸方向に展開してニューロンに入力するか、AdaptiveLIFが系列対応している場合はそのまま渡す。
        # AdaptiveLIFNeuronの実装を見ると、(Batch, Time, Features) の入力に対してループ処理が必要か
        # もしくは forward が (Batch, Features) 前提かを確認する必要がある。
        # 提供された __init__.py の AdaptiveLIFNeuron.forward は (Tensor) -> Tuple[Tensor, Tensor]
        # であり、内部で shape チェックをしている。
        # ただし、時間方向のループは networks/cortical_column.py のように外側で回すのが通例。
        # ここでは利便性のため、内部でループ処理を行う。
        
        out_spikes_list = []
        # ニューロンの状態リセット（非ステートフルモード時）
        # ステートフルモードの場合は呼び出し元で制御するが、ここでは毎回初期化と仮定
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