# ファイルパス: snn_research/models/transformer/dsa_transformer.py
# 日本語タイトル: DSA Spiking Transformer モデル
# 目的・内容:
#   Phase 8-2で実装した DynamicSparseAttention を用いた Transformer アーキテクチャ。
#   - DSATransformerBlock: DSA + FeedForward の基本ブロック
#   - DSASpikingTransformer: エンドツーエンドのSNNモデル
#   - Spike-Element-Wise (SEW) ResNetのような残差結合を採用し、深層化しても学習を安定化。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from snn_research.core.layers.dsa import DynamicSparseAttention
from snn_research.core.neurons import AdaptiveLIFNeuron

class FeedForwardBlock(nn.Module):
    """
    SNN用 Feed-Forward Network (FFN)
    Structure: Linear -> LIF -> Linear -> LIF
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1, neuron_params: Dict[str, Any] = None):
        super().__init__()
        if neuron_params is None:
            neuron_params = {}
            
        d_ff = d_model * expansion_factor
        self.fc1 = nn.Linear(d_model, d_ff)
        self.neuron1 = AdaptiveLIFNeuron(features=d_ff, **neuron_params)
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(d_ff, d_model)
        self.neuron2 = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Dim) Spikes
        out = self.fc1(x)
        out, _ = self.neuron1(out) # Spikes
        out = self.dropout(out)
        
        out = self.fc2(out)
        out, _ = self.neuron2(out) # Spikes
        return out
    
    def reset_state(self):
        self.neuron1.reset()
        self.neuron2.reset()

class DSATransformerBlock(nn.Module):
    """
    Dynamic Sparse Attention を含む Transformer Block。
    構成:
      1. DSA (Attention)
      2. Add (Residual) & Norm
      3. FFN
      4. Add (Residual) & Norm
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        top_k: int, 
        dropout: float = 0.1,
        neuron_params: Dict[str, Any] = None
    ):
        super().__init__()
        self.dsa = DynamicSparseAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            top_k=top_k, 
            dropout=dropout,
            neuron_params=neuron_params
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForwardBlock(
            d_model=d_model, 
            dropout=dropout, 
            neuron_params=neuron_params
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. DSA Sub-layer
        # 入力xはスパイク。DSAもスパイクを出力。
        # Residual Connection: x (0/1) + attn_out (0/1) -> (0/1/2)
        # LayerNormによりアナログ値に戻り、次の層の入力として適切にスケーリングされる
        attn_out, _ = self.dsa(x)
        x = self.norm1(x + attn_out) 
        
        # 2. FFN Sub-layer
        # Norm後のアナログ値をFFNに入力しても、FFNの最初のLinearがそれを受け取るのでOK
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x # Next block receives Analog (Normed) values -> OK for Linear input
    
    def reset_state(self):
        self.dsa.reset_state()
        self.ffn.reset_state()

class DSASpikingTransformer(nn.Module):
    """
    Phase 8-2: SNN-DSA搭載 スパイキングTransformerモデル。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        top_k: int = 4,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input Embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder Blocks
        # 発火しやすくするために閾値を調整
        neuron_params = {'base_threshold': 0.5, 'tau_mem': 5.0}
        
        self.layers = nn.ModuleList([
            DSATransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                top_k=top_k, 
                dropout=dropout,
                neuron_params=neuron_params
            )
            for _ in range(num_layers)
        ])
        
        # Output Head
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Time, InputDim) - Analog or Spike input
        Returns:
            logits: (Batch, OutputDim) - Last time step logits or Mean over time
        """
        B, T, _ = x.shape
        
        # Embedding
        x = self.embedding(x)
        
        # Add Positional Encoding
        # シーケンス長に合わせてスライス
        if T <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, :T, :]
        else:
            # 入力が想定より長い場合は繰り返し利用（簡易対応）
            pos_emb = self.pos_embedding.repeat(1, (T // self.pos_embedding.shape[1]) + 1, 1)
            x = x + pos_emb[:, :T, :]
            
        x = self.dropout(x)
        
        # Transformer Blocks
        for layer in self.layers:
            x = layer(x)
            
        # Final Norm
        x = self.norm_final(x)
        
        # Classification (Global Average Pooling over Time for classification)
        # または、Last Tokenを使用するが、SNNでは平均発火率（または平均活動）を使うのが安定する
        x_mean = x.mean(dim=1) 
        logits = self.classifier(x_mean)
        
        return logits

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()