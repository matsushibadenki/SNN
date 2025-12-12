# ファイルパス: snn_research/models/transformer/dsa_transformer.py
# Title: DSA Spiking Transformer モデル [Final Fix]
# Description:
#   - forward メソッドに return_spikes 引数を明示的に追加。
#   - ベンチマークスイートとの互換性を保証。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union

from snn_research.core.layers.dsa import DynamicSparseAttention
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.base import BaseModel

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1, neuron_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if neuron_params is None: neuron_params = {}
        d_ff = d_model * expansion_factor
        self.fc1 = nn.Linear(d_model, d_ff)
        self.neuron1 = AdaptiveLIFNeuron(features=d_ff, **neuron_params)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.neuron2 = AdaptiveLIFNeuron(features=d_model, **neuron_params)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out, _ = self.neuron1(out) 
        out = self.dropout(out)
        out = self.fc2(out)
        out, _ = self.neuron2(out)
        return out
    
    def reset_state(self):
        self.neuron1.reset()
        self.neuron2.reset()

class DSATransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, top_k: int, dropout: float = 0.1, neuron_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dsa = DynamicSparseAttention(
            d_model=d_model, num_heads=num_heads, top_k=top_k, 
            dropout=dropout, neuron_params=neuron_params
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(
            d_model=d_model, dropout=dropout, neuron_params=neuron_params
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.dsa(x)
        x = self.norm1(x + attn_out) 
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x 
    
    def reset_state(self):
        self.dsa.reset_state()
        self.ffn.reset_state()

class DSASpikingTransformer(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        top_k: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        neuron_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.d_model = d_model
        if neuron_params is None: neuron_params = {'base_threshold': 0.5, 'tau_mem': 5.0}
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DSATransformerBlock(
                d_model=d_model, num_heads=num_heads, top_k=top_k, 
                dropout=dropout, neuron_params=neuron_params
            )
            for _ in range(num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, output_dim)
        self._init_weights()

    def forward(
        self, 
        x: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        B, T = x.shape
        x = self.embedding(x)
        
        if T <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, :T, :]
        else:
            pos_emb = self.pos_embedding.repeat(1, (T // self.pos_embedding.shape[1]) + 1, 1)
            x = x + pos_emb[:, :T, :]
            
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_final(x)
        x_mean = x.mean(dim=1) 
        logits = self.classifier(x_mean)
        
        if return_spikes:
            total_spikes = self.get_total_spikes()
            # ダミーのスパイク情報（実際は各層から集計）
            avg_spikes = torch.tensor(total_spikes, device=x.device)
            mem = torch.tensor(0.0, device=x.device)
            return logits, avg_spikes, mem
        
        return logits

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
            
    def get_total_spikes(self) -> float:
        total = 0.0
        for module in self.modules():
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                total += module.total_spikes.item()
            elif hasattr(module, 'spikes') and isinstance(module.spikes, torch.Tensor):
                total += module.spikes.sum().item()
        return total