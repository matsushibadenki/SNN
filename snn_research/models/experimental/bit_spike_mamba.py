# snn_research/models/experimental/bit_spike_mamba.py
# 修正: d_model, d_state などの引数を受け入れるように __init__ を拡張
# 修正: norm と output_projection を追加して mypy エラーを回避

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any

def bit_quantize_weight(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_quant = (w_scaled).round().clamp(-1, 1)
    w_quant = (w_quant - w_scaled).detach() + w_scaled
    return w_quant, scale

class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant, scale = bit_quantize_weight(self.weight)
        return F.linear(x, w_quant) * scale + (self.bias if self.bias is not None else 0)

class BitSpikeMambaModel(nn.Module):
    def __init__(self, 
                 dim: int = 128, 
                 depth: int = 2, 
                 vocab_size: int = 100, 
                 # 互換性のための追加引数
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_layers: int = 2,
                 time_steps: int = 16,
                 neuron_config: Any = None,
                 **kwargs: Any):
        super().__init__()
        
        # 引数の優先順位処理
        self.dim = d_model if d_model is not None else dim
        self.depth = num_layers if num_layers is not None else depth
        
        self.embedding = nn.Embedding(vocab_size, self.dim)
        self.layers = nn.ModuleList([
            BitLinear(self.dim, self.dim) for _ in range(self.depth)
        ])
        
        # Mamba構造に必要な正規化層と射影層を追加
        self.norm = nn.LayerNorm(self.dim)
        self.head = BitLinear(self.dim, vocab_size)
        self.output_projection = self.head
        
        # 内部状態シミュレーション用
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Any:
        # x: (Batch, Seq)
        x_emb = self.embedding(x)
        out = x_emb
        for layer in self.layers:
            out = layer(out)
        
        # 正規化と射影
        out = self.norm(out)
        logits = self.output_projection(out)
        
        if return_spikes:
            # ダミーのスパイクと膜電位を返す
            batch, seq, _ = logits.shape
            spikes = torch.zeros(batch, seq, self.dim, device=x.device)
            mem = torch.zeros_like(spikes)
            return logits, spikes, mem
            
        return logits

    def print_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        print(f"Model Size: {param_size / 1024**2:.3f} MB")

# エイリアス
BitSpikeMamba = BitSpikeMambaModel