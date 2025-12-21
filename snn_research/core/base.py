# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (最適化版)
# Description: DDP対応のスパイク集計と標準的なPyTorch状態管理を統合。

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Optional

class SNNLayerNorm(nn.Module):
    """SNN用のLayerNorm。"""
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """すべてのSNNモデルが継承する基底クラス。"""
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_total_spikes(self) -> float:
        """全デバイスの総スパイク数を集計。"""
        total = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else None)
        
        for m in self.modules():
            if hasattr(m, 'total_spikes') and isinstance(m.total_spikes, torch.Tensor):
                total += m.total_spikes
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        return total.item()
    
    def reset_spike_stats(self) -> None:
        """統計情報のリセット。"""
        for m in self.modules():
            if hasattr(m, 'reset') and callable(m.reset):
                m.reset()
            for attr in ['total_spikes', 'spikes']:
                val = getattr(m, attr, None)
                if isinstance(val, torch.Tensor):
                    val.detach().zero_()
