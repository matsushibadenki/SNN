# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (汎用化・拡張版)
# Description:
# - すべてのSNNモデルが継承する基底クラス。
# - 修正: 特定のニューロンクラス（AdaptiveLIFNeuron等）へのハードコーディング依存を排除。
#   'total_spikes' バッファを持つ任意のモジュールを自動的に検出し、集計するように変更。
#   これにより、新しいニューロンタイプを追加してもこのファイルを修正する必要がなくなる。

import torch
import torch.nn as nn
from typing import Any, Optional

class SNNLayerNorm(nn.Module):
    """
    SNN用のLayerNorm。
    """
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """
    すべてのSNNモデルが継承する基底クラス。
    """
    def _init_weights(self) -> None:
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def get_total_spikes(self) -> float:
        """
        モデル全体の総スパイク数を計算する。
        特定のクラスに依存せず、'total_spikes' 属性を持つすべてのサブモジュールを集計する。
        """
        total = 0.0
        for module in self.modules():
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                total += module.total_spikes.item()
        return total
    
    def reset_spike_stats(self) -> None:
        """
        スパイク関連の統計情報をリセットする。
        """
        for module in self.modules():
            # spikingjelly準拠の reset()
            if hasattr(module, 'reset') and callable(module.reset):
                module.reset()
            
            # 明示的なカウンタリセット (reset() でクリアされない実装への保険)
            if hasattr(module, 'total_spikes') and isinstance(module.total_spikes, torch.Tensor):
                with torch.no_grad():
                    module.total_spikes.zero_()
            
            if hasattr(module, 'spikes') and isinstance(module.spikes, torch.Tensor):
                with torch.no_grad():
                    module.spikes.zero_()
