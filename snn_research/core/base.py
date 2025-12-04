# ファイルパス: snn_research/core/base.py
# Title: SNNモデル 基底クラス (全ニューロン対応版)
# Description:
# - BaseModelクラス。
#   修正点:
#   - get_total_spikes, reset_spike_stats の対象クラスに
#     EvolutionaryLeakLIF を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Type

# すべてのニューロンクラスをインポート
from .neurons import (
    AdaptiveLIFNeuron, 
    IzhikevichNeuron, 
    ProbabilisticLIFNeuron,
    GLIFNeuron,
    TC_LIF,
    DualThresholdNeuron,
    ScaleAndFireNeuron,
    BistableIFNeuron,
    EvolutionaryLeakLIF # 追加
)

# 型エイリアス
SNNNeuronType = Union[
    AdaptiveLIFNeuron, 
    IzhikevichNeuron, 
    ProbabilisticLIFNeuron,
    GLIFNeuron,
    TC_LIF,
    DualThresholdNeuron,
    ScaleAndFireNeuron,
    BistableIFNeuron,
    EvolutionaryLeakLIF
]

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
        """
        total = 0.0
        # 認識対象のニューロンクラス (修正: EvolutionaryLeakLIFを追加)
        target_classes = (
            AdaptiveLIFNeuron, 
            IzhikevichNeuron, 
            ProbabilisticLIFNeuron,
            GLIFNeuron,
            TC_LIF,
            DualThresholdNeuron,
            ScaleAndFireNeuron,
            BistableIFNeuron,
            EvolutionaryLeakLIF
        )
        
        for module in self.modules():
            if isinstance(module, target_classes):
                if hasattr(module, 'total_spikes'):
                    total += module.total_spikes.item()
        return total
    
    def reset_spike_stats(self) -> None:
        """
        スパイク関連の統計情報をリセットする。
        """
        target_classes = (
            AdaptiveLIFNeuron, 
            IzhikevichNeuron, 
            ProbabilisticLIFNeuron,
            GLIFNeuron,
            TC_LIF,
            DualThresholdNeuron,
            ScaleAndFireNeuron,
            BistableIFNeuron,
            EvolutionaryLeakLIF
        )

        for module in self.modules():
            if isinstance(module, target_classes):
                if hasattr(module, 'reset'):
                    module.reset()
                # total_spikes が reset() でクリアされない実装の場合に備えて明示的にゼロ化
                if hasattr(module, 'total_spikes'):
                    module.total_spikes.zero_()
