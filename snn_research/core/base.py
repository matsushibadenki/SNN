# ファイルパス: snn_research/core/base.py
# (更新: 全ニューロンタイプ対応)
# Title: SNNモデル 基底クラス
# Description:
# - 複数のモデルアーキテクチャで共有される基底クラス(BaseModel)と共通レイヤー(SNNLayerNorm)。
# - get_total_spikes および reset_spike_stats メソッドを更新し、
#   プロジェクトで定義されたすべてのニューロンタイプ (LIF, Izhikevich, GLIF, TC_LIF, etc.)
#   を正しく認識して処理できるようにする。

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
    BistableIFNeuron
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
    BistableIFNeuron
]

class SNNLayerNorm(nn.Module):
    """
    SNN用のLayerNorm。
    時間方向 (Time) とバッチ方向 (Batch) がある場合、
    通常は (B, T, D) に対して D の正規化を行う。
    nn.LayerNorm は最後の次元に対して正規化を行うため、そのまま使用可能。
    """
    def __init__(self, normalized_shape: Any, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class BaseModel(nn.Module):
    """
    すべてのSNNモデルが継承する基底クラス。
    重みの初期化やスパイク統計の共通メソッドを提供する。
    """
    def _init_weights(self) -> None:
        """
        重みの初期化を行うユーティリティ。
        """
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
        各ニューロン層が持つ 'total_spikes' バッファを集計する。
        """
        total = 0.0
        # 認識対象のニューロンクラス
        target_classes = (
            AdaptiveLIFNeuron, 
            IzhikevichNeuron, 
            ProbabilisticLIFNeuron,
            GLIFNeuron,
            TC_LIF,
            DualThresholdNeuron,
            ScaleAndFireNeuron,
            BistableIFNeuron
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
            BistableIFNeuron
        )

        for module in self.modules():
            if isinstance(module, target_classes):
                if hasattr(module, 'reset'):
                    module.reset()
                # total_spikes が reset() でクリアされない実装の場合に備えて明示的にゼロ化
                if hasattr(module, 'total_spikes'):
                    module.total_spikes.zero_()