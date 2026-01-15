# snn_research/learning_rules/base_rule.py
# 修正: 戻り値を Any にして、Tensor以外(タプル等)を返す既存実装を許容する

from abc import ABC, abstractmethod
import torch
from typing import Any

class BioLearningRule(ABC):
    """
    生物学的妥当性を持つ学習則の抽象基底クラス。
    """
    @abstractmethod
    def update(self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, **kwargs) -> Any:
        """
        シナプス重みを更新する。
        Returns:
            更新後の重み行列 (Tensor) または (重み, 変化量) のタプルなど
        """
        pass