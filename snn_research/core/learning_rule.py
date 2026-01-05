# ファイルパス: snn_research/core/learning_rule.py
# タイトル: 抽象学習アルゴリズムインターフェース (高機能版)
# 機能説明: 予測符号化やSTDPなどのカスタム学習則のための基底クラス。

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable, Optional, List
import torch.nn as nn
from torch import Tensor

Parameters = Iterable[nn.Parameter]

class AbstractLearningRule(ABC):
    """
    SNNの重み更新を制御する抽象クラス。
    標準的なBackpropではなく、局所的なエラーや活動に基づく更新をサポート。
    """
    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        self.params: List[nn.Parameter] = list(params)
        self.layer_name: Optional[str] = kwargs.get('layer_name')
        self.hparams: Dict[str, Any] = self._parse_hparams(kwargs)

    def _parse_hparams(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """ハイパーパラメータのパースとバリデーション。"""
        return {
            'learning_rate': float(kwargs.get('learning_rate', 0.01)),
            'weight_decay': float(kwargs.get('weight_decay', 0.0)),
            'meta_lr': float(kwargs.get('meta_lr', 0.001))
        }

    @abstractmethod
    def step(
        self,
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """学習ステップの実行。更新に関連するメトリクスを返す。"""
        pass

    def zero_grad(self) -> None:
        """勾配バッファの初期化。"""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def get_hparams(self) -> Dict[str, Any]:
        return self.hparams