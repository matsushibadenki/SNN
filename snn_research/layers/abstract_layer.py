# ファイルパス: snn_research/layers/abstract_layer.py
# 日本語タイトル: 抽象ネットワークレイヤー基底クラス (修正版)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 2, P2-1) に基づく、全てのネットワークレイヤーの基底クラス。
#   nn.Moduleを継承し、PyTorchのエコシステムと互換性を持ちつつ、
#   局所学習則 (LearningRule) を保持・適用するためのインターフェースを提供する。
#   
#   修正点:
#   - 循環インポートを避けるため、AbstractLearningRuleのインポートをTYPE_CHECKING内に移動。
#   - 絶対インポートを使用。

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterable, List, TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

# 設定クラスはデータクラスなので循環のリスクが低いため通常インポート
try:
    from snn_research.config.learning_config import BaseLearningConfig
except ImportError:
    BaseLearningConfig = Any # type: ignore

if TYPE_CHECKING:
    # 循環参照回避のため型チェック時のみインポート
    from snn_research.core.learning_rule import AbstractLearningRule

# 型エイリアス
Parameters = Iterable[nn.Parameter]
LayerOutput = Dict[str, Tensor]
UpdateMetrics = Dict[str, Tensor]

class AbstractLayer(nn.Module, ABC):
    """
    BPフリー学習およびSNNのための抽象ネットワークレイヤー (PyTorch準拠)。
    """

    def __init__(
        self, 
        input_shape: Any, 
        output_shape: Any,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "AbstractLayer"
    ) -> None:
        """
        レイヤーを初期化します。
        """
        super().__init__()
        
        self.name: str = name
        self.input_shape: Any = input_shape
        self.output_shape: Any = output_shape
        self.built: bool = False
        
        # レイヤーの学習可能なパラメータ (nn.Parameter)
        # 具象クラスの build() メソッド等で設定されることを期待
        self.params: List[nn.Parameter] = [] 

        # 学習規則の設定とインスタンス
        self.learning_config: Optional[BaseLearningConfig] = learning_config
        self.learning_rule: Optional[AbstractLearningRule] = None
        
    @abstractmethod
    def build(self) -> None:
        """
        レイヤーのパラメータ（重みなど）を初期化し、
        self.params に登録し、self.learning_rule をセットアップします。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        """
        (nn.Module.forward のオーバーライド)
        単一の時間ステップにおける順伝播を実行します。
        
        Args:
            inputs (Tensor): 入力テンソル。
            model_state (Dict[str, Tensor]): ネットワーク全体の現在の状態。
        
        Returns:
            LayerOutput: 次の層への入力となる活動や、記録すべき内部状態を含む辞書。
        """
        raise NotImplementedError

    def update_local(
        self, 
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> UpdateMetrics:
        """
        局所学習則に基づき、このレイヤーの重みを更新します。
        """
        if not self.built or self.learning_rule is None:
            return {}

        # 学習則に委譲
        metrics: UpdateMetrics = self.learning_rule.step(
            inputs, 
            targets, 
            model_state
        )
        
        return metrics