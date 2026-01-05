# ファイルパス: snn_research/core/layers/abstract_snn_layer.py
# 日本語タイトル: 抽象SNNレイヤー (修正版)
# 機能説明: 
#   SNN特有の機能（状態リセットなど）を追加した抽象レイヤー。
#   snn_research.layers.abstract_layer.AbstractLayer を継承する。
#   
#   修正点:
#   - 相対インポート (...) を廃止し、絶対インポートを使用。
#   - ダミー定義ブロックを削除し、依存関係を明確化。

from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Any, Optional
from torch import Tensor

# 絶対インポートを使用
from snn_research.layers.abstract_layer import AbstractLayer, LayerOutput
from snn_research.config.learning_config import BaseLearningConfig

class AbstractSNNLayer(AbstractLayer):
    """
    SNNレイヤーのための抽象基底クラス。
    内部状態（膜電位など）のリセット機能を追加。
    """

    def __init__(
        self, 
        input_shape: Any,
        output_shape: Any,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "AbstractSNNLayer"
    ) -> None:
        """
        抽象SNNレイヤーを初期化します。
        """
        super().__init__(input_shape, output_shape, learning_config, name)
        
        # SNNの内部状態 (例: 膜電位)
        self.membrane_potential: Optional[Tensor] = None

    @abstractmethod
    def build(self) -> None:
        """
        (AbstractLayerから継承)
        レイヤーのパラメータ（重み）と内部状態（膜電位）を初期化します。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        """
        (AbstractLayerから継承)
        単一の時間ステップ (t) における順伝播を実行します。
        """
        raise NotImplementedError

    @abstractmethod
    def reset_state(self) -> None:
        """
        レイヤーの内部状態 (膜電位など) をリセットします。
        時系列データのバッチ切り替え時などに呼び出されます。
        """
        raise NotImplementedError