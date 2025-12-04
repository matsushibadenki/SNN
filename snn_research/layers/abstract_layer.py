# ファイルパス: snn_research/layers/abstract_layer.py
# タイトル: 抽象ネットワークレイヤーインターフェース (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 2, P2-1相当) に基づく、
#   全てのネットワークレイヤーが準拠すべき共通インターフェース。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - PyTorch の 'torch.Tensor' および 'torch.nn.Module' を直接使用します。
#   - 'AbstractLayer' が 'nn.Module' を継承するように変更します。
#   - 'forward' は 'nn.Module' の 'forward' をオーバーライドします。
#
# 修正: 循環インポートを解消するため、AbstractLearningRule のインポートを
#       TYPE_CHECKING ブロック内に移動しました。

from abc import ABC, abstractmethod
# --- ▼ 修正 ▼ ---
from typing import Dict, Any, Optional, Iterable, TYPE_CHECKING
# --- ▲ 修正 ▲ ---

# --- ダミー実装の解消 (PyTorch のインポート) ---
import torch
import torch.nn as nn
from torch import Tensor # TypeVar の代わりに torch.Tensor を使用

# P1-4 (抽象化) と P1-3 (Config) で定義したモジュールをインポート
try:
    # --- ▼ 修正: 循環インポート対策 ▼ ---
    if TYPE_CHECKING:
        from ..core.learning_rule import AbstractLearningRule
    # --- ▲ 修正 ▲ ---
    from ..core.learning_rule import Parameters
    from ..config.learning_config import BaseLearningConfig
except ImportError:
    # (mypy フォールバック)
    Parameters = Iterable[nn.Parameter] # type: ignore[misc]
    BaseLearningConfig = Any # type: ignore[misc, assignment]
    class AbstractLearningRule: # type: ignore[no-redef]
        def __init__(self, params: Parameters, **kwargs: Any) -> None: pass
        def step(
            self, i: Tensor, t: Optional[Tensor], s: Dict[str, Tensor]
        ) -> Dict[str, Tensor]:
            return {}

# 型エイリアス (PyTorch準拠)
LayerOutput = Dict[str, Tensor]
UpdateMetrics = Dict[str, Tensor]


# 修正: Generic[Tensor] を削除し、nn.Module と ABC を継承
class AbstractLayer(nn.Module, ABC):
    """
    BPフリー学習のための抽象ネットワークレイヤー (PyTorch準拠)。
    """

    def __init__(
        self, 
        input_shape: Any, # (具象クラスでの形状推論のため保持)
        output_shape: Any,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "AbstractLayer"
    ) -> None:
        """
        レイヤーを初期化します。
        """
        # 修正: nn.Module の __init__ を呼び出し
        super().__init__()
        
        self.name: str = name
        self.input_shape: Any = input_shape
        self.output_shape: Any = output_shape
        self.built: bool = False
        
        # レイヤーの学習可能なパラメータ (nn.Parameter)
        # (具象クラスが build() または __init__ で設定することを期待)
        # 修正: 型を Parameters (Iterable[nn.Parameter]) に
        self.params: Parameters = [] 

        # P1-4 の学習規則
        self.learning_config: Optional[BaseLearningConfig] = learning_config
        # 修正: Tensor 型引数を削除
        # --- ▼ 修正: 循環インポート対策 ▼ ---
        self.learning_rule: Optional["AbstractLearningRule"] = None
        # --- ▲ 修正 ▲ ---
        
    # (P2-1) build() は具象クラス (LIFLayerなど) が
    #        パラメータ初期化のために実装する
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
        順伝播計算を実行します。
        """
        raise NotImplementedError

    def update_local(
        self, 
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> UpdateMetrics:
        """
        P1-2 (ローカル学習) に基づき、このレイヤーの重みを更新します。
        """
        if not self.built or self.learning_rule is None:
            return {}

        metrics: UpdateMetrics = self.learning_rule.step(
            inputs, 
            targets, 
            model_state
        )
        
        return metrics

    # (nn.Module が __call__ を提供するため、
    #  AbstractLayer での __call__ のオーバーライドは不要)
