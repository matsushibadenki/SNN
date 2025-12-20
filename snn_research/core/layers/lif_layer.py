# ファイルパス: snn_research/core/layers/lif_layer.py
# タイトル: Leaky Integrate-and-Fire (LIF) SNNレイヤー
# 機能説明: 
#   Project SNN4のロードマップ (Phase 4, P4-4) に基づく、
#   AbstractSNNLayer (P4-1) を継承する具象LIFレイヤー。
#
#   修正点:
#   - mypyエラー (Cannot assign to a type) を修正。
#   - PredictiveCodingRule のインポートを遅延実行に変更し、循環参照と型エラーを回避。

import logging
from typing import Dict, Any, Optional, Tuple, cast, List, TYPE_CHECKING
import torch
import torch.nn as nn
from torch import Tensor 

# プロジェクト内モジュールの絶対インポート
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer
from snn_research.layers.abstract_layer import LayerOutput
from snn_research.config.learning_config import BaseLearningConfig
from snn_research.core.learning_rule import Parameters
from snn_research.core.synapse_dynamics import apply_probabilistic_transmission

# 型チェック時のみインポート（mypy用）
if TYPE_CHECKING:
    from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule

# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

@torch.jit.script
def lif_update(
    inputs: Tensor, 
    V: Tensor, 
    W: Tensor, 
    b: Tensor, 
    decay: float, 
    threshold: float
) -> Tuple[Tensor, Tensor]:
    """ P4-4: 単一ステップのLIFダイナミクス (PyTorch 実装) """
    # 入力電流の計算
    I_t: Tensor = nn.functional.linear(inputs, W, b)
    
    # 電位の減衰と統合
    V_leaked: Tensor = V * decay
    V_new: Tensor = V_leaked + I_t
    
    # 発火判定
    spikes: Tensor = (V_new > threshold).float()
    
    # リセット (Soft Reset: 閾値を減算)
    V_reset: Tensor = V_new - (spikes * threshold)
    
    return V_reset, spikes


class LIFLayer(AbstractSNNLayer):
    """
    P4-4: Leaky Integrate-and-Fire (LIF) レイヤー (PyTorch実装)。
    """

    def __init__(
        self, 
        input_features: int, 
        neurons: int,
        learning_config: Optional[BaseLearningConfig] = None,
        name: str = "LIFLayer",
        decay: float = 0.95, 
        threshold: float = 1.0,
        synaptic_reliability: float = 1.0, # デフォルトは1.0 (信頼性100%)
    ) -> None:
        
        dummy_shape: Tuple[int, ...] = (0,)
        # (name は AbstractLayer の __init__ で設定される)
        super().__init__(dummy_shape, dummy_shape, learning_config, name)
        
        self.decay: float = decay
        self.threshold: float = threshold
        
        self.synaptic_reliability = synaptic_reliability
        
        self._input_features: int = input_features
        self._neurons: int = neurons
        
        # 重みとバイアスの定義 (requires_grad=False は手動更新またはカスタム学習則のため)
        self.W: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons, self._input_features), 
            requires_grad=False
        )
        self.b: nn.Parameter = nn.Parameter(
            torch.empty(self._neurons), 
            requires_grad=False
        )
        
        self.membrane_potential: Optional[Tensor] = None


    def build(self) -> None:
        """
        (P2-1) パラメータを初期化し、(P1-4) 学習規則をセットアップします。
        """
        if logger:
            logger.debug(f"Building layer: {self.name}")
            
        # パラメータの初期化 (Kaiming Uniform)
        nn.init.kaiming_uniform_(self.W, a=0.01)
        nn.init.zeros_(self.b)
        
        # P1-4: 学習可能なパラメータとして登録
        self.params = [self.W, self.b]
        
        # P1-4: 学習規則のインスタンス化
        if self.learning_config:
            # 遅延インポート: 循環参照回避とmypyエラー解消のためここでインポートする
            try:
                from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule
                rule_cls = PredictiveCodingRule
            except ImportError:
                logger.error("Could not import PredictiveCodingRule. Learning rule setup skipped.")
                # インポートできない場合は学習ルールなしで続行（または例外送出）
                rule_cls = None

            if rule_cls is not None:
                # P1-3 の設定を取得
                rule_kwargs: Dict[str, Any] = self.learning_config.to_dict()
                # P1-4 (AbstractLearningRule) のため、レイヤー名を渡す
                rule_kwargs['layer_name'] = self.name
                
                self.learning_rule = rule_cls(
                    self.params, 
                    **rule_kwargs
                )
            
        self.built = True

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        """
        内部状態（膜電位）の初期化
        """
        if logger:
            logger.debug(f"Initializing state for {self.name} with batch size {batch_size}")
        
        self.membrane_potential = torch.zeros(
            (batch_size, self._neurons), device=device
        )

    def forward(
        self, 
        inputs: Tensor, 
        model_state: Dict[str, Tensor]
    ) -> LayerOutput:
        """
        順伝播処理
        """
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")
        
        batch_size: int = inputs.shape[0]

        if self.membrane_potential is None:
            self._init_state(batch_size, inputs.device)
        
        V_t_minus_1: Tensor = cast(Tensor, self.membrane_potential)

        # --- シナプスゆらぎの適用 ---
        # 学習モード(self.training)に応じて確率的伝達を適用
        effective_W = apply_probabilistic_transmission(
            self.W, 
            reliability=self.synaptic_reliability,
            training=self.training 
        )

        # --- LIFダイナミクスの計算 ---
        V_t: Tensor
        spikes_t: Tensor
        V_t, spikes_t = lif_update(
            inputs, V_t_minus_1, effective_W, self.b, self.decay, self.threshold
        )
        
        # 状態の更新
        self.membrane_potential = V_t
        
        return {
            'activity': spikes_t, 
            'membrane_potential': V_t
        }

    def reset_state(self) -> None:
        if logger:
            logger.debug(f"Resetting state for {self.name}")
        self.membrane_potential = None