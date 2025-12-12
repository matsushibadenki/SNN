# ファイルパス: snn_research/core/learning_rules/predictive_coding_rule.py
# 日本語タイトル: 予測符号化学習規則 (Predictive Coding Learning Rule)
# 機能説明:
#   Hebbian学習の一種として予測符号化の重み更新を実装。
#   delta_W ∝ Error * State.T - Weight_Decay
#   局所的な情報のみを使用するため、生物学的妥当性が高い。

from typing import Dict, Any, Iterable, Optional, List
import logging
import torch
import torch.nn as nn
from torch import Tensor

# 親クラスのインポート (プロジェクト構成に依存)
try:
    from ..learning_rule import AbstractLearningRule, Parameters
except ImportError:
    # フォールバック定義
    Parameters = Iterable[nn.Parameter] # type: ignore
    from abc import ABC, abstractmethod
    class AbstractLearningRule(ABC): # type: ignore
        def __init__(self, params: Parameters, **kwargs: Any) -> None: 
            self.params = params
            self.layer_name = kwargs.get('layer_name')
            self.hparams = kwargs
        @abstractmethod
        def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]: pass
        @abstractmethod
        def zero_grad(self) -> None: pass

logger = logging.getLogger(__name__)

class PredictiveCodingRule(AbstractLearningRule):
    """
    Predictive Codingのための局所学習則。
    Generative Weight (W) を更新し、予測誤差を最小化する。
    """
    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.error_weight: float = float(kwargs.get('error_weight', 1.0))
        self.weight_decay: float = float(kwargs.get('weight_decay', 1e-4))
        self.hparams['error_weight'] = self.error_weight
        
        # パラメータリストの分解 (通常は [Weight, Bias])
        param_list: List[nn.Parameter] = list(self.params)
        self.W: nn.Parameter = param_list[0]
        self.b: nn.Parameter = param_list[1] if len(param_list) > 1 else None # type: ignore

    def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        1ステップの重み更新を実行。
        model_state から 'pre_activity' (状態) と 'prediction_error' (誤差) を取得して計算する。
        """
        lr: float = float(self.hparams.get('learning_rate', 0.01))
        if self.layer_name is None: return {'update_magnitude': torch.tensor(0.0)}

        # レイヤー名をキーにして状態を取得
        # pre_activity: この層の入力（＝上位層の状態 State）
        # post_error: この層の出力誤差（＝予測誤差 Error）
        pre_activity: Optional[Tensor] = model_state.get(f'pre_activity_{self.layer_name}')
        post_error: Optional[Tensor] = model_state.get(f'prediction_error_{self.layer_name}')
        
        if pre_activity is None or post_error is None:
            return {'update_magnitude': torch.tensor(0.0)}
        
        total_update_magnitude: Tensor = torch.tensor(0.0, device=inputs.device)

        with torch.no_grad():
            try:
                # バッチ次元がある場合は調整
                if pre_activity.dim() == 1: pre_activity = pre_activity.unsqueeze(0)
                if post_error.dim() == 1: post_error = post_error.unsqueeze(0)

                # Hebbian Term: Error * State^T
                # (Batch, Out, 1) @ (Batch, 1, In) -> (Batch, Out, In) -> Mean -> (Out, In)
                term_hebb = torch.bmm(post_error.unsqueeze(2), pre_activity.unsqueeze(1)).mean(dim=0)
                
                # Weight Decay Term
                term_decay = self.W * self.weight_decay
                
                # Delta W
                delta_W = (term_hebb - term_decay) * (lr * self.error_weight)
                
                # 勾配クリッピング（安定化のため）
                delta_W = torch.clamp(delta_W, -0.1, 0.1)
                
                # 重み更新 (In-place)
                if self.W.shape == delta_W.shape:
                    self.W.add_(delta_W)
                    total_update_magnitude += delta_W.abs().mean()
                
                # バイアス更新
                if self.b is not None:
                    delta_b = post_error.mean(dim=0) * (lr * self.error_weight)
                    if self.b.shape == delta_b.shape:
                        self.b.add_(delta_b)
                        total_update_magnitude += delta_b.abs().mean()
                
            except Exception as e:
                logger.error(f"Failed to update {self.layer_name}: {e}")

        return {'update_magnitude': total_update_magnitude}

    def zero_grad(self) -> None:
        pass