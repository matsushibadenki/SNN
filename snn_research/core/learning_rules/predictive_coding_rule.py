# ファイルパス: snn_research/core/learning_rules/predictive_coding_rule.py
# 日本語タイトル: 予測符号化学習規則 v3.0 (Precision Weighted & Dampening)
# 機能説明:
#   1. Precision Weighting: 予測の「不確実性(Sigma)」を学習し、誤差をスケーリングする。
#   2. Oscillation Dampening: 符号反転時の更新抑制による安定化。
#   3. Gated Hebbian: 入力がない時の誤学習防止。

from typing import Dict, Any, Iterable, Optional
import logging
import torch
import torch.nn as nn
from torch import Tensor

try:
    from ..learning_rule import AbstractLearningRule, Parameters
except ImportError:
    # Fallback definitions
    Parameters = Iterable[nn.Parameter]  # type: ignore
    from abc import ABC, abstractmethod

    class AbstractLearningRule(ABC):  # type: ignore
        def __init__(self, params: Parameters, **kwargs: Any) -> None:
            self.params = params
            self.layer_name = kwargs.get('layer_name')
            self.hparams = kwargs

        @abstractmethod
        def step(self, inputs: Tensor, targets: Optional[Tensor],
                 model_state: Dict[str, Tensor]) -> Dict[str, Tensor]: pass

        @abstractmethod
        def zero_grad(self) -> None: pass

logger = logging.getLogger(__name__)


class PredictiveCodingRule(AbstractLearningRule):
    """
    Predictive Coding Rule v3.0 (Precision Weighted)

    Update Rule:
    Delta W = lr * ( Precision * (Error * State^T) - Weight_Decay * W )

    Precision (Sigma^-1) is learned alongside weights to handle uncertainty.
    """

    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.error_weight: float = float(kwargs.get('error_weight', 1.0))
        self.weight_decay: float = float(kwargs.get('weight_decay', 1e-4))
        self.use_oja: bool = bool(kwargs.get('use_oja', True))

        # パラメータリストの分解
        param_list = list(self.params)
        self.W: nn.Parameter = param_list[0]
        self.b: Optional[nn.Parameter] = param_list[1] if len(
            param_list) > 1 else None

        # Precision (不確実性の逆数) パラメータの追加
        # log_sigma として保持し、exp()で正の値(Precision)に変換して使う
        # 各出力ニューロンごとに1つのPrecisionを持つ
        out_features = self.W.shape[0]
        self.log_sigma = nn.Parameter(torch.zeros(
            out_features, 1, device=self.W.device))

        # 前回更新方向のキャッシュ (Dampening用)
        self.prev_delta: Optional[Tensor] = None

    def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        lr: float = float(self.hparams.get('learning_rate', 0.01))

        pre_activity: Optional[Tensor] = model_state.get(
            f'pre_activity_{self.layer_name}')
        post_error: Optional[Tensor] = model_state.get(
            f'prediction_error_{self.layer_name}')

        if pre_activity is None or post_error is None:
            return {'update_magnitude': torch.tensor(0.0)}

        total_update: Tensor = torch.tensor(0.0, device=inputs.device)

        with torch.no_grad():
            if pre_activity.dim() == 1:
                pre_activity = pre_activity.unsqueeze(0)
            if post_error.dim() == 1:
                post_error = post_error.unsqueeze(0)

            # Precision Weighting
            # log_sigma を学習させる場合、勾配計算が必要だが、ここでは簡易的に固定パラメータとして扱うか
            # あるいは誤差の大きさに応じて動的に調整する
            precision = torch.exp(-self.log_sigma)  # sigma^-1

            # Weighted Error
            # (Batch, Out) * (Out, 1).T -> Broadcast
            weighted_error = post_error * precision.T

            # 1. Hebbian Term with Gating
            # 入力(pre_activity)が弱い時は更新しない (Gated)
            gating_factor = torch.tanh(pre_activity.abs().mean(
                dim=0, keepdim=True).T * 10)  # (Out, In)

            # (Batch, Out, 1) @ (Batch, 1, In) -> Mean -> (Out, In)
            term_hebb = torch.bmm(weighted_error.unsqueeze(
                2), pre_activity.unsqueeze(1)).mean(dim=0)

            # Gating適用
            # 次元が合わない場合はブロードキャストを試みる
            try:
                term_hebb = term_hebb * gating_factor.T
            except Exception:
                pass  # シェイプ不一致ならスキップ（安全策）

            # 2. Oja's Rule (Normalization)
            if self.use_oja:
                post_sq = (weighted_error ** 2).mean(dim=0).unsqueeze(1)
                term_decay = self.weight_decay * post_sq * self.W
            else:
                term_decay = self.weight_decay * self.W

            # Delta W
            delta_W = (term_hebb - term_decay) * (lr * self.error_weight)

            # 3. Oscillation Dampening
            # 前回の更新と符号が逆なら、更新量を減らす（振動防止）
            if self.prev_delta is not None:
                sign_change = (delta_W * self.prev_delta) < 0
                delta_W[sign_change] *= 0.5  # 減衰係数

            self.prev_delta = delta_W.clone()

            # Update
            self.W.add_(delta_W)
            total_update += delta_W.abs().mean()

            # Bias Update
            if self.b is not None:
                delta_b = weighted_error.mean(dim=0) * (lr * self.error_weight)
                self.b.add_(delta_b)

            # Sigma Update (Variance Learning) - 簡易的
            # 誤差が大きい場合、Sigmaを大きくしてPrecisionを下げる（学習を緩める）
            error_sq = (post_error ** 2).mean(dim=0, keepdim=True).T
            delta_sigma = (error_sq - torch.exp(self.log_sigma)) * (lr * 0.1)
            self.log_sigma.add_(delta_sigma)

        return {'update_magnitude': total_update}

    def zero_grad(self) -> None:
        pass
