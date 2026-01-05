# ファイルパス: snn_research/core/learning_rules/predictive_coding_rule.py
# 日本語タイトル: 予測符号化学習規則 v3.1 (Free Energy Minimization)
# 機能説明:
#   自由エネルギー最小化に基づく重み更新則。
#   不確実性(Precision)による重み付けと、局所的なヘブ則の組み合わせ。

from typing import Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from torch import Tensor

from snn_research.core.learning_rule import AbstractLearningRule, Parameters

logger = logging.getLogger(__name__)


class PredictiveCodingRule(AbstractLearningRule):
    """
    Predictive Coding Rule v3.1

    Based on Free Energy Principle:
    F = (Error^2 / Sigma) + KL(Posterior || Prior)

    Delta W ~ - dF/dW = Precision * Error * Pre_Activity
    """

    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.error_weight: float = float(kwargs.get('error_weight', 1.0))
        self.weight_decay: float = float(kwargs.get('weight_decay', 1e-4))
        self.use_oja: bool = bool(kwargs.get('use_oja', True))

        param_list = list(self.params)
        if len(param_list) == 0:
            raise ValueError(
                "PredictiveCodingRule requires at least one parameter (Weights).")

        self.W: nn.Parameter = param_list[0]
        self.b: Optional[nn.Parameter] = param_list[1] if len(
            param_list) > 1 else None

        out_features = self.W.shape[0]
        self.log_sigma = nn.Parameter(torch.zeros(
            out_features, 1, device=self.W.device))
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

            # Precision (Inverse Variance)
            precision = torch.exp(-self.log_sigma)

            # Precision-Weighted Error
            weighted_error = post_error * precision.T

            # Gating: Activityが低い入力からの学習を抑制 (Noise robustness)
            # 生体では発火していないシナプスのLTPは起きにくい
            gating_factor = torch.tanh(
                pre_activity.abs().mean(dim=0, keepdim=True).T * 10)

            # Calculate Gradient: Error * Input (Hebbian-like)
            term_gradient = torch.bmm(weighted_error.unsqueeze(
                2), pre_activity.unsqueeze(1)).mean(dim=0)
            term_gradient = term_gradient * gating_factor.T

            # Weight Decay / Oja's Rule
            if self.use_oja:
                post_sq = (weighted_error ** 2).mean(dim=0).unsqueeze(1)
                term_decay = self.weight_decay * post_sq * self.W
            else:
                term_decay = self.weight_decay * self.W

            # Total Delta
            delta_W = (term_gradient - term_decay) * (lr * self.error_weight)

            # Momentum / Dampening
            if self.prev_delta is not None:
                # 振動抑制: 符号が反転したら減衰
                sign_change = (delta_W * self.prev_delta) < 0
                delta_W[sign_change] *= 0.5
                # モメンタム
                delta_W = delta_W * 0.1 + self.prev_delta * 0.9

            self.prev_delta = delta_W.clone()

            self.W.add_(delta_W)
            total_update += delta_W.abs().mean()

            if self.b is not None:
                delta_b = weighted_error.mean(dim=0) * (lr * self.error_weight)
                self.b.add_(delta_b)

            # Sigma Learning (Variance estimation)
            error_sq = (post_error ** 2).mean(dim=0, keepdim=True).T
            # Sigmaは誤差の分散に近づくように学習
            delta_sigma = (error_sq - torch.exp(self.log_sigma)) * (lr * 0.01)
            self.log_sigma.add_(delta_sigma)

        return {'update_magnitude': total_update}

    def zero_grad(self) -> None:
        pass
