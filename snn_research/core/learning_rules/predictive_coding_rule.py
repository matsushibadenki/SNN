# ファイルパス: snn_research/core/learning_rules/predictive_coding_rule.py
# Title: 予測符号化学習規則 (安定化版)
# Description: 符号修正(add_)と重み減衰(weight_decay)を適用。

from typing import Dict, Any, Iterable, Optional, cast, List
import logging
import torch
import torch.nn as nn
from torch import Tensor

# 親ディレクトリのABCをインポート
try:
    from ..learning_rule import AbstractLearningRule, Parameters
except ImportError:
    Parameters = Iterable[nn.Parameter] # type: ignore[misc]
    from abc import ABC, abstractmethod
    class AbstractLearningRule(ABC): # type: ignore[no-redef]
        def __init__(self, params: Parameters, **kwargs: Any) -> None: pass
        @abstractmethod
        def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]: pass
        @abstractmethod
        def zero_grad(self) -> None: pass

logger = logging.getLogger(__name__)

class PredictiveCodingRule(AbstractLearningRule):
    def __init__(self, params: Parameters, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.error_weight: float = float(kwargs.get('error_weight', 1.0))
        self.weight_decay: float = float(kwargs.get('weight_decay', 1e-4))
        self.hparams['error_weight'] = self.error_weight
        
        param_list: List[nn.Parameter] = list(self.params)
        self.W: nn.Parameter = param_list[0]
        self.b: nn.Parameter = param_list[1] if len(param_list) > 1 else None # type: ignore[assignment]

    def step(self, inputs: Tensor, targets: Optional[Tensor], model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        lr: float = float(self.hparams.get('learning_rate', 0.01))
        if self.layer_name is None: return {'update_magnitude': torch.tensor(0.0)}

        pre_activity: Optional[Tensor] = model_state.get(f'pre_activity_{self.layer_name}')
        post_error: Optional[Tensor] = model_state.get(f'prediction_error_{self.layer_name}')
        
        if pre_activity is None or post_error is None:
            return {'update_magnitude': torch.tensor(0.0)}
        
        total_update_magnitude: Tensor = torch.tensor(0.0, device=inputs.device)

        with torch.no_grad():
            try:
                if pre_activity.dim() == 1: pre_activity = pre_activity.unsqueeze(0)
                if post_error.dim() == 1: post_error = post_error.unsqueeze(0)

                # Hebbian term: Error * State.T
                term_hebb = torch.bmm(post_error.unsqueeze(2), pre_activity.unsqueeze(1)).mean(dim=0)
                
                # Weight Decay
                term_decay = self.W * self.weight_decay
                
                delta_W = (term_hebb - term_decay) * (lr * self.error_weight)
                delta_W = torch.clamp(delta_W, -0.1, 0.1)
                
                if self.W.shape == delta_W.shape:
                    self.W.add_(delta_W) # 修正: 加算
                    total_update_magnitude += delta_W.abs().mean()
                
                if self.b is not None:
                    delta_b = post_error.mean(dim=0) * (lr * self.error_weight)
                    if self.b.shape == delta_b.shape:
                        self.b.add_(delta_b)
                        total_update_magnitude += delta_b.abs().mean()
                
            except Exception as e:
                logger.error(f"Failed to update {self.layer_name}: {e}")

        return {'update_magnitude': total_update_magnitude}

    def zero_grad(self) -> None: pass