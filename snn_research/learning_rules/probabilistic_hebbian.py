# snn_research/learning_rules/probabilistic_hebbian.py
# 修正: optional_params を明示的に受け取るように再確認

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_rule import BioLearningRule

class ProbabilisticHebbian(nn.Module, BioLearningRule):
    # ... __init__ はそのまま ...
    def __init__(self, learning_rate: float = 0.005, weight_decay: float = 0.0001):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # ... 中身は前回の回答と同じ ...
        # バッチ次元の処理
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)

        # ヘブ則
        hebbian_term = torch.bmm(post_spikes.unsqueeze(2), pre_spikes.unsqueeze(1))
        mean_hebbian = hebbian_term.mean(dim=0)
        decay_term = self.weight_decay * weights

        dw = self.learning_rate * (mean_hebbian - decay_term)
        return dw, None