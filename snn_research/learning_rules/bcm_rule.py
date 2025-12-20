# ファイルパス: snn_research/learning_rules/bcm_rule.py
# Title: BCM (Bienenstock-Cooper-Munro) 学習規則
# Description:
#   ニューロンの平均活動に基づいて可塑性閾値を動的に調整し、
#   ネットワークの恒常性を維持する教師なし学習則。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class BCMLearningRule(BioLearningRule):
    """
    BCM (Bienenstock-Cooper-Munro) 学習規則。
    """
    avg_post_activity: Optional[torch.Tensor]

    def __init__(
        self, 
        learning_rate: float, 
        tau_avg: float, # 平均活動を計算するための時定数
        target_rate: float, # 目標とする平均発火率
        dt: float = 1.0
    ):
        self.learning_rate = learning_rate
        if tau_avg <= 0:
            raise ValueError("tau_avg must be positive")
        self.tau_avg = tau_avg
        if not (0 < target_rate <= 1.0):
             raise ValueError("target_rate must be between 0 and 1.0")
        self.target_rate = target_rate
        self.dt = dt
        
        self.avg_post_activity = None
        self.avg_decay_factor = dt / self.tau_avg

        print(f"🧠 BCM Learning Rule initialized (Target Rate: {target_rate}, Tau Avg: {tau_avg})")

    def _initialize_traces(self, post_shape: int, device: torch.device):
        self.avg_post_activity = torch.full((post_shape,), self.target_rate, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if pre_spikes.dim() > 1:
            pre_spikes_avg = pre_spikes.mean(dim=0)
        else:
            pre_spikes_avg = pre_spikes
            
        if post_spikes.dim() > 1:
            post_spikes_avg = post_spikes.mean(dim=0)
        else:
            post_spikes_avg = post_spikes

        if self.avg_post_activity is None or self.avg_post_activity.shape[0] != post_spikes_avg.shape[0]:
            self._initialize_traces(post_spikes_avg.shape[0], pre_spikes.device)
        
        avg_post_activity = cast(torch.Tensor, self.avg_post_activity)

        # 平均活動の更新
        with torch.no_grad():
            self.avg_post_activity = (
                (1.0 - self.avg_decay_factor) * avg_post_activity + 
                self.avg_decay_factor * post_spikes_avg
            ).detach()

        # BCM閾値 (theta)
        theta = avg_post_activity.clone()
        
        # phi関数: post * (post - theta)
        phi = post_spikes_avg * (post_spikes_avg - theta)
        
        # 重み更新: dw = lr * phi * pre
        dw = self.learning_rate * torch.outer(phi, pre_spikes_avg)
        
        return dw, None