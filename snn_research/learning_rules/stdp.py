# ファイルパス: snn_research/learning_rules/stdp.py
# Title: 報酬変調型 Triplet STDP (v16.5)
# Description:
#   ペアベースの限界を超え、トリプレット(3スパイク)による時間相関の精密化。
#   Dopamine変調による「意味のある学習」への昇華。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .base_rule import BioLearningRule

class TripletSTDP(BioLearningRule):
    """
    Triplet STDP (トリプレットSTDP) 学習則 with Reward Modulation.
    の精度96%超えを目指すための高精度可塑性モデル。
    """
    def __init__(
        self, 
        learning_rate: float = 0.01,
        target_rate: float = 0.05,
        homeostasis_strength: float = 0.2,
        dt: float = 1.0
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.dt = dt
        self.target_rate = target_rate
        self.homeostasis_strength = homeostasis_strength

        # パラメータ設定 (生物学的妥当性に基づく)
        self.tau_plus = 16.8
        self.tau_minus = 33.7
        self.tau_x = 101.0  # Triplet用
        self.tau_y = 125.0  # Triplet用
        
        self.a2_plus = 7.5e-3
        self.a2_minus = 7.0e-3
        self.a3_plus = 9.3e-3  # Triplet LTP強化
        self.a3_minus = 2.3e-4

        # トレース情報の保持
        self.pre_trace = None
        self.post_trace = None
        self.pre_trace_triplet = None
        self.post_trace_triplet = None
        self.avg_firing_rate = None

    def _initialize_traces(self, pre_shape: torch.Size, post_shape: torch.Size, device: torch.device):
        self.pre_trace = torch.zeros(pre_shape, device=device)
        self.post_trace = torch.zeros(post_shape, device=device)
        self.pre_trace_triplet = torch.zeros(pre_shape, device=device)
        self.post_trace_triplet = torch.zeros(post_shape, device=device)
        self.avg_firing_rate = torch.full((post_shape[1],), self.target_rate, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)
        
        if self.pre_trace is None or self.pre_trace.shape != pre_spikes.shape:
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)

        # 1. トレースの減衰と更新
        with torch.no_grad():
            self.pre_trace = self.pre_trace * (1.0 - self.dt / self.tau_plus) + pre_spikes
            self.post_trace = self.post_trace * (1.0 - self.dt / self.tau_minus) + post_spikes
            self.pre_trace_triplet = self.pre_trace_triplet * (1.0 - self.dt / self.tau_x) + pre_spikes
            self.post_trace_triplet = self.post_trace_triplet * (1.0 - self.dt / self.tau_y) + post_spikes
            
            # 恒常性維持(Homeostasis)のための発火率追跡
            self.avg_firing_rate = 0.999 * self.avg_firing_rate + 0.001 * post_spikes.mean(dim=0)

        # 2. 報酬信号の取得 (Dopamine変調)
        # の⑤(非勾配型)を実現するため、報酬が正の時のみ学習を促進
        reward = optional_params.get("reward", 1.0) if optional_params else 1.0
        
        # 3. Triplet更新量の計算 (行列演算による高速化)
        # Pairwise LTP/LTD
        dw_2_plus = torch.matmul(post_spikes.t(), self.pre_trace)
        dw_2_minus = torch.matmul(self.post_trace.t(), pre_spikes)
        
        # Triplet LTP (pre-post-pre)
        dw_3_plus = torch.matmul(post_spikes.t(), self.pre_trace_triplet)
        
        # 重み増分 dw
        dw = self.a2_plus * dw_2_plus + self.a3_plus * dw_3_plus - self.a2_minus * dw_2_minus

        # 4. 恒常性(Homeostasis)による変調
        # 目標レートを超えると、重みの増加を抑制
        rate_error = (self.avg_firing_rate - self.target_rate) * self.homeostasis_strength
        homeostasis_mod = torch.clamp(1.0 - rate_error.unsqueeze(1), min=0.0, max=2.0)
        
        # 全体を学習率と報酬で統合
        final_dw = self.learning_rate * reward * dw * homeostasis_mod
        
        return final_dw, None
