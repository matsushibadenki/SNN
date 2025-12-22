# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: Meta-Cognitive Causal Trace (Mypy & Feature Fixed)
# Description: 不確実性駆動型学習と因果貢献度追跡を統合。

import torch
from typing import Dict, Any, Optional, Tuple, List, cast
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    メタ認知機能を統合した因果学習則。
    不確実性が高い（自信がない）時に学習を深める動的リソース配分を実装。
    """
    def __init__(self, **kwargs: Any):
        super().__init__(
            learning_rate=kwargs.get('learning_rate', 0.01),
            a_plus=kwargs.get('a_plus', 0.01),
            a_minus=kwargs.get('a_minus', 0.008),
            tau_trace=kwargs.get('tau_trace', 20.0),
            tau_eligibility=kwargs.get('tau_eligibility', 100.0)
        )
        # 因果的貢献度を保持する属性。外部から直接アクセス可能にする。
        self.causal_contribution: Optional[torch.Tensor] = None
        self.uncertainty_buffer: List[float] = []
        self.uncertainty_threshold: float = 0.7
        self.deep_thinking_multiplier: float = 2.5

        print("🧠 Causal Trace V16.5 (Meta-Cognitive & Causal-Analysis Enabled) initialized.")

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        params = optional_params or {}
        uncertainty = float(params.get("uncertainty", 0.5))
        
        # ⑮ 自信がない時だけ深く考える動的リソース配分
        meta_lr_factor = self.deep_thinking_multiplier if uncertainty > self.uncertainty_threshold else 1.0
        reward = float(params.get("reward", 0.0)) * meta_lr_factor
        
        # 親クラス(RewardModulatedSTDP)の更新処理
        dw, backward_credit = super().update(pre_spikes, post_spikes, weights, {**params, "reward": reward})
        
        # 因果的貢献度の更新 (絶対値の指数移動平均)
        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self.causal_contribution = torch.zeros_like(weights)
        
        with torch.no_grad():
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

        if backward_credit is None:
            backward_credit = torch.zeros_like(pre_spikes)
            
        # ③ 学習再現性のためのクランプ処理
        dw = torch.clamp(dw, -0.1, 0.1)
        
        return dw, backward_credit

    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        """シンプルネットワークからの呼び出し互換性を維持するためのメソッド。"""
        return self.causal_contribution
