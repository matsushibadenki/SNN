# ファイルパス: snn_research/learning_rules/causal_trace.py
# 日本語タイトル: 因果クレジット割当規則 (整合性強化版)
# 目的: Bio-RL での重み更新時に、形状不一致による RuntimeError を防止。

import torch
from typing import Dict, Any, Optional, Tuple, List
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    メタ認知機能を統合した因果学習則。
    [修正] backward_credit が Pre-synaptic ニューロン数に一致するように厳密に定義。
    """
    def __init__(self, **kwargs: Any):
        super().__init__(
            learning_rate=kwargs.get('learning_rate', 0.01),
            a_plus=kwargs.get('a_plus', 0.01),
            a_minus=kwargs.get('a_minus', 0.008),
            tau_trace=kwargs.get('tau_trace', 20.0),
            tau_eligibility=kwargs.get('tau_eligibility', 100.0)
        )
        self.causal_contribution: Optional[torch.Tensor] = None
        self.uncertainty_buffer: List[float] = []
        self.uncertainty_threshold: float = 0.7
        self.deep_thinking_multiplier: float = 2.5

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        params = optional_params or {}
        uncertainty = float(params.get("uncertainty", 0.5))
        
        # 学習率の変調
        meta_lr_factor = self.deep_thinking_multiplier if uncertainty > self.uncertainty_threshold else 1.0
        reward = float(params.get("reward", 0.0)) * meta_lr_factor
        
        # 親クラスの更新（形状調整済み STDP を呼び出し）
        dw, backward_credit = super().update(pre_spikes, post_spikes, weights, {**params, "reward": reward})
        
        # [修正] backward_credit が常に (Batch, Pre_Neurons) となるように保証
        if backward_credit is None or backward_credit.shape[-1] != pre_spikes.shape[-1]:
            backward_credit = torch.zeros_like(pre_spikes)
            
        # 因果貢献度の記録
        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self.causal_contribution = torch.zeros_like(weights)
        with torch.no_grad():
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01
            
        return dw, backward_credit

    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        return self.causal_contribution