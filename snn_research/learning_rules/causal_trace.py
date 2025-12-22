# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: Meta-Cognitive Causal Trace (Mypy Fixed)

import torch
from typing import Dict, Any, Optional, Tuple, List, cast
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    メタ認知機能を統合した因果学習則。
    Mypyエラー対応および不確実性駆動型学習率の実装。
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
        # 型アノテーションを追加してMypyエラーを解消
        self.uncertainty_buffer: List[float] = []
        self.uncertainty_threshold = 0.7
        self.deep_thinking_multiplier = 2.5

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        params = optional_params or {}
        uncertainty = float(params.get("uncertainty", 0.5))
        
        meta_lr_factor = self.deep_thinking_multiplier if uncertainty > self.uncertainty_threshold else 1.0
        
        # 報酬変調。親クラスの戻り値型(dw, backward_credit)に注意
        reward = float(params.get("reward", 0.0)) * meta_lr_factor
        
        # RewardModulatedSTDP.update を呼び出し
        dw, backward_credit = super().update(pre_spikes, post_spikes, weights, {**params, "reward": reward})
        
        # backward_credit が None の場合のフォールバック処理
        if backward_credit is None:
            backward_credit = torch.zeros_like(pre_spikes)
            
        # 制限をかけて数値的安定性を向上 (Objective.md ③)
        dw = torch.clamp(dw, -0.1, 0.1)
        
        # 戻り値を Tuple[Tensor, Tensor] に固定
        return dw, backward_credit
