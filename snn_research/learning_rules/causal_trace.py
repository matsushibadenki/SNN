# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: メタ認知統合型 Causal Trace Credit Assignment (v16.5)
# Description:
#   「思考の質」を担保するため、不確実性(Uncertainty)に基づく動的リソース配分を実装。
#   因果連鎖のクレジット割当精度を極大化。

import torch
from typing import Dict, Any, Optional, Tuple, cast
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    メタ認知機能を備えた因果学習エンジン。
    の目標 ⑮ を達成するため、不確実性に応じた学習率の動的制御を行う。
    """
    def __init__(self, **kwargs: Any):
        super().__init__(
            learning_rate=kwargs.get('learning_rate', 0.01),
            a_plus=kwargs.get('a_plus', 0.01),
            a_minus=kwargs.get('a_minus', 0.008),
            tau_trace=kwargs.get('tau_trace', 20.0),
            tau_eligibility=kwargs.get('tau_eligibility', 100.0)
        )
        self.causal_contribution = None
        self.uncertainty_buffer = []
        
        # メタ認知パラメータ
        self.uncertainty_threshold = 0.7  # これを超えると「深く考える」
        self.deep_thinking_multiplier = 2.5

        print("🧠 Causal Trace V16.5 (Meta-Cognitive Enabled) initialized.")

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if optional_params is None: optional_params = {}

        # 1. 不確実性 (Uncertainty) の評価
        # エントロピーや予測誤差から「自信のなさ」を算出
        uncertainty = optional_params.get("uncertainty", 0.5)
        
        # 2. 「深く考える」動的リソース配分 ⑮
        # 自信がない（不確実性が高い）時、学習率を高めて因果関係を強く刻み込む
        meta_lr_factor = 1.0
        if uncertainty > self.uncertainty_threshold:
            meta_lr_factor = self.deep_thinking_multiplier
            # ロギングやリソース要求(Astrocyteへの通知)をここに挟むことが可能

        # 3. 適格度トレースの更新 (親クラスのロジックを流用)
        # 報酬変調をメタ学習率で強化
        reward = optional_params.get("reward", 0.0) * meta_lr_factor
        
        # 4. 因果クレジットの計算
        # 既存のupdateメソッドを呼び出し、メタ変調を適用
        dw, backward_credit = super().update(pre_spikes, post_spikes, weights, {**optional_params, "reward": reward})
        
        # 5. 数値的安定性の確保 (Objective.md ③ 再現性向上)
        dw = torch.clamp(dw, -0.1, 0.1)
        
        return dw, backward_credit
