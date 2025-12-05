# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: 進化版 因果追跡クレジット割り当て学習則 (V2)
# Description:
# - CausalTraceCreditAssignmentEnhanced を基盤とし、さらなる機能向上を目指した実装
# - 修正: updateメソッド内の backward_credit 計算における torch.einsum の添字を修正。
#   'ij,ij->i' (post側で和を取る) ではなく 'ij,ij->j' (pre側、つまり入力側へ逆伝播) に変更し、
#   次元不整合を解消。

import torch
from typing import Dict, Any, Optional, Tuple
import math

from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    文脈変調、競合、高レベル因果連携を導入した、さらに進化した因果学習則。
    """
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float,
                 tau_trace: float, tau_eligibility: float, dt: float = 1.0,
                 credit_time_decay: float = 0.95,
                 dynamic_lr_factor: float = 2.0,
                 modulate_eligibility_tau: bool = True,
                 min_eligibility_tau: float = 10.0,
                 max_eligibility_tau: float = 200.0,
                 context_modulation_strength: float = 0.5,
                 competition_k_ratio: float = 0.1,
                 rule_based_lr_factor: float = 3.0
                 ):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        self.causal_contribution: Optional[torch.Tensor] = None
        self.base_learning_rate = learning_rate
        self.credit_time_decay = credit_time_decay
        self.dynamic_lr_factor = dynamic_lr_factor
        self.modulate_eligibility_tau = modulate_eligibility_tau
        self.min_eligibility_tau = min_eligibility_tau
        self.max_eligibility_tau = max_eligibility_tau
        self.base_tau_eligibility = tau_eligibility
        self.context_modulation_strength = context_modulation_strength
        self.competition_k_ratio = competition_k_ratio
        self.rule_based_lr_factor = rule_based_lr_factor
        
        print("🧠 V2 Enhanced Causal Trace Credit Assignment rule initialized.")

    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        self.causal_contribution = torch.zeros(weight_shape, device=device)

    def _apply_context_modulation(self, backward_credit: torch.Tensor, optional_params: Dict[str, Any]) -> torch.Tensor:
        """Global Workspace や Memory からの文脈情報でクレジット信号を変調する。"""
        modulated_credit = backward_credit.clone()

        workspace_context = optional_params.get("global_workspace_context")
        memory_context = optional_params.get("memory_context")

        modulation_factor = 1.0

        if workspace_context and isinstance(workspace_context, dict) and workspace_context.get("type") == "emotion":
            valence = workspace_context.get("valence", 0.0)
            if valence < -0.5:
                modulation_factor += self.context_modulation_strength * abs(valence)

        if memory_context and isinstance(memory_context, list) and len(memory_context) > 0:
            if any("FAILURE" in str(mem.get("result")) for mem in memory_context):
                 modulation_factor += self.context_modulation_strength * 0.5

        return modulated_credit * modulation_factor

    def _apply_competition(self, dw: torch.Tensor, eligibility_trace: torch.Tensor) -> torch.Tensor:
        """競合メカニズムを適用し、更新対象のシナプスを選択する。"""
        if self.competition_k_ratio >= 1.0:
            return dw

        num_synapses = dw.numel()
        k = max(1, int(num_synapses * self.competition_k_ratio))

        abs_eligibility = torch.abs(eligibility_trace)
        top_k_values, _ = torch.topk(abs_eligibility.view(-1), k)
        threshold = top_k_values[-1]

        mask = abs_eligibility >= threshold

        return dw * mask.float()

    def _apply_high_level_rules(self, dynamic_lr: torch.Tensor, optional_params: Dict[str, Any], weights: torch.Tensor) -> torch.Tensor:
        """CausalInferenceEngineからの抽象ルールに基づき学習率を調整する。"""
        rule = optional_params.get("abstract_causal_rule")

        if rule and isinstance(rule, dict):
            if rule.get("increase_lr"):
                # print(f"   - Applying high-level rule: Increasing LR for relevant synapses.")
                return dynamic_lr * self.rule_based_lr_factor

        return dynamic_lr


    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V2: 文脈変調、競合、ルール連携を含む更新プロセス。
        """
        if optional_params is None: optional_params = {}

        # --- 1. トレース初期化と更新 ---
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape[0] != pre_spikes.shape[0] or self.post_trace.shape[0] != post_spikes.shape[0]:
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        self._update_traces(pre_spikes, post_spikes)

        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)

        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)

        assert self.pre_trace is not None and self.post_trace is not None and self.eligibility_trace is not None and self.causal_contribution is not None

        # --- 2. 適格度トレースの更新 ---
        # eligibility_trace: (N_post, N_pre)
        potential_dw = self.a_plus * torch.outer(post_spikes, self.pre_trace) - self.a_minus * torch.outer(pre_spikes, self.post_trace).T
        self.eligibility_trace += potential_dw

        if self.modulate_eligibility_tau:
            contrib_norm = torch.sigmoid(self.causal_contribution * 10 - 5)
            current_tau_eligibility = self.min_eligibility_tau + (self.max_eligibility_tau - self.min_eligibility_tau) * contrib_norm
            eligibility_decay = (self.eligibility_trace / current_tau_eligibility.clamp(min=1e-6)) * self.dt
        else:
            eligibility_decay = (self.eligibility_trace / self.base_tau_eligibility) * self.dt
        self.eligibility_trace -= eligibility_decay

        # --- 3. 報酬/クレジット信号の処理 ---
        reward = optional_params.get("reward", 0.0)
        causal_credit_signal = optional_params.get("causal_credit", 0.0)
        effective_reward_signal = reward + causal_credit_signal

        dw = torch.zeros_like(weights)
        if abs(effective_reward_signal) > 1e-6:
            # --- 4. 動的学習率の計算 ---
            contrib_norm = torch.sigmoid(self.causal_contribution * 10 - 5)
            dynamic_lr = self.base_learning_rate * (1 + self.dynamic_lr_factor * contrib_norm)

            # --- 5. 高レベルルールによる学習率調整 ---
            dynamic_lr = self._apply_high_level_rules(dynamic_lr, optional_params, weights)

            # --- 6. 重み変化量の計算 ---
            dw = dynamic_lr * effective_reward_signal * self.eligibility_trace

            # --- 7. 競合的割り当て ---
            dw = self._apply_competition(dw, self.eligibility_trace)

            # --- 8. 長期貢献度の更新 ---
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

            # --- 9. 適格度トレースのリセット ---
            self.eligibility_trace *= 0.0

        # --- 10. クレジット信号の逆方向伝播 ---
        if self.eligibility_trace is not None:
            # credit_contribution: (N_post, N_pre)
            # weights: (N_post, N_pre)
            credit_contribution = self.eligibility_trace
            
            # --- ▼ 修正箇所: 添字を 'ij,ij->j' に変更 ▼ ---
            # これにより、プリシナプスニューロン (j) ごとの総クレジット和を計算します。
            # 結果の形状は (N_pre,) となり、前層の post_spikes と一致します。
            raw_backward_credit = torch.einsum('ij,ij->j', weights, credit_contribution) 
            # --- ▲ 修正箇所 ▲ ---

            # 文脈変調を適用
            backward_credit = self._apply_context_modulation(raw_backward_credit, optional_params)

        else:
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit
        
    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        """長期的な因果的貢献度を返す。"""
        return self.causal_contribution
