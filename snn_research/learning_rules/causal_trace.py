# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: Causal Trace Credit Assignment V2 [Batch & Type Fixed]
# Description:
#   mypyエラー修正: _initialize_traces に shape (Size) を渡すように変更。
#   RuntimeError修正: torch.outer を torch.matmul に変更し、バッチ処理に対応。

import torch
from typing import Dict, Any, Optional, Tuple, Union, cast
import math

from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignmentEnhancedV2(RewardModulatedSTDP):
    """
    文脈変調、競合、高レベル因果連携を導入した、さらに進化した因果学習則。
    数値的安定性を強化。
    """
    avg_reward: torch.Tensor

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
        
        self.avg_reward = torch.tensor(0.0)
        
        print("🧠 V2 Enhanced Causal Trace Credit Assignment rule initialized (Stabilized).")

    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        self.causal_contribution = torch.zeros(weight_shape, device=device)

    def _apply_context_modulation(self, backward_credit: torch.Tensor, optional_params: Dict[str, Any]) -> torch.Tensor:
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
        if self.competition_k_ratio >= 1.0:
            return dw

        num_synapses = dw.numel()
        k = max(1, int(num_synapses * self.competition_k_ratio))

        abs_eligibility = torch.abs(eligibility_trace)
        
        if k < num_synapses:
            top_k_values, _ = torch.topk(abs_eligibility.view(-1), k)
            threshold = top_k_values[-1]
            mask = abs_eligibility >= threshold
            return dw * mask.float()
        
        return dw

    def _apply_high_level_rules(self, dynamic_lr: torch.Tensor, optional_params: Dict[str, Any]) -> torch.Tensor:
        rule = optional_params.get("abstract_causal_rule")
        if rule and isinstance(rule, dict):
            if rule.get("increase_lr"):
                return dynamic_lr * self.rule_based_lr_factor
        return dynamic_lr

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if optional_params is None: optional_params = {}

        # 1D入力対応 (Batch次元追加)
        if pre_spikes.dim() == 1: pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1: post_spikes = post_spikes.unsqueeze(0)

        if self.avg_reward.device != weights.device:
            self.avg_reward = self.avg_reward.to(weights.device)

        # --- 1. トレース初期化と更新 ---
        # 修正: shape[0] (int) ではなく shape (Size) を渡す
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape != pre_spikes.shape or self.post_trace.shape != post_spikes.shape:
            self._initialize_traces(pre_spikes.shape, post_spikes.shape, pre_spikes.device)
        
        self._update_traces(pre_spikes, post_spikes)

        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)

        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)

        assert self.pre_trace is not None
        assert self.post_trace is not None
        assert self.eligibility_trace is not None
        assert self.causal_contribution is not None
        
        self.eligibility_trace = torch.nan_to_num(self.eligibility_trace, nan=0.0)
        self.eligibility_trace.clamp_(-10.0, 10.0)

        # --- 2. 適格度トレース (Eligibility Trace) の更新 ---
        # 修正: torch.outer -> torch.matmul (Batch対応)
        # LTP: Post.T @ Pre -> (Post, Pre)
        ltp = self.a_plus * torch.matmul(post_spikes.t(), self.pre_trace)
        # LTD: Post_Trace.T @ Pre -> (Post, Pre)
        ltd = self.a_minus * torch.matmul(self.post_trace.t(), pre_spikes)
        
        potential_dw = ltp - ltd
        self.eligibility_trace += potential_dw

        if self.modulate_eligibility_tau:
            contrib_norm = torch.sigmoid(self.causal_contribution * 5.0)
            current_tau_eligibility = self.min_eligibility_tau + (self.max_eligibility_tau - self.min_eligibility_tau) * contrib_norm
            eligibility_decay = (self.eligibility_trace / current_tau_eligibility.clamp(min=1e-6)) * self.dt
        else:
            eligibility_decay = (self.eligibility_trace / self.base_tau_eligibility) * self.dt
        
        self.eligibility_trace -= eligibility_decay

        # --- 3. 報酬/クレジット信号の処理 ---
        reward = optional_params.get("reward", 0.0)
        causal_credit_signal = optional_params.get("causal_credit", 0.0)
        
        if abs(reward) > 1e-6:
             with torch.no_grad():
                 self.avg_reward = 0.95 * self.avg_reward + 0.05 * reward
             effective_reward_signal = (reward - self.avg_reward.item()) + causal_credit_signal
        else:
             effective_reward_signal = causal_credit_signal

        dw = torch.zeros_like(weights)
        
        if abs(effective_reward_signal) > 1e-6:
            # --- 4. 動的学習率 ---
            contrib_norm = torch.sigmoid(self.causal_contribution)
            stability_factor = 1.1 - contrib_norm 
            
            dynamic_lr = torch.tensor(self.base_learning_rate, device=weights.device) * stability_factor * self.dynamic_lr_factor
            dynamic_lr = self._apply_high_level_rules(dynamic_lr, optional_params)

            # --- 6. 重み変化量 ---
            dw = dynamic_lr * effective_reward_signal * self.eligibility_trace
            dw = torch.nan_to_num(dw, nan=0.0)

            # --- 7. 競合 ---
            dw = self._apply_competition(dw, self.eligibility_trace)

            # --- 8. 貢献度更新 ---
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

            # --- 9. リセット ---
            self.eligibility_trace *= 0.5

        # --- 10. クレジット逆伝播 ---
        if self.eligibility_trace is not None:
            raw_backward_credit = torch.einsum('ij,ij->j', weights, self.eligibility_trace) 
            raw_backward_credit = torch.nan_to_num(raw_backward_credit, nan=0.0)
            backward_credit = self._apply_context_modulation(raw_backward_credit, optional_params)
            backward_credit *= self.credit_time_decay
        else:
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit
        
    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        return self.causal_contribution
