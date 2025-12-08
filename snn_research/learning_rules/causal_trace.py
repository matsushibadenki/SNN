# ファイルパス: snn_research/learning_rules/causal_trace.py
# Title: 進化版 因果追跡クレジット割り当て学習則 (V2 - Stabilized & Robust)
# Description:
# - CausalTraceCreditAssignmentEnhancedV2 の安定性向上版。
# - NaNチェック (torch.nan_to_num) を導入し、学習崩壊を防ぐ。
# - クレジット信号と適格性トレースのクリッピングを強化。
# - mypyエラー修正: register_buffer 削除、avg_reward の型ヒント、_apply_high_level_rules の型修正。
# - 文末の不要な '}' を削除。

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
        
        # 修正: nn.Moduleではないため register_buffer は使えない。
        # 属性として初期化する。デバイス移動は update 内で管理する。
        self.avg_reward = torch.tensor(0.0)
        
        print("🧠 V2 Enhanced Causal Trace Credit Assignment rule initialized (Stabilized).")

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
                # ネガティブ感情時は、原因となった行動へのクレジットを強める（または負の報酬なら抑制を強める）
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

        # 適格度が高いシナプスのみを更新対象とする
        abs_eligibility = torch.abs(eligibility_trace)
        
        # トップkの閾値を取得
        if k < num_synapses:
            top_k_values, _ = torch.topk(abs_eligibility.view(-1), k)
            threshold = top_k_values[-1]
            mask = abs_eligibility >= threshold
            return dw * mask.float()
        
        return dw

    # 修正: 引数と戻り値の型を torch.Tensor に変更
    def _apply_high_level_rules(self, dynamic_lr: torch.Tensor, optional_params: Dict[str, Any]) -> torch.Tensor:
        """CausalInferenceEngineからの抽象ルールに基づき学習率を調整する。"""
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
        """
        V2: 文脈変調、競合、ルール連携を含む更新プロセス。
        """
        if optional_params is None: optional_params = {}

        # デバイスの同期
        if self.avg_reward.device != weights.device:
            self.avg_reward = self.avg_reward.to(weights.device)

        # --- 1. トレース初期化と更新 ---
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape[0] != pre_spikes.shape[0] or self.post_trace.shape[0] != post_spikes.shape[0]:
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        self._update_traces(pre_spikes, post_spikes)

        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)

        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)

        # 型アサーション
        assert self.pre_trace is not None
        assert self.post_trace is not None
        assert self.eligibility_trace is not None
        assert self.causal_contribution is not None
        
        # --- 安定化: NaN除去とクリップ ---
        # トレースが爆発するのを防ぐ
        self.eligibility_trace = torch.nan_to_num(self.eligibility_trace, nan=0.0)
        self.eligibility_trace.clamp_(-10.0, 10.0)

        # --- 2. 適格度トレース (Eligibility Trace) の更新 ---
        # Hebbian項 (LTP): Post * Pre_Trace
        ltp = self.a_plus * torch.outer(post_spikes, self.pre_trace)
        # Anti-Hebbian項 (LTD): Pre * Post_Trace
        ltd = self.a_minus * torch.outer(pre_spikes, self.post_trace).T
        
        potential_dw = ltp - ltd
        self.eligibility_trace += potential_dw

        # 適格度の減衰 (長期貢献度に応じて時定数を変調)
        if self.modulate_eligibility_tau:
            # 貢献度が高いシナプスは記憶（適格度）が長く残る
            contrib_norm = torch.sigmoid(self.causal_contribution * 5.0) # スケーリング調整
            current_tau_eligibility = self.min_eligibility_tau + (self.max_eligibility_tau - self.min_eligibility_tau) * contrib_norm
            eligibility_decay = (self.eligibility_trace / current_tau_eligibility.clamp(min=1e-6)) * self.dt
        else:
            eligibility_decay = (self.eligibility_trace / self.base_tau_eligibility) * self.dt
        
        self.eligibility_trace -= eligibility_decay

        # --- 3. 報酬/クレジット信号の処理 ---
        reward = optional_params.get("reward", 0.0)
        causal_credit_signal = optional_params.get("causal_credit", 0.0)
        
        # 報酬のベースライン補正 (Advantage)
        # これにより、常に正の報酬だと重みが発散する問題を防ぐ
        # self.avg_reward は Tensor
        if abs(reward) > 1e-6:
             with torch.no_grad():
                 self.avg_reward = 0.95 * self.avg_reward + 0.05 * reward
             
             # 報酬が平均より高い場合に強化、低い場合に抑制
             effective_reward_signal = (reward - self.avg_reward.item()) + causal_credit_signal
        else:
             effective_reward_signal = causal_credit_signal # クレジットのみ

        dw = torch.zeros_like(weights)
        
        # 学習トリガー（報酬またはクレジットがある場合のみ重みを更新）
        if abs(effective_reward_signal) > 1e-6:
            # --- 4. 動的学習率の計算 ---
            # 貢献度が高いシナプスは学習率を下げる（安定化）
            contrib_norm = torch.sigmoid(self.causal_contribution)
            # 貢献度が高いほど学習率を下げる (1.0 -> 0.1)
            stability_factor = 1.1 - contrib_norm 
            
            # dynamic_lr は Tensor になる
            dynamic_lr = torch.tensor(self.base_learning_rate, device=weights.device) * stability_factor * self.dynamic_lr_factor

            # --- 5. 高レベルルールによる学習率調整 ---
            # 引数と戻り値を Tensor に統一したため修正
            dynamic_lr = self._apply_high_level_rules(dynamic_lr, optional_params)

            # --- 6. 重み変化量の計算 ---
            # dw = lr * Reward * Eligibility
            dw = dynamic_lr * effective_reward_signal * self.eligibility_trace
            
            # 安全策: NaN除去
            dw = torch.nan_to_num(dw, nan=0.0)

            # --- 7. 競合的割り当て ---
            dw = self._apply_competition(dw, self.eligibility_trace)

            # --- 8. 長期貢献度の更新 ---
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

            # --- 9. 学習後の適格度リセット（オプション）---
            # 報酬を受け取ったら、その原因となった活動の記録は一度クリアする戦略
            # 完全にゼロにするのではなく、減衰させる
            self.eligibility_trace *= 0.5

        # --- 10. クレジット信号の逆方向伝播 ---
        if self.eligibility_trace is not None:
            # 次の層（前段）に送るクレジット信号を計算
            # Post側の活動（適格度）に寄与したPre側ニューロンに対してクレジットを分配
            # 'ij,ij->j' : 重みと適格度の積をPreニューロンごとに合計
            raw_backward_credit = torch.einsum('ij,ij->j', weights, self.eligibility_trace) 
            raw_backward_credit = torch.nan_to_num(raw_backward_credit, nan=0.0)

            # 文脈変調を適用
            backward_credit = self._apply_context_modulation(raw_backward_credit, optional_params)
            
            # 信号の減衰とスケーリング
            backward_credit *= self.credit_time_decay
        else:
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit
        
    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        """長期的な因果的貢献度を返す。"""
        return self.causal_contribution
