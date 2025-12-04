# ファイルパス: snn_research/learning_rules/causal_trace.py
# コードの最も最初には、ファイルパス、ファイルの内容を示したタイトル、機能の説明を詳細に記述してください。 修正内容は記載する必要はありません。
# Title: 進化版 因果追跡クレジット割り当て学習則 (V2)
# Description:
# - CausalTraceCreditAssignmentEnhanced を基盤とし、さらなる機能向上を目指した実装
# - 文脈依存のクレジット変調、パス依存性（簡易版）、競合的クレジット割り当て、高レベル因果連携の概念を取り入れる

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
                 modulate_eligibility_tau: bool = True, # デフォルトで有効化
                 min_eligibility_tau: float = 10.0,
                 max_eligibility_tau: float = 200.0,
                 # --- ▼ V2 新パラメータ ▼ ---
                 context_modulation_strength: float = 0.5, # 文脈による変調強度
                 competition_k_ratio: float = 0.1,        # 競合で更新を適用するシナプスの割合 (上位10%)
                 rule_based_lr_factor: float = 3.0       # 高レベルルールによる学習率増加係数
                 # --- ▲ V2 新パラメータ ▲ ---
                 ):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        self.causal_contribution: Optional[torch.Tensor] = None
        self.base_learning_rate = learning_rate
        self.credit_time_decay = credit_time_decay # Note: V2では直接使わず、tau_eligibility変調で代替
        self.dynamic_lr_factor = dynamic_lr_factor
        self.modulate_eligibility_tau = modulate_eligibility_tau
        self.min_eligibility_tau = min_eligibility_tau
        self.max_eligibility_tau = max_eligibility_tau
        self.base_tau_eligibility = tau_eligibility
        # --- ▼ V2 新属性 ▼ ---
        self.context_modulation_strength = context_modulation_strength
        self.competition_k_ratio = competition_k_ratio
        self.rule_based_lr_factor = rule_based_lr_factor
        # --- ▲ V2 新属性 ▼ ---
        print("🧠 V2 Enhanced Causal Trace Credit Assignment rule initialized.")
        print(f"   - Context Modulation Strength: {self.context_modulation_strength}")
        print(f"   - Competition Ratio (Top K%): {self.competition_k_ratio * 100:.1f}%")
        print(f"   - Rule-based LR Factor: {self.rule_based_lr_factor}")

    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        # (変更なし)
        self.causal_contribution = torch.zeros(weight_shape, device=device)

    def _apply_context_modulation(self, backward_credit: torch.Tensor, optional_params: Dict[str, Any]) -> torch.Tensor:
        """Global Workspace や Memory からの文脈情報でクレジット信号を変調する。"""
        modulated_credit = backward_credit.clone() # 元のクレジットをコピー

        workspace_context = optional_params.get("global_workspace_context") # {'type': 'emotion', 'valence': -0.8,...} など
        memory_context = optional_params.get("memory_context") # 関連する過去の記憶情報など

        modulation_factor = 1.0

        # 例: ネガティブな感情が強い場合、関連するクレジットを増幅（回避学習のため）
        if workspace_context and isinstance(workspace_context, dict) and workspace_context.get("type") == "emotion":
            valence = workspace_context.get("valence", 0.0)
            if valence < -0.5: # 強いネガティブ感情
                modulation_factor += self.context_modulation_strength * abs(valence)

        # 例: 類似した失敗記憶が想起された場合、関連クレジットを増幅
        if memory_context and isinstance(memory_context, list) and len(memory_context) > 0:
            # memory_context に失敗経験が含まれるかチェックするロジック (簡易版)
            if any("FAILURE" in str(mem.get("result")) for mem in memory_context):
                 modulation_factor += self.context_modulation_strength * 0.5 # 失敗記憶による増幅

        return modulated_credit * modulation_factor

    def _apply_competition(self, dw: torch.Tensor, eligibility_trace: torch.Tensor) -> torch.Tensor:
        """競合メカニズムを適用し、更新対象のシナプスを選択する。"""
        if self.competition_k_ratio >= 1.0: # 競合なし
            return dw

        num_synapses = dw.numel()
        k = max(1, int(num_synapses * self.competition_k_ratio))

        # 適格度トレースの絶対値が大きい上位k個のシナプスを選択
        # eligibility_trace は学習の「ポテンシャル」を示すため、dw自体より適切
        abs_eligibility = torch.abs(eligibility_trace)
        top_k_values, _ = torch.topk(abs_eligibility.view(-1), k)
        threshold = top_k_values[-1] # 上位k個の最小値

        # 閾値より小さい更新はゼロにするマスクを作成
        mask = abs_eligibility >= threshold

        return dw * mask.float() # マスクを適用

    def _apply_high_level_rules(self, dynamic_lr: torch.Tensor, optional_params: Dict[str, Any], weights: torch.Tensor) -> torch.Tensor:
        """CausalInferenceEngineからの抽象ルールに基づき学習率を調整する。"""
        rule = optional_params.get("abstract_causal_rule") # 例: {'condition': 'context_A', 'cause': 'neuron_X', 'effect': 'neuron_Y', 'increase_lr': True}

        if rule and isinstance(rule, dict):
            # このルールが現在の接続 (weights) に関連するか判定するロジックが必要
            # (ここでは簡易的に、特定のルールが来たら全体の学習率を上げる)
            if rule.get("increase_lr"):
                print(f"   - Applying high-level rule: Increasing LR for relevant synapses.")
                # 本来はルールに関連するシナプスのみを選択的に変更すべき
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

        # --- 2. 適格度トレースの更新 (時定数変調付き) ---
        potential_dw = self.a_plus * torch.outer(post_spikes, self.pre_trace) - self.a_minus * torch.outer(pre_spikes, self.post_trace).T
        self.eligibility_trace += potential_dw

        # 時定数変調 (Path Dependency の簡易版)
        if self.modulate_eligibility_tau:
            contrib_norm = torch.sigmoid(self.causal_contribution * 10 - 5)
            current_tau_eligibility = self.min_eligibility_tau + (self.max_eligibility_tau - self.min_eligibility_tau) * contrib_norm
            eligibility_decay = (self.eligibility_trace / current_tau_eligibility.clamp(min=1e-6)) * self.dt # ゼロ除算防止
        else:
            eligibility_decay = (self.eligibility_trace / self.base_tau_eligibility) * self.dt
        self.eligibility_trace -= eligibility_decay

        # --- 3. 報酬/クレジット信号の処理 ---
        reward = optional_params.get("reward", 0.0)
        causal_credit_signal = optional_params.get("causal_credit", 0.0)
        effective_reward_signal = reward + causal_credit_signal

        dw = torch.zeros_like(weights)
        if abs(effective_reward_signal) > 1e-6: # 報酬/クレジットがある場合のみ
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

        # --- 10. クレジット信号の逆方向伝播 (文脈変調付き) ---
        if self.eligibility_trace is not None: # eligibility_trace は更新後も存在する
            # eligibility_trace はそのステップでの「変化の可能性」を示す
            # backward_credit は、その可能性がどれだけ後段の信号 (reward/credit) に影響を与えたか
            # effective_reward_signal を使うべきか？ -> No, それは局所的な更新に使う
            # backward_credit は、この層の「状態変化 (eligibility)」が後段にどれだけ影響しうるか、を示すべき
            # => eligibility_trace をそのまま使い、重みで逆伝播させるのが基本

            credit_contribution = self.eligibility_trace # (N_post, N_pre)
            raw_backward_credit = torch.einsum('ij,ij->i', weights, credit_contribution) # (N_pre,)

            # 文脈変調を適用
            backward_credit = self._apply_context_modulation(raw_backward_credit, optional_params)

        else: # 通常起こらないはず
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit
        
    def get_causal_contribution(self) -> Optional[torch.Tensor]:
        """長期的な因果的貢献度を返す。"""
        return self.causal_contribution
