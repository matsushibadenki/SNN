# ファイルパス: snn_research/models/bio/simple_network.py
# (修正: BaseModel継承による共通インターフェース準拠)
#
# Title: Bio-Inspired SNN (修正版)
# Description:
# - 生物学的学習則 (STDP, BCM等) を用いた多層SNNモデル。
# - 修正: BaseModelを継承し、get_total_spikesやreset_spike_statsに対応。
# - 修正: 学習則インスタンスをディープコピーし、層ごとの状態混線を防止。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast
import copy

from .lif_neuron_legacy import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
from snn_research.core.synapse_dynamics import apply_probabilistic_transmission
# 修正: BaseModelのインポート
from snn_research.core.base import BaseModel

class BioSNN(BaseModel):
    """
    生物学的学習則を用いたSNNモデル。
    各層ごとに独立した学習則インスタンスを管理する。
    """
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: dict, 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None, 
        sparsification_config: Optional[Dict[str, Any]] = None,
        synaptic_reliability: float = 0.9
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        
        self.synaptic_reliability = synaptic_reliability
        
        if self.sparsification_enabled:
            pass # ログ出力は省略可能

        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        
        self.synaptic_rules: List[BioLearningRule] = []
        self.homeostatic_rules: List[Optional[BioLearningRule]] = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(BioLIFNeuron(layer_sizes[i+1], neuron_params))
            weight = nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i]) * 0.5)
            self.weights.append(weight)
            
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))
            
            if homeostatic_rule:
                self.homeostatic_rules.append(copy.deepcopy(homeostatic_rule))
            else:
                self.homeostatic_rules.append(None)
        
        # BaseModelの初期化 (get_total_spikesなどのために必要なら)
        # self._init_weights() # BioSNNは独自に初期化しているため不要

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_spikes_history = []
        current_spikes = input_spikes
        
        for i, layer in enumerate(self.layers):
            # シナプスゆらぎの適用
            effective_weight = apply_probabilistic_transmission(
                self.weights[i],
                self.synaptic_reliability,
                training=self.training
            )
            
            # 電流計算
            current = torch.matmul(effective_weight, current_spikes)
            
            # ニューロン更新
            current_spikes = layer(current)
            hidden_spikes_history.append(current_spikes)
            
        return current_spikes, hidden_spikes_history
        
    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor],
        optional_params: Optional[Dict[str, Any]] = None
    ):
        """
        全層の重みを更新する。
        """
        if not self.training:
            return

        backward_credit: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        # 逆順（出力層 -> 入力層）で更新を行う（クレジット伝播のため）
        for i in reversed(range(len(self.weights))):
            # 入力スパイク (Pre)
            pre_spikes = all_layer_spikes[i]
            # 出力スパイク (Post)
            post_spikes = all_layer_spikes[i+1]

            if backward_credit is not None:
                # 階層的クレジット信号を報酬に加算
                reward_signal = current_params.get("reward", 0.0)
                # 簡易的な変調
                try:
                    modulated_reward = reward_signal + backward_credit.mean().item() * 0.1
                    current_params["reward"] = modulated_reward
                except Exception:
                    pass

            layer_synaptic_rule = self.synaptic_rules[i]
            layer_homeostatic_rule = self.homeostatic_rules[i]
            
            # 1. シナプス可塑性ルール
            dw_synaptic, backward_credit_new = layer_synaptic_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=current_params
            )
            backward_credit = backward_credit_new
            
            # 2. 恒常性維持ルール
            dw_homeostasis = torch.zeros_like(self.weights[i].data)
            if layer_homeostatic_rule:
                dw_homeo, _ = layer_homeostatic_rule.update(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    weights=self.weights[i],
                    optional_params=optional_params 
                )
                dw_homeostasis = dw_homeo

            dw = dw_synaptic + dw_homeostasis

            # 適応的因果スパース化
            if self.sparsification_enabled and isinstance(layer_synaptic_rule, CausalTraceCreditAssignmentEnhancedV2):
                causal_contribution = layer_synaptic_rule.get_causal_contribution()
                if causal_contribution is not None:
                    contribution_mask = causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask.float()

            # 重み更新適用
            with torch.no_grad():
                self.weights[i].add_(dw)
                self.weights[i].clamp_(min=0.0)
