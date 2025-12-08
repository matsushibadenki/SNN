# ファイルパス: snn_research/models/bio/simple_network.py
# Title: Bio-Inspired SNN (Robust Implementation & Parameter Logic Fixed)
# Description:
# - 生物学的学習則 (STDP, BCM等) を用いた多層SNNモデル。
# - 修正: ニューロン生成時のパラメータフィルタリングとマッピングを実装し、TypeErrorを解消。
# - 修正: v_threshold (Legacy) -> base_threshold (AdaptiveLIF) の自動変換を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Type, Union, cast
import copy
import logging

# 既存の BioLIFNeuron (Legacy) と新しい AdaptiveLIFNeuron をインポート
from .lif_neuron_legacy import BioLIFNeuron
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
from snn_research.core.synapse_dynamics import apply_probabilistic_transmission
from snn_research.core.base import BaseModel

logger = logging.getLogger(__name__)

class BioSNN(BaseModel):
    """
    生物学的学習則を用いたSNNモデル。
    各層ごとに独立した学習則インスタンスを管理し、柔軟な自己組織化を可能にする。
    """
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: dict, 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None, 
        sparsification_config: Optional[Dict[str, Any]] = None,
        synaptic_reliability: float = 0.9,
        neuron_type: str = "adaptive_lif" # "legacy_lif", "adaptive_lif", "izhikevich"
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        self.synaptic_reliability = synaptic_reliability
        
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        
        self.synaptic_rules: List[BioLearningRule] = []
        self.homeostatic_rules: List[Optional[BioLearningRule]] = []

        # --- ニューロン生成関数の定義 (パラメータ適応ロジックを含む) ---
        def create_neuron(size: int) -> nn.Module:
            p = neuron_params.copy()

            if neuron_type == "adaptive_lif":
                # マッピング: v_threshold -> base_threshold
                if 'v_threshold' in p:
                    if 'base_threshold' not in p:
                        p['base_threshold'] = p['v_threshold']
                    del p['v_threshold']
                
                # 不要なパラメータの削除 (AdaptiveLIFNeuronが受け付けないもの)
                valid_keys = [
                    'tau_mem', 'base_threshold', 'adaptation_strength', 
                    'target_spike_rate', 'noise_intensity', 'threshold_decay', 
                    'threshold_step', 'v_reset', 'homeostasis_rate'
                ]
                filtered_p = {k: v for k, v in p.items() if k in valid_keys}
                
                return AdaptiveLIFNeuron(features=size, **filtered_p)

            elif neuron_type == "izhikevich":
                # Izhikevich用のパラメータフィルタリング
                valid_keys = ['a', 'b', 'c', 'd', 'dt']
                filtered_p = {k: v for k, v in p.items() if k in valid_keys}
                return IzhikevichNeuron(features=size, **filtered_p)

            else: # legacy_lif or default
                # BioLIFNeuronは v_threshold 等をそのまま期待する
                # フィルタリングなしで渡す（または必要なものだけ渡す）
                return BioLIFNeuron(n_neurons=size, neuron_params=p)

        # -----------------------------------------------------------

        for i in range(len(layer_sizes) - 1):
            # ニューロン層の追加 (出力側)
            try:
                self.layers.append(create_neuron(layer_sizes[i+1]))
            except TypeError as e:
                logger.error(f"Failed to create neuron layer {i} (Type: {neuron_type}). Params: {neuron_params}")
                raise e
            
            # 重みの初期化 (Xavier init like)
            w_init = torch.randn(layer_sizes[i+1], layer_sizes[i]) * (1.0 / (layer_sizes[i] ** 0.5))
            # 興奮性結合として正の値で開始するバイアスをかける場合
            w_init = torch.abs(w_init) * 0.5 
            weight = nn.Parameter(w_init)
            self.weights.append(weight)
            
            # 学習則の複製 (層ごとに独立した状態を持つため)
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))
            
            if homeostatic_rule:
                self.homeostatic_rules.append(copy.deepcopy(homeostatic_rule))
            else:
                self.homeostatic_rules.append(None)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_spikes_history = []
        current_spikes = input_spikes
        
        for i, layer in enumerate(self.layers):
            # シナプスゆらぎの適用 (確率的伝達)
            effective_weight = apply_probabilistic_transmission(
                self.weights[i],
                self.synaptic_reliability,
                training=self.training
            )
            
            # 電流計算 (Linear projection)
            # input: (B, In), weight: (Out, In) -> output: (B, Out)
            # torch.matmul(input, weight.t()) or F.linear
            current = torch.nn.functional.linear(current_spikes, effective_weight)
            
            # ニューロン更新
            # AdaptiveLIFNeuronなどは (spikes, mem) を返すが、Legacyは spikes のみ
            out = layer(current)
            
            if isinstance(out, tuple):
                current_spikes = out[0]
            else:
                current_spikes = out
            
            hidden_spikes_history.append(current_spikes)
            
        return current_spikes, hidden_spikes_history
        
    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor],
        optional_params: Optional[Dict[str, Any]] = None
    ):
        """
        全層の重みを更新する。
        all_layer_spikes: [Input, Hidden1, Hidden2, ..., Output] の順
        """
        if not self.training:
            return

        backward_credit: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        # 逆順（出力層 -> 入力層）で更新を行う（クレジット伝播のため）
        # weights[i] は layers[i] (i+1番目のニューロン群) への入力重み
        # 対応するスパイクは all_layer_spikes[i] (Pre) と all_layer_spikes[i+1] (Post)
        
        for i in reversed(range(len(self.weights))):
            # 入力スパイク (Pre)
            pre_spikes = all_layer_spikes[i]
            # 出力スパイク (Post)
            post_spikes = all_layer_spikes[i+1]

            # 階層的クレジット信号を報酬に加算
            if backward_credit is not None:
                # クレジット信号による報酬変調 (Pre-synaptic側へのフィードバック)
                # backward_credit は (Batch, Pre_Neurons) なので平均を取ってスカラー化
                credit_scalar = backward_credit.mean().item()
                current_params["causal_credit"] = credit_scalar # 学習則内で使用
            else:
                current_params["causal_credit"] = 0.0

            layer_synaptic_rule = self.synaptic_rules[i]
            layer_homeostatic_rule = self.homeostatic_rules[i]
            
            # 1. シナプス可塑性ルール (STDP / Causal Trace)
            dw_synaptic, backward_credit_new = layer_synaptic_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=current_params
            )
            backward_credit = backward_credit_new
            
            # 2. 恒常性維持ルール (BCM / Synaptic Scaling)
            dw_homeostasis = torch.zeros_like(self.weights[i].data)
            if layer_homeostatic_rule:
                dw_homeo, _ = layer_homeostatic_rule.update(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    weights=self.weights[i],
                    optional_params=optional_params 
                )
                if dw_homeo is not None:
                    dw_homeostasis = dw_homeo

            # 統合
            dw = dw_synaptic + dw_homeostasis

            # 適応的因果スパース化 (貢献度の低いシナプスの刈り込み)
            if self.sparsification_enabled and isinstance(layer_synaptic_rule, CausalTraceCreditAssignmentEnhancedV2):
                causal_contribution = layer_synaptic_rule.get_causal_contribution()
                if causal_contribution is not None:
                    contribution_mask = causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask.float()

            # 重み更新適用
            with torch.no_grad():
                self.weights[i].add_(dw)
                
                # 重みのクリッピング (発散防止 & 生物学的制約)
                # 興奮性なら正、抑制性なら負に保つなどの制約もここに入れるべきだが
                # ここでは単純な範囲制限とする
                self.weights[i].clamp_(min=-2.0, max=2.0)
                
    def get_total_spikes(self) -> float:
        total = 0.0
        for layer in self.layers:
            if hasattr(layer, 'total_spikes'):
                # 修正: castを使用してTensorであることを明示してからitem()を呼ぶ
                spikes_tensor = cast(torch.Tensor, layer.total_spikes)
                total += spikes_tensor.item()
        return total

    def reset_spike_stats(self):
        for layer in self.layers:
             if hasattr(layer, 'reset'): layer.reset()
