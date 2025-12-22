# ファイルパス: snn_research/models/bio/simple_network.py
# Title: Bio-Inspired SNN (Mypy Final Fixed)
# Description: homeostatic_rulesのNoneチェックとcausal_contributionへの安全なアクセスを実装。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast
import copy
import logging

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
    """
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: Dict[str, Any], 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None, 
        sparsification_config: Optional[Dict[str, Any]] = None,
        synaptic_reliability: float = 0.9,
        neuron_type: str = "adaptive_lif"
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        config = sparsification_config or {}
        self.sparsification_enabled = config.get("enabled", False)
        self.contribution_threshold = config.get("contribution_threshold", 0.0)
        self.synaptic_reliability = synaptic_reliability
        
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        self.synaptic_rules: List[BioLearningRule] = []
        self.homeostatic_rules: List[Optional[BioLearningRule]] = []

        def create_neuron(size: int) -> nn.Module:
            p = neuron_params.copy()
            if neuron_type == "adaptive_lif":
                if 'v_threshold' in p:
                    p.setdefault('base_threshold', p.pop('v_threshold'))
                valid_keys = [
                    'tau_mem', 'base_threshold', 'adaptation_strength', 
                    'target_spike_rate', 'noise_intensity', 'threshold_decay', 
                    'threshold_step', 'v_reset', 'homeostasis_rate'
                ]
                return AdaptiveLIFNeuron(features=size, **{k: v for k, v in p.items() if k in valid_keys})
            elif neuron_type == "izhikevich":
                return IzhikevichNeuron(features=size, **{k: v for k, v in p.items() if k in ['a', 'b', 'c', 'd', 'dt']})
            return BioLIFNeuron(n_neurons=size, neuron_params=p)

        for i in range(len(layer_sizes) - 1):
            self.layers.append(create_neuron(layer_sizes[i+1]))
            w_init = torch.abs(torch.randn(layer_sizes[i+1], layer_sizes[i]) * (1.0 / (layer_sizes[i] ** 0.5))) * 0.5
            self.weights.append(nn.Parameter(w_init))
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))
            self.homeostatic_rules.append(copy.deepcopy(homeostatic_rule) if homeostatic_rule else None)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_spikes_history = []
        current_spikes = input_spikes
        for i, layer in enumerate(self.layers):
            effective_weight = apply_probabilistic_transmission(self.weights[i], self.synaptic_reliability, training=self.training)
            current = torch.nn.functional.linear(current_spikes, effective_weight)
            out = layer(current)
            current_spikes = out[0] if isinstance(out, tuple) else out
            hidden_spikes_history.append(current_spikes)
        return current_spikes, hidden_spikes_history

    def update_weights(self, all_layer_spikes: List[torch.Tensor], optional_params: Optional[Dict[str, Any]] = None):
        if not self.training: return
        backward_credit: Optional[torch.Tensor] = None
        current_params = (optional_params or {}).copy()

        for i in reversed(range(len(self.weights))):
            pre_spikes, post_spikes = all_layer_spikes[i], all_layer_spikes[i+1]
            current_params["causal_credit"] = backward_credit.mean().item() if backward_credit is not None else 0.0
            
            # シナプス可塑性更新
            dw_synaptic, backward_credit = self.synaptic_rules[i].update(
                pre_spikes=pre_spikes, post_spikes=post_spikes, weights=self.weights[i], optional_params=current_params
            )
            
            # 恒常性維持更新 (Noneチェックを追加してmypyエラーを解消)
            dw_homeo = torch.zeros_like(self.weights[i])
            h_rule = self.homeostatic_rules[i]
            if h_rule is not None:
                res, _ = h_rule.update(
                    pre_spikes=pre_spikes, post_spikes=post_spikes, weights=self.weights[i], optional_params=optional_params
                )
                if res is not None: dw_homeo = res

            dw = dw_synaptic + dw_homeo

            # ⑮ スパース化 (型安全にget_causal_contributionを呼び出し)
            if self.sparsification_enabled:
                rule = self.synaptic_rules[i]
                contrib = None
                if isinstance(rule, CausalTraceCreditAssignmentEnhancedV2):
                    contrib = rule.get_causal_contribution()
                elif hasattr(rule, 'get_causal_contribution'):
                    contrib = getattr(rule, 'get_causal_contribution')()
                
                if contrib is not None:
                    dw = dw * (contrib > self.contribution_threshold).float()

            with torch.no_grad():
                self.weights[i].add_(dw)
                self.weights[i].clamp_(min=-2.0, max=2.0)
