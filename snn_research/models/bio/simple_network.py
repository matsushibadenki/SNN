# ファイルパス: snn_research/models/bio/simple_network.py
# Title: 生物学的SNN (High-Gain Init for STDP)
# Description: STDP学習を促進するため、重み初期値を大きく設定し、発火しやすく改良。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast
import logging
from snn_research.core.base import BaseModel
from snn_research.learning_rules.base_rule import BioLearningRule

logger = logging.getLogger(__name__)

class BioSNN(BaseModel):
    """
    生物学的妥当性を備えた多層SNN。
    STDP学習のために、初期重みを活性化しやすい値に設定。
    """
    def __init__(
        self, 
        layer_sizes: List[int], 
        neuron_params: Dict[str, Any], 
        synaptic_rule: BioLearningRule, 
        homeostatic_rule: Optional[BioLearningRule] = None,
        sparsification_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.neuron_params = neuron_params
        
        self.tau_mem = neuron_params.get('tau_mem', 10.0)
        self.v_threshold = neuron_params.get('v_threshold', 1.0)
        self.v_reset = neuron_params.get('v_reset', 0.0)
        self.dt = neuron_params.get('dt', 1.0)
        
        self.weights = nn.ParameterList()
        self.synaptic_rules: List[BioLearningRule] = []
        self.mem_potentials: List[torch.Tensor] = []

        import copy
        for i in range(len(layer_sizes) - 1):
            # [修正] 重み初期化の強化
            # 通常の Xavier (1/sqrt(N)) ではSNNの発火には不十分な場合があるため、
            # ゲインを 3.0 倍にして初期アクティビティを確保する。
            w_init = torch.randn(layer_sizes[i], layer_sizes[i+1]) * (3.0 / (layer_sizes[i] ** 0.5))
            
            # 正負のバランスを少し崩して（興奮性を強く）発火を促すオプション
            w_init += 0.05 
            
            self.weights.append(nn.Parameter(w_init))
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))
            
        # 刈り込み設定
        config = sparsification_config or {}
        self.sparsification_enabled = config.get("enabled", False)

    def reset_state(self, batch_size: int, device: torch.device):
        """膜電位のリセット"""
        self.mem_potentials = []
        for size in self.layer_sizes[1:]:
            self.mem_potentials.append(torch.zeros(batch_size, size, device=device))

    def apply_causal_pruning(self, layer_idx: int) -> None:
        """因果貢献度に基づく刈り込み"""
        rule = self.synaptic_rules[layer_idx]
        if hasattr(rule, 'get_causal_contribution'):
            contribution = cast(Any, rule).get_causal_contribution()
            if contribution is not None:
                threshold = torch.quantile(contribution.abs(), 0.1)
                mask = (contribution.abs() >= threshold).float()
                with torch.no_grad():
                    if contribution.shape == self.weights[layer_idx].shape:
                        self.weights[layer_idx].data.mul_(mask)
                    elif contribution.t().shape == self.weights[layer_idx].shape:
                         self.weights[layer_idx].data.mul_(mask.t())

    def update_weights(self, all_layer_spikes: List[torch.Tensor], optional_params: Optional[Dict[str, Any]] = None) -> None:
        """STDP学習則による重み更新"""
        uncertainty = (optional_params or {}).get("uncertainty", 1.0)
        
        for i in range(len(self.weights)):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            rule = self.synaptic_rules[i]
            
            # STDPルールは (Pre, Post) に合わせた dw を返すように修正済み
            dw, _ = rule.update(pre_spikes, post_spikes, self.weights[i], optional_params)
            
            with torch.no_grad():
                self.weights[i].add_(dw)
                # クランプ範囲を拡大して表現力を維持
                self.weights[i].clamp_(-3.0, 3.0)
            
            if self.sparsification_enabled and uncertainty < 0.3:
                self.apply_causal_pruning(i)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """順伝播 (LIF)"""
        batch_size = x.shape[0]
        device = x.device
        
        if not self.mem_potentials or self.mem_potentials[0].shape[0] != batch_size:
            self.reset_state(batch_size, device)
            
        spikes_history = [x]
        current_input = x
        
        for i, weight in enumerate(self.weights):
            # I = Input @ W
            current = torch.matmul(current_input, weight)
            
            # V(t) = V(t-1)*decay + I
            decay = 1.0 - (self.dt / self.tau_mem)
            self.mem_potentials[i] = self.mem_potentials[i] * decay + current
            
            # Spike generation
            spikes = (self.mem_potentials[i] >= self.v_threshold).float()
            
            # Reset (Soft)
            self.mem_potentials[i] = self.mem_potentials[i] - (spikes * self.v_threshold)
            
            current_input = spikes
            spikes_history.append(spikes)
            
        return current_input, spikes_history
