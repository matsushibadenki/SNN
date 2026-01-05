# ファイルパス: snn_research/models/bio/simple_network.py
# Title: BioSNN with Learning Reset
# Description: 学習則の状態をリセットする機能を追加し、バッチ学習に対応。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
from snn_research.core.base import BaseModel
from snn_research.learning_rules.base_rule import BioLearningRule

logger = logging.getLogger(__name__)


class BioSNN(BaseModel):
    """
    生物学的妥当性を備えた多層SNN。
    Noise Injectionにより、学習初期のスパースな状態でも活動を保証する。
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

        self.tau_mem = neuron_params.get('tau_mem', 20.0)
        self.v_threshold = neuron_params.get('v_threshold', 1.0)
        self.v_reset = neuron_params.get('v_reset', 0.0)
        self.dt = neuron_params.get('dt', 1.0)
        # ノイズレベル (標準偏差)
        self.noise_std = neuron_params.get('noise_std', 0.1)

        self.weights = nn.ParameterList()
        self.synaptic_rules: List[BioLearningRule] = []
        self.mem_potentials: List[torch.Tensor] = []

        import copy
        for i in range(len(layer_sizes) - 1):
            # 初期重み: 強めの初期化で信号伝播を助ける
            gain = 2.0
            w_init = torch.randn(
                layer_sizes[i], layer_sizes[i+1]) * (gain / (layer_sizes[i] ** 0.5))
            self.weights.append(nn.Parameter(w_init))
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))

        config = sparsification_config or {}
        self.sparsification_enabled = config.get("enabled", False)

    def reset_state(self, batch_size: int, device: torch.device):
        self.mem_potentials = []
        for size in self.layer_sizes[1:]:
            self.mem_potentials.append(
                torch.zeros(batch_size, size, device=device))

    def reset_learning_rules(self):
        """全てのシナプス学習則の内部状態をリセットする"""
        for rule in self.synaptic_rules:
            if hasattr(rule, 'reset'):
                rule.reset()

    def update_weights(self, all_layer_spikes: List[torch.Tensor], optional_params: Optional[Dict[str, Any]] = None) -> None:

        for i in range(len(self.weights)):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            rule = self.synaptic_rules[i]

            dw, _ = rule.update(pre_spikes, post_spikes,
                                self.weights[i], optional_params)

            with torch.no_grad():
                self.weights[i].add_(dw)
                self.weights[i].clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.shape[0]
        device = x.device

        if not self.mem_potentials or self.mem_potentials[0].shape[0] != batch_size:
            self.reset_state(batch_size, device)

        spikes_history = [x]
        current_input = x

        for i, weight in enumerate(self.weights):
            # 電流入力
            current = torch.matmul(current_input, weight)

            # ノイズ注入 (確率的発火の促進)
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(current) * self.noise_std
                current = current + noise

            decay = 1.0 - (self.dt / self.tau_mem)
            self.mem_potentials[i] = self.mem_potentials[i] * decay + current

            # 発火
            spikes = (self.mem_potentials[i] >= self.v_threshold).float()

            # リセット
            self.mem_potentials[i] = self.mem_potentials[i] - \
                (spikes * self.v_threshold)

            current_input = spikes
            spikes_history.append(spikes)

        return current_input, spikes_history
