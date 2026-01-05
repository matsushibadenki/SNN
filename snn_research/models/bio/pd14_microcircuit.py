# ファイルパス: snn_research/models/bio/pd14_microcircuit.py
# Title: Potjans-Diesmann Cortical Microcircuit Model (PD14) - Validated v16.1
# Description:
#   大脳皮質の正準回路モデル。NMDAゲインと抑制バランスを調整済み。

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from snn_research.core.base import BaseModel
from snn_research.core.neurons.multi_compartment import TwoCompartmentLIF
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F

logger = logging.getLogger(__name__)


class PD14Microcircuit(BaseModel):
    def __init__(
        self,
        scale_factor: float = 0.1,
        time_steps: int = 16,
        neuron_type: str = "two_compartment",
        input_dim: int = 100,
        output_dim: int = 10
    ):
        super().__init__()
        self.time_steps = time_steps
        self.scale_factor = scale_factor

        # Layer Definitions
        base_counts = {
            "L23e": 20683, "L23i": 5834, "L4e": 21915, "L4i": 5479,
            "L5e": 4850, "L5i": 1065, "L6e": 14395, "L6i": 2948
        }
        self.pop_counts = {k: max(10, int(v * scale_factor))
                           for k, v in base_counts.items()}
        self.populations = nn.ModuleDict()

        for name, count in self.pop_counts.items():
            if neuron_type == "two_compartment":
                # v16.1 Stability Tuning
                nmda_gain = 0.15 if 'e' in name else 0.05
                self.populations[name] = TwoCompartmentLIF(
                    features=count,
                    nmda_gain=nmda_gain,
                    tau_soma=20.0,
                    v_threshold=1.0,
                    nmda_threshold=1.2
                )
            else:
                self.populations[name] = AdaptiveLIFNeuron(features=count)

        # Connectivity
        pop_names = list(base_counts.keys())
        self.connections = nn.ModuleDict()

        for src in pop_names:
            for tgt in pop_names:
                prob = self._get_connection_prob(src, tgt)
                if prob > 0:
                    src_dim, tgt_dim = self.pop_counts[src], self.pop_counts[tgt]
                    layer = nn.Linear(src_dim, tgt_dim, bias=False)
                    limit = 1.0 / np.sqrt(src_dim)

                    with torch.no_grad():
                        mask = (torch.rand(tgt_dim, src_dim) < prob).float()
                        # Inhibitory reinforcement
                        if 'i' in src:
                            nn.init.uniform_(
                                layer.weight, -limit * 8.0, -limit * 2.0)
                        else:
                            nn.init.uniform_(
                                layer.weight, limit * 0.05, limit * 0.8)
                        layer.weight *= mask

                    self.connections[f"{src}_to_{tgt}"] = layer

        # I/O Layers
        in_limit = 1.0 / np.sqrt(input_dim)
        self.thalamic_input_L4 = nn.Linear(input_dim, self.pop_counts["L4e"])
        nn.init.uniform_(self.thalamic_input_L4.weight,
                         in_limit*0.5, in_limit*2.0)
        self.thalamic_input_L6 = nn.Linear(input_dim, self.pop_counts["L6e"])
        nn.init.uniform_(self.thalamic_input_L6.weight,
                         in_limit*0.5, in_limit*2.0)
        self.feedback_input_L23 = nn.Linear(input_dim, self.pop_counts["L23e"])
        nn.init.uniform_(self.feedback_input_L23.weight,
                         in_limit*0.2, in_limit*1.5)
        self.readout = nn.Linear(self.pop_counts["L5e"], output_dim)

        logger.info(
            f"🧠 PD14 Microcircuit initialized. Neurons: {sum(self.pop_counts.values())}")

    def _get_connection_prob(self, src: str, tgt: str) -> float:
        if src == tgt:
            return 0.05
        if src[:3] == tgt[:3]:
            return 0.1
        if "L4" in src and "L23" in tgt:
            return 0.15
        if "L23" in src and "L5" in tgt:
            return 0.15
        if "L5" in src and "L6" in tgt:
            return 0.1
        if "L6" in src and "L4" in tgt:
            return 0.05
        if 'i' in src:
            return 0.2
        return 0.01

    def forward(self, thalamic_input: torch.Tensor, topdown_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = thalamic_input.shape[0]
        device = thalamic_input.device
        SJ_F.reset_net(self)

        spike_counts = {name: 0.0 for name in self.populations.keys()}
        readout_accum = torch.zeros(
            B, self.readout.out_features, device=device)
        prev_spikes = {name: torch.zeros(
            B, count, device=device) for name, count in self.pop_counts.items()}

        for t in range(self.time_steps):
            current_spikes = {}
            for name, neuron in self.populations.items():
                internal = torch.zeros(B, self.pop_counts[name], device=device)
                for src in prev_spikes:
                    if f"{src}_to_{name}" in self.connections:
                        internal += self.connections[f"{src}_to_{name}"](
                            prev_spikes[src])

                ext_soma = torch.zeros_like(internal)
                ext_dend = torch.zeros_like(internal)

                if name == "L4e":
                    ext_soma += self.thalamic_input_L4(thalamic_input)
                if name == "L6e":
                    ext_soma += self.thalamic_input_L6(thalamic_input)
                if name == "L23e" and topdown_input is not None:
                    ext_dend += self.feedback_input_L23(topdown_input)

                if isinstance(neuron, TwoCompartmentLIF):
                    spikes, _ = neuron(input_soma=internal +
                                       ext_soma, input_dend=ext_dend)
                else:
                    spikes, _ = neuron(internal+ext_soma+ext_dend)

                current_spikes[name] = spikes
                spike_counts[name] += spikes.sum().item() / B

            readout_accum += self.readout(current_spikes["L5e"])
            prev_spikes = current_spikes

        return readout_accum, {k: v/self.time_steps for k, v in spike_counts.items()}
