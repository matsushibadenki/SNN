# snn_research/bio_models/lif_neuron.py
# Title: Leaky Integrate-and-Fire (LIF) ニューロンモデル
# Description: シンプルなLIFニューロンを実装します。

import torch
import torch.nn as nn

class BioLIFNeuron(nn.Module):
    """生物学的学習則のためのシンプルなLIFニューロン。"""
    def __init__(self, n_neurons: int, neuron_params: dict, dt: float = 1.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_mem = neuron_params['tau_mem']
        self.v_thresh = neuron_params['v_threshold']
        self.v_reset = neuron_params['v_reset']
        self.v_rest = neuron_params['v_rest']
        self.dt = dt
        
        self.voltages = torch.full((n_neurons,), self.v_rest)

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        if self.voltages.device != input_current.device:
            self.voltages = self.voltages.to(input_current.device)

        # 膜電位の漏れ
        leak = (self.voltages - self.v_rest) / self.tau_mem
        
        # 膜電位の更新
        self.voltages += (-leak + input_current) * self.dt
        
        # 発火
        spikes = (self.voltages >= self.v_thresh).float()
        
        # リセット
        self.voltages[spikes.bool()] = self.v_reset
        
        return spikes