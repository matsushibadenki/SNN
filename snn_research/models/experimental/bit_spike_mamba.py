# ファイルパス: snn_research/models/experimental/bit_spike_mamba.py
# 日本語タイトル: Bit-Spike Mamba Model (Brain v20 Prototype) - Fixed v3
# 目的・内容:
#   SpikingMambaBlockの線形射影層をBitSpikeLinearに置換。
#   修正: get_model_size_mb メソッドを復元。

import torch
import torch.nn as nn
from typing import Dict, Any, Type, cast

from snn_research.core.mamba_core import SpikingMambaBlock, SpikingMamba
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
from snn_research.core.neurons import AdaptiveLIFNeuron

class BitSpikeMambaBlock(SpikingMambaBlock):
    """
    BitSpikeLinearを採用した軽量版Mambaブロック。
    """
    def __init__(self, d_model, d_state, d_conv, expand, neuron_class, neuron_params):
        super().__init__(d_model, d_state, d_conv, expand, neuron_class, neuron_params)
        # --- Layer Replacement with BitSpike ---
        self.in_proj = BitSpikeLinear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = BitSpikeLinear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.out_proj = BitSpikeLinear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        return super().forward(x)

class BitSpikeMamba(SpikingMamba):
    """
    Brain v20 アーキテクチャの中核となるモデル。
    """
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, num_layers, time_steps, neuron_config, **kwargs):
        super().__init__(vocab_size, d_model, d_state, d_conv, expand, num_layers, time_steps, neuron_config, **kwargs)
        
        # パラメータフィルタリング
        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        
        filtered_params = {}
        if neuron_type == 'lif':
            valid_keys = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 
                          'target_spike_rate', 'noise_intensity', 'threshold_decay', 
                          'threshold_step', 'v_reset']
            filtered_params = {k: v for k, v in neuron_params.items() if k in valid_keys}
        else:
            filtered_params = neuron_params

        # レイヤーの再構築（BitSpikeMambaBlockを使用）
        self.layers = nn.ModuleList([
            BitSpikeMambaBlock(
                d_model, d_state, d_conv, expand, 
                cast(Type[nn.Module], AdaptiveLIFNeuron), 
                filtered_params
            )
            for _ in range(num_layers)
        ])
        self._init_weights()

    def get_model_size_mb(self) -> float:
        """モデルサイズ(MB)を計算。1.58bit化による圧縮効果を表示"""
        param_count = sum(p.numel() for p in self.parameters())
        # BitSpikeなら理論上 2 bits/param (0.25 bytes)
        bit_size = param_count * 0.25 / (1024 * 1024) 
        return bit_size