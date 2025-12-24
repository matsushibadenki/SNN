# ファイルパス: snn_research/models/experimental/bit_spike_mamba.py
# 日本語タイトル: Bit-Spike Mamba Model (Fix: Strict Return Values)
# 目的: forwardの戻り値を厳密に定義し、アンパックエラーを防ぐ。

import torch
import torch.nn as nn
from typing import Dict, Any, Type, cast, Tuple, Union

from snn_research.core.mamba_core import SpikingMambaBlock, SpikingMamba
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
from snn_research.core.neurons import AdaptiveLIFNeuron

class BitSpikeMambaBlock(SpikingMambaBlock):
    """
    BitSpikeLinearを採用した軽量版Mambaブロック。
    """
    def __init__(self, d_model, d_state, d_conv, expand, neuron_class, neuron_params):
        super().__init__(d_model, d_state, d_conv, expand, neuron_class, neuron_params)
        # BitSpikeLayerへの置換
        self.in_proj = BitSpikeLinear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = BitSpikeLinear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.out_proj = BitSpikeLinear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # 親クラスのforward実装を利用
        # 注意: 親クラスが (out, state) を返す場合でも、ここではTensorのみを期待して処理する
        out = super().forward(x)
        if isinstance(out, tuple):
            return out[0]
        return out

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

        # レイヤー再構築
        self.layers = nn.ModuleList([
            BitSpikeMambaBlock(
                d_model, d_state, d_conv, expand, 
                cast(Type[nn.Module], AdaptiveLIFNeuron), 
                filtered_params
            )
            for _ in range(num_layers)
        ])
        self._init_weights()

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        安全なForward実装。常に (logits, avg_spikes, mem) の3要素を返す。
        """
        # Embedding
        if hasattr(self, 'embedding'):
            x = self.embedding(x)

        # Layers Loop
        for layer in self.layers:
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        
        # Norm
        if hasattr(self, 'norm_f'):
            x = self.norm_f(x)
        elif hasattr(self, 'norm'): # SpikingMamba might use 'norm'
            x = self.norm(x)
            
        # Head (Logits)
        if hasattr(self, 'lm_head'):
            x = self.lm_head(x)
        elif hasattr(self, 'output_projection'):
            x = self.output_projection(x)
            
        logits = x
        
        # ダミーのスパイク統計 (計算コスト削減のため0埋め)
        avg_spikes = torch.tensor(0.0, device=x.device)
        mem = torch.tensor(0.0, device=x.device)
        
        # 確実に3要素を返す
        return logits, avg_spikes, mem

    def get_model_size_mb(self) -> float:
        param_count = sum(p.numel() for p in self.parameters())
        bit_size = param_count * 0.25 / (1024 * 1024) 
        return bit_size
