# ファイルパス: snn_research/core/cortical_column.py
# (修正: mypy [assignment] エラー修正)

import torch
import torch.nn as nn
from typing import Dict, Any, Type, Tuple, List, Optional, cast, Union

from .base import BaseModel, SNNLayerNorm
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron

class CorticalLayer(nn.Module):
    """
    皮質カラム内の1つの層 (例: L4, L2/3)。
    """
    def __init__(self, features: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any], name: str):
        super().__init__()
        self.name = name
        self.neuron = neuron_class(features=features, **neuron_params)
        self.norm = SNNLayerNorm(features)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes, mem = self.neuron(x)
        return spikes, mem

class CorticalColumn(BaseModel):
    """
    3層構造 (L4, L2/3, L5/6) を持つ簡易化された皮質カラムモデル。
    """
    def __init__(
        self, 
        input_dim: int, 
        column_dim: int, 
        output_dim: int, 
        neuron_config: Dict[str, Any],
        **kwargs: Any
    ):
        super().__init__()
        self.column_dim = column_dim
        
        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        
        # --- 修正: 型ヒントを一般化 ---
        neuron_class: Type[nn.Module]
        
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            if 'v_threshold' in neuron_params:
                if 'base_threshold' not in neuron_params:
                    neuron_params['base_threshold'] = neuron_params['v_threshold']
            
            valid_keys = [
                'tau_mem', 'base_threshold', 'adaptation_strength', 
                'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step'
            ]
            filtered_params = {k: v for k, v in neuron_params.items() if k in valid_keys}
            neuron_params = filtered_params
            
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
            valid_keys = ['a', 'b', 'c', 'd', 'dt']
            filtered_params = {k: v for k, v in neuron_params.items() if k in valid_keys}
            neuron_params = filtered_params
        else:
            neuron_class = AdaptiveLIFNeuron

        self.L4 = CorticalLayer(column_dim, neuron_class, neuron_params, "L4")
        self.L23 = CorticalLayer(column_dim, neuron_class, neuron_params, "L23")
        self.L56 = CorticalLayer(column_dim, neuron_class, neuron_params, "L56")

        self.proj_input_L4 = nn.Linear(input_dim, column_dim)
        self.proj_L4_L23 = nn.Linear(column_dim, column_dim)
        self.proj_L23_L56 = nn.Linear(column_dim, column_dim)
        self.proj_L56_L4 = nn.Linear(column_dim, column_dim)
        
        self.rec_L4 = nn.Linear(column_dim, column_dim)
        self.rec_L23 = nn.Linear(column_dim, column_dim)
        self.rec_L56 = nn.Linear(column_dim, column_dim)

        self.proj_out_ff = nn.Linear(column_dim, output_dim)
        self.proj_out_fb = nn.Linear(column_dim, output_dim)

        self._init_weights()

    def forward(
        self, 
        input_signal: torch.Tensor, 
        prev_states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = input_signal.shape[0]
        device = input_signal.device
        
        if prev_states is None:
            prev_states = {
                "L4": torch.zeros(batch_size, self.column_dim, device=device),
                "L23": torch.zeros(batch_size, self.column_dim, device=device),
                "L56": torch.zeros(batch_size, self.column_dim, device=device)
            }
            
        spikes_L4_prev = prev_states["L4"]
        spikes_L23_prev = prev_states["L23"]
        spikes_L56_prev = prev_states["L56"]

        in_L4 = self.proj_input_L4(input_signal) + \
                self.proj_L56_L4(spikes_L56_prev) + \
                self.rec_L4(spikes_L4_prev)
        spikes_L4, _ = self.L4(in_L4)

        in_L23 = self.proj_L4_L23(spikes_L4) + \
                 self.rec_L23(spikes_L23_prev)
        spikes_L23, _ = self.L23(in_L23)

        in_L56 = self.proj_L23_L56(spikes_L23) + \
                 self.rec_L56(spikes_L56_prev)
        spikes_L56, _ = self.L56(in_L56)

        out_ff = self.proj_out_ff(spikes_L23)
        out_fb = self.proj_out_fb(spikes_L56)
        
        current_states = {
            "L4": spikes_L4,
            "L23": spikes_L23,
            "L56": spikes_L56
        }
        
        return out_ff, out_fb, current_states