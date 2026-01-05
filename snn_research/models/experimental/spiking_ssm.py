# ファイルパス: snn_research/models/experimental/spiking_ssm.py
# Title: Spiking State Space Model (SpikingSSM) - ロジック修正版
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast, Union

from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]

import logging
logger = logging.getLogger(__name__)

class S4DLIFBlock(sj_base.MemoryModule):
    """
    S4D (Structured State Space) の計算を
    LIFニューロンのダイナミクスで近似するコアブロック。
    """
    lif_h: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]
    lif_y: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]

    def __init__(
        self,
        d_model: int, 
        d_state: int, 
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        d_conv: int = 4, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model, 
            bias=True
        )
        
        self.in_proj_B = nn.Linear(d_model, d_state)
        self.in_proj_D = nn.Linear(d_model, d_model)
        self.state_proj_A = nn.Linear(d_state, d_state)
        self.out_proj_C = nn.Linear(d_state, d_model)

        neuron_params_state = neuron_params.copy()
        neuron_params_state.pop('features', None)
        self.lif_h = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron], neuron_class(features=d_state, **neuron_params_state))
        
        neuron_params_out = neuron_params.copy()
        neuron_params_out.pop('features', None)
        self.lif_y = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron], neuron_class(features=d_model, **neuron_params_out))

        self.norm = SNNLayerNorm(d_model)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        if hasattr(self.lif_h, 'set_stateful'):
            cast(Any, self.lif_h).set_stateful(stateful)
        if hasattr(self.lif_y, 'set_stateful'):
            cast(Any, self.lif_y).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.lif_h, 'reset'):
            cast(Any, self.lif_h).reset()
        if hasattr(self.lif_y, 'reset'):
            cast(Any, self.lif_y).reset()

    def forward(
        self, 
        x_t: torch.Tensor, 
        h_t_prev: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h_transition_current: torch.Tensor = self.state_proj_A(h_t_prev) 
        h_input_current: torch.Tensor = self.in_proj_B(x_t)
        h_t_spike, h_t_mem = self.lif_h(h_transition_current + h_input_current) 
        
        y_state_current: torch.Tensor = self.out_proj_C(h_t_spike)
        y_input_current: torch.Tensor = self.in_proj_D(x_t)
        
        y_t_spike, _ = self.lif_y(y_state_current + y_input_current) 
        
        y_t_out: torch.Tensor = self.norm(x_t + y_t_spike)
        
        return y_t_out, h_t_spike


class SpikingSSM(BaseModel):
    """
    Spiking State Space Model (SpikingSSM) アーキテクチャ。
    """
    embedding: nn.Embedding
    pos_encoder: nn.Parameter
    layers: nn.ModuleList
    final_norm: SNNLayerNorm
    output_projection: nn.Linear
    conv1d_input: nn.Conv1d

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_state: int = 64,
        num_layers: int = 6,
        time_steps: int = 16, 
        d_conv: int = 4, 
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.time_steps = time_steps
        
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}
            
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[nn.Module]
        filtered_params: Dict[str, Any] = {}

        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_mem', 'base_threshold']}
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'a', 'b', 'c', 'd']}
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'base_threshold']}
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']}
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']}
        else:
            raise ValueError(f"Unknown neuron type for SpikingSSM: {neuron_type_str}")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        self.conv1d_input = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True
        )

        self.layers = nn.ModuleList([
            S4DLIFBlock(d_model, d_state, neuron_class, filtered_params, d_conv=d_conv)
            for _ in range(num_layers)
        ])
        
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logger.info(f"✅ SpikingSSM (S4D-LIF RNN Mode) initialized. (Layers: {num_layers}, D_State: {d_state})")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        output_hidden_states: bool = False,
        return_full_hiddens: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T_seq = input_ids.shape
        device: torch.device = input_ids.device
        
        SJ_F.reset_net(self)
        
        x: torch.Tensor = self.embedding(input_ids) 
        x = x + self.pos_encoder[:, :T_seq, :]
        
        x_conv: torch.Tensor = self.conv1d_input(x.transpose(1, 2)) 
        x_conv = x_conv[..., :T_seq].transpose(1, 2) 
        
        for layer_module in self.layers:
            layer_to_set: S4DLIFBlock = cast(S4DLIFBlock, layer_module)
            layer_to_set.set_stateful(True)

        outputs: List[torch.Tensor] = []
        
        h_states: List[torch.Tensor] = [
            torch.zeros(B, self.d_state, device=device) for _ in range(self.num_layers)
        ]
        
        for t_idx in range(T_seq):
            x_t: torch.Tensor = x_conv[:, t_idx, :]
            x_t_layer: torch.Tensor = x_t
            
            for i in range(self.num_layers):
                layer: S4DLIFBlock = cast(S4DLIFBlock, self.layers[i])
                y_t, h_t_new = layer(x_t_layer, h_states[i])
                x_t_layer = y_t
                h_states[i] = h_t_new
            
            outputs.append(x_t_layer)
        
        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * T_seq)
        # ------------------------------------

        for layer_module in self.layers:
            layer_to_reset: S4DLIFBlock = cast(S4DLIFBlock, layer_module)
            layer_to_reset.set_stateful(False)

        x_final_seq: torch.Tensor = torch.stack(outputs, dim=1)
        x_norm_final: torch.Tensor = self.final_norm(x_final_seq)
        
        output: torch.Tensor
        if output_hidden_states or return_full_hiddens:
            output = x_norm_final
        else:
            logits: torch.Tensor = self.output_projection(x_norm_final)
            output = logits
        
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device) 

        return output, avg_spikes, mem