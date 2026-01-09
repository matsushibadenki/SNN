# ファイルパス: snn_research/core/layers/adapters.py
# (修正: AnalogToSpikesのrepeatロジック修正)
#
# Title: ANN-SNN アダプタレイヤー
# Description:
# - アナログ値とスパイク時系列を相互変換するためのアダプタレイヤー。
# - 修正: バッチサイズが1以外の場合に対応するため、repeatの引数を動的に生成。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder
# type: ignore[import-untyped]
from spikingjelly.activation_based import functional, base as sj_base


class AnalogToSpikes(BaseModel):
    """
    アナログ値をスパイク時系列に変換するアダプタ。
    DifferentiableTTFSEncoder (DTTFS) のロジックも含む。
    """
    neuron: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
                  DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron]

    def __init__(self, in_features: int, out_features: int, time_steps: int, activation: Type[nn.Module], neuron_config: Dict[str, Any]):
        super().__init__()
        self.time_steps = time_steps
        self.projection = nn.Linear(in_features, out_features)

        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)

        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron,
                                 GLIFNeuron, DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron]]

        filtered_params: Dict[str, Any]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = out_features
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type_str == 'dttfs':
            neuron_class = DifferentiableTTFSEncoder
            neuron_params['num_neurons'] = out_features
            neuron_params['duration'] = time_steps
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['num_neurons', 'duration', 'initial_sensitivity']
            }
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF  # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron  # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
            raise ValueError(
                f"Unknown neuron type for AnalogToSpikes: {neuron_type_str}")

        self.neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
                           DifferentiableTTFSEncoder, TC_LIF, DualThresholdNeuron], neuron_class(**filtered_params))
        self.output_act = activation()

    def forward(self, x_analog: torch.Tensor, return_full_mems: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (B, L, D_in) or (B, D_in)
        x: torch.Tensor = self.projection(x_analog)
        # (B, L, D_out) or (B, D_out)
        x = self.output_act(x)

        if isinstance(self.neuron, DifferentiableTTFSEncoder):
            # (B, L, D_out) -> (B*L, D_out)
            B = x.shape[0]
            dims = x.shape[1:-1]
            D_out = x.shape[-1]
            x_flat = x.reshape(-1, D_out)

            # (B*L, D_out) -> (B*L, T_steps, D_out)
            dttfs_spikes_stacked = self.neuron(x_flat)

            dttfs_output_shape: Tuple[int, ...]
            if x_analog.dim() == 3:  # (B, L, D_in)
                dttfs_output_shape = (B, dims[0], self.time_steps, D_out)
            else:  # (B, D_in)
                dttfs_output_shape = (B, self.time_steps, D_out)

            spikes_out = dttfs_spikes_stacked.reshape(dttfs_output_shape)

            dummy_mem = torch.zeros_like(spikes_out, requires_grad=False)

            return spikes_out, dummy_mem

        # --- 従来のLIF/Izhikevich/GLIF/TC_LIF/DualThreshold (外部T_stepsループ) ---
        # [Fix] repeat logic for variable batch sizes
        # x: (..., D_out) -> x_unsqueezed: (..., 1, D_out)
        x_unsqueezed = x.unsqueeze(-2)
        # Prepare repeat dimensions: [1, ..., 1, time_steps, 1]
        repeat_dims = [1] * x_unsqueezed.dim()
        repeat_dims[-2] = self.time_steps
        x_repeated = x_unsqueezed.repeat(*repeat_dims)

        if isinstance(self.neuron, sj_base.MemoryModule):
            cast(sj_base.MemoryModule, self.neuron).set_stateful(True)
        functional.reset_net(self.neuron)

        local_mems_history: List[torch.Tensor] = []

        hook: Optional[torch.utils.hooks.RemovableHandle] = None
        if return_full_mems:
            def _hook_mem_local(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                local_mems_history.append(output[1])  # mem
            hook = self.neuron.register_forward_hook(_hook_mem_local)

        spikes_history: List[torch.Tensor] = []

        neuron_features: int = -1
        if isinstance(self.neuron, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)):
            neuron_features = self.neuron.features
        else:
            neuron_features = self.projection.out_features  # Fallback

        x_time_batched: torch.Tensor = x_repeated.reshape(
            -1, self.time_steps, neuron_features)

        for t in range(self.time_steps):
            current_input: torch.Tensor = x_time_batched[:, t, :]
            spike_t, _ = self.neuron(current_input)
            spikes_history.append(spike_t)

        if isinstance(self.neuron, sj_base.MemoryModule):
            cast(sj_base.MemoryModule, self.neuron).set_stateful(False)

        full_mems: Optional[torch.Tensor] = None
        if return_full_mems and hook is not None:
            hook.remove()
            if local_mems_history:
                full_mems = torch.stack(local_mems_history, dim=1)

        spikes_stacked: torch.Tensor = torch.stack(spikes_history, dim=1)

        original_shape: Tuple[int, ...] = x_repeated.shape
        output_shape: Tuple[int, ...]

        if x_analog.dim() == 3:  # (B, L, D_in)
            output_shape = (
                original_shape[0], original_shape[1], self.time_steps, neuron_features)
        else:  # (B, D_in)
            output_shape = (original_shape[0],
                            self.time_steps, neuron_features)

        if full_mems is not None:
            full_mems = full_mems.reshape(output_shape)

        return spikes_stacked.reshape(output_shape), full_mems
