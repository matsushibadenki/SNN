# ファイルパス: snn_research/core/layers/complex_attention.py
# Title: Bit-Spike Multi-Level Attention
# Description:
# - SpikingTransformer 用アテンションレイヤー。
# - 修正: BitSpikeLinear を採用し、乗算フリーの1.58bitアテンションを実現。
# - 修正: スケール不一致時にゼロパディングを追加し、out_projへの入力次元を保証。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Type, cast, Union
import logging

from snn_research.core.base import SNNLayerNorm
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)
from snn_research.io.spike_encoder import DifferentiableTTFSEncoder 
# BitNetインポート
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    BitSpikeLinear = nn.Linear # type: ignore

logger = logging.getLogger(__name__)

class MultiLevelSpikeDrivenSelfAttention(nn.Module):
    """
    BitNet 1.58bit 統合型 Multi-Level Spike Self-Attention。
    """
    neuron_out: nn.Module 
    mem_history: List[torch.Tensor]
    neuron_q: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]
    neuron_k: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]

    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any], time_scales: List[int] = [1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.time_scales = time_scales
        self.mem_history = []
        
        # BitSpikeLinear への置き換え
        # これにより Q, K, V の射影が {-1, 0, 1} の重みで行われる
        self.q_proj = BitSpikeLinear(d_model, d_model)
        self.k_proj = BitSpikeLinear(d_model, d_model)
        self.v_proj = BitSpikeLinear(d_model, d_model)
        self.out_proj = BitSpikeLinear(d_model * len(time_scales), d_model)
        
        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        
        self.neuron_q = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **filtered_params))
        self.neuron_k = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **filtered_params))
        self.neuron_out = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **filtered_params))
        
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.01))

    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        elif neuron_class == DifferentiableTTFSEncoder:
            valid_params = ['num_neurons', 'duration', 'initial_sensitivity']
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        
        filtered_params: Dict[str, Any] = {k: v for k, v in neuron_params.items() if k in valid_params}
        return filtered_params

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.mem_history.append(output[1])
    
    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.append(self.neuron_q.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_k.register_forward_hook(self._hook_mem))
        hooks.append(self.neuron_out.register_forward_hook(self._hook_mem))
        return hooks

    def clear_mem_history(self) -> None:
        self.mem_history = []

    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        # XNOR演算はビットワイズ演算の近似として浮動小数点で行う
        # (0,1)入力に対して、一致すれば1, 不一致なら0に近い挙動
        # ここではL2距離の逆数的なアプローチを継続
        q_ext: torch.Tensor = q_spikes.unsqueeze(3)
        k_ext: torch.Tensor = k_spikes.unsqueeze(2)
        xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1)
        return attn_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q_raw, _ = self.neuron_q(self.q_proj(x))
        k_raw, _ = self.neuron_k(self.k_proj(x))
        v = self.v_proj(x)

        q_gate = torch.sigmoid(q_raw - self.sparsity_threshold)
        k_gate = torch.sigmoid(k_raw - self.sparsity_threshold)
        q = q_raw * q_gate
        k = k_raw * k_gate

        outputs: List[torch.Tensor] = []
        for scale in self.time_scales:
            # 修正: Tがscaleで割り切れない場合でも、out_projの次元に合わせるためにゼロテンソルを追加する
            if T >= scale and T % scale == 0:
                q_scaled = F.avg_pool1d(q.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                k_scaled = F.avg_pool1d(k.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                v_scaled = F.avg_pool1d(v.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                
                T_scaled = q_scaled.shape[1]

                q_h = q_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                k_h = k_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                v_h = v_scaled.view(B, T_scaled, self.n_head, self.d_head).permute(0, 2, 1, 3) 
                
                attn_scores_xnor = self._xnor_similarity(q_h, k_h) 
                attn_weights = torch.sigmoid(attn_scores_xnor) 
                attn_output = torch.matmul(attn_weights, v_h) 
                
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_scaled, C)
                attn_output_upsampled = F.interpolate(attn_output.transpose(1, 2), size=T, mode='nearest').transpose(1, 2)
                outputs.append(attn_output_upsampled)
            else:
                # scale条件を満たさない場合はゼロ埋めして次元数を維持
                outputs.append(torch.zeros(B, T, C, device=x.device))

        if not outputs:
             # time_scalesが空の場合など（通常ありえないが念のため）
             neuron_out_spikes, _ = self.neuron_out(x)
             return cast(torch.Tensor, neuron_out_spikes)

        concatenated_output = torch.cat(outputs, dim=-1)
        final_output = self.out_proj(concatenated_output)
        final_spikes, _ = self.neuron_out(final_output.reshape(B*T, -1))
        return final_spikes.reshape(B, T, C)

class STAttenBlock(nn.Module):
    """
    Spiking Transformer Block with BitNet.
    """
    mem_history: List[torch.Tensor]
    lif1: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]
    lif2: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]
    lif3: Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF]
    attn: MultiLevelSpikeDrivenSelfAttention
    learned_delays: nn.Parameter

    def __init__(self, d_model: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = MultiLevelSpikeDrivenSelfAttention(d_model, n_head, neuron_class, neuron_params)
        self.learned_delays = nn.Parameter(torch.zeros(d_model))

        filtered_params = self._filter_neuron_params(neuron_class, neuron_params)
        
        self.lif1 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **filtered_params))
        self.norm2 = SNNLayerNorm(d_model)
        
        # FFN部分もBitSpikeLinearに変更して完全BitNet化
        self.fc1 = BitSpikeLinear(d_model, d_model * 4)
        self.lif2 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model * 4, **filtered_params))
        self.fc2 = BitSpikeLinear(d_model * 4, d_model)
        self.lif3 = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF], neuron_class(features=d_model, **filtered_params))
        self.mem_history = []

    def _filter_neuron_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        else:
             # Default fallback
             valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']
        return {k: v for k, v in neuron_params.items() if k in valid_params}

    def _hook_mem(self, module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.mem_history.append(output[1])

    def register_mem_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        hooks.extend(self.attn.register_mem_hooks())
        hooks.append(self.lif1.register_forward_hook(self._hook_mem))
        hooks.append(self.lif2.register_forward_hook(self._hook_mem))
        hooks.append(self.lif3.register_forward_hook(self._hook_mem))
        return hooks
    
    def clear_mem_history(self) -> None:
        self.mem_history = []
        self.attn.clear_mem_history()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        attn_out = self.attn(self.norm1(x))
        x_attn = x + attn_out
        x_flat = x_attn.reshape(B * T, D)
        
        x_delayed = x_flat + self.learned_delays
        spike_flat, _ = self.lif1(x_delayed)
        
        x_res = spike_flat.reshape(B, T, D)
        ffn_in = self.norm2(x_res)
        ffn_flat = ffn_in.reshape(B * T, D)
        ffn_hidden, _ = self.lif2(self.fc1(ffn_flat))
        ffn_out_flat = self.fc2(ffn_hidden)
        ffn_out = ffn_out_flat.reshape(B, T, D)
        x_ffn = x_res + ffn_out
        x_ffn_flat = x_ffn.reshape(B * T, D)
        out_flat, _ = self.lif3(x_ffn_flat)
        out = out_flat.reshape(B, T, D)
        return out