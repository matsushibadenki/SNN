# ファイルパス: snn_research/models/transformer/spiking_rwkv.py
# (修正: ニューロンパラメータのフィルタリング追加 & v_reset対応 & スパイク集計順序修正)
# Title: Spiking RWKV (Standard & 1.58bit BitNet)
# Description:
# - 修正: ニューロンクラスの初期化時にパラメータフィルタリングを適用し、堅牢性を向上。
# - 修正: forwardメソッド内で、総ニューロン数を考慮した正確な平均発火率を返すように変更。
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動 (重要)。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, Optional, List, cast, Union

import logging

from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
# type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F
# type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base
from snn_research.training.quantization import BitLinear

logger = logging.getLogger(__name__)


def _filter_neuron_params(neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする。
    """
    valid_params: List[str] = []

    if neuron_class == AdaptiveLIFNeuron:
        valid_params = [
            'features', 'tau_mem', 'base_threshold', 'adaptation_strength',
            'target_spike_rate', 'noise_intensity', 'threshold_decay',
            'threshold_step', 'v_reset'
        ]
    elif neuron_class == IzhikevichNeuron:
        valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
    else:
        # デフォルト (LIF互換と仮定)
        valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']

    return {k: v for k, v in neuron_params.items() if k in valid_params}


class SpikingRWKVBlock(sj_base.MemoryModule):
    """
    標準的な Spiking RWKV ブロック。
    """

    def __init__(
        self,
        d_model: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # --- Time-mixing ---
        self.ln_time = SNNLayerNorm(d_model)
        self.time_mix_k = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_v = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_r = nn.Linear(d_model, d_model, bias=False)

        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model) * 0.1)

        self.time_key_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))
        self.time_value_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))
        self.time_receptance_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))

        # --- Channel-mixing ---
        self.ln_channel = SNNLayerNorm(d_model)
        d_ffn: int = int(d_model * 3.5)
        self.channel_mix_k = nn.Linear(d_model, d_ffn, bias=False)
        self.channel_mix_r = nn.Linear(d_model, d_model, bias=False)
        self.channel_mix_v = nn.Linear(d_ffn, d_model, bias=False)

        self.channel_key_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_ffn, **neuron_params))
        self.channel_receptance_lif = cast(
            Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **neuron_params))

        # 総ニューロン数を計算 (3*d_model + d_ffn + d_model)
        self.total_neurons_in_block = 3 * d_model + d_ffn + d_model

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        for module in [self.time_key_lif, self.time_value_lif, self.time_receptance_lif,
                       self.channel_key_lif, self.channel_receptance_lif]:
            if hasattr(module, 'set_stateful'):
                cast(Any, module).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        for module in [self.time_key_lif, self.time_value_lif, self.time_receptance_lif,
                       self.channel_key_lif, self.channel_receptance_lif]:
            if hasattr(module, 'reset'):
                cast(Any, module).reset()

    def forward(
        self,
        x: torch.Tensor,
        time_mixing_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Time-mixing
        x_norm_time = self.ln_time(x)
        k_current = self.time_mix_k(x_norm_time)
        v_current = self.time_mix_v(x_norm_time)
        r_current = self.time_mix_r(x_norm_time)

        k, _ = self.time_key_lif(k_current)
        v, _ = self.time_value_lif(v_current)
        r_spike, _ = self.time_receptance_lif(r_current)
        r = torch.sigmoid(r_spike)

        w = self.time_decay.sigmoid()
        new_time_mixing_state = (time_mixing_state * w) + \
            (k * (1 - w)) * self.time_first

        rwkv_out = r * new_time_mixing_state
        x = x + rwkv_out

        # 2. Channel-mixing
        x_norm_channel = self.ln_channel(x)
        k_current_ch = self.channel_mix_k(x_norm_channel)
        r_current_ch = self.channel_mix_r(x_norm_channel)

        k_spike, _ = self.channel_key_lif(k_current_ch)
        k_spike_activated = F.relu(k_spike)

        r_spike, _ = self.channel_receptance_lif(r_current_ch)
        r_gate = torch.sigmoid(r_spike)

        v_out = self.channel_mix_v(k_spike_activated)

        ffn_out = r_gate * v_out
        x = x + ffn_out

        return x, new_time_mixing_state


class SpikingRWKV(BaseModel):
    """
    標準的な Spiking RWKV モデル。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.time_steps = time_steps

        if neuron_config is None:
            neuron_config = {'type': 'lif',
                             'tau_mem': 10.0, 'base_threshold': 1.0}

        neuron_type_str = neuron_config.get("type", "lif")
        neuron_params_raw = neuron_config.copy()
        neuron_params_raw.pop('type', None)
        neuron_class: Type[nn.Module] = AdaptiveLIFNeuron if neuron_type_str == 'lif' else IzhikevichNeuron

        # パラメータフィルタリング適用
        filtered_params = _filter_neuron_params(
            neuron_class, neuron_params_raw)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model))

        self.layers = nn.ModuleList([
            SpikingRWKVBlock(d_model, neuron_class, filtered_params)
            for _ in range(num_layers)
        ])

        self.time_mixing_neurons = nn.ModuleList([
            cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                 neuron_class(features=d_model, **filtered_params))
            for _ in range(num_layers)
        ])

        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # 全ニューロン数計算
        neurons_per_block = cast(
            SpikingRWKVBlock, self.layers[0]).total_neurons_in_block
        self.total_model_neurons = neurons_per_block * num_layers + \
            d_model * num_layers  # blocks + extra mixing

        self._init_weights()
        logger.info(
            f"✅ SpikingRWKV initialized (Layers: {num_layers}, D: {d_model}).")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, context_embeds: Optional[torch.Tensor] = None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_seq = input_ids.shape
        device = input_ids.device

        SJ_F.reset_net(self)

        x = self.embedding(input_ids)
        if T_seq <= self.pos_encoder.shape[1]:
            x = x + self.pos_encoder[:, :T_seq, :]
        else:
            x = x + self.pos_encoder[:, :self.pos_encoder.shape[1], :]

        outputs = []

        layer_states = [torch.zeros(B, self.d_model, device=device)
                        for _ in range(self.num_layers)]

        for layer in self.layers:
            cast(Any, layer).set_stateful(True)
        for neuron in self.time_mixing_neurons:
            cast(Any, neuron).set_stateful(True)

        for t_idx in range(T_seq):
            x_t = x[:, t_idx, :]

            for i in range(self.num_layers):
                layer_block = cast(SpikingRWKVBlock, self.layers[i])
                x_t, new_state = layer_block(x_t, layer_states[i])
                layer_states[i] = new_state

            outputs.append(x_t)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            total_spikes = self.get_total_spikes()
            # self.total_model_neurons は1ステップあたりのニューロン数
            # T_seq だけ時間があるので、分母は B * T_seq * TotalNeurons
            avg_spikes_val = total_spikes / \
                (B * T_seq * self.total_model_neurons)
        # ------------------------------------

        for layer in self.layers:
            cast(Any, layer).set_stateful(False)
        for neuron in self.time_mixing_neurons:
            cast(Any, neuron).set_stateful(False)

        x_final_seq = torch.stack(outputs, dim=1)
        x_norm_final = self.final_norm(x_final_seq)
        logits = self.output_projection(x_norm_final)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem


class BitSpikingRWKVBlock(sj_base.MemoryModule):
    """
    1.58bit (BitNet) 化された Spiking RWKV ブロック。
    """

    def __init__(
        self,
        d_model: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        weight_bits: float = 1.58
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # --- Time-mixing (BitLinear) ---
        self.ln_time = SNNLayerNorm(d_model)
        self.time_mix_k = BitLinear(
            d_model, d_model, bias=False, weight_bits=weight_bits)
        self.time_mix_v = BitLinear(
            d_model, d_model, bias=False, weight_bits=weight_bits)
        self.time_mix_r = BitLinear(
            d_model, d_model, bias=False, weight_bits=weight_bits)

        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model) * 0.1)

        self.time_key_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))
        self.time_value_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))
        self.time_receptance_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_model, **neuron_params))

        # --- Channel-mixing (BitLinear) ---
        self.ln_channel = SNNLayerNorm(d_model)
        d_ffn: int = int(d_model * 3.5)
        self.channel_mix_k = BitLinear(
            d_model, d_ffn, bias=False, weight_bits=weight_bits)
        self.channel_mix_r = BitLinear(
            d_model, d_model, bias=False, weight_bits=weight_bits)
        self.channel_mix_v = BitLinear(
            d_ffn, d_model, bias=False, weight_bits=weight_bits)

        self.channel_key_lif = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(
            features=d_ffn, **neuron_params))
        self.channel_receptance_lif = cast(
            Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **neuron_params))

        # 総ニューロン数
        self.total_neurons_in_block = 3 * d_model + d_ffn + d_model

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        for module in [self.time_key_lif, self.time_value_lif, self.time_receptance_lif,
                       self.channel_key_lif, self.channel_receptance_lif]:
            if hasattr(module, 'set_stateful'):
                cast(Any, module).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        for module in [self.time_key_lif, self.time_value_lif, self.time_receptance_lif,
                       self.channel_key_lif, self.channel_receptance_lif]:
            if hasattr(module, 'reset'):
                cast(Any, module).reset()

    def forward(
        self,
        x: torch.Tensor,
        time_mixing_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Time-mixing
        x_norm_time = self.ln_time(x)
        k_current = self.time_mix_k(x_norm_time)
        v_current = self.time_mix_v(x_norm_time)
        r_current = self.time_mix_r(x_norm_time)

        k, _ = self.time_key_lif(k_current)
        v, _ = self.time_value_lif(v_current)
        r_spike, _ = self.time_receptance_lif(r_current)
        r = torch.sigmoid(r_spike)

        w = self.time_decay.sigmoid()
        new_time_mixing_state = (time_mixing_state * w) + \
            (k * (1 - w)) * self.time_first

        rwkv_out = r * new_time_mixing_state
        x = x + rwkv_out

        # 2. Channel-mixing
        x_norm_channel = self.ln_channel(x)
        k_current_ch = self.channel_mix_k(x_norm_channel)
        r_current_ch = self.channel_mix_r(x_norm_channel)

        k_spike, _ = self.channel_key_lif(k_current_ch)
        k_spike_activated = F.relu(k_spike)

        r_spike, _ = self.channel_receptance_lif(r_current_ch)
        r_gate = torch.sigmoid(r_spike)

        v_out = self.channel_mix_v(k_spike_activated)

        ffn_out = r_gate * v_out
        x = x + ffn_out

        return x, new_time_mixing_state


class BitSpikingRWKV(BaseModel):
    """
    1.58bit (BitNet) Spiking RWKV モデル。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.time_steps = time_steps

        model_config = kwargs.get('config', {})
        quant_config = model_config.get(
            'quantization', {}) if model_config else {}
        weight_bits = quant_config.get('weight_bits', 1.58)

        if neuron_config is None:
            neuron_config = {'type': 'lif',
                             'tau_mem': 10.0, 'base_threshold': 1.0}

        neuron_type_str = neuron_config.get("type", "lif")
        neuron_params_raw = neuron_config.copy()
        neuron_params_raw.pop('type', None)
        neuron_class: Type[nn.Module] = AdaptiveLIFNeuron if neuron_type_str == 'lif' else IzhikevichNeuron

        # パラメータフィルタリング適用
        filtered_params = _filter_neuron_params(
            neuron_class, neuron_params_raw)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model))

        self.layers = nn.ModuleList([
            BitSpikingRWKVBlock(d_model, neuron_class,
                                filtered_params, weight_bits=weight_bits)
            for _ in range(num_layers)
        ])

        self.time_mixing_neurons = nn.ModuleList([
            cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                 neuron_class(features=d_model, **filtered_params))
            for _ in range(num_layers)
        ])

        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = BitLinear(
            d_model, vocab_size, weight_bits=weight_bits)

        # 全ニューロン数計算
        neurons_per_block = cast(
            BitSpikingRWKVBlock, self.layers[0]).total_neurons_in_block
        self.total_model_neurons = neurons_per_block * num_layers + d_model * num_layers

        self._init_weights()
        logger.info(
            f"✅ BitSpikingRWKV initialized (weight_bits={weight_bits}).")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, context_embeds: Optional[torch.Tensor] = None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_seq = input_ids.shape
        device = input_ids.device

        SJ_F.reset_net(self)

        x = self.embedding(input_ids)
        if T_seq <= self.pos_encoder.shape[1]:
            x = x + self.pos_encoder[:, :T_seq, :]
        else:
            x = x + self.pos_encoder[:, :self.pos_encoder.shape[1], :]

        outputs = []
        layer_states = [torch.zeros(B, self.d_model, device=device)
                        for _ in range(self.num_layers)]

        for layer in self.layers:
            cast(Any, layer).set_stateful(True)
        for neuron in self.time_mixing_neurons:
            cast(Any, neuron).set_stateful(True)

        for t_idx in range(T_seq):
            x_t = x[:, t_idx, :]
            for i in range(self.num_layers):
                layer = cast(BitSpikingRWKVBlock, self.layers[i])
                x_t, new_state = layer(x_t, layer_states[i])
                layer_states[i] = new_state

            outputs.append(x_t)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            total_spikes = self.get_total_spikes()
            avg_spikes_val = total_spikes / \
                (B * T_seq * self.total_model_neurons)
        # ------------------------------------

        for layer in self.layers:
            cast(Any, layer).set_stateful(False)
        for neuron in self.time_mixing_neurons:
            cast(Any, neuron).set_stateful(False)

        x_final_seq = torch.stack(outputs, dim=1)
        x_norm_final = self.final_norm(x_final_seq)
        logits = self.output_projection(x_norm_final)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem
