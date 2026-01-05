# ファイルパス: snn_research/models/temporal_snn.py
# (修正: スパイク集計順序の修正)
# Title: 時系列データ特化 SNN モデル (RSNN / GatedSNN)
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast
import logging

# 既存のニューロンクラスをインポート
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from snn_research.core.base import BaseModel

# spikingjellyのfunctionalをリセットに利用
# type: ignore[import-untyped]
from spikingjelly.activation_based import functional

logger = logging.getLogger(__name__)


class SimpleRSNN(BaseModel):
    """
    時系列データ処理に特化したシンプルな再帰型SNN (RSNN) モデル。
    """
    hidden_neuron: nn.Module
    output_neuron: Optional[nn.Module]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        output_spikes: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_steps = time_steps
        self.output_spikes = output_spikes

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)

        neuron_class: Type[nn.Module]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_neuron = neuron_class(features=hidden_dim, **neuron_params)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        if self.output_spikes:
            self.output_neuron = neuron_class(
                features=output_dim, **neuron_params)
        else:
            self.output_neuron = None

        self._init_weights()
        logger.info(
            f"✅ SimpleRSNN initialized (In: {input_dim}, Hidden: {hidden_dim}, Out: {output_dim})")

    def forward(
        self,
        input_sequence: torch.Tensor,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T_input, _ = input_sequence.shape
        device = input_sequence.device

        functional.reset_net(self)

        hidden_spikes = torch.zeros(B, self.hidden_dim, device=device)
        output_history: List[torch.Tensor] = []

        for t in range(T_input):
            input_t = input_sequence[:, t, :]
            recurrent_input = self.recurrent(hidden_spikes)
            hidden_input_current = self.input_to_hidden(
                input_t) + recurrent_input

            hidden_spikes, _ = self.hidden_neuron(hidden_input_current)

            output_input_current = self.hidden_to_output(hidden_spikes)

            output_t: torch.Tensor
            if self.output_neuron:
                output_t, _ = self.output_neuron(output_input_current)
            else:
                output_t = output_input_current

            output_history.append(output_t)

        final_output_sequence = torch.stack(output_history, dim=1)
        final_logits = final_output_sequence[:, -1, :]

        # 統計情報
        avg_spikes_val = self.get_total_spikes() / (B * T_input) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return final_logits, avg_spikes, mem


class GatedSNN(BaseModel):
    """
    Gated Spiking Recurrent Neural Network (GSRNN)。
    """
    lif_z: nn.Module
    lif_r: nn.Module
    lif_h: nn.Module

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        **kwargs: Any
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_steps = time_steps

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)

        neuron_class: Type[nn.Module]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=True)

        gate_params = neuron_params.copy()
        if 'base_threshold' in gate_params:
            gate_params['base_threshold'] = 0.5

        self.lif_z = neuron_class(features=hidden_dim, **gate_params)
        self.lif_r = neuron_class(features=hidden_dim, **gate_params)
        self.lif_h = neuron_class(features=hidden_dim, **neuron_params)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()
        logger.info("✅ GatedSNN (Spiking GRU) initialized.")

    def forward(
        self,
        input_sequence: torch.Tensor,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T_input, _ = input_sequence.shape
        device = input_sequence.device

        functional.reset_net(self)

        if hasattr(self.lif_z, 'set_stateful'):
            cast(Any, self.lif_z).set_stateful(True)
        if hasattr(self.lif_r, 'set_stateful'):
            cast(Any, self.lif_r).set_stateful(True)
        if hasattr(self.lif_h, 'set_stateful'):
            cast(Any, self.lif_h).set_stateful(True)

        h_prev = torch.zeros(B, self.hidden_dim, device=device)
        output_history: List[torch.Tensor] = []

        for t in range(T_input):
            input_t = input_sequence[:, t, :]
            x_t = self.input_proj(input_t)

            z_t_input = self.W_z(x_t) + self.U_z(h_prev)
            z_t_spike, _ = self.lif_z(z_t_input)

            r_t_input = self.W_r(x_t) + self.U_r(h_prev)
            r_t_spike, _ = self.lif_r(r_t_input)

            h_hat_input = self.W_h(x_t) + self.U_h(r_t_spike * h_prev)
            h_hat_spike, _ = self.lif_h(h_hat_input)

            h_t = h_prev * (1.0 - z_t_spike) + h_hat_spike * z_t_spike
            h_prev = h_t
            output_history.append(h_t)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = self.get_total_spikes(
        ) / (B * T_input) if return_spikes and T_input > 0 else 0.0
        # ------------------------------------

        if hasattr(self.lif_z, 'set_stateful'):
            cast(Any, self.lif_z).set_stateful(False)
        if hasattr(self.lif_r, 'set_stateful'):
            cast(Any, self.lif_r).set_stateful(False)
        if hasattr(self.lif_h, 'set_stateful'):
            cast(Any, self.lif_h).set_stateful(False)

        final_hidden_state = output_history[-1]
        final_logits = self.output_proj(final_hidden_state)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return final_logits, avg_spikes, mem
