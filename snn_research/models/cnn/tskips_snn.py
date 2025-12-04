# ファイルパス: snn_research/architectures/tskips_snn.py
# (修正: スパイク集計順序の修正)
#
# Title: TSkipsSNN (Temporal Skips SNN)
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Type, Optional, cast

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]

import logging
logger = logging.getLogger(__name__)

class TSkipsBlock(sj_base.MemoryModule):
    """
    時間的遅延接続 (TSkips) を持つSNNブロック。
    """
    lif1: nn.Module
    fc1: nn.Linear
    
    forward_skip_buffer: List[torch.Tensor]
    backward_skip_buffer: List[torch.Tensor]

    def __init__(
        self,
        features: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        forward_delays: Optional[List[int]] = None,
        backward_delays: Optional[List[int]] = None
    ):
        super().__init__()
        self.features = features
        self.fc1 = nn.Linear(features, features)
        self.lif1 = neuron_class(features=features, **neuron_params)
        
        self.forward_delays = sorted(forward_delays) if forward_delays else []
        self.backward_delays = sorted(backward_delays) if backward_delays else []
        
        self.max_f_delay = max(self.forward_delays) if self.forward_delays else 0
        self.max_b_delay = max(self.backward_delays) if self.backward_delays else 0
        
        if self.forward_delays:
            self.f_skip_weights = nn.Parameter(torch.randn(len(self.forward_delays), features) * 0.1)
        if self.backward_delays:
            self.b_skip_weights = nn.Parameter(torch.randn(len(self.backward_delays), features) * 0.1)

        self.reset_buffers()
        
    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        if hasattr(self.lif1, 'set_stateful'):
            cast(Any, self.lif1).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.lif1, 'reset'):
            cast(Any, self.lif1).reset()
        self.reset_buffers()
        
    def reset_buffers(self) -> None:
        self.forward_skip_buffer = []
        self.backward_skip_buffer = []

    def forward(
        self, 
        x_t: torch.Tensor, 
        backward_inputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        1タイムステップ分の処理。
        """
        B = x_t.shape[0]
        
        if not self.forward_skip_buffer or self.forward_skip_buffer[0].shape[0] != B or self.forward_skip_buffer[0].device != x_t.device:
            self.forward_skip_buffer = [torch.zeros(B, self.features, device=x_t.device) for _ in range(self.max_f_delay + 1)]
        if not self.backward_skip_buffer or self.backward_skip_buffer[0].shape[0] != B or self.backward_skip_buffer[0].device != x_t.device:
            self.backward_skip_buffer = [torch.zeros(B, self.features, device=x_t.device) for _ in range(self.max_b_delay + 1)]

        # 1. 順方向 (Forward) スキップ接続
        forward_skip_input = torch.zeros_like(x_t)
        if self.forward_delays:
            for i, delay in enumerate(self.forward_delays):
                forward_skip_input += self.forward_skip_buffer[delay] * self.f_skip_weights[i]

        # 2. 逆方向 (Backward) スキップ接続
        backward_skip_input = torch.zeros_like(x_t)
        if self.backward_delays and backward_inputs:
            for i, delay in enumerate(self.backward_delays):
                if i < len(backward_inputs):
                    backward_skip_input += backward_inputs[i] * self.b_skip_weights[i]

        # 3. メインパス
        current_input = self.fc1(x_t) + forward_skip_input + backward_skip_input
        spikes_t, _ = self.lif1(current_input) 

        # 4. バッファの更新
        self.forward_skip_buffer.insert(0, spikes_t)
        self.forward_skip_buffer.pop()
        
        self.backward_skip_buffer.insert(0, spikes_t)
        self.backward_skip_buffer.pop()

        # 5. 出力
        backward_outputs = []
        if self.backward_delays:
             backward_outputs = [self.backward_skip_buffer[delay] for delay in self.backward_delays]
        
        return spikes_t, backward_outputs

class TSkipsSNN(BaseModel):
    """
    TSkipsBlockを複数層スタックしたSNNモデル。
    """
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_features: int,
        num_layers: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        forward_delays_per_layer: List[Optional[List[int]]],
        backward_delays_per_layer: List[Optional[List[int]]],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['tau_mem', 'base_threshold']}
        else:
            neuron_class = IzhikevichNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['a', 'b', 'c', 'd']}

        self.input_proj = nn.Linear(input_features, hidden_features)
        
        self.layers = nn.ModuleList()
        
        f_delays = forward_delays_per_layer if forward_delays_per_layer else [None] * num_layers
        b_delays = backward_delays_per_layer if backward_delays_per_layer else [None] * num_layers
        
        for i in range(num_layers):
            self.layers.append(
                TSkipsBlock(
                    features=hidden_features,
                    neuron_class=neuron_class,
                    neuron_params=neuron_params,
                    forward_delays=f_delays[i] if i < len(f_delays) else None,
                    backward_delays=b_delays[i] if i < len(b_delays) else None
                )
            )
            
        self.output_proj = nn.Linear(hidden_features, num_classes)
        self.output_lif = AdaptiveLIFNeuron(features=num_classes, **neuron_params)

        self._init_weights()
        logger.info(f"✅ TSkipsSNN initialized with {num_layers} layers.")

    def forward(
        self, 
        input_sequence: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if input_sequence.dim() == 2: 
             B, F_in = input_sequence.shape
             T_seq = self.time_steps
             input_sequence = input_sequence.unsqueeze(1).repeat(1, T_seq, 1)
        else:
             B, T_seq, F_in = input_sequence.shape
             
        device: torch.device = input_sequence.device
        
        SJ_F.reset_net(self)
        
        b_inputs: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        
        output_voltages: List[torch.Tensor] = []
        
        for layer in self.layers:
            cast(TSkipsBlock, layer).set_stateful(True)
        cast(Any, self.output_lif).set_stateful(True)

        for t in range(T_seq):
            x_t: torch.Tensor = input_sequence[:, t, :] 
            x_t = self.input_proj(x_t) 
            
            for i in range(self.num_layers):
                layer_block: TSkipsBlock = cast(TSkipsBlock, self.layers[i])
                
                current_b_inputs: List[torch.Tensor] = b_inputs[i]
                
                x_t, b_outputs = layer_block(x_t, current_b_inputs)
                
                if i > 0:
                    # clone() を使用して参照渡しを回避
                    b_inputs[i-1] = [t.clone() for t in b_outputs]
            
            final_output_current = self.output_proj(x_t)
            _, v_out_t = self.output_lif(final_output_current)
            output_voltages.append(v_out_t)
            
        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * T_seq)
        # ------------------------------------
        
        for layer in self.layers:
            cast(TSkipsBlock, layer).set_stateful(False)
        cast(Any, self.output_lif).set_stateful(False)

        logits: torch.Tensor = torch.stack(output_voltages, dim=1).mean(dim=1) 
        
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem