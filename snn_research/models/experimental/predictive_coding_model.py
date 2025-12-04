# ファイルパス: snn_research/models/experimental/predictive_coding_model.py
# Title: Predictive Coding SNN (BreakthroughSNN) - Stateful修正版
# Description:
# - 予測符号化を行うSNNモデル。
# - 修正: forwardメソッド内で、時間ループの実行中にニューロンの状態（膜電位）を保持するように
#   set_stateful(True) を呼び出すロジックを追加。これにより時間的統合が正しく機能する。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class BreakthroughSNN(BaseModel):
    """
    PredictiveCodingLayerを使用したSNNモデル。
    """
    token_embedding: nn.Embedding
    input_encoder: nn.Linear
    pc_layers: nn.ModuleList
    output_projection: nn.Linear

    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        d_state: int, 
        num_layers: int, 
        time_steps: int, 
        n_head: int, 
        neuron_config: Optional[Dict[str, Any]] = None, 
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)

        neuron_params: Dict[str, Any] = neuron_config.copy() if neuron_config is not None else {}
        neuron_type_str: str = neuron_params.pop('type', 'lif')
        
        neuron_params.pop('num_branches', None)
        neuron_params.pop('branch_features', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]]
        
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = d_model 
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF 
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron
            neuron_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
            raise ValueError(f"Unknown neuron type for BreakthroughSNN: {neuron_type_str}")

        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, cast(Type[nn.Module], neuron_class), neuron_params) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        
        self._init_weights()

    def _set_stateful(self, stateful: bool):
        """モデル内の全ニューロンのstatefulモードを切り替える"""
        for layer in self.pc_layers:
            if hasattr(layer, 'generative_neuron') and hasattr(layer.generative_neuron, 'set_stateful'):
                layer.generative_neuron.set_stateful(stateful)
            if hasattr(layer, 'inference_neuron') and hasattr(layer.inference_neuron, 'set_stateful'):
                layer.inference_neuron.set_stateful(stateful)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        output_hidden_states: bool = False, 
        return_full_hiddens: bool = False, 
        return_full_mems: bool = False, 
        context_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device: torch.device = input_ids.device
        
        # ネットワークのリセット
        SJ_F.reset_net(self)
        
        token_emb: torch.Tensor = self.token_embedding(input_ids)
        embedded_sequence: torch.Tensor = self.input_encoder(token_emb)
        
        inference_neuron = self.pc_layers[0].inference_neuron
        inference_neuron_features: int
        if hasattr(inference_neuron, 'features'):
             inference_neuron_features = cast(int, getattr(inference_neuron, 'features'))
        else:
             inference_neuron_features = self.d_state
        
        # 内部状態の初期化
        states: List[torch.Tensor] = [torch.zeros(batch_size, inference_neuron_features, device=device) for _ in range(self.num_layers)]
        
        all_timestep_outputs: List[torch.Tensor] = []
        all_timestep_mems: List[torch.Tensor] = []

        # --- 時間ループ開始前に Stateful モードを有効化 ---
        self._set_stateful(True)

        for _ in range(self.time_steps):
            sequence_outputs: List[torch.Tensor] = []
            sequence_mems: List[torch.Tensor] = []
            
            for i in range(seq_len):
                bottom_up_input: torch.Tensor = embedded_sequence[:, i, :]
                layer_mems: List[torch.Tensor] = []
                
                for j in range(self.num_layers):
                    layer = cast(PredictiveCodingLayer, self.pc_layers[j])
                    
                    new_state, error, combined_mem = layer(bottom_up_input, states[j])
                    
                    states[j] = new_state
                    bottom_up_input = error 
                    layer_mems.append(combined_mem)
                
                sequence_outputs.append(torch.cat(states, dim=1))
                sequence_mems.append(torch.cat(layer_mems, dim=1)) 

            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
            all_timestep_mems.append(torch.stack(sequence_mems, dim=1))
        
        full_hiddens: torch.Tensor = torch.stack(all_timestep_outputs, dim=2) 
        full_mems: torch.Tensor = torch.stack(all_timestep_mems, dim=2) 
        
        final_hidden_states: torch.Tensor = all_timestep_outputs[-1] 

        output: torch.Tensor
        mem_to_return: torch.Tensor
        
        if output_hidden_states:
             output = final_hidden_states
        elif return_full_hiddens:
             mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
             # ステートフル解除の前に返すわけにはいかないので、ここでは値を保持して後で返す
        else:
             output = self.output_projection(final_hidden_states)
        
        # スパイク集計 (リセット前に行う)
        avg_spikes_val = 0.0
        if return_spikes:
            total_spikes = self.get_total_spikes()
            total_neurons = self.num_layers * (self.d_model + self.d_state)
            avg_spikes_val = total_spikes / (batch_size * seq_len * self.time_steps * total_neurons)
        
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
        
        # --- 時間ループ終了後に Stateful モードを解除 ---
        self._set_stateful(False)
        
        # return_full_hiddens の場合の分岐処理
        if return_full_hiddens and not output_hidden_states:
             return full_hiddens, avg_spikes, mem_to_return

        return output, avg_spikes, mem_to_return
