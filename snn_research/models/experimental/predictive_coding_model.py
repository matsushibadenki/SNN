# ファイルパス: snn_research/models/experimental/predictive_coding_model.py
# 日本語タイトル: 予測符号化SNNモデル (BreakthroughSNN)
# 機能説明:
#   PredictiveCodingLayerを多層に積み重ねたSNNモデル。
#   Transformerのようにトークン埋め込みを入力とし、時間ステップごとの再帰処理を行う。
#   SNNCore経由で呼び出し可能。

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
    時系列入力に対して、各層がトップダウン予測とボトムアップ誤差伝播を行う。
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
        
        # 入力埋め込み
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)

        # ニューロン設定
        neuron_params: Dict[str, Any] = neuron_config.copy() if neuron_config is not None else {}
        neuron_type_str: str = neuron_params.pop('type', 'lif')
        # 不要なパラメータを削除
        neuron_params.pop('num_branches', None)
        neuron_params.pop('branch_features', None)
        
        neuron_class: Type[nn.Module] = AdaptiveLIFNeuron # Default
        if neuron_type_str == 'izhikevich': neuron_class = IzhikevichNeuron
        elif neuron_type_str == 'glif': neuron_class = GLIFNeuron
        elif neuron_type_str == 'tc_lif': neuron_class = TC_LIF
        elif neuron_type_str == 'dual_threshold': neuron_class = DualThresholdNeuron

        # PCレイヤーの積層
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, neuron_class, neuron_params) for _ in range(num_layers)]
        )
        
        # 出力層 (全層の状態を結合して分類に使用)
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        
        self._init_weights()

    def _set_stateful(self, stateful: bool):
        """モデル内の全ニューロンのstatefulモードを切り替える"""
        for layer in self.pc_layers:
            if hasattr(layer, 'generative_neuron'):
                cast(Any, layer.generative_neuron).set_stateful(stateful)
            if hasattr(layer, 'inference_neuron'):
                cast(Any, layer.inference_neuron).set_stateful(stateful)

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
        
        SJ_F.reset_net(self)
        
        # 入力エンコーディング
        token_emb: torch.Tensor = self.token_embedding(input_ids)
        embedded_sequence: torch.Tensor = self.input_encoder(token_emb)
        
        # 状態初期化
        states: List[torch.Tensor] = [torch.zeros(batch_size, self.d_state, device=device) for _ in range(self.num_layers)]
        
        all_timestep_outputs: List[torch.Tensor] = []
        all_timestep_mems: List[torch.Tensor] = []

        self._set_stateful(True)

        # 時間ステップループ
        for _ in range(self.time_steps):
            sequence_outputs: List[torch.Tensor] = []
            sequence_mems: List[torch.Tensor] = []
            
            # シーケンスループ
            for i in range(seq_len):
                bottom_up_input: torch.Tensor = embedded_sequence[:, i, :]
                layer_mems: List[torch.Tensor] = []
                
                for j in range(self.num_layers):
                    layer = cast(PredictiveCodingLayer, self.pc_layers[j])
                    # Layer Forward
                    new_state, error, combined_mem = layer(bottom_up_input, states[j])
                    
                    states[j] = new_state
                    bottom_up_input = error # 次の層への入力は「誤差」
                    layer_mems.append(combined_mem)
                
                # 全層の状態を結合して保存
                sequence_outputs.append(torch.cat(states, dim=1))
                sequence_mems.append(torch.cat(layer_mems, dim=1)) 

            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
            all_timestep_mems.append(torch.stack(sequence_mems, dim=1))
        
        # (Batch, Seq, Features)
        full_hiddens: torch.Tensor = torch.stack(all_timestep_outputs, dim=2) 
        full_mems: torch.Tensor = torch.stack(all_timestep_mems, dim=2) 
        
        # 最終ステップの出力を使用
        final_hidden_states: torch.Tensor = all_timestep_outputs[-1] 

        output: torch.Tensor
        if output_hidden_states:
             output = final_hidden_states
        else:
             output = self.output_projection(final_hidden_states)
        
        # スパイク統計
        avg_spikes_val = 0.0
        if return_spikes:
            total_spikes = self.get_total_spikes()
            avg_spikes_val = total_spikes / (batch_size * seq_len * self.time_steps)
        
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem_to_return = full_mems if return_full_mems else torch.tensor(0.0, device=device)
        
        self._set_stateful(False)
        
        if return_full_hiddens and not output_hidden_states:
             return full_hiddens, avg_spikes, mem_to_return

        return output, avg_spikes, mem_to_return
