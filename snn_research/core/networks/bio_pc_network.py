# ファイルパス: snn_research/core/networks/bio_pc_network.py
# Title: Bio-PCNet (ニューロン選択対応・修正版)
# Description:
#   生物学的妥当性を重視した予測符号化ネットワーク。
#   修正点:
#   - ニューロンクラスが AdaptiveLIFNeuron に固定されていた問題を修正。
#     config の 'type' に基づいて Izhikevich, GLIF, TC_LIF などを選択可能にしました。
#   - PredictiveCodingLayer の初期化時に、選択された neuron_class を渡すように変更。

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, cast, Type, Union

from snn_research.core.networks.abstract_snn_network import AbstractSNNNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule

# 必要なニューロンクラスをすべてインポート
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)

class BioPCNetwork(AbstractSNNNetwork):
    def __init__(
        self, 
        layer_dims: List[int], 
        time_steps: int, 
        neuron_config: Dict[str, Any],
        learning_rate: float = 0.001,
        sparsity: float = 0.1
    ):
        super().__init__()
        self.layer_dims = layer_dims
        self.time_steps = time_steps
        self.layers = nn.ModuleList()
        
        # --- ニューロンクラスの解決ロジックを追加 ---
        neuron_type = neuron_config.get("type", "lif")
        neuron_class: Type[nn.Module]
        
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron
        elif neuron_type == 'scale_and_fire':
            neuron_class = ScaleAndFireNeuron
        else:
            # デフォルト
            neuron_class = AdaptiveLIFNeuron
            
        # パラメータのコピー (PredictiveCodingLayer内でフィルタリングされるためそのまま渡す)
        neuron_params = neuron_config.copy()
        if 'type' in neuron_params:
            del neuron_params['type']

        for i in range(len(layer_dims) - 1):
            layer = PredictiveCodingLayer(
                d_model=layer_dims[i], 
                d_state=layer_dims[i+1],
                neuron_class=neuron_class, # 選択されたクラスを渡す
                neuron_params=neuron_params, 
                weight_tying=True,
                sparsity=sparsity
            )
            self.layers.append(layer)
            
            # パラメータリストの作成 (WeightとBias)
            params = [cast(nn.Parameter, layer.generative_fc.weight)]
            if layer.generative_fc.bias is not None: 
                params.append(cast(nn.Parameter, layer.generative_fc.bias))
                
            rule = PredictiveCodingRule(params=params, learning_rate=learning_rate, layer_name=f"layer_{i}")
            self.add_learning_rule(rule)

        self.layer_states: List[torch.Tensor] = []
        self.layer_errors: List[Optional[torch.Tensor]] = []

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, device = x.shape[0], x.device
        if not self.layer_states or self.layer_states[0].shape[0] != B:
            self.layer_states = [torch.zeros(B, self.layer_dims[i+1], device=device) for i in range(len(self.layer_dims)-1)]
            self.layer_errors = [None] * len(self.layers)

        if targets is not None:
            self.layer_states[-1] = targets.clone()

        final_output = torch.zeros(B, self.layer_dims[-1], device=device)

        for t in range(self.time_steps):
            curr_in = x
            for i, layer in enumerate(self.layers):
                td_state = self.layer_states[i]
                fb_err = self.layer_errors[i+1] if i + 1 < len(self.layers) else None
                
                # 学習則のために活動を記録
                self.model_state[f"pre_activity_layer_{i}"] = td_state.detach()
                
                upd_state, pred_err, _ = layer(curr_in, td_state, fb_err)
                
                # 学習則のために誤差を記録
                self.model_state[f"prediction_error_layer_{i}"] = pred_err.detach()
                
                self.layer_errors[i] = pred_err
                
                if targets is not None and i == len(self.layers) - 1:
                    self.layer_states[i] = targets
                else:
                    self.layer_states[i] = upd_state
                curr_in = self.layer_states[i]
            final_output = curr_in
        return final_output

    def reset_state(self) -> None:
        super().reset_state()
        self.layer_states = []
        self.layer_errors = []
        for l in self.layers:
             if hasattr(l, 'generative_neuron'): l.generative_neuron.reset() # type: ignore
             if hasattr(l, 'inference_neuron'): l.inference_neuron.reset() # type: ignore