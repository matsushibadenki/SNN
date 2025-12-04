# ファイルパス: snn_research/core/layers/predictive_coding.py
# Title: 予測符号化レイヤー (ニューロンパラメータ対応版)
# Description:
#   予測符号化（Predictive Coding）を行うSNNレイヤー。
#   修正点:
#     - _filter_params メソッドを拡張し、Izhikevich, GLIF, TC-LIF などの
#       多様なニューロンモデルのパラメータを正しく通過させるように修正。
#     - これにより、configで指定したニューロンタイプが正しく機能するようになる。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import logging

from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron
)

logger = logging.getLogger(__name__)

class PredictiveCodingLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        neuron_class: Type[nn.Module], 
        neuron_params: Dict[str, Any],
        weight_tying: bool = True,
        sparsity: float = 0.05
    ):
        super().__init__()
        self.weight_tying = weight_tying
        self.sparsity = sparsity
        
        # ニューロンパラメータのフィルタリング
        filtered_params = self._filter_params(neuron_class, neuron_params)

        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_model, **filtered_params))
        
        if self.weight_tying:
            self.inference_fc = None
        else:
            self.inference_fc = nn.Linear(d_model, d_state)
            
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], neuron_class(features=d_state, **filtered_params))
        
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_error = nn.LayerNorm(d_model)
        
        self.error_scale = nn.Parameter(torch.tensor(1.0))
        self.feedback_strength = nn.Parameter(torch.tensor(1.0))

    def _filter_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたニューロンクラスの__init__が受け入れるパラメータのみをフィルタリングする。
        """
        valid_params: List[str] = []
        
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features', 'base_threshold', 'gate_input_features']
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        else:
             # デフォルト (LIF互換)
             valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']
        
        return {k: v for k, v in neuron_params.items() if k in valid_params}

    def _apply_lateral_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard k-WTA: 絶対値の上位k%のみを残す。
        """
        if self.sparsity >= 1.0 or self.sparsity <= 0.0:
            return x
            
        # 活動の大きさ(絶対値)で評価
        x_abs = x.abs()
        
        B, N = x.shape
        k = int(N * self.sparsity)
        if k == 0: k = 1
        
        # 上位k個の値を取得
        topk_values, _ = torch.topk(x_abs, k, dim=1)
        
        # 閾値 (k番目の値)
        threshold = topk_values[:, -1].unsqueeze(1)
        
        # 閾値未満をゼロにする (元の符号は維持)
        # 閾値が0の場合は全て通してしまうので、微小値を加える
        threshold = torch.max(threshold, torch.tensor(1e-6, device=x.device))
        
        mask = (x_abs >= threshold).float()
        
        return x * mask

    def forward(
        self, 
        bottom_up_input: torch.Tensor, 
        top_down_state: torch.Tensor,
        top_down_error: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Generative Pass
        pred_input = self.generative_fc(self.norm_state(top_down_state))
        pred, gen_mem = self.generative_neuron(pred_input)
        
        # Error Calculation
        raw_error = bottom_up_input - pred
        error = raw_error * self.error_scale
        
        # Inference Pass
        norm_error = self.norm_error(error)
        if self.weight_tying:
            bu_input = F.linear(norm_error, self.generative_fc.weight.t())
        else:
            if self.inference_fc is None:
                 raise RuntimeError("inference_fc is None but weight_tying is False")
            bu_input = self.inference_fc(norm_error)

        total_input = bu_input - (top_down_error * self.feedback_strength) if top_down_error is not None else bu_input
        
        # ニューロン更新
        state_update, inf_mem = self.inference_neuron(total_input)
        
        # --- Hard k-WTA 適用 ---
        state_update = self._apply_lateral_inhibition(state_update)
        
        # 状態更新
        updated_state = top_down_state * 0.9 + state_update * 0.1
        
        combined_mem = torch.cat((gen_mem, inf_mem), dim=1) 
        
        return updated_state, error, combined_mem
