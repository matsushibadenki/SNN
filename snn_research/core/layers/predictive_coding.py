# ファイルパス: snn_research/core/layers/predictive_coding.py
# 日本語タイトル: 予測符号化レイヤー (Predictive Coding Layer - Relaxed Inference)
# 機能説明: 
#   ドキュメントに基づき、推論プロセスを「反復的な緩和過程」として実装。
#   Generative PathとInference Pathを反復させることで、エネルギー最小状態を探索する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import logging

try:
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
        TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron,
        BistableIFNeuron, EvolutionaryLeakLIF
    )
except ImportError:
    AdaptiveLIFNeuron = Any 
    IzhikevichNeuron = Any 
    GLIFNeuron = Any 
    TC_LIF = Any 
    DualThresholdNeuron = Any 
    ScaleAndFireNeuron = Any 
    BistableIFNeuron = Any 
    EvolutionaryLeakLIF = Any 

logger = logging.getLogger(__name__)

class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding (PC) を実行するSNNレイヤー。
    
    Biomimetic Enhancement:
    - Iterative Inference: 1ステップではなく、複数ステップの緩和(Relaxation)を行う。
    - Energy Minimization: 予測誤差(自由エネルギー)を最小化するように状態を更新。
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        neuron_class: Type[nn.Module], 
        neuron_params: Dict[str, Any],
        weight_tying: bool = True,
        sparsity: float = 0.05,
        inference_steps: int = 5, # Default to 5 steps of relaxation
        inference_lr: float = 0.1
    ):
        super().__init__()
        self.weight_tying = weight_tying
        self.sparsity = sparsity
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        
        filtered_params = self._filter_params(neuron_class, neuron_params)

        # 1. Generative Path (Top-Down: State -> Prediction)
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], 
                                      neuron_class(features=d_model, **filtered_params))
        
        # 2. Inference Path (Bottom-Up: Error -> State Update)
        if self.weight_tying:
            self.inference_fc = None 
        else:
            self.inference_fc = nn.Linear(d_model, d_state)
            
        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron], 
                                     neuron_class(features=d_state, **filtered_params))
        
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_error = nn.LayerNorm(d_model)
        
        self.error_scale = nn.Parameter(torch.tensor(1.0))
        self.feedback_strength = nn.Parameter(torch.tensor(1.0))

    def _filter_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        # (Parameter filtering logic remains same)
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        else:
             valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']
        return {k: v for k, v in neuron_params.items() if k in valid_params}

    def _apply_lateral_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """Hard k-WTA"""
        if self.sparsity >= 1.0 or self.sparsity <= 0.0:
            return x
        x_abs = x.abs()
        B, N = x.shape
        k = int(N * self.sparsity)
        if k == 0: k = 1
        topk_values, _ = torch.topk(x_abs, k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        threshold = torch.max(threshold, torch.tensor(1e-6, device=x.device))
        mask = (x_abs >= threshold).float()
        return x * mask

    def forward(
        self, 
        bottom_up_input: torch.Tensor, 
        top_down_state: torch.Tensor,
        top_down_error: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference Phase as Relaxation:
        内部状態を反復的に更新し、予測誤差を最小化する平衡状態(Equilibrium)を見つける。
        """
        
        # 初期状態
        current_state = top_down_state.clone()
        final_error = torch.zeros_like(bottom_up_input)
        combined_mem_list = []

        # --- Relaxation Loop (Fast Inference) ---
        # ドキュメント: "Inference... is a fast relaxation process"
        for step in range(self.inference_steps):
            # 1. Generative Pass
            pred_input = self.generative_fc(self.norm_state(current_state))
            pred, gen_mem = self.generative_neuron(pred_input)
            
            # 2. Error Calculation
            raw_error = bottom_up_input - pred
            error = raw_error * self.error_scale
            
            # 最終ステップの誤差を保存
            if step == self.inference_steps - 1:
                final_error = error

            # 3. Inference Pass (State Update)
            norm_error = self.norm_error(error)
            
            if self.weight_tying:
                bu_input = F.linear(norm_error, self.generative_fc.weight.t())
            else:
                if self.inference_fc is None: raise RuntimeError("inference_fc is None")
                bu_input = self.inference_fc(norm_error)

            total_input = bu_input
            if top_down_error is not None:
                total_input = total_input - (top_down_error * self.feedback_strength)
            
            # 状態更新の計算
            state_update, inf_mem = self.inference_neuron(total_input)
            
            # スパース性制約
            state_update = self._apply_lateral_inhibition(state_update)
            
            # 状態の更新 (Relaxation)
            # current_state += lr * update (Gradient Ascent on Free Energy approx)
            current_state = current_state * (1.0 - self.inference_lr) + state_update * self.inference_lr
            
            if step == self.inference_steps - 1:
                combined_mem_list.append(torch.cat((gen_mem, inf_mem), dim=1))

        # 最終的な膜電位 (可視化用)
        combined_mem = combined_mem_list[-1] if combined_mem_list else torch.tensor(0.0)
        
        return current_state, final_error, combined_mem
