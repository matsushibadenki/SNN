# Filepath: snn_research/core/models/visual_cortex.py
# Title: Visual Cortex Model (Causal Perception & FEP-ready)
# Description:
#   予測符号化(Predictive Coding)を用いた視覚野モデル。
#   DVS入力や動画フレームから因果的な内部状態(State)を抽出する。
#   【修正】単なる状態出力だけでなく、FEP（自由エネルギー原理）エージェントが必要とする
#   「予測誤差 (Surprise/Free Energy approximation)」を明示的に計算して返す。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import logging

from snn_research.core.base import BaseModel
from snn_research.core.layers.frequency_encoding import FrequencyEncodingLayer
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.neurons import AdaptiveLIFNeuron

logger = logging.getLogger(__name__)

class VisualCortex(BaseModel):
    """
    動的入力から因果構造を抽出する視覚野モデル。
    Active Inference Agentの「Perception」モジュールとして機能する。
    """
    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        d_model: int = 256,
        d_state: int = 128,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.d_state = d_state
        
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 20.0, 'base_threshold': 1.0}
            
        # 1. Frequency Encoding (FEEL) - ロバストな特徴抽出
        self.feel_layer = FrequencyEncodingLayer(time_steps=time_steps)
        
        self.feat_dim = input_channels * height * width
        self.input_projection = nn.Linear(self.feat_dim, d_model)
        
        # 2. Predictive Coding Layer (PCL) - 因果推論の核
        neuron_params = neuron_config.copy()
        if 'type' in neuron_params: del neuron_params['type']
            
        self.pcl = PredictiveCodingLayer(
            d_model=d_model,
            d_state=d_state,
            neuron_class=AdaptiveLIFNeuron,
            neuron_params=neuron_params
        )
        
        self._init_weights()
        logger.info(f"✅ VisualCortex initialized (FEEL -> PCL). D_Model={d_model}, D_State={d_state}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width) or Video (B, T, C, H, W).

        Returns:
            causal_state (torch.Tensor): 内部因果状態 (B, T, D_State). エージェントのBeliefとして使用。
            prediction_error (torch.Tensor): 予測誤差 (B, T, D_Model). 自由エネルギー(Surprise)の近似。
            reconstruction (torch.Tensor): 予測された入力 (B, T, D_Model).
        """
        B = x.shape[0]
        device = x.device
        
        # 1. Encoding
        if x.ndim == 4: # (B, C, H, W) -> Static -> FEEL -> (B, T, C, H, W)
            encoded_seq = self.feel_layer(x)
        elif x.ndim == 5: # (B, T, C, H, W) -> Video -> Use as is
             encoded_seq = x
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
            
        # Flatten & Project: (B, T, D_Model)
        encoded_seq_flat = encoded_seq.reshape(B, self.time_steps, -1)
        projected_seq = self.input_projection(encoded_seq_flat) 
        
        # 2. Predictive Coding Loop
        causal_states = []
        errors = []
        reconstructions = []
        
        # 初期状態
        current_state = torch.zeros(B, self.d_state, device=device)
        
        for t in range(self.time_steps):
            current_input = projected_seq[:, t, :]
            
            # PCL Forward: Input, TopDownState -> NewState, Error
            # PCL内部で「予測(Prediction)」と「入力」の比較が行われる
            # updated_state: 事後信念 (Posterior)
            # error: 予測誤差 (Surprise)
            updated_state, error, _ = self.pcl(current_input, current_state)
            
            # 予測（Reconstruction）を計算
            # PCLは内部で予測を生成しているが、戻り値に含まれていないため、ここで再現
            # Prediction = GenerativeFC(State)
            # (注: updated_stateを使うかcurrent_stateを使うかはタイミングによる。
            #  PC理論では事前予測(prior)との誤差を取るため、current_stateからの予測が妥当)
            prediction = self.pcl.generative_fc(self.pcl.norm_state(current_state))
            
            causal_states.append(updated_state)
            errors.append(error)
            reconstructions.append(prediction)
            
            current_state = updated_state
            
        # Stack outputs
        causal_states_stack = torch.stack(causal_states, dim=1) # (B, T, D_State)
        errors_stack = torch.stack(errors, dim=1) # (B, T, D_Model)
        reconstructions_stack = torch.stack(reconstructions, dim=1) # (B, T, D_Model)
        
        return causal_states_stack, errors_stack, reconstructions_stack

    def reset_spike_stats(self):
        if hasattr(self.pcl.generative_neuron, 'reset'):
            self.pcl.generative_neuron.reset()
        if hasattr(self.pcl.inference_neuron, 'reset'):
            self.pcl.inference_neuron.reset()