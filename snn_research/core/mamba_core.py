# ファイルパス: snn_research/core/mamba_core.py
# (修正)
# Title: Spiking-MAMBAモデル コア実装
# Description:
# - 循環インポートエラーを解消するため、BaseModelとSNNLayerNormの
#   インポート元を `snn_core` から新しい `base` モジュールに変更。
# - ニューロンのタイプをハードコーディングせず、コンストラクタで
#   neuron_classを受け取るように修正。
# - 多様なニューロンタイプに対応。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Type, Union, cast
from omegaconf import DictConfig
import math

from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from .base import BaseModel, SNNLayerNorm

class SpikingMambaBlock(nn.Module):
    """
    Spiking-MAMBAの基本ブロック。
    選択的SSMをスパイクベースで実装。
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        neuron_class: Type[nn.Module], 
        neuron_params: Dict[str, Any]
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.lif_conv = neuron_class(features=self.d_inner, **neuron_params)
        
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = SNNLayerNorm(d_model)
        
        self.lif_out = neuron_class(features=d_model, **neuron_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        # Conv1D: (B, D_inner, L)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        
        # LIF Conv
        x_conv_flat = x_conv.reshape(B * L, -1)
        x_conv_spikes, _ = self.lif_conv(x_conv_flat) # type: ignore[operator]
        x_conv_spikes = x_conv_spikes.reshape(B, L, -1)
        
        # SSM Parameters
        x_ssm_params = self.x_proj(x_conv_spikes)
        delta, B_param, C_param = x_ssm_params.split(split_size=[self.d_inner, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        
        # Discrete SSM
        A_bar = torch.exp(A * delta.unsqueeze(-1))
        B_bar = B_param.unsqueeze(-1) * delta.unsqueeze(-1)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_scan = []
        
        # Sequential Scan (Simple recurrence)
        for i in range(L):
            h = A_bar[:, i] * h + B_bar[:, i] * x_conv_spikes[:, i].unsqueeze(-1)
            y = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            y_scan.append(y)
            
        y = torch.stack(y_scan, dim=1) + x_conv_spikes * self.D
        y = y * F.silu(res)
        
        out = self.norm(x + self.out_proj(y))
        
        # Output LIF
        out_spikes, _ = self.lif_out(out.reshape(B * L, -1)) # type: ignore[operator]
        return out_spikes.reshape(B, L, -1)

class SpikingMamba(BaseModel):
    """
    SpikingMambaBlockを複数層重ねた、完全なSpiking-MAMBAモデル。
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        num_layers: int, 
        time_steps: int, 
        neuron_config: Dict[str, Any], 
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = time_steps

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]]

        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_mem', 'base_threshold']}
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'a', 'b', 'c', 'd', 'dt']}
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
            # GLIFはゲート入力次元が必要だが、ここでは簡易的にfeaturesと同じとする
            neuron_params['gate_input_features'] = d_model * expand
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'base_threshold', 'gate_input_features']}
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']}
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron
            filtered_params = {k: v for k, v in neuron_params.items() if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']}
        else:
             raise ValueError(f"Unknown neuron type for SpikingMamba: {neuron_type}")

        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SpikingMambaBlock(
                d_model, d_state, d_conv, expand, 
                cast(Type[nn.Module], neuron_class), 
                filtered_params
            )
            for _ in range(num_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        
        # 時間ステップループ
        # (注: Mamba自体がシーケンスモデルなので、SNNの時間ステップは外部ループとして扱うか、
        #  内部状態の更新に使うかは設計による。ここでは外部ループとして適用)
        
        for _ in range(self.time_steps):
            for layer in self.layers:
                x = layer(x)
                
        x = self.norm(x)
        logits = self.output_projection(x)
        
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (L * self.time_steps * B) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        mem = torch.tensor(0.0, device=input_ids.device) 
        
        return logits, avg_spikes, mem