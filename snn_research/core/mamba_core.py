# ファイルパス: snn_research/core/mamba_core.py
# Title: Spiking-MAMBA (BitNet Integrated & Optimized)
# Description:
#   Spiking-MAMBAモデルの実装。
#   修正: BitSpikeLinear を採用し、SSMスキャンのロジックを整理。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, cast

from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from .base import BaseModel, SNNLayerNorm

# BitNetのインポート (存在チェック付き)
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # BitSpikeLinearがない場合は通常のLinearを使用し、警告を出さないようにする
    class BitSpikeLinear(nn.Linear): # type: ignore
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

class SpikingMambaBlock(sj_base.MemoryModule):
    """
    Spiking-MAMBA Block with BitNet Weights.
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

        # BitSpikeLinearを使用 (乗算フリー化への布石)
        self.in_proj = BitSpikeLinear(d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.lif_conv = neuron_class(features=self.d_inner, **neuron_params)
        
        # BitSpikeLinear
        self.x_proj = BitSpikeLinear(self.d_inner, self.d_inner + 2 * d_state)
        self.dt_proj = BitSpikeLinear(self.d_inner, self.d_inner)
        
        # A_logの初期化 (Mamba本来の初期化: 負の値になるように調整)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # BitSpikeLinear
        self.out_proj = BitSpikeLinear(self.d_inner, d_model)
        self.norm = SNNLayerNorm(d_model)
        
        self.lif_out = neuron_class(features=d_model, **neuron_params)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        # 子モジュールへの再帰的な設定
        for m in self.modules():
            if isinstance(m, sj_base.MemoryModule) and m is not self:
                m.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        # 子モジュールのリセット
        for m in self.modules():
            if isinstance(m, sj_base.MemoryModule) and m is not self:
                m.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Length, Dim)
        """
        B, L, D = x.shape
        
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        # Causal Conv1d
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        
        # LIF Neuronへの入力 (FlattenしてBatchとして処理)
        # 注意: lif_convが状態を持つ場合、B*L次元の状態を保持することになる。
        # これは「各トークン位置が独立したニューロンを持つ」のと同義であり、
        # 配列全体に対する変換としては有効。
        x_conv_flat = x_conv.reshape(B * L, -1)
        output = self.lif_conv(x_conv_flat) 
        if isinstance(output, tuple):
            x_conv_spikes = output[0]
        else:
            x_conv_spikes = output
            
        x_conv_spikes = x_conv_spikes.reshape(B, L, -1)
        
        # SSM Parameters
        x_ssm_params = self.x_proj(x_conv_spikes)
        delta, B_param, C_param = x_ssm_params.split(split_size=[self.d_inner, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        
        # Discretization
        # A must be negative. exp(A_log) is positive, so -exp is negative.
        A = -torch.exp(self.A_log.float())
        
        # A_bar: (B, L, D_inner, D_state)
        A_bar = torch.exp(A * delta.unsqueeze(-1))
        # B_bar: (B, L, D_inner, D_state)
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(-2)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_scan = []
        
        # SSM Scan (Selective Scan)
        # Python loop is used here. For optimization, custom CUDA kernel or associative scan is needed.
        # Here we prioritize logic correctness for SNN context.
        for i in range(L):
            # x_term: (B, D_inner, D_state)
            x_term = B_bar[:, i] * x_conv_spikes[:, i].unsqueeze(-1)
            h = A_bar[:, i] * h + x_term
            # y: (B, D_inner)
            y = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            y_scan.append(y)
            
        y = torch.stack(y_scan, dim=1) + x_conv_spikes * self.D
        y = y * F.silu(res) # Gating with residual
        
        out = self.norm(x + self.out_proj(y))
        
        out_output = self.lif_out(out.reshape(B * L, -1))
        if isinstance(out_output, tuple):
            out_spikes = out_output[0]
        else:
            out_spikes = out_output
            
        return out_spikes.reshape(B, L, -1)

class SpikingMamba(BaseModel):
    """
    SpikingMamba: BitNet + SNN + SSM
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
        
        neuron_class: Any
        filtered_params: Dict[str, Any]
        
        # Neuron Parameter Filtering Logic
        valid_keys_map = {
            'lif': ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset', 'detach_reset'],
            'izhikevich': ['features', 'a', 'b', 'c', 'd', 'dt'],
            'glif': ['features', 'base_threshold', 'gate_input_features'],
            'tc_lif': ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset'],
            'dual_threshold': ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
        }

        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = d_model * expand
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron
        else:
             raise ValueError(f"Unknown neuron type for SpikingMamba: {neuron_type}")
             
        valid_keys = valid_keys_map.get(neuron_type, [])
        filtered_params = {k: v for k, v in neuron_params.items() if k in valid_keys}

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

    def _init_weights(self):
        # カスタム重み初期化ロジックがあればここに記述
        # 現状はPyTorchのデフォルトに任せるが、SNN向けに調整も可能
        pass

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_ids: (Batch, Length) - Static Sequence
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # ネットワークの状態をリセット (SpikingJelly機能)
        SJ_F.reset_net(self)
        
        x_embed = self.embedding(input_ids)
        x = x_embed
        
        # Statefulnessの有効化
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                cast(SpikingMambaBlock, layer).set_stateful(True)

        # 時間ステップ (SNN simulation steps)
        # 静的な入力を time_steps 回ネットワークに通し、ニューロンの状態を遷移させる
        for _ in range(self.time_steps):
            x_step = x_embed # 入力は各ステップで同じ（静的画像のRate Coding等の場合はここが変わる）
            for layer in self.layers:
                x_step = layer(x_step)
            x = x_step # 最終層の出力を保持（もしくは蓄積）
        
        # Statefulnessの解除
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                cast(SpikingMambaBlock, layer).set_stateful(False)

        x_out = self.norm(x)
        logits = self.output_projection(x_out)
        
        # スパイク統計の取得 (オプション)
        total_spikes = self.get_total_spikes() if hasattr(self, 'get_total_spikes') else torch.tensor(0.0)
        avg_spikes_val = total_spikes / (L * self.time_steps * B) if return_spikes else 0.0
        avg_spikes = torch.as_tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device) 
        
        return logits, avg_spikes, mem