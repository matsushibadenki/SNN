# ファイルパス: snn_research/models/experimental/thermo_mamba.py
# 日本語タイトル: Thermodynamic Spiking Mamba (System 1/2 Hybrid Engine)
# Description:
#   SpikingJellyの限界を超えるための実験的アーキテクチャ。
#   高速なSSM（System 1）と熱力学的サンプリング（System 2）を動的に切り替え、
#   自由エネルギー原理に基づく自己組織化を行う次世代SNN。
#   修正: super().reset() の呼び出しを削除 (BaseModelにresetがないため)。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type, cast

# 依存関係のインポート (環境に合わせて調整)
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # BitSpikeLinearがない場合のフォールバック
    class BitSpikeLinear(nn.Linear):  # type: ignore
        def __init__(
            self, in_features: int, out_features: int, bias: bool = True, **kwargs: Any
        ) -> None:
            super().__init__(in_features, out_features, bias=bias)

from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.evolution.structural_plasticity import StructuralPlasticity

# SpikingJelly
from spikingjelly.activation_based import base as sj_base  # type: ignore
from spikingjelly.activation_based import functional as SJ_F  # type: ignore


class ThermodynamicStateEngine(nn.Module):
    """
    System 2: 熱力学的状態エンジン
    Langevin Dynamicsを用いて、エネルギー地形上の安定点（記憶/正解）を探索する。
    """

    def __init__(
        self, d_inner: int, d_state: int, temperature: float = 1.0, steps: int = 5
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.base_temperature = temperature
        self.steps = steps

        # 内部エネルギー重み (Recurrent Energy Matrix)
        self.energy_weight = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)

    def energy_grad(self, h: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """
        エネルギー関数の勾配を計算: E(h, x) = ||h - Wh @ x||^2 + Regularization
        """
        # 簡易的なエネルギー地形: 現在の状態と、コンテキストから予測される状態の乖離
        predicted_h = torch.tanh(x_context @ self.energy_weight)
        grad = h - predicted_h
        return grad

    def forward(
        self, h_init: torch.Tensor, x_context: torch.Tensor, surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Langevin Samplingによる状態更新
        surprise (自由エネルギー) が高いほど、温度(探索範囲)を上げる
        """
        h = h_init.clone()
        dt = 0.1

        # サプライズによる温度スケーリング (動的温度調整)
        current_temp = self.base_temperature * (
            1.0 + torch.sigmoid(surprise).mean().item() * 5.0
        )
        noise_scale = math.sqrt(2 * dt * current_temp)

        for _ in range(self.steps):
            # 勾配によるドリフト項
            grad = self.energy_grad(h, x_context)
            drift = -grad * dt

            # 拡散項 (ブラウン運動)
            diffusion = torch.randn_like(h) * noise_scale

            # 状態更新
            h = h + drift + diffusion
            h = torch.tanh(h)  # 安定化

        return h


class TS_MambaBlock(sj_base.MemoryModule):
    """
    Thermodynamic Spiking Mamba Block
    System 1 (SSM Scan) と System 2 (Thermodynamic Sampling) を統合。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        neuron_params: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # --- System 1 Components (Fast Path) ---
        self.in_proj = BitSpikeLinear(d_model, self.d_inner * 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = BitSpikeLinear(
            self.d_inner, self.dt_rank + self.d_state * 2)
        self.dt_proj = BitSpikeLinear(self.dt_rank, self.d_inner)

        A = torch.arange(
            1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = BitSpikeLinear(self.d_inner, d_model)

        # Neurons
        self.lif_conv = AdaptiveLIFNeuron(
            features=self.d_inner, **neuron_params)
        self.lif_out = AdaptiveLIFNeuron(features=d_model, **neuron_params)

        # --- System 2 Components (Slow/Deep Path) ---
        self.thermo_engine = ThermodynamicStateEngine(
            self.d_inner, self.d_state)

        # Mode Controller
        self.thinking_mode = False
        self.surprise_threshold = 0.5  # System 2を発動させる閾値

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # 1. Projection & Conv (Common)
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(
            split_size=[self.d_inner, self.d_inner], dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)

        # Spike Encoding
        x_conv_flat = x_conv.reshape(B * L, -1)
        x_conv_spikes = self.lif_conv(x_conv_flat)
        if isinstance(x_conv_spikes, tuple):
            x_conv_spikes = x_conv_spikes[0]
        x_conv_spikes = x_conv_spikes.reshape(B, L, -1)

        # 2. SSM Parameters
        x_ssm_params = self.x_proj(x_conv_spikes)
        dt_in, B_param, C_param = x_ssm_params.split(
            split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_in))

        A = -torch.exp(self.A_log.float())

        # 3. Hybrid State Scan
        # System 1 (Vectorized Scan) と System 2 (Iterative Thinking) の分岐

        if self.thinking_mode:
            # System 2: Iterative + Thermodynamic Correction
            y = self._system2_process(
                x_conv_spikes, dt, A, B_param, C_param, L)
        else:
            # System 1: Standard SSM Scan (Approximated for SNN)
            y = self._system1_process(
                x_conv_spikes, dt, A, B_param, C_param, L)

        # 4. Output Projection
        y = y + x_conv_spikes * self.D
        y = y * F.silu(res)
        out = self.out_proj(y)

        out_spikes = self.lif_out(out.reshape(B * L, -1))
        if isinstance(out_spikes, tuple):
            out_spikes = out_spikes[0]

        return out_spikes.reshape(B, L, -1)

    def _system1_process(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        """
        高速処理モード: 離散化と単純な状態更新
        """
        # A_bar: (B, L, D_inner, D_state)
        A_bar = torch.exp(A * dt.unsqueeze(-1))
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(-2)

        h = torch.zeros(x.shape[0], self.d_inner,
                        self.d_state, device=x.device)
        y_scan = []

        for i in range(L):
            x_term = B_bar[:, i] * x[:, i].unsqueeze(-1)
            h = A_bar[:, i] * h + x_term
            y_t = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            y_scan.append(y_t)

        return torch.stack(y_scan, dim=1)

    def _system2_process(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        """
        熟考モード: SSM更新後に熱力学的サンプリングを行い、状態をリファインする
        """
        A_bar = torch.exp(A * dt.unsqueeze(-1))
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(-2)

        h = torch.zeros(x.shape[0], self.d_inner,
                        self.d_state, device=x.device)
        y_scan = []

        for i in range(L):
            # 1. Standard Update prediction
            x_term = B_bar[:, i] * x[:, i].unsqueeze(-1)
            h_pred = A_bar[:, i] * h + x_term

            # 2. Compute Surprise (Free Energy proxy)
            # 過去の状態からの予測と、現在の入力による更新の乖離をサプライズとする
            surprise = torch.mean((h - h_pred) ** 2)

            # 3. Thermodynamic Refinement (Langevin Sampling)
            # 状態hをエネルギー地形に従って沈降させる
            if surprise > self.surprise_threshold:
                h = self.thermo_engine(h_pred, x[:, i], surprise)
            else:
                h = h_pred

            y_t = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            y_scan.append(y_t)

        return torch.stack(y_scan, dim=1)


class ThermodynamicSpikingMamba(BaseModel):
    """
    Main Model Wrapper
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        neuron_config: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TS_MambaBlock(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    neuron_params=neuron_config,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = SNNLayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # 構造的可塑性マネージャ (Structural Plasticity)
        self.plasticity_engine = StructuralPlasticity(
            self, config={"pruning_rate": 0.05,
                          "growth_rate": 0.05, "noise_std": 0.02}
        )

    def forward(
        self, input_ids: torch.Tensor, thinking: bool = False
    ) -> torch.Tensor:
        """
        thinking: Trueの場合、System 2 (熟考モード) を強制起動
        """
        x = self.embedding(input_ids)

        # モード切替
        for layer in self.layers:
            if isinstance(layer, TS_MambaBlock):
                layer.thinking_mode = thinking

        # SNNループ（ここではTimeStep=1として簡略化または外部ループを想定）
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

    def evolve(self) -> Dict[str, int]:
        """
        ネットワーク構造を進化させる
        学習ループのepoch終了時などに呼び出す
        """
        return self.plasticity_engine.evolve_structure()

    def reset(self) -> None:
        """
        状態のリセット (SpikingJelly準拠)
        BaseModelにはresetがないため、super().reset()は呼び出さず、
        子モジュールのresetのみを行う。
        """
        for layer in self.layers:
            if isinstance(layer, sj_base.MemoryModule):
                layer.reset()


# 使用例のデモ
if __name__ == "__main__":
    # 設定
    neuron_cfg = {
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "tau": 2.0,
        "surrogate_function": SJ_F.surrogate.ATan(),
    }

    model = ThermodynamicSpikingMamba(
        vocab_size=1000, d_model=64, num_layers=2, neuron_config=neuron_cfg
    )

    # 1. System 1: Fast Inference
    input_data = torch.randint(0, 1000, (4, 32))  # (Batch, Length)
    output_fast = model(input_data, thinking=False)
    print(f"System 1 Output: {output_fast.shape}")

    # 2. System 2: Deep Thinking (Thermodynamic Sampling)
    output_slow = model(input_data, thinking=True)
    print(f"System 2 Output: {output_slow.shape}")

    # 3. Structural Evolution (Rewiring)
    stats = model.evolve()
    print(f"Evolution Stats: {stats}")
