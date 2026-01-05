# ファイルパス: snn_research/core/neurons/feel_neuron.py
# Title: Evolutionary Leak LIF (EL-LIF) ニューロン (修正版)
# Description:
#   FEEL-SNN (NeurIPS 2024) に基づく、学習可能な漏れ係数 (Evolutionary Leak) を持つLIFニューロン。
#   修正: spikes バッファの更新ロジックをベクトル対応に修正。
#   修正: _view_params による多次元入力対応。

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union
# type: ignore[import-untyped]
from spikingjelly.activation_based import base, surrogate


class EvolutionaryLeakLIF(base.MemoryModule):
    """
    Evolutionary Leak Leaky Integrate-and-Fire (EL-LIF) Neuron.
    """
    v_threshold: Union[torch.Tensor, nn.Parameter]
    decay_logit: nn.Parameter

    def __init__(
        self,
        features: int,
        initial_tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Optional[nn.Module] = None,
        detach_reset: bool = False,
        learn_threshold: bool = False
    ):
        super().__init__()
        self.features = features
        self.v_reset = v_reset
        self.detach_reset = detach_reset

        # サロゲート勾配関数 (デフォルト: ATan)
        self.surrogate_function = surrogate_function if surrogate_function is not None else surrogate.ATan()

        # 閾値 (学習可能にするオプションあり)
        if learn_threshold:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.register_buffer('v_threshold', torch.tensor(v_threshold))

        # --- Evolutionary Leak Factor (EL) ---
        initial_decay = math.exp(-1.0 / initial_tau)
        initial_decay = max(0.01, min(0.99, initial_decay))  # クリップ
        logit_initial = math.log(initial_decay / (1.0 - initial_decay))

        # 各ニューロン（特徴量）ごとに独立した減衰率を持つ
        self.decay_logit = nn.Parameter(torch.full((features,), logit_initial))

        # 状態変数
        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def _view_params(self, param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """入力テンソルの形状に合わせてパラメータをブロードキャストする"""
        if param.ndim != 1:
            return param
        # (B, C, H, W) -> (1, C, 1, 1)
        if x.ndim == 4 and x.shape[1] == self.features:
            return param.view(1, -1, 1, 1)
        # (B, T, C) or (B, C, L) depends on usage, assume (B, C) part matches
        if x.shape[-1] == self.features:  # Last dim
            return param
        if x.ndim == 3 and x.shape[1] == self.features:  # (B, C, L)
            return param.view(1, -1, 1)
        return param

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.stateful:
            self.mem = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)

        # 1. 進化的減衰率の計算
        decay = torch.sigmoid(self.decay_logit)
        decay_expanded = self._view_params(decay, x)

        threshold_expanded: torch.Tensor
        if isinstance(self.v_threshold, nn.Parameter):
            threshold_expanded = self._view_params(self.v_threshold, x)
        else:
            threshold_expanded = self._view_params(self.v_threshold, x)

        # 2. 膜電位の更新 (Leak + Input)
        self.mem = self.mem * decay_expanded + x

        # 3. スパイク生成
        spike = self.surrogate_function(self.mem - threshold_expanded)

        # 4. リセット
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        # Hard Reset: 発火したら v_reset に戻す
        self.mem = self.mem * (1.0 - spike_d) + self.v_reset * spike_d

        # 統計更新 (ベクトルとして保持)
        if spike.ndim > 1:
            if x.ndim == 4:  # (B, C, H, W)
                self.spikes = spike.mean(dim=(0, 2, 3))
            elif x.ndim == 3 and x.shape[2] == self.features:  # (B, T, C)
                self.spikes = spike.mean(dim=(0, 1))
            elif x.ndim == 3 and x.shape[1] == self.features:  # (B, C, L)
                self.spikes = spike.mean(dim=(0, 2))
            elif x.ndim == 2:  # (B, C)
                self.spikes = spike.mean(dim=0)
            else:
                self.spikes = spike.mean()
        else:
            self.spikes = spike

        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        return spike, self.mem
