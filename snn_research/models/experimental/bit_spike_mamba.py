# ファイルパス: snn_research/models/experimental/bit_spike_mamba.py
# 日本語タイトル: Bit-Spike Mamba Model (Fix: Time Loop & Input Handling)
# 目的: SNNの時間発展ループを復元し、トークンIDと特徴量入力の両方に対応する。

import torch
import torch.nn as nn
from typing import Any, Type, cast, Tuple

from snn_research.core.mamba_core import SpikingMambaBlock, SpikingMamba
from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F  # type: ignore


class BitSpikeMambaBlock(SpikingMambaBlock):
    """
    BitSpikeLinearを採用した軽量版Mambaブロック。
    """

    def __init__(self, d_model, d_state, d_conv, expand, neuron_class, neuron_params):
        super().__init__(d_model, d_state, d_conv, expand, neuron_class, neuron_params)
        # BitSpikeLayerへの置換
        self.in_proj = BitSpikeLinear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = BitSpikeLinear(
            self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.out_proj = BitSpikeLinear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        out = super().forward(x)
        # タプルならTensorのみ抽出
        if isinstance(out, tuple):
            return out[0]
        return out


class BitSpikeMamba(SpikingMamba):
    """
    Brain v20 アーキテクチャの中核となるモデル。
    """

    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, num_layers, time_steps, neuron_config, **kwargs):
        super().__init__(vocab_size, d_model, d_state, d_conv,
                         expand, num_layers, time_steps, neuron_config, **kwargs)

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)

        filtered_params = {}
        if neuron_type == 'lif':
            valid_keys = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength',
                          'target_spike_rate', 'noise_intensity', 'threshold_decay',
                          'threshold_step', 'v_reset']
            filtered_params = {k: v for k,
                               v in neuron_params.items() if k in valid_keys}
        else:
            filtered_params = neuron_params

        self.layers = nn.ModuleList([
            BitSpikeMambaBlock(
                d_model, d_state, d_conv, expand,
                cast(Type[nn.Module], AdaptiveLIFNeuron),
                filtered_params
            )
            for _ in range(num_layers)
        ])
        self._init_weights()

    # [Fix] Aligned signature with SpikingMamba (x -> input_ids)
    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SNN Forward Pass.
        input_ids: (Batch, Length) if Long (Token IDs)
                   (Batch, Length, Dim) if Float (Embeddings/Features)
        """

        device = input_ids.device

        # 1. 状態リセット
        SJ_F.reset_net(self)

        # 2. 入力処理 (Embedding or Direct)
        if input_ids.dtype == torch.long:
            if hasattr(self, 'embedding'):
                x_input = self.embedding(input_ids)  # (B, L, D)
            else:
                raise ValueError(
                    "Model has no embedding layer but received LongTensor input.")
        else:
            # 既に埋め込み済み、あるいは視覚特徴量など
            x_input = input_ids  # (B, L, D)

        # 3. Statefulness有効化
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                cast(SpikingMambaBlock, layer).set_stateful(True)

        # 4. SNN時間発展ループ (Time Loop)
        # 入力が静的(Static)であると仮定し、各タイムステップで同じ入力を流す
        # (動的入力の場合はここを (B, T, L, D) 等にする必要があるが、現状はStatic想定)
        x_last = x_input

        for _ in range(self.time_steps):
            x_step = x_input
            for layer in self.layers:
                x_step = layer(x_step)
            x_last = x_step  # 最終層の出力を更新

        # 5. Statefulness解除
        for layer in self.layers:
            if hasattr(layer, 'set_stateful'):
                cast(SpikingMambaBlock, layer).set_stateful(False)

        # 6. 出力層
        # [Fix] Cast self.norm_f/norm to Any to avoid "Tensor not callable"
        if hasattr(self, 'norm_f'):
            x_out = cast(Any, self.norm_f)(x_last)
        elif hasattr(self, 'norm'):
            x_out = cast(Any, self.norm)(x_last)
        else:
            x_out = x_last

        # [Fix] Cast self.lm_head/output_projection to Any
        if hasattr(self, 'lm_head'):
            logits = cast(Any, self.lm_head)(x_out)
        elif hasattr(self, 'output_projection'):
            logits = cast(Any, self.output_projection)(x_out)
        else:
            logits = x_out

        # ダミー統計
        avg_spikes = torch.tensor(0.0, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem

    def get_model_size_mb(self) -> float:
        param_count = sum(p.numel() for p in self.parameters())
        bit_size = param_count * 0.25 / (1024 * 1024)
        return bit_size
