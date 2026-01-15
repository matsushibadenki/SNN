# snn_research/core/layers/predictive_coding.py
# ファイルパス: snn_research/core/layers/predictive_coding.py
# 修正内容: mypy型エラー修正（eps引数を位置引数に変更）

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
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # 開発環境等でのフォールバック
    AdaptiveLIFNeuron = Any  # type: ignore
    IzhikevichNeuron = Any  # type: ignore
    GLIFNeuron = Any  # type: ignore
    TC_LIF = Any  # type: ignore
    DualThresholdNeuron = Any  # type: ignore
    ScaleAndFireNeuron = Any  # type: ignore
    BistableIFNeuron = Any  # type: ignore
    EvolutionaryLeakLIF = Any  # type: ignore
    # BitSpikeLinearがない場合は通常のLinearで代用するダミー

    class BitSpikeLinear(nn.Linear):  # type: ignore
        def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
            super().__init__(in_features, out_features, bias=bias)

logger = logging.getLogger(__name__)


class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding (PC) を実行するSNNレイヤー。

    Biomimetic Enhancement:
    - Iterative Inference: 複数ステップの緩和(Relaxation)による推論。
    - Energy Minimization: 自由エネルギー最小化。
    - BitNet Integration: 重みを {-1, 0, 1} に量子化し高速化。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        weight_tying: bool = True,
        sparsity: float = 0.05,
        inference_steps: int = 5,
        inference_lr: float = 0.1,
        use_bitnet: bool = True  # 【追加】デフォルトでBitNet有効化
    ):
        super().__init__()
        self.weight_tying = weight_tying
        self.sparsity = sparsity
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.use_bitnet = use_bitnet

        filtered_params = self._filter_params(neuron_class, neuron_params)

        # 線形層のクラス選択
        LinearLayer = BitSpikeLinear if use_bitnet else nn.Linear
        linear_kwargs = {'quantize_inference': True} if use_bitnet else {}

        # 1. Generative Path (Top-Down: State -> Prediction)
        self.generative_fc = LinearLayer(d_state, d_model, **linear_kwargs)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                                      neuron_class(features=d_model, **filtered_params))

        # 2. Inference Path (Bottom-Up: Error -> State Update)
        if self.weight_tying:
            self.inference_fc = None
        else:
            self.inference_fc = LinearLayer(d_model, d_state, **linear_kwargs)

        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                                     neuron_class(features=d_state, **filtered_params))

        self.norm_state = nn.LayerNorm(d_state)
        self.norm_error = nn.LayerNorm(d_model)

        self.error_scale = nn.Parameter(torch.tensor(1.0))
        self.feedback_strength = nn.Parameter(torch.tensor(1.0))

    def _filter_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスが受け入れるパラメータのみを抽出する"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength',
                            'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        elif neuron_class == IzhikevichNeuron:
            valid_params = ['features', 'a', 'b', 'c', 'd', 'dt']
        elif neuron_class == GLIFNeuron:
            valid_params = ['features',
                            'base_threshold', 'gate_input_features']
        elif neuron_class == TC_LIF:
            valid_params = ['features', 'tau_s_init', 'tau_d_init',
                            'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
        elif neuron_class == DualThresholdNeuron:
            valid_params = ['features', 'tau_mem',
                            'threshold_high_init', 'threshold_low_init', 'v_reset']
        elif neuron_class == ScaleAndFireNeuron:
            valid_params = ['features', 'num_levels', 'base_threshold']
        elif neuron_class == BistableIFNeuron:
            valid_params = ['features', 'v_threshold_high', 'v_reset', 'tau_mem',
                            'bistable_strength', 'v_rest', 'unstable_equilibrium_offset']
        elif neuron_class == EvolutionaryLeakLIF:
            valid_params = ['features', 'initial_tau', 'v_threshold',
                            'v_reset', 'detach_reset', 'learn_threshold']
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
        if k == 0:
            k = 1
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
        Inference Phase as Relaxation with BitNet Acceleration
        """

        # 初期状態
        current_state = top_down_state.clone()
        final_error = torch.zeros_like(bottom_up_input)
        combined_mem_list = []

        # --- Relaxation Loop ---
        for step in range(self.inference_steps):
            # 1. Generative Pass
            # BitSpikeLinearが自動的に重みを量子化して計算
            pred_input = self.generative_fc(self.norm_state(current_state))
            pred, gen_mem = self.generative_neuron(pred_input)

            # 2. Error Calculation
            raw_error = bottom_up_input - pred
            error = raw_error * self.error_scale

            if step == self.inference_steps - 1:
                final_error = error

            # 3. Inference Pass (State Update)
            norm_error = self.norm_error(error)

            if self.weight_tying:
                # 重み共有時の転置行列計算
                if self.use_bitnet and hasattr(self.generative_fc, 'weight'):
                    from snn_research.core.layers.bit_spike_layer import bit_quantize_weight
                    # mypy修正: epsをキーワード引数ではなく位置引数として渡す
                    # (bit_spike_layerの実装に準拠)
                    w_quant = bit_quantize_weight(
                        self.generative_fc.weight, 1e-5)
                    bu_input = F.linear(norm_error, w_quant.t())
                else:
                    bu_input = F.linear(
                        norm_error, self.generative_fc.weight.t())
            else:
                if self.inference_fc is None:
                    raise RuntimeError("inference_fc is None")
                bu_input = self.inference_fc(norm_error)

            total_input = bu_input
            if top_down_error is not None:
                total_input = total_input - \
                    (top_down_error * self.feedback_strength)

            # 状態更新
            state_update, inf_mem = self.inference_neuron(total_input)
            state_update = self._apply_lateral_inhibition(state_update)

            # 状態変数の緩和更新
            current_state = current_state * \
                (1.0 - self.inference_lr) + state_update * self.inference_lr

            if step == self.inference_steps - 1:
                combined_mem_list.append(torch.cat((gen_mem, inf_mem), dim=1))

        combined_mem = combined_mem_list[-1] if combined_mem_list else torch.tensor(
            0.0)

        return current_state, final_error, combined_mem
