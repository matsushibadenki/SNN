# ファイルパス: snn_research/architectures/hybrid_neuron_network.py
# (修正: outputs_sum (float) を outputs_list (List[Tensor]) に変更して型エラーを解消)
#
# Title: Hybrid Spiking CNN (Mixed Neuron Types)
# Description:
# - 層ごとに異なるニューロンタイプ（LIF, Izhikevichなど）を使用できるCNNモデル。
# - 視覚野のように、初期層は単純なLIF、深層は複雑なバースト発火ニューロン（Izhikevich）
#   といった構成を実験するために使用する。

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, cast, Tuple

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron
# type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F


class HybridSpikingCNN(BaseModel):
    """
    層ごとにニューロンモデルを選択可能なSpiking CNN。
    """

    def __init__(
        self,
        num_classes: int,
        time_steps: int,
        neuron_config: Dict[str, Any],  # デフォルト設定
        # 各層のニューロンタイプ名 ['lif', 'izhikevich', ...]
        layer_specific_neurons: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.time_steps = time_steps

        # デフォルトのパラメータを準備
        default_neuron_type = neuron_config.get('type', 'lif')
        default_params = neuron_config.copy()
        default_params.pop('type', None)

        # 層構成 (VGG-like small)
        # Conv -> Neuron -> Pool -> Conv -> Neuron -> Pool -> Flatten -> FC -> Neuron -> FC
        self.features_layers = nn.ModuleList()
        self.classifier_layers = nn.ModuleList()

        # 各層のニューロンタイプ決定ロジック
        # layer_specific_neurons がなければ全てデフォルト
        types = layer_specific_neurons if layer_specific_neurons else [
            default_neuron_type] * 3
        if len(types) < 3:
            types.extend([default_neuron_type] * (3 - len(types)))

        # --- Layer 1 (Conv) ---
        self.features_layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.features_layers.append(
            self._create_neuron(types[0], 32, default_params))
        self.features_layers.append(nn.AvgPool2d(2))

        # --- Layer 2 (Conv) ---
        self.features_layers.append(
            nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.features_layers.append(
            self._create_neuron(types[1], 64, default_params))
        self.features_layers.append(nn.AvgPool2d(2))

        # --- Classifier ---
        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        fc_input_dim = 64 * 4 * 4
        self.classifier_layers.append(nn.Linear(fc_input_dim, 128))
        self.classifier_layers.append(
            self._create_neuron(types[2], 128, default_params))
        self.classifier_layers.append(nn.Linear(128, num_classes))

        self._init_weights()

    def _create_neuron(self, type_name: str, features: int, base_params: Dict[str, Any]) -> nn.Module:
        """ニューロンインスタンスを生成するヘルパー"""
        if type_name == 'lif':
            valid_params = {k: v for k, v in base_params.items() if k in [
                'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
            return AdaptiveLIFNeuron(features=features, **valid_params)
        elif type_name == 'izhikevich':
            valid_params = {k: v for k, v in base_params.items() if k in [
                'a', 'b', 'c', 'd', 'dt']}
            return IzhikevichNeuron(features=features, **valid_params)
        elif type_name == 'glif':
            valid_params = {k: v for k, v in base_params.items(
            ) if k in ['base_threshold', 'gate_input_features']}
            if 'gate_input_features' not in valid_params:
                valid_params['gate_input_features'] = features
            return GLIFNeuron(features=features, **valid_params)
        else:
            return AdaptiveLIFNeuron(features=features, tau_mem=10.0, base_threshold=1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = x.device
        B = x.shape[0]

        SJ_F.reset_net(self)

        outputs_list: List[torch.Tensor] = []

        # 時間ループ
        for t in range(self.time_steps):
            out = x

            # Features
            for layer in self.features_layers:
                out = layer(out)
                if isinstance(out, tuple):
                    out = out[0]  # ニューロン出力 (spikes, mem)

            # Pooling & Flatten
            out = self.adaptive_pool(out)
            out = self.flatten(out)

            # Classifier
            for layer in self.classifier_layers:
                out = layer(out)
                if isinstance(out, tuple):
                    out = out[0]

            # out は最後のLinear層の出力 (Logits)
            outputs_list.append(out)

        # 時間平均
        if outputs_list:
            logits = torch.stack(outputs_list).mean(dim=0)
        else:
            # time_steps が 0 の場合のフェイルセーフ
            out_features = cast(
                nn.Linear, self.classifier_layers[-1]).out_features
            logits = torch.zeros(B, out_features, device=device)

        # 統計
        avg_spikes_val = self.get_total_spikes(
        ) / (B * self.time_steps) if return_spikes and self.time_steps > 0 else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem
