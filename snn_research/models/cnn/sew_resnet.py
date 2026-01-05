# ファイルパス: snn_research/models/cnn/sew_resnet.py
# Title: SEW (Spike-Element-Wise) ResNet - 修正版
# Description:
# - 修正: forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動。
# - 修正: AdaptiveLIFNeuron のパラメータフィルタに v_reset を追加。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast
import logging

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
# type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F

logger = logging.getLogger(__name__)


class SEWResidualBlock(nn.Module):
    """
    SEW (Spike-Element-Wise) 残差ブロック。
    """
    conv1: nn.Conv2d
    norm1: nn.Module  # SNNLayerNorm or BatchNorm
    lif1: nn.Module
    conv2: nn.Conv2d
    norm2: nn.Module
    lif_shortcut: nn.Module
    lif_out: nn.Module
    downsample: Optional[nn.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()

        # 1. Main Path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.lif1 = neuron_class(features=out_channels, **neuron_params)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # 2. Shortcut Path (残差接続)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.lif_shortcut = neuron_class(
            features=out_channels, **neuron_params)

        # 3. Final Activation (ADD -> FIRE)
        self.lif_out = neuron_class(features=out_channels, **neuron_params)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        B, T, C_in, H, W = x_spike.shape

        # --- 1. ショートカットパス ---
        shortcut_spikes: List[torch.Tensor] = []

        neuron_sc: nn.Module = cast(nn.Module, self.lif_shortcut)
        if hasattr(neuron_sc, 'set_stateful'):
            getattr(neuron_sc, 'set_stateful')(True)

        identity: torch.Tensor = x_spike

        if self.downsample:
            identity_flat = identity.reshape(B * T, C_in, H, W)
            identity_downsampled_flat = self.downsample(identity_flat)
            _, C_out, H_out, W_out = identity_downsampled_flat.shape
            identity_downsampled = identity_downsampled_flat.reshape(
                B, T, C_out, H_out, W_out)
        else:
            identity_downsampled = identity
            _, C_out, H_out, W_out = identity_downsampled.shape

        for t_idx in range(T):
            x_sc_t: torch.Tensor = identity_downsampled[:, t_idx, ...]
            x_sc_t_flat: torch.Tensor = x_sc_t.permute(
                0, 2, 3, 1).reshape(-1, C_out)
            spike_sc_t_flat, _ = self.lif_shortcut(x_sc_t_flat)
            spike_sc_t: torch.Tensor = spike_sc_t_flat.reshape(
                B, H_out, W_out, C_out).permute(0, 3, 1, 2)
            shortcut_spikes.append(spike_sc_t)

        if hasattr(neuron_sc, 'set_stateful'):
            getattr(neuron_sc, 'set_stateful')(False)

        # --- 2. メインパス ---
        main_path_spikes: List[torch.Tensor] = []

        neuron1: nn.Module = cast(nn.Module, self.lif1)
        neuron_out: nn.Module = cast(nn.Module, self.lif_out)
        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(True)
        if hasattr(neuron_out, 'set_stateful'):
            getattr(neuron_out, 'set_stateful')(True)

        for t_idx in range(T):
            x_t: torch.Tensor = x_spike[:, t_idx, ...]
            y_t: torch.Tensor = self.conv1(x_t)
            y_t = self.norm1(y_t)

            B_t, C_t, H_t, W_t = y_t.shape
            y_t_flat: torch.Tensor = y_t.permute(0, 2, 3, 1).reshape(-1, C_t)

            spike1_t_flat, _ = self.lif1(y_t_flat)
            spike1_t: torch.Tensor = spike1_t_flat.reshape(
                B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)

            y_t = self.conv2(spike1_t)
            y_t = self.norm2(y_t)

            residual_input: torch.Tensor = y_t + shortcut_spikes[t_idx]

            res_in_flat: torch.Tensor = residual_input.permute(
                0, 2, 3, 1).reshape(-1, C_out)
            spike_out_t_flat, _ = self.lif_out(res_in_flat)
            spike_out_t: torch.Tensor = spike_out_t_flat.reshape(
                B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)

            main_path_spikes.append(spike_out_t)

        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(False)
        if hasattr(neuron_out, 'set_stateful'):
            getattr(neuron_out, 'set_stateful')(False)

        return torch.stack(main_path_spikes, dim=1)


class SEWResNet(BaseModel):
    """
    SEW (Spike-Element-Wise) ResNet アーキテクチャ。
    """

    def __init__(
        self,
        num_classes: int = 10,
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.time_steps = time_steps

        if neuron_config is None:
            neuron_config = {'type': 'lif',
                             'tau_mem': 10.0, 'base_threshold': 1.0}

        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)

        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            # --- 修正: v_reset を追加 ---
            neuron_params = {k: v for k, v in neuron_params.items() if k in [
                'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']}
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in [
                'a', 'b', 'c', 'd', 'dt']}
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type_str}")

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(self.in_channels)
        self.lif1 = neuron_class(features=self.in_channels, **neuron_params)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            self.in_channels, 64, 2, 1, neuron_class, neuron_params)
        self.layer2 = self._make_layer(
            64, 128, 2, 2, neuron_class, neuron_params)
        self.layer3 = self._make_layer(
            128, 256, 2, 2, neuron_class, neuron_params)
        self.layer4 = self._make_layer(
            256, 512, 2, 2, neuron_class, neuron_params)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()
        logger.info(f"✅ SEWResNet initialized (T={time_steps}).")

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any]
    ) -> nn.Sequential:

        layers: List[nn.Module] = []
        layers.append(SEWResidualBlock(in_channels, out_channels,
                      stride, neuron_class, neuron_params))

        for _ in range(1, num_blocks):
            layers.append(SEWResidualBlock(
                out_channels, out_channels, 1, neuron_class, neuron_params))

        return nn.Sequential(*layers)

    def forward(
        self,
        input_images: torch.Tensor,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, C, H, W = input_images.shape
        device: torch.device = input_images.device

        x_spikes_t: torch.Tensor = input_images.unsqueeze(
            1).repeat(1, self.time_steps, 1, 1, 1)

        SJ_F.reset_net(self)

        # --- 1. 入力層 (時間軸ループ) ---
        spikes_history: List[torch.Tensor] = []

        neuron1: nn.Module = cast(nn.Module, self.lif1)
        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(True)

        for t in range(self.time_steps):
            x_t: torch.Tensor = x_spikes_t[:, t, ...]
            y_t: torch.Tensor = self.conv1(x_t)
            y_t = self.norm1(y_t)
            B_t, C_t, H_t, W_t = y_t.shape
            y_t_flat: torch.Tensor = y_t.permute(0, 2, 3, 1).reshape(-1, C_t)
            spike_t_flat, _ = self.lif1(y_t_flat)
            spike_t: torch.Tensor = spike_t_flat.reshape(
                B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)
            spikes_history.append(spike_t)

        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(False)

        x_spikes: torch.Tensor = torch.stack(spikes_history, dim=1)

        x_flat_time: torch.Tensor = x_spikes.reshape(
            B * self.time_steps, self.in_channels, H, W)
        pooled_flat: torch.Tensor = self.pool1(x_flat_time)
        _, C_pool, H_pool, W_pool = pooled_flat.shape
        x_spikes = pooled_flat.reshape(
            B, self.time_steps, C_pool, H_pool, W_pool)

        # --- 2. 残差ブロック層 ---
        x_spikes = self.layer1(x_spikes)
        x_spikes = self.layer2(x_spikes)
        x_spikes = self.layer3(x_spikes)
        x_spikes = self.layer4(x_spikes)

        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes_val = 0.0
        if return_spikes:
            avg_spikes_val = self.get_total_spikes() / (B * self.time_steps)
        # ------------------------------------

        # --- 3. 分類層 ---
        x_analog: torch.Tensor = x_spikes.mean(dim=1)

        x_analog = self.avgpool(x_analog)
        x_analog = torch.flatten(x_analog, 1)
        logits: torch.Tensor = self.fc(x_analog)

        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem
