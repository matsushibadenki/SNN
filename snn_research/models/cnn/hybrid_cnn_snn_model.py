# ファイルパス: snn_research/models/cnn/hybrid_cnn_snn_model.py
# Title: Hybrid CNN-SNN Model (Fixed Config & Dimensions)
# Description:
# - Adapter初期化時に neuron_config に 'features' を確実に含めるよう修正。
# - full_mems の結合軸を dim=4 から dim=1 (Layers) に修正。

import torch
import torch.nn as nn
from torchvision import models  # type: ignore[import-untyped]
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.layers.adapters import AnalogToSpikes
from snn_research.core.layers.complex_attention import STAttenBlock
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF
)

# type: ignore[import-untyped]
from spikingjelly.activation_based import functional, base as sj_base


class HybridCnnSnnModel(BaseModel):
    """
    ANN(CNN)フロントエンドとSNN(Transformer)バックエンドを持つハイブリッドモデル。
    """

    def __init__(self, vocab_size: int, time_steps: int, ann_frontend: Dict[str, Any], snn_backend: Dict[str, Any], neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps

        if ann_frontend['name'] == 'mobilenet_v2':
            weights: Optional[models.MobileNet_V2_Weights] = models.MobileNet_V2_Weights.DEFAULT if ann_frontend.get(
                'pretrained', True) else None
            mobilenet: nn.Module = models.mobilenet_v2(weights=weights)
            # type: ignore[assignment]
            self.ann_feature_extractor: nn.Module = cast(
                nn.Module, mobilenet.features)
        else:
            raise ValueError(
                f"Unsupported ANN frontend: {ann_frontend['name']}")

        for param in self.ann_feature_extractor.parameters():
            param.requires_grad = True

        # 修正: neuron_config に 'features' を明示的に追加
        adapter_neuron_config = neuron_config.copy()
        adapter_neuron_config['features'] = snn_backend['d_model']

        self.adapter_a2s = AnalogToSpikes(
            in_features=ann_frontend['output_features'],
            out_features=snn_backend['d_model'],
            time_steps=time_steps,
            activation=nn.ReLU,
            neuron_config=adapter_neuron_config
        )

        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[Union[AdaptiveLIFNeuron,
                                 IzhikevichNeuron, GLIFNeuron, TC_LIF]]

        filtered_params: Dict[str, Any]
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
            }
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = snn_backend['d_model']
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF  # type: ignore[assignment]
            filtered_params = {
                k: v for k, v in neuron_params.items()
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        else:
            raise ValueError(
                f"Unknown neuron type for HybridCnnSnnModel backend: {neuron_type}")

        self.snn_backend = nn.ModuleList([
            STAttenBlock(
                snn_backend['d_model'], snn_backend['n_head'], neuron_class, filtered_params)
            for _ in range(snn_backend['num_layers'])
        ])

        self.output_projection = nn.Linear(snn_backend['d_model'], vocab_size)
        self._init_weights()

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        functional.reset_net(self)

        ann_features: torch.Tensor = self.ann_feature_extractor(input_images)
        ann_features = ann_features.mean([2, 3])

        snn_input_spikes: torch.Tensor
        adapter_mems: Optional[torch.Tensor]
        snn_input_spikes, adapter_mems = self.adapter_a2s(
            ann_features, return_full_mems=return_full_mems)

        local_mems_history: List[torch.Tensor] = []
        if return_full_mems and adapter_mems is not None:
            # adapter_mems: (B, T, D) -> (B, 1, T, D) for layer stack
            local_mems_history.append(adapter_mems.unsqueeze(1))

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            for layer_module in self.snn_backend:
                block = cast(STAttenBlock, layer_module)
                block.clear_mem_history()
                hooks.extend(block.register_mem_hooks())

        x: torch.Tensor = snn_input_spikes
        functional.reset_net(self.snn_backend)

        full_hiddens_list: List[torch.Tensor] = []

        for layer_module in self.snn_backend:
            block = cast(STAttenBlock, layer_module)
            cast(sj_base.MemoryModule, block.lif1).set_stateful(True)
            cast(sj_base.MemoryModule, block.lif2).set_stateful(True)
            cast(sj_base.MemoryModule, block.lif3).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_q).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_k).set_stateful(True)
            cast(sj_base.MemoryModule, block.attn.neuron_out).set_stateful(True)

            x = layer_module(x)
            full_hiddens_list.append(x)

        full_mems: torch.Tensor
        if return_full_mems:
            for layer_idx, layer_module in enumerate(self.snn_backend):
                block = cast(STAttenBlock, layer_module)
                num_neurons_in_block = 6
                block_mems = block.mem_history
                lif3_mems: List[torch.Tensor] = [block_mems[t*num_neurons_in_block + 5]
                                                 for t in range(self.time_steps) if (t*num_neurons_in_block + 5) < len(block_mems)]
                if len(lif3_mems) == self.time_steps:
                    lif3_mems_stacked: torch.Tensor = torch.stack(
                        lif3_mems, dim=1).unsqueeze(1)
                    local_mems_history.append(lif3_mems_stacked)

            for hook in hooks:
                hook.remove()

            if local_mems_history:
                # [Fix] Concatenate on dim=1 (Layers) instead of dim=4
                # Each element is (B, 1, T, D) -> Concat to (B, L, T, D)
                full_mems_cat: torch.Tensor = torch.cat(
                    local_mems_history, dim=1)
                full_mems = full_mems_cat
            else:
                full_mems = torch.zeros(
                    B, 1, self.time_steps, 1, device=device)
        else:
            full_mems = torch.tensor(0.0, device=device)

        total_spikes: float = 0.0
        if return_spikes:
            total_spikes = self.get_total_spikes()
        avg_spikes_val: float = total_spikes / \
            (B * self.time_steps) if return_spikes else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)

        # Reset state
        for layer_module in self.snn_backend:
            block = cast(STAttenBlock, layer_module)
            cast(sj_base.MemoryModule, block.lif1).set_stateful(False)
            cast(sj_base.MemoryModule, block.lif2).set_stateful(False)
            cast(sj_base.MemoryModule, block.lif3).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_q).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_k).set_stateful(False)
            cast(sj_base.MemoryModule, block.attn.neuron_out).set_stateful(False)

        full_hiddens: torch.Tensor = torch.stack(full_hiddens_list, dim=1)

        final_features: torch.Tensor = x.mean(dim=1)

        if return_full_hiddens:
            return full_hiddens, torch.tensor(0.0, device=device), full_mems

        logits: torch.Tensor = self.output_projection(final_features)

        return logits, avg_spikes, full_mems
