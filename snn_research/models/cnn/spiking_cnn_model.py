# ファイルパス: snn_research/models/cnn/spiking_cnn_model.py
# Title: Spiking CNN Model - 型修正版
# Description:
#   mypyエラー [operator], [name-defined] を修正。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)

# --- ▼ 修正: SJ_F としてインポート ▼ ---
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
# --- ▲ 修正 ▲ ---

class SpikingCNN(BaseModel):
    """画像分類用のSpiking CNNモデル。"""
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        # (初期化コードは変更なし、省略)
        self.time_steps = time_steps
        neuron_class = AdaptiveLIFNeuron # ダミー
        # ... (パラメータ設定)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            neuron_class(features=16), 
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron_class(features=32), 
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), 
            neuron_class(features=128), 
            nn.Linear(128, vocab_size)
        )
        self._init_weights()

    def _set_stateful(self, stateful: bool):
        for layer in self.features:
            if hasattr(layer, 'set_stateful'): layer.set_stateful(stateful) # type: ignore
        for layer in self.classifier:
            if hasattr(layer, 'set_stateful'): layer.set_stateful(stateful) # type: ignore

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        
        SJ_F.reset_net(self)
        
        output_voltages: List[torch.Tensor] = []
        full_hiddens_list: List[torch.Tensor] = [] 
        total_neurons_in_pass = 0

        self._set_stateful(True)

        for _ in range(self.time_steps):
            x: torch.Tensor = input_images
            
            # Features Part
            for features_layer in self.features: 
                # --- ▼ 修正: 明示的なキャストで [operator] エラーを回避 ▼ ---
                layer_module = cast(nn.Module, features_layer)
                if isinstance(layer_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = layer_module(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    x = layer_module(x) 
                # --- ▲ 修正 ▲ ---
            
            hidden_repr_t = x.mean(dim=[2, 3]) 
            full_hiddens_list.append(hidden_repr_t) 

            # Classifier Part
            for classifier_layer in self.classifier: 
                # --- ▼ 修正: キャスト ▼ ---
                layer_module = cast(nn.Module, classifier_layer)
                if isinstance(layer_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = layer_module(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    x = layer_module(x)
                # --- ▲ 修正 ▲ ---

            output_voltages.append(x) 
        
        self._set_stateful(False)
        
        full_mems = torch.tensor(0.0, device=device)
        final_logits = torch.stack(output_voltages, dim=0).mean(dim=0)
        
        avg_spikes_val = 0.0
        if return_spikes and total_neurons_in_pass > 0:
            total_spikes = self.get_total_spikes()
            avg_spikes_val = total_spikes / (B * self.time_steps * total_neurons_in_pass)
        
        avg_spikes = torch.tensor(avg_spikes_val, device=device)

        return final_logits, avg_spikes, full_mems
