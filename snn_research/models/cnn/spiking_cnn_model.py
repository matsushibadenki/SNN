# ファイルパス: snn_research/models/cnn/spiking_cnn_model.py
# Title: Spiking CNN Model - Stateful修正版
# Description:
# - 画像分類用のSpiking CNN。
# - 修正: 時間ループ内で膜電位を保持するため、set_stateful(True) を適用するように修正。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)

from spikingjelly.activation_based import functional # type: ignore[import-untyped]

class SpikingCNN(BaseModel):
    """
    画像分類用のSpiking CNNモデル。
    """
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        num_classes: int = vocab_size
        self.time_steps = time_steps
        
        neuron_type: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[Union[AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron]]
        
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
            neuron_params['gate_input_features'] = None 
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'base_threshold', 'gate_input_features']
            }
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF 
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron 
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
             raise ValueError(f"Unknown neuron type for SpikingCNN: {neuron_type}")

        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            neuron_class(features=16, **filtered_params), 
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron_class(features=32, **filtered_params), 
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), 
            neuron_class(features=128, **filtered_params), 
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _set_stateful(self, stateful: bool):
        """モデル内の全ニューロンのstatefulモードを切り替える"""
        # Features内のニューロン
        for layer in self.features:
            if hasattr(layer, 'set_stateful'):
                layer.set_stateful(stateful)
        # Classifier内のニューロン
        for layer in self.classifier:
            if hasattr(layer, 'set_stateful'):
                layer.set_stateful(stateful)

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, return_full_hiddens: bool = False, return_full_mems: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        
        # ネットワークのリセット
        SJ_F.reset_net(self)
        
        output_voltages: List[torch.Tensor] = []
        full_hiddens_list: List[torch.Tensor] = [] 
        
        local_mems_history: List[torch.Tensor] = []
        
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        if return_full_mems:
            def _hook_mem(module: nn.Module, input: Any, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
                if isinstance(output, tuple) and len(output) > 1:
                    local_mems_history.append(output[1]) 
            
            # フック登録ロジック（変更なし）
            target_module = None
            neuron_types = (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)
            for module in reversed(self.classifier):
                if isinstance(module, neuron_types):
                    target_module = module
                    break
            if target_module is None:
                for module in reversed(self.features):
                    if isinstance(module, neuron_types):
                        target_module = module
                        break
            if target_module is not None:
                hooks.append(target_module.register_forward_hook(_hook_mem))

        total_neurons_in_pass = 0

        # --- 時間ループ開始前に Stateful モードを有効化 ---
        self._set_stateful(True)

        # 時間ステップループ
        for _ in range(self.time_steps):
            x: torch.Tensor = input_images
            
            # Features Part
            for features_layer in self.features: 
                if isinstance(features_layer, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = features_layer(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    x = features_layer(x) # type: ignore[operator]
            
            hidden_repr_t = x.mean(dim=[2, 3]) 
            full_hiddens_list.append(hidden_repr_t) 

            # Classifier Part
            for classifier_layer in self.classifier: 
                if isinstance(classifier_layer, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = classifier_layer(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    x = classifier_layer(x) # type: ignore[operator]

            output_voltages.append(x) 
        
        # --- 時間ループ終了 ---

        full_mems: torch.Tensor
        if return_full_mems:
            for hook in hooks: hook.remove()
            if local_mems_history:
                last_layer_mems = local_mems_history[-self.time_steps:] 
                full_mems = torch.stack(last_layer_mems, dim=1).unsqueeze(2) 
            else:
                full_mems = torch.zeros(B, self.time_steps, 1, 1, device=device)
        else:
            full_mems = torch.tensor(0.0, device=device)
            

        full_hiddens_stacked: torch.Tensor = torch.stack(full_hiddens_list, dim=1) 
        full_hiddens: torch.Tensor = full_hiddens_stacked.unsqueeze(1) 

        if return_full_hiddens:
             # stateful解除を忘れずに
             self._set_stateful(False)
             return full_hiddens, torch.tensor(0.0, device=device), full_mems

        final_logits: torch.Tensor = torch.stack(output_voltages, dim=0).mean(dim=0)
        
        avg_spikes_val = 0.0
        if return_spikes and total_neurons_in_pass > 0:
            # リセット前に集計
            total_spikes = self.get_total_spikes()
            avg_spikes_val = total_spikes / (B * self.time_steps * total_neurons_in_pass)
        
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)

        # --- Stateful モードを解除 ---
        self._set_stateful(False)

        return final_logits, avg_spikes, full_mems
