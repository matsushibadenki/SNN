# ファイルパス: snn_research/models/cnn/spiking_cnn_model.py
# Title: Spiking CNN Model - 完全修正版
# Description:
#   画像分類用のSpiking CNNモデル。
#   修正点:
#   - __init__ メソッド内の省略・ダミー実装を修正。
#   - neuron_config に基づいて適切なニューロンクラス (LIF, Izhikevich, GLIFなど) を選択し、
#     パラメータを正しく渡すロジックを実装。
#   - forward メソッドの型ヒントとキャスト処理を維持。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union

from snn_research.core.base import BaseModel
from snn_research.core.neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)

# SJ_F としてインポート
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

class SpikingCNN(BaseModel):
    """画像分類用のSpiking CNNモデル。"""
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        
        # --- ニューロンクラスとパラメータの解決 ---
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[nn.Module]
        filtered_params: Dict[str, Any] = {}

        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
            }
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['a', 'b', 'c', 'd', 'dt']
            }
        elif neuron_type_str == 'glif':
            neuron_class = GLIFNeuron
            # GLIFはゲート入力次元が必要だが、ここでは簡易的にfeaturesと同じとする
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['base_threshold', 'gate_input_features']
            }
        elif neuron_type_str == 'tc_lif':
            neuron_class = TC_LIF
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']
            }
        elif neuron_type_str == 'dual_threshold':
            neuron_class = DualThresholdNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
            }
        else:
            # デフォルト
            neuron_class = AdaptiveLIFNeuron
            filtered_params = {
                k: v for k, v in neuron_params.items() 
                if k in ['tau_mem', 'base_threshold']
            }

        # --- レイヤー構築 ---
        # GLIFのためのgate_input_features補完
        def get_params(features: int) -> Dict[str, Any]:
            p = filtered_params.copy()
            if neuron_type_str == 'glif' and 'gate_input_features' not in p:
                p['gate_input_features'] = features
            return p

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            neuron_class(features=16, **get_params(16)), 
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron_class(features=32, **get_params(32)), 
            nn.AvgPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), 
            neuron_class(features=128, **get_params(128)), 
            nn.Linear(128, vocab_size)
        )
        
        self._init_weights()

    def _set_stateful(self, stateful: bool) -> None:
        """モデル内の全ニューロンのステートフルモードを設定"""
        for layer in self.features:
            if hasattr(layer, 'set_stateful'):
                cast(Any, layer).set_stateful(stateful)
        for layer in self.classifier:
            if hasattr(layer, 'set_stateful'):
                cast(Any, layer).set_stateful(stateful)

    def forward(
        self, 
        input_images: torch.Tensor, 
        return_spikes: bool = False, 
        output_hidden_states: bool = False, 
        return_full_hiddens: bool = False, 
        return_full_mems: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        
        SJ_F.reset_net(self)
        
        output_voltages: List[torch.Tensor] = []
        # full_hiddens_list: List[torch.Tensor] = [] # 未使用のため削除または必要に応じて復活
        total_neurons_in_pass = 0

        self._set_stateful(True)

        for _ in range(self.time_steps):
            x: torch.Tensor = input_images
            
            # Features Part
            for features_layer in self.features: 
                layer_module = cast(nn.Module, features_layer)
                if isinstance(layer_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    # ニューロン層の出力は (spikes, mem)
                    spikes, _ = layer_module(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    # Conv, Pool などの通常層
                    x = layer_module(x)
            
            # 特徴マップの平均などを保存する場合
            # hidden_repr_t = x.mean(dim=[2, 3]) 
            # full_hiddens_list.append(hidden_repr_t) 

            # Classifier Part
            for classifier_layer in self.classifier: 
                layer_module = cast(nn.Module, classifier_layer)
                if isinstance(layer_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, TC_LIF, DualThresholdNeuron)): 
                    spikes, _ = layer_module(x) # type: ignore[operator]
                    x = spikes
                    if return_spikes:
                        total_neurons_in_pass += x[0].numel()
                else:
                    x = layer_module(x)

            # 最終層の出力 (Linearの出力など) を蓄積
            output_voltages.append(x) 
        
        self._set_stateful(False)
        
        full_mems = torch.tensor(0.0, device=device)
        # 時間方向の平均を最終的なロジットとする
        final_logits = torch.stack(output_voltages, dim=0).mean(dim=0)
        
        avg_spikes_val = 0.0
        if return_spikes and total_neurons_in_pass > 0:
            total_spikes = self.get_total_spikes()
            # バッチサイズ * タイムステップ * 1ステップあたりのニューロン数 で正規化
            avg_spikes_val = total_spikes / (B * self.time_steps * total_neurons_in_pass)
        
        avg_spikes = torch.tensor(avg_spikes_val, device=device)

        return final_logits, avg_spikes, full_mems