# ファイルパス: snn_research/models/experimental/feel_snn.py
# Title: FEEL-SNN (Frequency Encoded Evolutionary Leak SNN) (修正版)
# Description:
#   周波数エンコーディングと進化的リークニューロンを組み合わせた堅牢なSNNモデル。
#   修正内容: 
#   - nn.Sequential 内でタプル (spike, mem) を返すニューロン層を使用するとエラーになるため、
#     nn.ModuleList を使用して forward メソッドで明示的に制御するように構造を変更。
#   - forwardメソッド内で、get_total_spikes() を set_stateful(False) の前に移動 (重大バグ修正)。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from snn_research.core.base import BaseModel
from snn_research.core.neurons.feel_neuron import EvolutionaryLeakLIF
from snn_research.core.layers.frequency_encoding import FrequencyEncodingLayer
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class FEELSNN(BaseModel):
    """
    FEEL-SNN Architecture.
    [Input] -> [Frequency Encoding] -> [Conv/EL-LIF Layers] -> [Output]
    """
    def __init__(
        self,
        num_classes: int,
        time_steps: int,
        in_channels: int = 3,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = time_steps
        
        if neuron_config is None:
            neuron_config = {}
            
        initial_tau = neuron_config.get('initial_tau', 2.0)
        v_threshold = neuron_config.get('v_threshold', 1.0)
        learn_threshold = neuron_config.get('learn_threshold', False)

        # 1. Frequency Encoding Layer (FEEL)
        self.freq_encoder = FrequencyEncodingLayer(time_steps=time_steps)
        
        # 2. Feature Extractor (Conv + EL-LIF)
        # nn.Sequential ではなく ModuleList を使用して個別に定義
        self.features_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            EvolutionaryLeakLIF(features=32, initial_tau=initial_tau, v_threshold=v_threshold, learn_threshold=learn_threshold),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            EvolutionaryLeakLIF(features=64, initial_tau=initial_tau, v_threshold=v_threshold, learn_threshold=learn_threshold),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            EvolutionaryLeakLIF(features=128, initial_tau=initial_tau, v_threshold=v_threshold, learn_threshold=learn_threshold),
            nn.AdaptiveAvgPool2d((4, 4))
        ])
        
        # 3. Classifier
        self.flatten = nn.Flatten()
        self.classifier_layers = nn.ModuleList([
            nn.Linear(128 * 4 * 4, 256),
            EvolutionaryLeakLIF(features=256, initial_tau=initial_tau, v_threshold=v_threshold, learn_threshold=learn_threshold),
            nn.Linear(256, num_classes)
        ])

        self._init_weights()

    def _forward_layers(self, layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """
        レイヤーリストを順次実行し、ニューロン層からのタプル戻り値 (spike, mem) を処理する。
        """
        for layer in layers:
            x = layer(x)
            if isinstance(x, tuple):
                # ニューロン層の場合、(spike, mem) が返るため spike のみ次へ渡す
                x = x[0]
        return x

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Any:
        # リセット
        SJ_F.reset_net(self)
        
        encoded_inputs = self.freq_encoder(input_images)
        
        # ステートフルモード設定
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(True) # type: ignore

        outputs = []
        last_mem = torch.tensor(0.0, device=input_images.device)

        for t in range(self.time_steps):
            x_t = encoded_inputs[:, t, ...]
            
            # 特徴抽出
            feat = self._forward_layers(self.features_layers, x_t)
            
            flat = self.flatten(feat)
            
            # 分類器
            out = self._forward_layers(self.classifier_layers, flat)
            
            outputs.append(out)
            
        # ループ終了後に最後のLIF層の膜電位を取得
        # classifier_layers[1] が EvolutionaryLeakLIF であることを前提
        if len(self.classifier_layers) > 1:
            lif_layer = self.classifier_layers[1]
            if hasattr(lif_layer, 'mem'):
                last_mem = lif_layer.mem # type: ignore
            elif hasattr(lif_layer, 'v'):
                last_mem = lif_layer.v # type: ignore
        
        # 時間平均ロジット
        logits = torch.stack(outputs).mean(dim=0)
        
        # --- 修正: リセット前にスパイクを集計 ---
        avg_spikes = 0.0
        if return_spikes:
            avg_spikes = self.get_total_spikes() / (input_images.shape[0] * self.time_steps)
        # ------------------------------------

        # ステートフル解除 (リセット)
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(False) # type: ignore
            
        return logits, torch.tensor(avg_spikes, device=input_images.device), last_mem