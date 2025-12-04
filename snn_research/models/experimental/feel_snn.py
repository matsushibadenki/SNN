# ファイルパス: snn_research/architectures/feel_snn.py
# Title: FEEL-SNN (Frequency Encoded Evolutionary Leak SNN) (修正版)
# Description:
#   周波数エンコーディングと進化的リークニューロンを組み合わせた堅牢なSNNモデル。
#   修正: mypyエラー [syntax] を解消するため、type: ignore の記述を修正。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Union, cast

from snn_research.core.base import BaseModel
from snn_research.core.neurons.feel_neuron import EvolutionaryLeakLIF
from snn_research.core.layers.frequency_encoding import FrequencyEncodingLayer
# --- ▼ 修正: [import-untyped] を削除し、汎用的な ignore に変更 ▼ ---
from spikingjelly.activation_based import functional as SJ_F # type: ignore
# --- ▲ 修正 ▲ ---

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
            
        # EL-LIF パラメータの抽出
        initial_tau = neuron_config.get('initial_tau', 2.0)
        v_threshold = neuron_config.get('v_threshold', 1.0)
        learn_threshold = neuron_config.get('learn_threshold', False)

        # 1. Frequency Encoding Layer (FEEL)
        self.freq_encoder = FrequencyEncodingLayer(time_steps=time_steps)
        
        # 2. Feature Extractor (Conv + EL-LIF)
        # シンプルなVGGライクな構造
        self.features = nn.Sequential(
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
        )
        
        # 3. Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            EvolutionaryLeakLIF(features=256, initial_tau=initial_tau, v_threshold=v_threshold, learn_threshold=learn_threshold),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Any:
        # input_images: (B, C, H, W)
        
        # リセット
        SJ_F.reset_net(self)
        
        # 1. 周波数エンコーディング (B, T, C, H, W)
        encoded_inputs = self.freq_encoder(input_images)
        
        # ステートフルモード設定
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(True) # type: ignore

        # 時間ステップループ
        outputs = []
        for t in range(self.time_steps):
            # (B, C, H, W)
            x_t = encoded_inputs[:, t, ...]
            
            feat = self.features(x_t)
            flat = self.flatten(feat)
            out = self.classifier(flat)
            outputs.append(out)
            
        # ステートフル解除
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(False) # type: ignore
        
        # 時間平均ロジット
        logits = torch.stack(outputs).mean(dim=0)
        
        # 統計
        avg_spikes = 0.0
        if return_spikes:
            avg_spikes = self.get_total_spikes() / (input_images.shape[0] * self.time_steps)
            
        mem = torch.tensor(0.0, device=input_images.device)
        
        return logits, torch.tensor(avg_spikes, device=input_images.device), mem