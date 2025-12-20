# ファイルパス: snn_research/models/cnn/spiking_cnn_model.py
# 日本語タイトル: Spiking CNN Model (Refactored)
# ファイルの目的・内容:
#   画像分類用SNN。NeuronFactoryを使用してニューロン生成を簡素化したバージョン。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List

from snn_research.core.base import BaseModel
from snn_research.core.factories import NeuronFactory # 新しいファクトリを使用
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class SpikingCNN(BaseModel):
    """
    画像分類用のSpiking CNNモデル。
    NeuronFactoryを使用して構成可能。
    """
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        
        # --- ファクトリを使用したレイヤー構築 ---
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            NeuronFactory.create(features=16, config=neuron_config),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            NeuronFactory.create(features=32, config=neuron_config),
            nn.AvgPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), 
            NeuronFactory.create(features=128, config=neuron_config),
            nn.Linear(128, vocab_size)
        )
        
        self._init_weights()

    def forward(
        self, 
        input_images: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Any:
        
        B = input_images.shape[0]
        device = input_images.device
        SJ_F.reset_net(self)
        
        outputs = []
        total_spikes = 0.0

        # ニューロン層をステートフルにする
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(True) # type: ignore

        for _ in range(self.time_steps):
            x = input_images
            
            # 特徴抽出
            for layer in self.features:
                if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                    x = layer(x)
                else:
                    # ニューロン層 (spikes, mem)
                    x, _ = layer(x)
                    if return_spikes:
                         total_spikes += x.sum().item()

            # 分類器
            for layer in self.classifier:
                if isinstance(layer, (nn.Linear, nn.Flatten)):
                    x = layer(x)
                else:
                    x, _ = layer(x)
                    if return_spikes:
                         total_spikes += x.sum().item()

            outputs.append(x) # 最後のLinear出力

        # ステートフル解除
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                m.set_stateful(False) # type: ignore
        
        # 時間平均
        logits = torch.stack(outputs).mean(dim=0)
        
        avg_spikes = 0.0
        if return_spikes:
            # 概算: バッチ * 時間 * 全ニューロン数 で割るのが正しいが
            # 簡易的に合計を返すか、適切な正規化を行う
            avg_spikes = total_spikes / (B * self.time_steps)
            
        return logits, torch.tensor(avg_spikes, device=device), torch.tensor(0.0)