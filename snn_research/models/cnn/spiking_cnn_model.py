# ファイルパス: snn_research/models/cnn/spiking_cnn_model.py
# 日本語タイトル: Spiking CNN Model (Fix: Dynamic Input Size)
# 目的: 画像サイズに合わせてLinear層の次元を自動計算し、次元不一致エラーを防ぐ。

import torch
import torch.nn as nn
from typing import Dict, Any, cast

from snn_research.core.base import BaseModel
from snn_research.core.factories import NeuronFactory
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class SpikingCNN(BaseModel):
    """
    画像分類用のSpiking CNNモデル。
    入力画像サイズに応じてネットワーク構造を動的に調整する。
    """
    
    def __init__(self, vocab_size: int, time_steps: int, neuron_config: Dict[str, Any], img_size: int = 32, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        
        # 特徴抽出部
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            NeuronFactory.create(features=16, config=neuron_config),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            NeuronFactory.create(features=32, config=neuron_config),
            nn.AvgPool2d(2),
        )
        
        # Linear層の入力次元を自動計算
        # ダミー入力を使って畳み込み後のサイズを取得
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            # ニューロン層はステートレスとして扱われるか、初期化前なのでスルーされることを期待
            # ここではnn.Sequentialを通すが、NeuronFactoryが生成するモジュールが
            # forwardでstateful前提だとエラーになる可能性があるため、簡易計算を行う
            # Conv(32) -> Pool(16) -> Conv(16) -> Pool(8) for 32x32
            # 実際にはSequentialを通して形状確認するのが確実
            x = dummy
            for layer in self.features:
                if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                    x = layer(x)
                # ニューロン層は形状を変えない前提でスキップ（または恒等写像とみなす）
            
            flat_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128), 
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
                # [Fix] Cast to Any to satisfy mypy
                cast(Any, m).set_stateful(True)

        for _ in range(self.time_steps):
            x = input_images
            
            # 特徴抽出
            for layer in self.features:
                if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                    x = layer(x)
                else:
                    x, _ = layer(x) # (spikes, mem)
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

            outputs.append(x)

        # ステートフル解除
        for m in self.modules():
            if hasattr(m, 'set_stateful'):
                # [Fix] Cast to Any
                cast(Any, m).set_stateful(False)
        
        # 時間平均 (Logits)
        logits = torch.stack(outputs).mean(dim=0)
        
        avg_spikes = 0.0
        if return_spikes:
            avg_spikes = total_spikes / (B * self.time_steps)
            
        return logits, torch.tensor(avg_spikes, device=device), torch.tensor(0.0)