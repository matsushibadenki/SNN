# ファイルパス: snn_research/core/layers/active_predictive_layer.py
# 日本語タイトル: 能動的予測サンプリングレイヤー (感度調整版)

import torch
import torch.nn as nn
import math
from typing import Optional
from snn_research.core.layers.thermodynamic import ThermodynamicSamplingLayer

class ActivePredictiveLayer(nn.Module):
    def __init__(self, features: int, base_temperature: float = 0.5, steps: int = 10) -> None:
        super().__init__()
        self.features = features
        # 修正1: ベース温度を少し上げ、初期の探索範囲を広げる
        self.tsu = ThermodynamicSamplingLayer(features, temperature=base_temperature, steps=steps)
        
        self.internal_weights = nn.Parameter(torch.randn(features, features) * 0.1)
        self.last_error: Optional[torch.Tensor] = None

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        # 入力を正規化してスケールを安定させる
        x_norm = nn.functional.normalize(x_input, p=2, dim=-1)
        
        prediction = torch.tanh(torch.matmul(x_norm, self.internal_weights))
        error = x_norm - prediction
        
        # 修正2: 誤差感度を調整
        free_energy = torch.mean(error**2)
        dynamic_temp: float = self.tsu.temperature * (1.0 + float(free_energy.item()) * 20.0)
        
        sampled_output = self._langevin_sampling(x_norm, dynamic_temp)
        self.last_error = error.detach()
        return sampled_output

    def _langevin_sampling(self, external_field: torch.Tensor, temp: float) -> torch.Tensor:
        x = external_field.clone() # 入力からスタートすることで収束を速める
        dt = 0.2 # ステップサイズを拡大
        steps: int = self.tsu.steps
        
        for _ in range(steps):
            grad = self.tsu.energy_grad(x) - external_field
            drift = -grad * dt
            
            noise_scale = math.sqrt(2 * dt * temp)
            diffusion = torch.randn_like(x) * noise_scale
            
            # ソフトな非線形性を維持しつつ更新
            x = torch.tanh(x + drift + diffusion)
            
        return x