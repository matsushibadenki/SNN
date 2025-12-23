# ファイルパス: snn_research/core/layers/active_predictive_layer.py
# 日本語タイトル: 能動的予測サンプリングレイヤー (型安全版)
# 目的: 自由エネルギーに基づく動的温度制御を、mypyの型チェックをパスする形で実装する。

import torch
import torch.nn as nn
import math
from typing import Optional, cast
from snn_research.core.layers.thermodynamic import ThermodynamicSamplingLayer

class ActivePredictiveLayer(nn.Module):
    """
    Active Predictive Layer.
    予測誤差に基づく熱力学的サンプリング。
    """
    def __init__(self, features: int, base_temperature: float = 0.1, steps: int = 10) -> None:
        super().__init__()
        self.features = features
        self.tsu = ThermodynamicSamplingLayer(features, temperature=base_temperature, steps=steps)
        
        self.internal_weights = nn.Parameter(torch.randn(features, features) * 0.01)
        self.precision = nn.Parameter(torch.ones(features))
        
        # 型アノテーションを追加
        self.last_error: Optional[torch.Tensor] = None

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """推論プロセス"""
        # 1. 予測の生成
        prediction = torch.tanh(torch.matmul(x_input, self.internal_weights))
        
        # 2. 予測誤差の計算
        error = x_input - prediction
        free_energy = torch.mean(error**2) # スカラーTensor
        
        # 3. 動的温度制御
        # dynamic_temp を明示的に float に変換
        dynamic_temp: float = self.tsu.temperature * (1.0 + float(free_energy.item()) * 10.0)
        
        # 4. 熱力学的サンプリング
        sampled_output = self._langevin_sampling(x_input, dynamic_temp)
        
        self.last_error = error.detach()
        return sampled_output

    def _langevin_sampling(self, external_field: torch.Tensor, temp: float) -> torch.Tensor:
        """Langevin Dynamicsサンプリング"""
        x = torch.zeros_like(external_field)
        dt = 0.1
        steps: int = self.tsu.steps
        
        for _ in range(steps):
            # エネルギー勾配の取得
            grad = self.tsu.energy_grad(x) - external_field
            
            # ドリフト項
            drift = -grad * dt
            
            # 拡散項
            noise_scale = math.sqrt(2 * dt * temp)
            diffusion = torch.randn_like(x) * noise_scale
            
            x = torch.tanh(x + drift + diffusion)
            
        return x

    def extra_repr(self) -> str:
        return f'features={self.features}, steps={self.tsu.steps}'
