# ファイルパス: snn_research/core/layers/thermodynamic.py
# 日本語タイトル: Thermodynamic Sampling Layer (TSU)
# 機能説明: 
#   ドキュメントにある「熱力学的コンピューティング」を実装するレイヤー。
#   決定論的な演算ではなく、エネルギー障壁と熱ノイズを用いた確率的サンプリングを行う。

import torch
import torch.nn as nn
import math
from typing import Optional # 修正: 追加

class ThermodynamicSamplingLayer(nn.Module):
    """
    Thermodynamic Sampling Layer.
    
    Operation:
        Output ~ Boltzmann(Energy(x) / Temperature)
    
    Dynamics:
        Langevin Dynamicsを用いて、エネルギー地形上のサンプリングを行う。
        x_new = x_old - step * grad(E) + sqrt(2 * step * T) * noise
    """
    def __init__(self, features: int, temperature: float = 1.0, steps: int = 5, dt: float = 0.1):
        super().__init__()
        self.features = features
        self.temperature = temperature
        self.steps = steps
        self.dt = dt
        
        # エネルギー地形を定義する重み (Energy Function Parameters)
        # E(x) = 0.5 * x^T W x + b^T x などの二次形式を想定
        self.weight = nn.Parameter(torch.randn(features, features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(features))
        
        # 対称性を強制するための工夫（エネルギー関数は対称行列で定義されることが多い）
        self.register_buffer('mask', 1 - torch.eye(features))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        状態 x におけるエネルギー E(x) を計算。
        Hopfield Network的なエネルギー定義: E = -0.5 * x W x - b x
        """
        # 重みの対称化: W_sym = 0.5 * (W + W^T)
        W_sym = 0.5 * (self.weight + self.weight.t())
        
        # 二次項: x^T W x
        # (B, F) @ (F, F) -> (B, F) * (B, F) -> sum -> (B,)
        quad = 0.5 * torch.sum(torch.matmul(x, W_sym) * x, dim=1)
        
        # 線形項: b^T x
        lin = torch.sum(self.bias * x, dim=1)
        
        # エネルギー (低いほど安定)
        return - (quad + lin)

    def energy_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        エネルギーの勾配 dE/dx を計算。
        dE/dx = - (W x + b)
        """
        W_sym = 0.5 * (self.weight + self.weight.t())
        return - (torch.matmul(x, W_sym) + self.bias)

    def forward(self, x_init: torch.Tensor, external_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Langevin Dynamicsによるサンプリング。
        
        Args:
            x_init: 初期状態 (入力)
            external_field: 外部からの入力 (Bias項に加算される)
        """
        x = x_init.clone()
        
        # Langevin Sampling Loop
        for _ in range(self.steps):
            # 1. 決定論的ドリフト (Gradient Descent on Energy)
            # 勾配方向へ移動（エネルギーを下げる）
            grad = self.energy_grad(x)
            if external_field is not None:
                # 外部入力は x をその方向へ引っ張る力（ポテンシャル）として作用
                # E_total = E_internal - x * external
                # dE_total/dx = grad - external
                grad = grad - external_field

            drift = -grad * self.dt
            
            # 2. 確率的拡散 (Thermal Noise)
            # 温度 T に比例したノイズ
            noise_scale = math.sqrt(2 * self.dt * self.temperature)
            diffusion = torch.randn_like(x) * noise_scale
            
            # 更新
            x = x + drift + diffusion
            
            # 3. 境界条件 (例えば -1 ~ 1 に制限する場合)
            x = torch.tanh(x) # 状態をソフトに制限

        return x

    def extra_repr(self) -> str:
        return f'features={self.features}, temp={self.temperature}, steps={self.steps}'