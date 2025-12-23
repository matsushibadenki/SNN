# ファイルパス: snn_research/core/layers/active_predictive_layer.py
# 日本語タイトル: 能動的予測サンプリングレイヤー (ActivePredictiveLayer)
# 目的: 予測符号化の誤差を用いて熱力学的サンプリングの温度を動的に制御し、不確実性下での推論精度を向上させる。

import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from snn_research.core.layers.thermodynamic import ThermodynamicSamplingLayer

class ActivePredictiveLayer(nn.Module):
    """
    Active Predictive Layer.
    
    Predictive Coding (PC) と Thermodynamic Sampling (TSU) を統合。
    内部モデルの予測誤差 F (Free Energy) に基づいてサンプリング温度 T を制御する。
    T ∝ F (誤差が大きいほど、熱ノイズによる探索を強化)
    """
    def __init__(self, features: int, base_temperature: float = 0.1, steps: int = 10):
        super().__init__()
        self.features = features
        # 熱力学的サンプリングユニット
        self.tsu = ThermodynamicSamplingLayer(features, temperature=base_temperature, steps=steps)
        
        # 予測のための内部結合 (1.58-bit的な粗い重み)
        self.internal_weights = nn.Parameter(torch.randn(features, features) * 0.01)
        self.precision = nn.Parameter(torch.ones(features)) # 信頼度 (逆分散)
        
        self.last_error = None

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        推論プロセス:
        1. 入力から予測値を生成
        2. 予測誤差 (エネルギー) の計算
        3. 誤差に基づいて TSU の温度を調整
        4. TSU による確率的サンプリングを実行
        """
        batch_size = x_input.size(0)
        
        # 1. 予測の生成 (内部モデル)
        # 簡易的な線形予測: x_pred = f(W * x_prev)
        # 本来は時間軸に沿った推論が必要だが、ここでは入力をポテンシャルとして扱う
        prediction = torch.tanh(torch.matmul(x_input, self.internal_weights))
        
        # 2. 予測誤差の計算 (L2ノルム)
        error = x_input - prediction
        free_energy = torch.mean(error**2, dim=1) # (Batch,)
        
        # 3. 動的温度制御 (能動的推論の核)
        # 誤差が大きいほど温度を上げ、熱ノイズを増やす
        # T_active = T_base * (1 + alpha * FreeEnergy)
        dynamic_temp = self.tsu.temperature * (1.0 + free_energy.mean() * 10.0)
        
        # 4. 熱力学的サンプリング (Langevin Dynamics)
        # 外部場 (external_field) として入力を与え、エネルギー最小値を探る
        sampled_output = self._langevin_sampling(x_input, dynamic_temp)
        
        self.last_error = error.detach()
        return sampled_output

    def _langevin_sampling(self, external_field: torch.Tensor, temp: float) -> torch.Tensor:
        """TSUのサンプリングループを動的温度で実行"""
        x = torch.zeros_like(external_field)
        dt = 0.1
        
        for _ in range(self.tsu.steps):
            # エネルギー勾配 (内部結合による安定化)
            grad = self.tsu.energy_grad(x) - external_field
            
            # ドリフト項
            drift = -grad * dt
            
            # 確率的拡散項 (動的に計算された温度を使用)
            noise_scale = math.sqrt(2 * dt * temp)
            diffusion = torch.randn_like(x) * noise_scale
            
            x = torch.tanh(x + drift + diffusion)
            
        return x

    def extra_repr(self) -> str:
        return f'features={self.features}, steps={self.tsu.steps}'