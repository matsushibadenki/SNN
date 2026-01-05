# ファイルパス: snn_research/core/layers/logic_gated_snn_v2_1.py
# タイトル: SCAL v3.1 - Stabilized Perception Layer
# 内容: ゲイン整合性の確保とホメオスタシス制御による過剰発火防止
# 修正: Mypy型エラー修正 (Explicit typing, Unchained operations)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Dict, cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

class SCALPerceptionLayer(nn.Module):
    """
    SCAL v3.1 Stabilized Phase-Critical知覚層
    仕様書の高ゲイン特性を維持しつつ、閾値制御を安定化
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'readout',
        time_steps: int = 10,
        gain: float = 50.0,
        beta_membrane: float = 0.9,
        v_th_init: float = 25.0,    # 修正: Gain(50)に合わせて引き上げ
        v_th_min: float = 5.0,      # 修正: 下限も引き上げ
        v_th_max: float = 100.0,    # 修正: 上限も引き上げ
        gamma_th: float = 0.01,
        target_spike_rate: float = 0.15, # 目標スパイク率
        spike_rate_control_strength: float = 0.05, # 制御強度
        use_multiscale: bool = True,
        # 互換性引数
        gamma: float = 0.005,
        temperature_base: float = 1.0,
        variance_ema_factor: float = 0.95,
        inhibition_strength: float = 0.2,
        sharpening_factor: float = 1.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.time_steps = time_steps
        self.gain = gain
        self.beta_membrane = beta_membrane
        
        self.v_th_min = v_th_min
        self.v_th_max = v_th_max
        self.gamma_th = gamma_th
        self.target_spike_rate = target_spike_rate
        self.spike_rate_control_strength = spike_rate_control_strength
        
        # 投影層
        self.projection = nn.Linear(in_features, in_features, bias=False)
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        
        # SCAL重心
        self.register_buffer('centroids', torch.randn(out_features, in_features))
        self.centroids: torch.Tensor # Explicit type
        self.centroids.div_(math.sqrt(in_features))
        
        # 適応パラメータ
        self.register_buffer('adaptive_threshold', torch.full((out_features,), v_th_init))
        self.adaptive_threshold: torch.Tensor # Explicit type
        
        self.register_buffer('spike_rate_ema', torch.full((out_features,), target_spike_rate))
        self.spike_rate_ema: torch.Tensor # Explicit type
        
        # 状態モニタリング
        self.stats = {
            'spike_rate': target_spike_rate,
            'mean_threshold': v_th_init,
            'temperature': 1.0,
            'threshold_min': v_th_min,
            'threshold_max': v_th_max,
            'mean_variance': 0.0
        }

    def reset_state(self):
        pass # ステートレス実装のため不要だが互換性のため残す

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        
        # 1. Bipolar変換 & 特徴抽出
        x_bipolar = 2.0 * x - 1.0
        features = self.projection(x_bipolar)
        features = torch.tanh(features)
        
        # 2. 類似度計算
        # Explicit types to avoid "Tensor | Module" confusion
        centroids_t = cast(torch.Tensor, self.centroids)
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        centroids_norm = F.normalize(centroids_t, p=2, dim=1, eps=1e-8)
        cosine_sim = torch.mm(features_norm, centroids_norm.t())
        
        # 3. 高ゲイン入力
        s_prime = self.gain * cosine_sim
        
        # 4. 時系列積分
        spike_trains = []
        V_mem_history = []
        V_current = torch.zeros(batch_size, self.out_features, device=device)
        
        th_tensor = cast(torch.Tensor, self.adaptive_threshold)
        
        for t in range(self.time_steps):
            V_current = self.beta_membrane * V_current + s_prime
            
            # 発火判定 (Soft-Spike for Grad, Hard-Spike for Output)
            # 閾値近傍での挙動を少し滑らかにする
            threshold = th_tensor.unsqueeze(0)
            surrogate_grad = torch.sigmoid((V_current - threshold))
            
            if self.training:
                # Gumbel Noise for stochasticity during training
                spikes = (surrogate_grad > torch.rand_like(surrogate_grad)).float()
            else:
                spikes = (V_current > threshold).float()
            
            # Reset
            V_current = V_current * (1.0 - spikes)
            
            spike_trains.append(spikes)
            V_mem_history.append(V_current.clone())
        
        spike_trains_stack = torch.stack(spike_trains, dim=2)
        V_mem_stack = torch.stack(V_mem_history, dim=2)
        output_rate = spike_trains_stack.mean(dim=2)
        
        # 統計更新 (EMA)
        with torch.no_grad():
            current_mean_rate = output_rate.mean(dim=0) # (out_features,)
            # Unchain to satisfy mypy
            self.spike_rate_ema.mul_(0.95)
            self.spike_rate_ema.add_(current_mean_rate * 0.05)
            
            self.stats['spike_rate'] = output_rate.mean().item()
            self.stats['mean_threshold'] = th_tensor.mean().item()
            
        return {
            'output': output_rate,
            'spikes': spike_trains_stack,
            'membrane_potential': V_mem_stack,
            'features': features
        }

    def update_plasticity(self, pre_activity, post_output, target, learning_rate=0.01):
        with torch.no_grad():
            # 特徴量再計算
            x_bipolar = 2.0 * pre_activity - 1.0
            features = torch.tanh(self.projection(x_bipolar))
            
            unique_classes = torch.unique(target)
            
            # Cast buffers
            centroids_t = cast(torch.Tensor, self.centroids)
            th_tensor = cast(torch.Tensor, self.adaptive_threshold)
            ema_tensor = cast(torch.Tensor, self.spike_rate_ema)
            
            for c in unique_classes:
                c_idx = int(c.item())
                mask = (target == c)
                class_features = features[mask]
                
                # --- 1. 重心更新 (SCAL) ---
                current_centroid = centroids_t[c_idx]
                new_centroid = class_features.mean(dim=0)
                centroids_t[c_idx] = current_centroid + learning_rate * (new_centroid - current_centroid)
                
                # --- 2. 閾値制御 (Hybrid: Variance + Homeostasis) ---
                if class_features.size(0) > 1:
                    variance = torch.var(class_features, dim=0)
                    variance_norm = torch.norm(variance)
                    var_factor = 1.0 - self.gamma_th * variance_norm
                else:
                    var_factor = torch.tensor(1.0, device=features.device)

                # B. ホメオスタシス制御 (過剰発火防止)
                rate_error = ema_tensor[c_idx] - self.target_spike_rate
                homeo_factor = 1.0 + self.spike_rate_control_strength * rate_error
                
                # 統合係数
                total_factor = var_factor * homeo_factor
                
                # 急激な変動を抑える
                total_factor = torch.clamp(total_factor, 0.95, 1.05)
                
                th_tensor[c_idx] *= total_factor
                th_tensor[c_idx] = torch.clamp(
                    th_tensor[c_idx], self.v_th_min, self.v_th_max
                )
            
            centroids_t.div_(centroids_t.norm(dim=1, keepdim=True) + 1e-8)

    def get_phase_critical_metrics(self) -> Dict[str, float]:
        th_tensor = cast(torch.Tensor, self.adaptive_threshold)
        return {
            'spike_rate': self.stats['spike_rate'],
            'mean_threshold': self.stats['mean_threshold'],
            'temperature': 1.0,
            'threshold_min': th_tensor.min().item(),
            'threshold_max': th_tensor.max().item(),
            'mean_variance': self.stats.get('mean_variance', 0.0)
        }

# 互換用
ImprovedPhaseCriticalSCAL = SCALPerceptionLayer