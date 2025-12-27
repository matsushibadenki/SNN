# ファイルパス: snn_research/core/layers/logic_gated_snn_v2_1.py
# タイトル: SCAL v2.1 - バランス調整版
# 内容: 過度な抑制を排除し、信号透過率を向上

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Optional, Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from snn_research.utils.advanced_encoding import HybridEncoder
except ImportError:
    print("Warning: advanced_encoding.py not found. Using fallback encoder.")
    class HybridEncoder(nn.Module):
        def __init__(self, input_dim, **kwargs):
            super().__init__()
            self.output_dim = input_dim
        def forward(self, x):
            return (x - 0.5) * 2.0

class ImprovedPhaseCriticalSCAL(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'readout',
        gamma: float = 0.005,              # 0.008 -> 0.005 (閾値上昇をマイルドに)
        v_th_init: float = 1.0,
        v_th_min: float = 0.1,
        v_th_max: float = 15.0,
        temperature_base: float = 0.15,
        target_spike_rate: float = 0.15,
        use_multiscale: bool = True,
        variance_ema_factor: float = 0.95,
        spike_rate_control_strength: float = 0.1  # 0.05 -> 0.1 (目標スパイク率への追従を強化)
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        
        self.gamma = gamma
        self.v_th_init = v_th_init
        self.v_th_min = v_th_min
        self.v_th_max = v_th_max
        self.temperature_base = temperature_base
        self.target_spike_rate = target_spike_rate
        self.variance_ema_factor = variance_ema_factor
        self.spike_rate_control_strength = spike_rate_control_strength
        
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.encoder = HybridEncoder(in_features)
            effective_in_features = self.encoder.output_dim
        else:
            self.encoder = None
            effective_in_features = in_features
        
        w = torch.empty(out_features, effective_in_features)
        nn.init.orthogonal_(w, gain=1.0)
        w = w * (1.0 / math.sqrt(effective_in_features))
        
        self.register_buffer('synapse_weights', w)
        self.register_buffer('momentum_buffer', torch.zeros_like(w))
        
        self.register_buffer('adaptive_threshold', torch.full((out_features,), v_th_init))
        self.register_buffer('class_variance_memory', torch.ones(out_features))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('spike_history', torch.zeros(out_features))
        self.register_buffer('spike_rate_ema', torch.tensor(target_spike_rate))
        
        self.stats = {
            'spike_rate': 0.0,
            'mean_threshold': v_th_init,
            'mean_variance': 1.0,
            'temperature': temperature_base
        }
    
    def reset_state(self):
        self.membrane_potential.zero_()
        self.spike_history.zero_()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.encoder is not None:
            x_features = self.encoder(x)
        else:
            x_features = (x - 0.5) * 2.0
        
        x_norm = F.normalize(x_features, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.synapse_weights, p=2, dim=1, eps=1e-8)
        cosine_sim = torch.matmul(x_norm, w_norm.t())
        
        # 適応的ゲイン (信号強度に応じて調整)
        signal_strength = cosine_sim.abs().max(dim=1, keepdim=True)[0]
        # 信号が弱い(ノイズが多い)ときはゲインを上げて補償する
        adaptive_gain = 10.0 + 20.0 * (1.0 - signal_strength.clamp(0.0, 1.0))
        
        v_current = cosine_sim * adaptive_gain
        
        base_temp = self.temperature_base
        variance_factor = (1.0 + 0.2 * self.class_variance_memory.unsqueeze(0)) # 0.5 -> 0.2 (温度上昇を抑える)
        threshold_factor = (self.adaptive_threshold.unsqueeze(0) / self.v_th_init)
        temperature = base_temp * variance_factor * threshold_factor
        
        spike_logits = (v_current - self.adaptive_threshold.unsqueeze(0)) / (temperature + 1e-8)
        spike_prob = torch.sigmoid(spike_logits)
        
        # k-WTA (側抑制) は廃止し、純粋なSoftmax/Sigmoid確率分布を使用する
        # これによりアンサンブル時の「迷い」を有効活用できる
        
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            spikes = (torch.rand_like(spike_prob) < spike_prob).float()
        
        output = F.softmax(v_current, dim=1)
        
        with torch.no_grad():
            self.membrane_potential.copy_(v_current.mean(dim=0))
            self.spike_history.copy_(spikes.mean(dim=0))
            current_spike_rate = spikes.mean().item()
            self.spike_rate_ema.mul_(0.9).add_(current_spike_rate * 0.1)
            self.stats['spike_rate'] = current_spike_rate
            self.stats['mean_threshold'] = self.adaptive_threshold.mean().item()
            self.stats['mean_variance'] = self.class_variance_memory.mean().item()
            self.stats['temperature'] = temperature.mean().item()
        
        return {
            'output': output,
            'spikes': spikes,
            'membrane_potential': v_current,
            'spike_prob': spike_prob
        }
    
    def update_plasticity(self, pre_activity, post_output, target, learning_rate=0.02):
        with torch.no_grad():
            batch_size = pre_activity.size(0)
            target_onehot = torch.zeros(batch_size, self.out_features, device=pre_activity.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            
            if self.encoder is not None:
                pre_features = self.encoder(pre_activity)
            else:
                pre_features = (pre_activity - 0.5) * 2.0
            
            # Weight Update
            class_sums = torch.matmul(target_onehot.t(), pre_features)
            class_counts = target_onehot.sum(dim=0, keepdim=True).t() + 1e-8
            batch_centroids = F.normalize(class_sums / class_counts, p=2, dim=1)
            
            current_weights_norm = F.normalize(self.synapse_weights, p=2, dim=1)
            delta = batch_centroids - current_weights_norm
            update_mask = (class_counts.squeeze() > 0).float().unsqueeze(1)
            
            self.momentum_buffer.mul_(0.95).add_(delta * update_mask)
            self.synapse_weights.add_(self.momentum_buffer * learning_rate)
            self.synapse_weights.div_(self.synapse_weights.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # Threshold Adaptation
            class_variance = torch.zeros(self.out_features, device=pre_activity.device)
            for c in range(self.out_features):
                mask = (target == c)
                if mask.sum() > 1:
                    class_samples_norm = F.normalize(pre_features[mask], p=2, dim=1)
                    var = class_samples_norm.var(dim=0).sum()
                    class_variance[c] = var.clamp(0.0, 5.0)
            
            self.class_variance_memory.mul_(self.variance_ema_factor).add_(
                class_variance * (1.0 - self.variance_ema_factor)
            )
            
            # ロジック改善: 分散による閾値上昇を抑制
            variance_effect = torch.sigmoid((self.class_variance_memory - 1.0))
            # gamma係数を小さくしたため、上昇幅が減る
            threshold_factor_variance = 1.0 + self.gamma * variance_effect
            
            # スパイク率によるフィードバックを強化
            spike_rate_error = self.spike_rate_ema.item() - self.target_spike_rate
            # 制御強度を上げたため、スパイク率低下時(error < 0)に強力に閾値を下げる
            threshold_factor_spike = 1.0 + self.spike_rate_control_strength * spike_rate_error
            
            combined_factor = (threshold_factor_variance * threshold_factor_spike).clamp(0.98, 1.02)
            self.adaptive_threshold.mul_(combined_factor).clamp_(self.v_th_min, self.v_th_max)
    
    def get_phase_critical_metrics(self) -> Dict[str, float]:
        return {
            'spike_rate': self.stats['spike_rate'],
            'spike_rate_ema': self.spike_rate_ema.item(),
            'mean_threshold': self.stats['mean_threshold'],
            'mean_variance': self.stats['mean_variance'],
            'temperature': self.stats['temperature'],
            'threshold_min': self.adaptive_threshold.min().item(),
            'threshold_max': self.adaptive_threshold.max().item(),
        }

class PhaseCriticalSCAL(ImprovedPhaseCriticalSCAL): pass