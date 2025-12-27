# ファイルパス: snn_research/core/layers/logic_gated_snn_v2_1.py
# タイトル: SCAL v2.1 - Spatial Optimized
# 内容: エンコーダ変更への対応と、適応的学習率の導入

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
        gamma: float = 0.005,
        v_th_init: float = 1.0,
        v_th_min: float = 0.1,
        v_th_max: float = 15.0,
        temperature_base: float = 0.15,
        target_spike_rate: float = 0.15,
        use_multiscale: bool = True,
        variance_ema_factor: float = 0.95,
        spike_rate_control_strength: float = 0.1
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
        
        # ゲイン設定: Pattern Separatorの出力は既に整形されているので
        # 過度なゲインは不要だが、決定境界を鋭くするために適度にかける
        signal_strength = cosine_sim.abs().max(dim=1, keepdim=True)[0]
        adaptive_gain = 15.0 + 15.0 * (1.0 - signal_strength.clamp(0.0, 1.0))
        
        v_current = cosine_sim * adaptive_gain
        
        base_temp = self.temperature_base
        variance_factor = (1.0 + 0.2 * self.class_variance_memory.unsqueeze(0))
        threshold_factor = (self.adaptive_threshold.unsqueeze(0) / self.v_th_init)
        temperature = base_temp * variance_factor * threshold_factor
        
        spike_logits = (v_current - self.adaptive_threshold.unsqueeze(0)) / (temperature + 1e-8)
        spike_prob = torch.sigmoid(spike_logits)
        
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
            
            # === 適応的学習率 ===
            # ノイズが多く、確信度が低い場合は学習率を下げる
            # output shape: [batch, classes]
            output = post_output['output']
            confidence, _ = output.max(dim=1)
            # 確信度が低いサンプルに対する学習率を減衰させる (間違った教師信号への過学習防止)
            # confidence: 0.1~1.0 -> adaptive_lr_factor: 0.1~1.0
            adaptive_lr_factor = confidence.unsqueeze(1)
            
            class_sums = torch.matmul(target_onehot.t(), pre_features * adaptive_lr_factor)
            # 重み付きカウント
            class_counts = torch.matmul(target_onehot.t(), adaptive_lr_factor).squeeze() + 1e-8
            class_counts = class_counts.unsqueeze(1)
            
            batch_centroids = F.normalize(class_sums / class_counts, p=2, dim=1)
            
            current_weights_norm = F.normalize(self.synapse_weights, p=2, dim=1)
            delta = batch_centroids - current_weights_norm
            update_mask = (class_counts.squeeze() > 0.01).float().unsqueeze(1)
            
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
            
            variance_effect = torch.sigmoid((self.class_variance_memory - 1.0))
            threshold_factor_variance = 1.0 + self.gamma * variance_effect
            
            spike_rate_error = self.spike_rate_ema.item() - self.target_spike_rate
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
