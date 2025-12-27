# ファイルパス: snn_research/core/layers/logic_gated_snn_v2_1.py
# タイトル: SCAL v2.1 - 改善版（閾値制御・マルチスケール）
# 改善点: 閾値適応の安定化、マルチスケール特徴、温度制御
# 修正履歴: v_th_init の保存漏れを修正

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

class MultiScaleFeatureExtractor(nn.Module):
    """
    マルチスケール特徴抽出
    異なる解像度でパターンを捕捉し、ノイズ耐性を向上
    """
    
    def __init__(self, input_dim: int, scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] in {0, 1}
        Returns:
            features: [batch, input_dim * len(scales)]
        """
        # バイポーラ変換
        x_bipolar = (x - 0.5) * 2.0
        
        features = []
        
        for scale in self.scales:
            if scale == 1:
                features.append(x_bipolar)
            else:
                # ダウンサンプリング -> アップサンプリング
                # 異なるスケールでのスムージング効果
                pooled = F.avg_pool1d(
                    x_bipolar.unsqueeze(1), 
                    kernel_size=scale, 
                    stride=scale
                )
                upsampled = F.interpolate(
                    pooled, 
                    size=self.input_dim, 
                    mode='linear', 
                    align_corners=False
                )
                features.append(upsampled.squeeze(1))
        
        return torch.cat(features, dim=1)


class ImprovedPhaseCriticalSCAL(nn.Module):
    """
    SCAL v2.1: 改善版Phase-Critical実装
    
    主要改善:
    1. 閾値適応の安定化（指数移動平均、非線形制御）
    2. スパイク率調整の強化
    3. 温度制御の改善
    4. マルチスケール特徴（オプション）
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'readout',
        # Phase-Critical パラメータ
        gamma: float = 0.008,              # 閾値適応率（低減）
        v_th_init: float = 0.8,            # 初期閾値（上昇）
        v_th_min: float = 0.3,             # 閾値下限（上昇）
        v_th_max: float = 1.5,             # 閾値上限
        temperature_base: float = 0.15,
        target_spike_rate: float = 0.15,
        # 新規パラメータ
        use_multiscale: bool = False,      # マルチスケール特徴
        variance_ema_factor: float = 0.95, # 分散EMA係数
        spike_rate_control_strength: float = 0.02  # スパイク率制御強度
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        
        # パラメータ保存 (ここを修正: self.v_th_init を追加)
        self.gamma = gamma
        self.v_th_init = v_th_init  # 追加
        self.v_th_min = v_th_min
        self.v_th_max = v_th_max
        self.temperature_base = temperature_base
        self.target_spike_rate = target_spike_rate
        self.variance_ema_factor = variance_ema_factor
        self.spike_rate_control_strength = spike_rate_control_strength
        
        # マルチスケール特徴
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale = MultiScaleFeatureExtractor(in_features, scales=(1, 2, 4))
            effective_in_features = in_features * 3
        else:
            effective_in_features = in_features
        
        # 重みの初期化
        w = torch.empty(out_features, effective_in_features)
        nn.init.orthogonal_(w, gain=1.0)
        w = w * (0.15 / math.sqrt(effective_in_features))
        
        self.register_buffer('synapse_weights', w)
        self.register_buffer('momentum_buffer', torch.zeros_like(w))
        
        # Phase-Critical バッファ
        self.register_buffer('adaptive_threshold', 
                           torch.full((out_features,), v_th_init))
        self.register_buffer('class_variance_memory', 
                           torch.ones(out_features))
        self.register_buffer('membrane_potential', 
                           torch.zeros(out_features))
        self.register_buffer('spike_history', 
                           torch.zeros(out_features))
        
        # 追加: スパイク率履歴（EMA）
        self.register_buffer('spike_rate_ema', 
                           torch.tensor(target_spike_rate))
        
        # 統計
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
        batch_size = x.size(0)
        
        # マルチスケール特徴抽出（オプション）
        if self.use_multiscale:
            x_features = self.multiscale(x)
        else:
            x_features = (x - 0.5) * 2.0  # バイポーラ変換
        
        # 正規化コサイン類似度
        x_norm = F.normalize(x_features, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.synapse_weights, p=2, dim=1, eps=1e-8)
        
        cosine_sim = torch.matmul(x_norm, w_norm.t())
        
        # 適応的ゲイン
        # 高ノイズ時に自動的にゲインを上げる
        signal_strength = cosine_sim.abs().max(dim=1, keepdim=True)[0]
        adaptive_gain = 30.0 + 70.0 * (1.0 - signal_strength.clamp(0.0, 1.0))
        
        scaled_sim = cosine_sim * adaptive_gain
        
        # 膜電位
        v_current = scaled_sim
        
        # 改善された温度計算
        # 分散だけでなく、現在の閾値も考慮
        base_temp = self.temperature_base
        variance_factor = (1.0 + 0.5 * self.class_variance_memory.unsqueeze(0))
        # ここで self.v_th_init が必要
        threshold_factor = (self.adaptive_threshold.unsqueeze(0) / self.v_th_init)
        
        temperature = base_temp * variance_factor * threshold_factor
        
        threshold_broadcast = self.adaptive_threshold.unsqueeze(0)
        
        # スパイク確率
        spike_logits = (v_current - threshold_broadcast) / (temperature + 1e-8)
        spike_prob = torch.sigmoid(spike_logits)
        
        # 確率的発火
        if self.training:
            # Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid((torch.log(spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            # ベルヌーイサンプリング
            spikes = (torch.rand_like(spike_prob) < spike_prob).float()
        
        # 出力
        output = F.softmax(v_current, dim=1)
        
        # 統計更新
        with torch.no_grad():
            self.membrane_potential.copy_(v_current.mean(dim=0))
            self.spike_history.copy_(spikes.mean(dim=0))
            
            current_spike_rate = spikes.mean().item()
            # EMA更新
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
    
    def update_plasticity(
        self,
        pre_activity: torch.Tensor,
        post_output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        learning_rate: Optional[float] = 0.02
    ):
        with torch.no_grad():
            batch_size = pre_activity.size(0)
            
            # Target one-hot
            target_onehot = torch.zeros(batch_size, self.out_features,
                                       device=pre_activity.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # === Mode 1: Weight Update ===
            
            # マルチスケール特徴
            if self.use_multiscale:
                pre_features = self.multiscale(pre_activity)
            else:
                pre_features = (pre_activity - 0.5) * 2.0
            
            # クラス別重心
            class_sums = torch.matmul(target_onehot.t(), pre_features)
            class_counts = target_onehot.sum(dim=0, keepdim=True).t() + 1e-8
            
            batch_centroids = class_sums / class_counts
            batch_centroids = F.normalize(batch_centroids, p=2, dim=1)
            
            current_weights_norm = F.normalize(self.synapse_weights, p=2, dim=1)
            delta = batch_centroids - current_weights_norm
            
            update_mask = (class_counts.squeeze() > 0).float().unsqueeze(1)
            delta = delta * update_mask
            
            # Momentum更新
            self.momentum_buffer.mul_(0.95).add_(delta)
            self.synapse_weights.add_(self.momentum_buffer * learning_rate)
            
            # 正規化
            self.synapse_weights.div_(
                self.synapse_weights.norm(p=2, dim=1, keepdim=True) + 1e-8
            )
            
            # === Mode 2: Improved Threshold Adaptation ===
            
            # クラス内分散計算（より安定）
            class_variance = torch.zeros(self.out_features, device=pre_activity.device)
            
            for c in range(self.out_features):
                mask = (target == c)
                count = mask.sum()
                if count > 1:
                    class_samples = pre_features[mask]
                    # 正規化後の分散
                    class_samples_norm = F.normalize(class_samples, p=2, dim=1)
                    variance = class_samples_norm.var(dim=0).mean()
                    class_variance[c] = variance.clamp(0.0, 2.0)
            
            # 分散メモリ更新（より強いEMA）
            self.class_variance_memory.mul_(self.variance_ema_factor).add_(
                class_variance * (1.0 - self.variance_ema_factor)
            )
            
            # 改善された閾値適応
            # 1. 分散ベース調整（非線形）
            variance_norm = self.class_variance_memory.clamp(0.0, 2.0)
            # sigmoid変換で過度な変化を抑制
            variance_effect = torch.sigmoid((variance_norm - 1.0) * 2.0)
            threshold_factor_variance = 1.0 - self.gamma * variance_effect
            
            # 2. スパイク率ベース調整（グローバル）
            spike_rate_error = self.spike_rate_ema.item() - self.target_spike_rate
            threshold_factor_spike = 1.0 + self.spike_rate_control_strength * spike_rate_error
            
            # 統合
            combined_factor = threshold_factor_variance * threshold_factor_spike
            combined_factor = combined_factor.clamp(0.98, 1.02)
            
            self.adaptive_threshold.mul_(combined_factor)
            self.adaptive_threshold.clamp_(self.v_th_min, self.v_th_max)
    
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


# Backward compatibility wrapper
class PhaseCriticalSCAL(ImprovedPhaseCriticalSCAL):
    """Alias for compatibility"""
    pass


class LogicGatedSNN(nn.Module):
    """Legacy wrapper"""
    
    def __init__(self, in_features: int, out_features: int,
                 max_states: int = 100, mode: str = 'reservoir'):
        super().__init__()
        
        if mode == 'readout':
            self.core = ImprovedPhaseCriticalSCAL(
                in_features, out_features, mode='readout',
                gamma=0.008,
                v_th_init=0.8,
                v_th_min=0.3,
                use_multiscale=False  # デフォルトはFalse（高速）
            )
        else:
            self.core = ImprovedPhaseCriticalSCAL(
                in_features, out_features, mode='reservoir',
                gamma=0.005,
                v_th_init=0.6,
                use_multiscale=False
            )
        
        self.mode = mode
    
    @property
    def membrane_potential(self):
        return self.core.membrane_potential
    
    def reset_state(self):
        self.core.reset_state()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.core(x)
        return result['output']
    
    def update_plasticity(self, pre_spikes, post_spikes, reward, learning_rate=0.02):
        if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
            target = reward.argmax(dim=1)
        else:
            target = post_spikes.argmax(dim=1)
        
        result = self.core(pre_spikes)
        self.core.update_plasticity(pre_spikes, result, target, learning_rate)
