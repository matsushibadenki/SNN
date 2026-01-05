# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# タイトル: Phase-Critical SCAL - 真の閾値適応実装
# 修正: ドキュメントの数式を忠実に実装し、スパイク発火を実現

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class PhaseCriticalSCAL(nn.Module):
    """
    Phase-Critical Statistical Centroid Alignment Learning

    Key Features:
    1. Bipolar transformation for noise cancellation
    2. Variance-driven threshold adaptation
    3. True stochastic spiking behavior
    4. Temperature-scaled spike probability
    """

    membrane_potential: torch.Tensor
    synapse_weights: torch.Tensor
    adaptive_threshold: torch.Tensor
    class_variance_memory: torch.Tensor
    momentum_buffer: torch.Tensor
    spike_history: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = 'readout',
        # Phase-Critical パラメータ
        gamma: float = 0.01,           # 閾値適応率
        v_th_init: float = 0.5,        # 初期閾値
        v_th_min: float = 0.1,         # 閾値下限
        v_th_max: float = 2.0,         # 閾値上限
        temperature_base: float = 0.1,  # 温度パラメータ
        target_spike_rate: float = 0.15,  # 目標発火率 (15%)
        # SCAL パラメータ
        momentum: float = 0.95,
        learning_rate: float = 0.02
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # Phase-Critical パラメータ
        self.gamma = gamma
        self.v_th_min = v_th_min
        self.v_th_max = v_th_max
        self.temperature_base = temperature_base
        self.target_spike_rate = target_spike_rate

        # SCAL パラメータ
        self.momentum_coef = momentum
        self.lr = learning_rate

        # 重みの初期化（直交基底）
        w = torch.empty(out_features, in_features)
        nn.init.orthogonal_(w, gain=1.0)
        w = w * (0.1 / math.sqrt(in_features))

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

        # 統計量追跡
        self.stats = {
            'spike_rate': 0.0,
            'mean_threshold': v_th_init,
            'mean_variance': 1.0,
            'temperature': temperature_base
        }

    def reset_state(self):
        """状態のリセット"""
        self.membrane_potential.zero_()
        self.spike_history.zero_()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with true spiking behavior

        Returns:
            Dict with keys: 'output', 'spikes', 'membrane_potential'
        """

        # Step 1: バイポーラ変換
        x_bipolar = (x - 0.5) * 2.0  # {0,1} -> {-1,1}

        # Step 2: 正規化コサイン類似度
        x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.synapse_weights, p=2, dim=1, eps=1e-8)

        cosine_sim = torch.matmul(x_norm, w_norm.t())

        # Step 3: 線形ゲイン（高コントラスト）
        # ドキュメントの "High-Gain Linear Contrast"
        gain = 50.0
        scaled_sim = cosine_sim * gain

        # Step 4: 膜電位の更新
        # V(t+1) = β*V(t) + scaled_sim
        beta = 0.0 if not self.training else 0.0  # リーク無し（簡略化）
        v_current = beta * self.membrane_potential.unsqueeze(0) + scaled_sim

        # Step 5: Temperature-scaled spike probability
        # P(spike) = σ((V - V_th) / T_c)

        # クラス分散に比例する温度
        temperature = self.temperature_base * \
            (1.0 + self.class_variance_memory.unsqueeze(0))

        threshold_broadcast = self.adaptive_threshold.unsqueeze(0)

        spike_logits = (v_current - threshold_broadcast) / (temperature + 1e-8)
        spike_prob = torch.sigmoid(spike_logits)

        # Step 6: 確率的発火
        if self.training:
            # Gumbel-Softmax trick for differentiable sampling
            gumbel_noise = - \
                torch.log(-torch.log(torch.rand_like(spike_prob) + 1e-8) + 1e-8)
            spikes = torch.sigmoid(
                (torch.log(spike_prob + 1e-8) + gumbel_noise) / 0.1)
        else:
            # テスト時: 確率的サンプリング
            spikes = (torch.rand_like(spike_prob) < spike_prob).float()

        # Step 7: 出力（Softmax分布）
        output = F.softmax(v_current, dim=1)

        # 統計更新
        with torch.no_grad():
            self.membrane_potential.copy_(v_current.mean(dim=0))
            self.spike_history.copy_(spikes.mean(dim=0))

            self.stats['spike_rate'] = spikes.mean().item()
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
        learning_rate: Optional[float] = None
    ):
        """
        Phase-Critical Plasticity Update

        Dual-Mode:
        1. Weight update (Statistical centroid)
        2. Threshold adaptation (Variance-driven)
        """

        if learning_rate is None:
            learning_rate = self.lr

        with torch.no_grad():
            batch_size = pre_activity.size(0)

            # Target one-hot
            target_onehot = torch.zeros(batch_size, self.out_features,
                                        device=pre_activity.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            # === Mode 1: Weight Update (Centroid) ===

            # バイポーラ変換
            pre_bipolar = (pre_activity - 0.5) * 2.0

            # クラス別重心計算
            class_sums = torch.matmul(target_onehot.t(), pre_bipolar)
            class_counts = target_onehot.sum(dim=0, keepdim=True).t() + 1e-8

            batch_centroids = class_sums / class_counts
            batch_centroids = F.normalize(batch_centroids, p=2, dim=1)

            # 重み更新（Momentum SGD）
            current_weights_norm = F.normalize(
                self.synapse_weights, p=2, dim=1)
            delta = batch_centroids - current_weights_norm

            # 更新マスク（サンプルがあるクラスのみ）
            update_mask = (class_counts.squeeze() > 0).float().unsqueeze(1)
            delta = delta * update_mask

            # Momentum更新
            self.momentum_buffer.mul_(self.momentum_coef).add_(delta)
            self.synapse_weights.add_(self.momentum_buffer * learning_rate)

            # 正規化
            self.synapse_weights.div_(
                self.synapse_weights.norm(p=2, dim=1, keepdim=True) + 1e-8
            )

            # === Mode 2: Threshold Adaptation (Variance-driven) ===

            # クラス内分散の計算
            class_variance = torch.zeros(
                self.out_features, device=pre_activity.device)

            for c in range(self.out_features):
                mask = (target == c)
                if mask.sum() > 1:
                    class_samples = pre_bipolar[mask]
                    # 分散 = E[x^2] - E[x]^2
                    variance = class_samples.var(dim=0).mean()
                    class_variance[c] = variance

            # 分散メモリの更新（EMA）
            self.class_variance_memory.mul_(0.9).add_(class_variance * 0.1)

            # 閾値適応則（ドキュメントの数式）
            # V_th(t+1) = V_th(t) * (1 - γ * ||Σ_c||_F)

            variance_norm = self.class_variance_memory.clamp(min=0.0, max=5.0)

            # 分散が大きい → 閾値を下げる（より発火しやすく）
            threshold_factor = 1.0 - self.gamma * variance_norm
            threshold_factor = threshold_factor.clamp(min=0.95, max=1.05)

            self.adaptive_threshold.mul_(threshold_factor)
            self.adaptive_threshold.clamp_(self.v_th_min, self.v_th_max)

            # === Spike Rate Regulation ===
            # 目標発火率に近づけるための追加調整

            current_spike_rate = self.spike_history.mean()
            target_rate = self.target_spike_rate

            if current_spike_rate < target_rate * 0.5:
                # 発火率が低すぎる → 閾値を下げる
                self.adaptive_threshold.mul_(0.99)
            elif current_spike_rate > target_rate * 2.0:
                # 発火率が高すぎる → 閾値を上げる
                self.adaptive_threshold.mul_(1.01)

            self.adaptive_threshold.clamp_(self.v_th_min, self.v_th_max)

    def get_phase_critical_metrics(self) -> Dict[str, float]:
        """Phase-Critical指標の取得"""
        return {
            'spike_rate': self.stats['spike_rate'],
            'mean_threshold': self.stats['mean_threshold'],
            'mean_variance': self.stats['mean_variance'],
            'temperature': self.stats['temperature'],
            'threshold_min': self.adaptive_threshold.min().item(),
            'threshold_max': self.adaptive_threshold.max().item(),
        }


class LogicGatedSNN(nn.Module):
    """
    Legacy wrapper for backward compatibility
    """

    def __init__(self, in_features: int, out_features: int,
                 max_states: int = 100, mode: str = 'reservoir'):
        super().__init__()

        if mode == 'readout':
            self.core = PhaseCriticalSCAL(
                in_features, out_features, mode='readout',
                gamma=0.01, v_th_init=0.5
            )
        else:
            # Reservoir mode: 簡易版
            self.core = PhaseCriticalSCAL(
                in_features, out_features, mode='reservoir',
                gamma=0.005, v_th_init=0.3
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
        # reward を target として解釈
        if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
            target = reward.argmax(dim=1)
        else:
            # fallback
            target = post_spikes.argmax(dim=1)

        result = self.core(pre_spikes)
        self.core.update_plasticity(pre_spikes, result, target, learning_rate)
