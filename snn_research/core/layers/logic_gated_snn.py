# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final SOTA: Annealed Relaxation)
# 修正: Top-K緩和係数のアニーリング処理を追加し、収束を保証

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファ型ヒント
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor
    adaptive_threshold: torch.Tensor
    topk_relaxation: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        SCAL (Statistical Centroid Alignment Learning) ベースのニューロモルフィック層。
        SOTA Edition: Annealed Top-K Relaxation.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 学習可能層 (Readout)
            std_dev = 0.05
            trainable = True
            
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            
            # 初期ゲイン: 10.0
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 10.0)
            self.register_buffer('topk_relaxation', torch.tensor(1.0)) 
        else:
            # リザーバー層 (Fixed)
            std_dev = 3.0 / math.sqrt(in_features)
            trainable = False
            
            if out_features >= in_features:
                w = torch.empty(out_features, in_features)
                nn.init.orthogonal_(w, gain=1.0)
                mask = (torch.rand_like(w) > 0.7).float()
                raw_states = w * mask * (std_dev * 4.0)
            else:
                raw_states = torch.randn(out_features, in_features) * std_dev

            effective_w = self._quantize_weights(raw_states)
            self.register_buffer('frozen_weight', effective_w)
            self.register_buffer('synapse_states', torch.zeros(1))
            self.register_buffer('momentum_buffer', torch.zeros(1))
            self.register_buffer('adaptive_threshold', torch.ones(1))
            
            # 初期緩和係数: 0.7 (やや緩和)
            self.register_buffer('topk_relaxation', torch.tensor(0.7)) 
            
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def _quantize_weights(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.zeros_like(x)
        threshold_val = 0.01 
        w[x > threshold_val] = 1.0
        w[x < -threshold_val] = -1.0
        return w * 0.5

    def get_effective_weights(self) -> torch.Tensor:
        if self.mode == 'readout':
            return self.states
        else:
            return self.frozen_weight

    def reset_state(self):
        """内部状態（膜電位）のリセット"""
        self.membrane_potential.zero_()
        # [重要] リセット時は緩和を戻すが、アニーリングの進行状況によっては戻さない制御も可能
        # ここではエピソード独立性を重視してリセット
        if self.mode != 'readout':
             self.topk_relaxation.fill_(0.7) 

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        v_mem = None
        spikes = None
        
        if self.mode == 'readout':
            # 1. Bipolar Transformation
            x_bipolar = (x - 0.5) * 2.0
            
            # 2. Normalization
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # 3. Cosine Similarity
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # 4. Adaptive Gain
            gain = self.adaptive_threshold.mean().clamp(1.0, 200.0)
            scaled_sim = cosine_sim * gain
            
            if self.training:
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            # Reservoir Mode: Top-K & Energy Gating Hybrid with Annealed Relaxation
            input_energy = x.norm(dim=1, keepdim=True)
            energy_threshold = input_energy.mean() * 1.0
            gate = torch.sigmoid((input_energy - (energy_threshold + 1e-6)) * 10.0)
            x_gated = x * gate
            
            v_mem = torch.matmul(x_gated, w.t())
            
            k = int(self.out_features * 0.2)
            if k < 1: k = 1
            
            topk_vals, _ = torch.topk(v_mem, k, dim=1)
            kth_val = topk_vals[:, -1].unsqueeze(1)
            
            # 緩和係数の適用
            dynamic_threshold = torch.maximum(torch.tensor(0.5, device=x.device), kth_val)
            dynamic_threshold = dynamic_threshold * self.topk_relaxation
            
            spikes = (v_mem >= dynamic_threshold).float()
            
            # [修正] アニーリング付き自己調整
            # 学習中は徐々に relaxation を 1.0 (厳格) に近づける
            # これにより、初期は探索(緩和)、後半は活用(厳格)へと自然に移行する
            if self.training:
                target_relaxation = 1.0
                # 現在値と目標値の差分を少しずつ埋める (Exponential Moving Average的な挙動)
                diff = target_relaxation - self.topk_relaxation
                # 0.001 の微小ステップで近づける
                self.topk_relaxation.add_(diff * 0.001)
                
                # 発火率による微調整も併用（暴走防止）
                firing_rate = spikes.mean()
                if firing_rate > 0.15:
                    self.topk_relaxation.add_(0.005) # 閾値を上げて抑制
                elif firing_rate < 0.05:
                    self.topk_relaxation.sub_(0.005) # 閾値を下げて促進
                
                self.topk_relaxation.clamp_(0.5, 1.2)
        
        if v_mem is None:
             v_mem = torch.zeros((x.shape[0], self.out_features), device=x.device)
             spikes = torch.zeros_like(v_mem)

        with torch.no_grad():
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                target_onehot = reward
                
                class_sums = torch.matmul(target_onehot.t(), pre_spikes_bipolar)
                class_counts = target_onehot.sum(dim=0).unsqueeze(1) + 1e-8
                
                batch_centroids = class_sums / class_counts
                batch_centroids = F.normalize(batch_centroids, p=2, dim=1)
                
                delta = batch_centroids - F.normalize(self.states, p=2, dim=1)
                update_mask = (target_onehot.sum(dim=0).unsqueeze(1) > 0).float()
                delta = delta * update_mask
                
            else:
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size

            # Momentum Update
            self.momentum_buffer.mul_(0.9).add_(delta)
            self.states.add_(self.momentum_buffer * learning_rate)
            
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            self.states.clamp_(-20.0, 20.0)
            
            # Auto-Tuning Gain
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.1
                gain_delta = 0.5 * (entropy - target_entropy)
                self.adaptive_threshold.add_(gain_delta)
                self.adaptive_threshold.clamp_(5.0, 200.0)
