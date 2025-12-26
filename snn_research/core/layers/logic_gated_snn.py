# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final SOTA: Balanced Momentum)
# 修正: MomentumとGain Limitを適正値に戻し、安定性を回復

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
    path_mixer: torch.Tensor 

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        SCAL (Statistical Centroid Alignment Learning) ベースのニューロモルフィック層。
        SOTA Edition: Balanced Momentum & Adaptive Thresholding.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        # [修正] 次元数に応じたハイパーパラメータの自動設定
        # 0.999は過剰反応を引き起こしたため、0.995に調整し、Gain Limitも抑制
        if in_features > 64:
            self.hparams = {
                'momentum': 0.995,         # 0.999 -> 0.995: 安定性と慣性のバランス
                'target_entropy': 0.25,    # 0.1 -> 0.25: 過学習/過信を防ぐ
                'gain_limit': 50.0,        # 200.0 -> 50.0: ノイズ増幅を防ぐ
                'gain_update_rate': 0.01   # 標準的な更新速度に戻す
            }
        else:
            self.hparams = {
                'momentum': 0.90,          
                'target_entropy': 0.60,    
                'gain_limit': 50.0,
                'gain_update_rate': 0.05
            }

        if self.mode == 'readout':
            # 学習可能層 (Readout)
            std_dev = 0.05
            trainable = True
            
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            
            # 初期ゲイン
            initial_gain = 8.0 if in_features > 64 else 5.0
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * initial_gain)
            self.register_buffer('path_mixer', torch.tensor(0.0)) # 未使用
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
            
            # 初期状態: 探索(Explore)寄り
            self.register_buffer('path_mixer', torch.tensor(0.2))
            
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
        if self.mode != 'readout':
             self.path_mixer.fill_(0.2)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        v_mem = None
        spikes = None
        
        if self.mode == 'readout':
            # 1. Bipolar Transformation
            x_bipolar = (x - 0.5) * 2.0
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # パラメータに基づいたゲイン制限
            gain_limit = self.hparams['gain_limit'] if hasattr(self, 'hparams') else 50.0
            gain = self.adaptive_threshold.mean().clamp(1.0, gain_limit)
            scaled_sim = cosine_sim * gain
            
            if self.training:
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            # Reservoir Mode: Dual-Path Gating
            
            input_energy = x.norm(dim=1, keepdim=True)
            energy_threshold = input_energy.mean() * 1.0
            if energy_threshold == 0: energy_threshold = 1.0
            
            # Path A: Exploration (High Sensitivity)
            gate_explore = torch.sigmoid((input_energy - (energy_threshold * 0.5)) * 2.0)
            v_explore = torch.matmul(x * gate_explore, w.t())
            
            # 入力エネルギーに応じた適応的閾値
            ratio = input_energy.mean() / (energy_threshold + 1e-8)
            adaptive_thresh_explore = 0.1 * ratio
            adaptive_thresh_explore = torch.clamp(adaptive_thresh_explore, 0.05, 0.2)
            spikes_explore = (v_explore >= adaptive_thresh_explore).float()
            
            # Path B: Exploitation (High Precision)
            gate_exploit = torch.sigmoid((input_energy - (energy_threshold * 1.5)) * 10.0)
            v_exploit = torch.matmul(x * gate_exploit, w.t())
            
            # Top-Kの緩和
            k = int(self.out_features * 0.15)
            if k < 1: k = 1
            topk_vals, _ = torch.topk(v_exploit, k, dim=1)
            kth_val = topk_vals[:, -1].unsqueeze(1)
            dynamic_thresh = torch.maximum(torch.tensor(0.6, device=x.device), kth_val)
            spikes_exploit = (v_exploit >= dynamic_thresh).float()
            
            # Path Mixing
            mixer = self.path_mixer.clamp(0.0, 1.0)
            
            # v_mem の混合
            v_mem = v_explore * (1.0 - mixer) + v_exploit * mixer
            
            # 閾値混合
            mixed_threshold = adaptive_thresh_explore * (1.0 - mixer) + 0.6 * mixer
            spikes = (v_mem >= mixed_threshold).float()
            
            if self.training:
                self.path_mixer.add_(0.0001).clamp_(0.0, 1.0)
        
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

            # 次元数に応じたMomentum設定を使用
            momentum_val = self.hparams['momentum'] if hasattr(self, 'hparams') else 0.95
            
            self.momentum_buffer.mul_(momentum_val).add_(delta)
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
                
                # 次元数に応じたパラメータを使用
                target_ent = self.hparams['target_entropy'] if hasattr(self, 'hparams') else 0.4
                update_rate = self.hparams['gain_update_rate'] if hasattr(self, 'hparams') else 0.05
                limit = self.hparams['gain_limit'] if hasattr(self, 'hparams') else 50.0

                gain_delta = update_rate * (entropy - target_ent)
                self.adaptive_threshold.add_(gain_delta)
                self.adaptive_threshold.clamp_(5.0, limit)
