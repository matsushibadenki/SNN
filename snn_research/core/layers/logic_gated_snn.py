# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final SOTA: Adaptive Offset)
# 修正: 低信号時の適応型オフセットと動的Top-Kにより、Acc 88%突破を目指す

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
        SOTA Edition: Adaptive Offset & Dynamic Top-K.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if in_features > 64:
            self.hparams = {
                'momentum': 0.99,          
                'target_entropy': 0.25,    
                'gain_limit': 200.0,       
                'gain_update_rate': 0.01   
            }
        else:
            self.hparams = {
                'momentum': 0.90,          
                'target_entropy': 0.60,    
                'gain_limit': 50.0,
                'gain_update_rate': 0.05
            }

        if self.mode == 'readout':
            std_dev = 0.05
            trainable = True
            
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            
            initial_gain = 20.0 if in_features > 64 else 5.0
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * initial_gain)
            self.register_buffer('path_mixer', torch.tensor(0.0)) 
        else:
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
        self.membrane_potential.zero_()
        if self.mode != 'readout':
             self.path_mixer.fill_(0.2)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        v_mem = None
        spikes = None
        
        if self.mode == 'readout':
            x_bipolar = (x - 0.5) * 2.0
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            sharpened_sim = F.relu(cosine_sim).pow(2.0)

            gain_limit = self.hparams['gain_limit'] if hasattr(self, 'hparams') else 50.0
            gain = self.adaptive_threshold.mean().clamp(1.0, gain_limit)
            scaled_sim = sharpened_sim * gain
            
            if self.training:
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            input_energy = x.norm(dim=1, keepdim=True)
            energy_threshold = input_energy.mean() * 1.0
            if energy_threshold == 0: energy_threshold = 1.0
            
            # [修正] Adaptive Offset: エネルギー不足時(V_Mean低下時)に底上げを行う
            energy_ratio = input_energy.mean() / (energy_threshold + 1e-8)
            low_energy_boost = torch.clamp(0.1 - energy_ratio, min=0.0) * 2.0 # 0.1以下で発動
            
            gate_explore = torch.sigmoid((input_energy - (energy_threshold * 0.5)) * 2.0)
            v_explore = torch.matmul(x * gate_explore, w.t()) + low_energy_boost # オフセット加算
            
            adaptive_thresh_explore = 0.1 * energy_ratio
            adaptive_thresh_explore = torch.clamp(adaptive_thresh_explore, 0.05, 0.2)
            spikes_explore = (v_explore >= adaptive_thresh_explore).float()
            
            gate_exploit = torch.sigmoid((input_energy - (energy_threshold * 1.5)) * 10.0)
            v_exploit = torch.matmul(x * gate_exploit, w.t())
            
            # [修正] Dynamic Top-K: 低エネルギー時はTop-Kを20%に拡大して情報を拾う
            base_k = 0.15
            adaptive_k = base_k + (0.05 if energy_ratio < 0.2 else 0.0)
            k = int(self.out_features * adaptive_k)
            if k < 1: k = 1
            
            topk_vals, _ = torch.topk(v_exploit, k, dim=1)
            kth_val = topk_vals[:, -1].unsqueeze(1)
            
            dynamic_thresh = torch.maximum(torch.tensor(0.58, device=x.device), kth_val)
            spikes_exploit = (v_exploit >= dynamic_thresh).float()
            
            mixer = self.path_mixer.clamp(0.0, 1.0)
            v_mem = v_explore * (1.0 - mixer) + v_exploit * mixer
            
            mixed_threshold = adaptive_thresh_explore * (1.0 - mixer) + 0.58 * mixer
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
                delta.clamp_(-0.05, 0.05)
                
                update_mask = (target_onehot.sum(dim=0).unsqueeze(1) > 0).float()
                delta = delta * update_mask
                
            else:
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size
                delta.clamp_(-0.05, 0.05)

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
            
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                
                target_ent = self.hparams['target_entropy'] if hasattr(self, 'hparams') else 0.4
                update_rate = self.hparams['gain_update_rate'] if hasattr(self, 'hparams') else 0.05
                limit = self.hparams['gain_limit'] if hasattr(self, 'hparams') else 50.0

                gain_delta = update_rate * (entropy - target_ent)
                self.adaptive_threshold.add_(gain_delta)
                self.adaptive_threshold.clamp_(5.0, limit)
