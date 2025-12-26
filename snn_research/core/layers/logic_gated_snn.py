# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final SOTA: Hyper-Contrast Boosting)
# 修正: 低信頼度時の極端なコントラスト強調(Power Scaling)により、Acc 88%の壁を物理的にこじ開ける

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
        SOTA Edition: Hyper-Contrast Boosting & Dynamic Top-K.
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
            
            # [修正] Hyper-Contrast Boosting Strategy
            # 1. まず類似度のピークを1.0に正規化する (Auto-Gain)
            sim_max, _ = cosine_sim.max(dim=1, keepdim=True)
            auto_gain = 1.0 / (sim_max.detach().clamp(min=0.01))
            norm_sim = cosine_sim * auto_gain
            
            # 2. 信号強度(sim_max)に基づいて「べき乗(Power)」を動的に決定する
            # 強度が高い(0.5以上) -> Power 2.0 (通常のSquared ReLU)
            # 強度が低い(0.1近辺) -> Power 6.0以上 (極端なコントラスト強調)
            # これにより、ノイズに埋もれた微差を強制的に拡大する。
            
            signal_strength = sim_max.detach().clamp(min=0.05, max=0.5)
            # 線形補間: 0.05のときPower=7.0, 0.5のときPower=2.0
            # slope = (2.0 - 7.0) / (0.5 - 0.05) = -5.0 / 0.45 ≈ -11.1
            adaptive_power = 7.55 - 11.1 * signal_strength
            adaptive_power = adaptive_power.clamp(min=2.0, max=8.0)
            
            # 3. 適用
            # norm_simは最大1.0なので、何乗しても発散しない。
            # しかし小さい値(0.9など)はべき乗で急速に減衰し、トップとの差が開く。
            sharpened_sim = F.relu(norm_sim).pow(adaptive_power)

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
            
            # [修正] Dynamic Top-K:
