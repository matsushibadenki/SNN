# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Contrastive Hebbian)
# 内容: 教師あり対照ヘブ学習(Attraction/Repulsion)、バイポーラ信号処理、超高感度温度制御

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

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            trainable = True
            
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 初期温度を少し高めに設定
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 50.0)
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

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        if self.mode == 'readout':
            # Bipolar Transform: 0/1 -> -1/1
            x_bipolar = (x - 0.5) * 2.0
            
            # Normalization
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # Cosine Similarity
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # Temperature Scaling
            temperature = self.adaptive_threshold.mean()
            scaled_sim = cosine_sim * temperature
            
            if self.training:
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            v_mem = torch.matmul(x, w.t())
            spikes = (v_mem >= 1.0).float()
        
        with torch.no_grad():
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            # Bipolar Input for Learning
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                target_onehot = reward
                
                # --- Contrastive Hebbian Update ---
                # Positive Phase (Attraction): 正解クラスの重みを入力へ近づける
                pos_input = torch.matmul(target_onehot.t(), pre_spikes_bipolar)
                
                # Negative Phase (Repulsion): 不正解クラスの重みを入力から遠ざける
                # (1 - target) が不正解クラスのマスク
                neg_mask = 1.0 - target_onehot
                neg_input = torch.matmul(neg_mask.t(), pre_spikes_bipolar)
                
                # Update Rule
                # Repulsion係数 (beta): あまり強くしすぎると重みが発散するので控えめに (0.2程度)
                beta = 0.2
                
                # Delta = (Pos_Signal - beta * Neg_Signal) / Batch
                # 正解クラスには pre_spikes が加算され、不正解クラスには pre_spikes が減算される
                delta = (pos_input - beta * neg_input) / batch_size

            else:
                # Legacy support
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size

            # Direct Update (Simpler is better for noise cancellation)
            self.states.add_(delta * learning_rate)
            
            # --- Normalization & Constraints ---
            
            # 1. Centering
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Norm Scaling (Crucial)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # Clamp
            self.states.clamp_(-20.0, 20.0)
            
            # Temperature Auto-tuning (High Sensitivity)
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.1 
                
                # 0.48ノイズ(相関0.04)を検知するため、温度は非常に高く設定できるようにする
                # エントロピーが高い＝迷っている＝温度不足
                temp_delta = 1.0 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                # 上限を300まで開放
                self.adaptive_threshold.clamp_(10.0, 300.0)
