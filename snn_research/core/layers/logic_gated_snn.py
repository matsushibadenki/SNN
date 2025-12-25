# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Power Law Dynamics)
# 内容: べき乗則(Power Law)によるS/N比増幅、重心蓄積学習、バイポーラ信号処理

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
            std_dev = 0.05
            trainable = True
            
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # Power Lawを使うため、温度は低めで良い（べき乗が温度の代わりになる）
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 1.0)
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
            # 1. Bipolar Transformation (-1/1)
            x_bipolar = (x - 0.5) * 2.0
            
            # 2. Normalization
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # 3. Cosine Similarity
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # 4. Power Law Dynamics (The Key Fix)
            # コサイン類似度(-1 ~ 1)を、奇数乗することで、
            # 0付近のノイズを急速に減衰させ、相関が高い部分だけを際立たせる。
            # 例: 0.1^9 = 1e-9 (ほぼ0), 0.5^9 = 0.002
            # これによりS/N比が劇的に改善する。
            power_degree = 9.0 
            
            # 符号を維持したままべき乗
            powered_sim = cosine_sim.sign() * cosine_sim.abs().pow(power_degree)
            
            # スケールを戻すために定数倍（学習安定化のため）
            scaled_sim = powered_sim * 100.0
            
            if self.training:
                # Temperature Scalingも併用
                temp = self.adaptive_threshold.mean()
                spikes = F.softmax(scaled_sim * temp, dim=1)
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
        """
        Centroid Accumulation Rule
        """
        if not self.trainable:
            return

        with torch.no_grad():
            # Bipolar Input
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                target_onehot = reward
                
                # --- Centroid Accumulation ---
                # 正解クラスの入力ベクトルを単純に加算平均していくイメージ
                # 余計なRepulsion（引き剥がし）は高ノイズ下では逆効果なので行わない
                
                # 正解クラスごとの入力和
                class_sums = torch.matmul(target_onehot.t(), pre_spikes_bipolar)
                
                # バッチ内のクラス出現数
                class_counts = target_onehot.sum(dim=0).unsqueeze(1) + 1e-8
                
                # バッチ内重心
                batch_centroids = class_sums / class_counts
                batch_centroids = F.normalize(batch_centroids, p=2, dim=1)
                
                # 重心を現在の重みに近づける (EMA)
                delta = batch_centroids - F.normalize(self.states, p=2, dim=1)
                
                # 更新マスク
                update_mask = (target_onehot.sum(dim=0).unsqueeze(1) > 0).float()
                delta = delta * update_mask
                
            else:
                # Legacy
                if isinstance(reward, float):
                    reward_tensor = torch.full((pre_spikes.size(0), self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / pre_spikes.size(0)

            # Update
            self.states.add_(delta * learning_rate)
            
            # --- Constraints ---
            
            # Centering
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # Norm Scaling
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            self.states.clamp_(-20.0, 20.0)
            
            # Temperature Adjustment (Power Lawがあるので控えめに)
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.5 
                temp_delta = 0.1 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                self.adaptive_threshold.clamp_(1.0, 20.0)
