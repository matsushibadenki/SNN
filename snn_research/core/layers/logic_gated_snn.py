# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Delta Rule & Adaptive Temperature)
# 内容: 教師ありデルタ則による直接誤差学習、適応型温度スケーリング、チャネル中心化

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
            # 読み出し層 (学習可能)
            std_dev = 0.05
            trainable = True
            
            # 直交初期化
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            
            # Adaptive Thresholdの代わりに Temperature parameter として使用
            # 初期温度は高め(50.0)に設定し、学習とともに調整可能にする
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 50.0)
        else:
            # リザーバー層 (固定)
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
        """3値量子化"""
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
            # 1. Centered Cosine Similarity
            # ノイズ耐性の要。クラス間の平均を引き、相対的な信号強度を取り出す。
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            cosine_sim = torch.matmul(x_norm, w_norm.t()) # [Batch, Out]
            
            mean_sim = cosine_sim.mean(dim=1, keepdim=True)
            centered_sim = cosine_sim - mean_sim
            
            # 2. Temperature Scaling
            # 学習可能な温度パラメータを使用
            temperature = self.adaptive_threshold.mean() # 全ニューロン共通温度とする
            scaled_sim = centered_sim * temperature
            
            # 3. Activation (Softmax or Hard WTA)
            if self.training:
                # 学習時はSoftmaxで微分可能な確率分布にする
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                # 推論時はArgmaxで明確な判定
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            # リザーバー層
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
            
            # Reward引数を柔軟に処理
            # Tensorかつ形状が[Batch, Out]の場合、それは「誤差信号 (Error Signal)」として扱う
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                error_signal = reward
                # Delta Rule: Error * Input
                # Batch平均をとる
                delta = torch.matmul(error_signal.t(), pre_spikes) / batch_size
            
            else:
                # 従来のスカラー報酬の場合（後方互換性）
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                elif reward.dim() == 1:
                    reward_tensor = reward.unsqueeze(1).expand(-1, self.out_features)
                else:
                    reward_tensor = reward
                
                # Hebbian-like update scaled by reward * prob
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes) / batch_size

            # モメンタム更新
            momentum_factor = 0.95
            self.momentum_buffer.mul_(momentum_factor).add_(delta)
            
            # 重み更新
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Constraints ---
            # 1. Centering (平均除去) - コサイン類似度のために重要
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Norm Scaling (球面射影)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            self.states.clamp_(-20.0, 20.0)
            
            # 温度調整 (Automatic Temperature Scaling)
            # 予測分布が極端（全部0または1）にならないように、エントロピーに基づいて温度を微調整
            if self.mode == 'readout':
                # post_spikesはSoftmax出力
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.5 # 適度な分散を維持
                temp_delta = 0.1 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                self.adaptive_threshold.clamp_(10.0, 100.0) # 温度範囲制限
