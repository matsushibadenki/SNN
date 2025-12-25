# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Normalized Potential)
# 内容: 膜電位のZ-Score正規化、適応型閾値、およびWTAフォールバックを備えたSNNレイヤー

import torch
import torch.nn as nn
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor
    adaptive_threshold: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化。
          - 'readout': 学習可能、連続値重み。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            # 正規化後の閾値なので、標準偏差単位で設定 (例: 1.5シグマ)
            self.base_threshold = 1.5
            trainable = True
            states = torch.randn(out_features, in_features) * std_dev
            # クランプ範囲 [-20, 20]
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 適応閾値
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * self.base_threshold)
        else:
            # リザーバー層 (std=3.0)
            std_dev = 3.0 / math.sqrt(in_features)
            self.base_threshold = 1.0
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
        """3値量子化 (-1, 0, 1) * 0.5"""
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
        
        # 1. 生の膜電位を計算
        raw_current = torch.matmul(x, w.t())
        
        # 2. レイヤー正規化 (Z-Score Normalization)
        # これにより、V_Mean が -4600 でも強制的に 平均0, 分散1 に補正され、
        # 相対的に強いニューロンが浮かび上がる。
        if self.out_features > 1:
            mean = raw_current.mean(dim=1, keepdim=True)
            std = raw_current.std(dim=1, keepdim=True) + 1e-6
            v_mem = (raw_current - mean) / std
        else:
            v_mem = raw_current

        if self.mode == 'readout':
            threshold = self.adaptive_threshold.unsqueeze(0)
            
            if self.training:
                with torch.no_grad():
                    # 正規化されているため、発火率はより安定する
                    fire_rate = (v_mem >= threshold).float().mean(dim=0)
                    target_rate = 0.1
                    
                    # 閾値の動的調整
                    delta = 0.01 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    # 正規化空間なので閾値は 0.0 (平均) 〜 5.0 (5シグマ) 程度に収まる
                    self.adaptive_threshold.clamp_(0.1, 5.0)
        else:
            threshold = self.base_threshold

        spikes = (v_mem >= threshold).float()
        
        # --- Robustness Fix: Winner-Take-All Fallback (Hard Top-K) ---
        # 正規化により自律発火しやすくなっているが、それでも発火しない場合の保険
        if self.mode == 'readout':
            has_spike = spikes.sum(dim=1) > 0
            
            if not has_spike.all():
                no_spike_mask = ~has_spike
                # 正規化後の値で最大のものを選ぶ＝最も相対的にマッチしているニューロン
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0

        if self.training or not self.training:
            # ログ出力用には正規化後の値を保存（これによりV_Meanは0付近になるはず）
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            momentum = 0.95
            
            # 勾配の計算
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Weight Centering & Normalization ---
            # 重みの平均を0に保つ (バイアスのドリフト防止)
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # カオス的摂動
            if random.random() < 0.01: 
                noise = torch.randn_like(self.states) * 0.01 * learning_rate
                self.states.add_(noise)
            
            # 重みノルムの正規化
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features) 
            scale_factor = torch.clamp(target_norm / (norm + 1e-6), max=1.0)
            self.states.mul_(scale_factor)
            
            # 重みの値を制限
            self.states.clamp_(-20.0, 20.0)
