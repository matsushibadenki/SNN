# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Orthogonal & High Contrast)
# 内容: 直交初期化、高コントラストコサイン類似度、および強化された可塑性を備えたSNNレイヤー

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            # コサイン類似度(x20スケール)用の閾値初期値
            # スケールを上げたため、閾値も少し高めに設定
            self.base_threshold = 1.0 
            trainable = True
            
            # 変更点1: 直交初期化 (Orthogonal Initialization)
            # クラス間の分離を最大化し、ノイズ耐性の基礎を作る
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            # スケールを調整
            states = states * std_dev
            
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
        
        # --- Cosine Similarity Logic ---
        if self.mode == 'readout':
            # 1. 入力ベクトルの正規化 (L2 Norm)
            x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # 2. 重みベクトルの正規化
            w_norm = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # 3. コサイン類似度の計算 (-1.0 ~ 1.0)
            cosine_sim = torch.matmul(x_norm, w_norm.t())
            
            # 4. スケーリング (High Contrast)
            # 変更点2: スケールを 20.0 に倍増し、コントラストを高める
            v_mem = cosine_sim * 20.0
        else:
            # リザーバー層は従来通りのドット積
            v_mem = torch.matmul(x, w.t())
        
        # --- Thresholding ---
        if self.mode == 'readout':
            threshold = self.adaptive_threshold.unsqueeze(0)
            
            if self.training:
                with torch.no_grad():
                    fire_rate = (v_mem >= threshold).float().mean(dim=0)
                    target_rate = 0.1
                    
                    # 閾値の動的調整
                    delta = 0.01 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    # スケール20.0に合わせて上限も緩和
                    self.adaptive_threshold.clamp_(0.1, 15.0)
        else:
            threshold = self.base_threshold

        spikes = (v_mem >= threshold).float()
        
        # --- Robustness Fix: Winner-Take-All Fallback (Hard Top-K) ---
        if self.mode == 'readout':
            has_spike = spikes.sum(dim=1) > 0
            
            if not has_spike.all():
                no_spike_mask = ~has_spike
                # コサイン類似度が最も高い（＝角度が最も近い）ニューロンを強制発火
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0

        if self.training or not self.training:
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
            
            # 変更点3: 負の報酬（失敗）を強調して、間違ったパターンから強く引き離す
            # 正の報酬はそのまま、負の報酬を 1.5倍 に重み付け
            effective_reward = reward.clone()
            effective_reward[reward < 0] *= 1.5
            
            momentum = 0.95
            
            # 勾配計算
            delta = torch.matmul(effective_reward.t(), pre_spikes) / batch_size
            
            self.momentum_buffer.mul_(momentum).add_(delta)
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Weight Normalization & Centering ---
            
            # 1. Centering (ドリフト防止)
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Chaos (停滞防止)
            if random.random() < 0.01: 
                noise = torch.randn_like(self.states) * 0.01 * learning_rate
                self.states.add_(noise)
            
            # 3. Norm Scaling (正則化)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features) 
            scale_factor = torch.clamp(target_norm / (norm + 1e-6), min=0.5, max=1.5)
            self.states.mul_(scale_factor)
            
            # 値の暴走を防ぐクランプ
            self.states.clamp_(-20.0, 20.0)
