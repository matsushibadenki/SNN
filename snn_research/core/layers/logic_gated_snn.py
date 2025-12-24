# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: モメンタム学習・高接続許容版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値
        self.threshold = 5.0 # 感度を高めるために少し下げる
        
        # 初期化: 閾値付近に分散させ、最初からある程度の接続を持たせる
        # 分散を大きくして多様な特徴を捉える
        states = torch.randn(out_features, in_features) * 10.0
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        # 学習用バッファ
        self.register_buffer('weight_momentum', torch.zeros(out_features, in_features))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        w = torch.zeros_like(self.states)
        w[self.states > self.threshold] = 1.0
        w[self.states < -self.threshold] = -1.0
        return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力電流
        current = torch.matmul(x, w.t())
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位更新
        new_v_mem = v_mem.unsqueeze(0) * 0.6 + effective_current # リークを少し弱める
        
        # 発火
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値調整
        with torch.no_grad():
            target_activity = 0.2
            th_update = (mean_spikes - target_activity) * 0.02
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(0.5, 10.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        モメンタム付きサロゲート勾配学習則
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 1. 勾配計算 (Correlation)
            # 報酬(誤差信号)と入力の相関をとる
            raw_grad = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 2. サロゲート勾配マスク (広めに設定)
            # 閾値から遠くてもある程度更新させる (Dead Neuron防止)
            dist_from_th_pos = (self.states - self.threshold).abs()
            dist_from_th_neg = (self.states + self.threshold).abs()
            surrogate_scale = 10.0 # 広げる
            
            grad_mask_pos = torch.relu(1.0 - dist_from_th_pos / surrogate_scale)
            grad_mask_neg = torch.relu(1.0 - dist_from_th_neg / surrogate_scale)
            grad_mask = torch.max(grad_mask_pos, grad_mask_neg)
            
            # マスクが0の部分にもわずかに勾配を通す (Leaky Surrogate)
            grad_mask = grad_mask + 0.1
            
            # 3. モメンタム更新
            # 勾配の方向を蓄積し、ノイズを打ち消す
            momentum = cast(torch.Tensor, self.weight_momentum)
            momentum.mul_(0.9).add_(raw_grad * grad_mask, alpha=0.1)
            
            # 学習率
            lr = 10.0 * (1.0 - self.proficiency.item() * 0.5)
            
            # 重み更新
            self.states.add_(momentum * lr)
            
            # 4. 構造恒常性 (Sparsity Control)
            # 許容値を高める (80%)
            active_links = (self.states.abs() > self.threshold).float()
            conn_ratio = active_links.mean()
            target_conn = 0.80 
            
            if conn_ratio > target_conn:
                decay = 0.001
                self.states.sub_(self.states * decay)
            
            # 忘却とノイズ
            self.states.add_(torch.randn_like(self.states) * 0.05)
            self.states.clamp_(-self.max_states, self.max_states)
            
            self.proficiency.add_(0.0005)
            self.proficiency.clamp_(0.0, 1.0)
