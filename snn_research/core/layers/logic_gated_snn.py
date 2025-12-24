# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 活性化優先・剪定緩行版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 重みの閾値: ここを下げて「繋がりやすく」する
        self.threshold = 0.5 
        
        # 初期化: 閾値の手前まで値を持ち上げておく
        # これにより、少しの学習で接続がONになる
        states = torch.randn(out_features, in_features) * 0.3
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        
        # 発火閾値: 最初は非常に低くして、とにかく発火させる
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 0.5))
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
        
        # 膜電位更新: 減衰を少し弱めて(0.8)、入力を保持しやすくする
        new_v_mem = v_mem.unsqueeze(0) * 0.8 + effective_current
        
        # 発火
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値恒常性: 目標発火率を高め(20%)に設定
        with torch.no_grad():
            target_activity = 0.20
            th_update = (mean_spikes - target_activity) * 0.01
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(0.1, 10.0) 
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        活性化優先学習則
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 学習率: 少し上げる (0.02 -> 0.1)
            # 初期は大胆に動かす
            lr = 0.1 * (1.0 - self.proficiency.item() * 0.3)
            
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            self.states.add_(delta * lr)
            
            # --- 構造的恒常性 (緩やかに) ---
            active_links = (self.states.abs() > self.threshold).float()
            conn_ratio = active_links.mean()
            
            # 目標接続率 50% (以前は20%で厳しすぎた)
            target_conn = 0.50
            
            if conn_ratio > target_conn:
                # 目標を超えたら減衰させるが、極端にはしない
                decay = 0.999
            else:
                # 目標以下なら減衰なし（あるいは負の減衰で成長促進）
                decay = 1.0
            
            self.states.mul_(decay)
            
            # ノイズ注入
            self.states.add_(torch.randn_like(self.states) * 0.01)
            
            self.states.clamp_(-self.max_states, self.max_states)
            self.proficiency.add_(0.0002)
            self.proficiency.clamp_(0.0, 1.0)
