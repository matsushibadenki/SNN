# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 構造恒常性・スパース維持版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値設定
        self.threshold = max_states * 0.2
        
        # --- 初期化の改善 ---
        # 平均0, 分散大きめで初期化し、初期接続率を自然な確率(約50%)に任せる
        states = torch.randn(out_features, in_features) * (self.threshold * 2.0)
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 初期閾値を少し下げて、学習初期の不感症を防ぐ
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.5))
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重み: 閾値を超えたら1、負の閾値を下回ったら-1
        w = torch.zeros_like(self.states)
        w[self.states > self.threshold] = 1.0
        w[self.states < -self.threshold] = -1.0
        return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力電流の計算
        current = torch.matmul(x, w.t())
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位の更新 (リーク強め: 0.5)
        new_v_mem = v_mem.unsqueeze(0) * 0.5 + effective_current
        
        # 発火判定
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値の恒常性維持 (目標発火率: 10-20%)
        with torch.no_grad():
            target_activity = 0.15
            th_update = (mean_spikes - target_activity) * 0.05
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(1.0, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        構造恒常性を取り入れた可塑性更新則
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 1. 重みの更新 (Delta Rule / Hebbian)
            lr = 2.0 * (1.0 - self.proficiency.item() * 0.5)
            
            # Feedback Alignmentによる勾配近似
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            self.states.add_(delta * lr)
            
            # 2. 構造恒常性 (Structural Homeostasis) - これが最重要修正点
            # 現在の接続率を計算
            active_links = (self.states.abs() > self.threshold).float()
            conn_ratio = active_links.mean()
            
            target_conn = 0.50 # 目標接続率 50%
            
            # 接続率が高すぎる場合、全体的に値を減衰させて接続を切る
            # 逆に低すぎる場合は、減衰を弱める（または加算する）
            if conn_ratio > target_conn:
                decay = 0.99 # 強力な減衰
            else:
                decay = 0.9995 # 弱い減衰
                
            self.states.mul_(decay)
            
            # ノイズ注入 (探索)
            noise = torch.randn_like(self.states) * 0.05
            self.states.add_(noise)
            
            # クランプ
            self.states.clamp_(-self.max_states, self.max_states)
            
            # 熟練度の更新
            self.proficiency.add_(0.0001)
            self.proficiency.clamp_(0.0, 1.0)
