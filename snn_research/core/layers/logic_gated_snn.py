# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: デルタ則・強制学習版)

import torch
import torch.nn as nn
from typing import cast, Union

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 初期状態: 少し高めの分散で初期化
        states = torch.randn(out_features, in_features) * 20.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
        # traceバッファは維持するが、今回の主要学習ロジックからは外す
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # ゲイン調整: 8.0で維持
        conn_count = w.sum(dim=1).clamp(min=1.0)
        gain = 8.0 / torch.log1p(conn_count * 0.5)
        
        current = torch.matmul(x, w.t()) * gain.unsqueeze(0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        noise = torch.randn_like(current) * 0.5
        
        new_v_mem = v_mem.unsqueeze(0) * 0.8 + effective_current + noise
        spikes = (new_v_mem >= v_th.unsqueeze(0)).to(torch.float32)
        
        mean_spikes = spikes.mean(dim=0)
        
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        with torch.no_grad():
            # 閾値ホメオスタシス
            self.adaptive_threshold.mul_(0.98) # 自然減衰
            target_activity = 0.15 
            th_update = (mean_spikes - target_activity) * 0.5
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(3.0, 40.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        """ 
        修正版学習則: デルタ則に基づく強制学習
        pre_spikes: (in_features,) またはバッチ平均
        reward: (out_features,) のベクトル教示信号
        """
        with torch.no_grad():
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                
            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            # Modulation (報酬/教示信号)
            if isinstance(reward, torch.Tensor) and reward.ndim == 1:
                modulation = reward.unsqueeze(1) # (out, 1)
            else:
                modulation = reward
            
            # 学習率
            lr = 4.0 * (1.0 - prof * 0.5) # 初期学習率を倍増 (2.0 -> 4.0)
            
            # --- 核心的な変更 ---
            # 旧: delta = Trace(Pre*Post) * Reward
            # 新: delta = Pre * Reward * LR
            # これにより、Postが発火していなくても、Reward(正解信号)があればPreが接続されているシナプスを強化できる。
            
            # pre_spikes: (in,) -> (1, in)
            pre_activity = pre_spikes.unsqueeze(0) if pre_spikes.dim() == 1 else pre_spikes.mean(dim=0, keepdim=True)
            
            # delta: (out, 1) * (1, in) -> (out, in)
            # 正解ニューロン(Reward>0)は、入力があったシナプスを強化。
            # 不正解ニューロン(Reward<0)は、入力があったシナプスを抑制。
            delta = modulation * pre_activity * lr
            
            # 罰の緩和 (死の螺旋防止)
            if isinstance(modulation, torch.Tensor):
                delta = delta.clamp(min=-5.0, max=15.0) 
            
            self.states.add_(delta)
            
            # 構造恒常性 (接続維持)
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.20
            conn_error = target_conn - current_conn
            self.states.add_(conn_error * 5.0)
            
            # 忘却とノイズ
            self.states.mul_(0.9995)
            # ノイズを少し増やして探索を維持
            self.states.add_(torch.randn_like(self.states) * 0.1)

            self.states.clamp_(1, self.max_states)
