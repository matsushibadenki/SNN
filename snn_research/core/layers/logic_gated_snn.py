# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 恒常性覚醒版)

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
        
        # 初期状態
        states = torch.randn(out_features, in_features) * 20.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 初期閾値を低く設定してスタートダッシュを決める
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
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
        
        # ゲイン調整: 5.0 -> 8.0 に戻して「刺激不足」を解消
        conn_count = w.sum(dim=1).clamp(min=1.0)
        gain = 8.0 / torch.log1p(conn_count * 0.5)
        
        # 電流計算
        current = torch.matmul(x, w.t()) * gain.unsqueeze(0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        noise = torch.randn_like(current) * 0.5
        
        # 膜電位更新
        new_v_mem = v_mem.unsqueeze(0) * 0.8 + effective_current + noise
        
        # 発火判定
        spikes = (new_v_mem >= v_th.unsqueeze(0)).to(torch.float32)
        
        # ステート更新
        mean_spikes = spikes.mean(dim=0)
        
        # 不応期: 3.0 -> 2.0 に短縮
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        # 膜電位リセット
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        with torch.no_grad():
            # トレース更新
            avg_x = x.mean(dim=0)
            self.eligibility_trace.mul_(0.9).add_(torch.outer(mean_spikes, avg_x))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # --- 閾値のホメオスタシス（ここが重要） ---
            
            # 1. 自然減衰: 発火しなくても勝手に下がる (Wait & Decay)
            # これにより「張り付き」から必ず回復する
            self.adaptive_threshold.mul_(0.98)
            
            # 2. 活動依存の調整
            target_activity = 0.15 
            th_update = (mean_spikes - target_activity) * 0.5
            self.adaptive_threshold.add_(th_update)
            
            # 下限を3.0まで下げて、再発火しやすくする
            self.adaptive_threshold.clamp_(3.0, 40.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        """ 可塑性更新 """
        with torch.no_grad():
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                
            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            if isinstance(reward, torch.Tensor) and reward.ndim == 1:
                modulation = reward.unsqueeze(1)
            else:
                modulation = reward
            
            lr = 2.0 * (1.0 - prof * 0.8)
            
            delta = trace * modulation * lr
            
            # 罰の緩和クリップ
            if isinstance(modulation, torch.Tensor):
                delta = delta.clamp(min=-5.0, max=10.0) 
            
            self.states.add_(delta)
            
            # --- 救済措置: サイレント・レスキュー ---
            # もしこの層が全く発火しておらず(Traceがゼロに近い)、かつ報酬が負(失敗)なら
            # ランダムにシナプスを強化して「何かを感じ取れる」ようにする
            if trace.sum() < 0.1 and avg_reward < 0:
                 # ランダムな刺激（生存本能）
                 survival_noise = torch.randn_like(self.states) * 2.0
                 self.states.add_(survival_noise)
            
            # 構造恒常性
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.20
            conn_error = target_conn - current_conn
            
            self.states.add_(conn_error * 5.0)
            
            # 忘却とノイズ
            self.states.mul_(0.9995)
            self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
