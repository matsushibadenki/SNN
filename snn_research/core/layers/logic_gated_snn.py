# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 密結合初期化・高感度版)

import torch
import torch.nn as nn
from typing import cast, Union

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        # 閾値を少し下げる
        self.threshold = max_states * 0.4
        
        # --- 決定的な変更: 初期化 ---
        # 最初から疎（Sparse）にするのではなく、初期は「全結合に近い状態」からスタートし、
        # 学習によって不要な接続を削ぎ落とす。
        # mean=threshold + 5.0 なので、初期状態で50%以上の確率で接続される
        states = torch.randn(out_features, in_features) * 10.0 + (self.threshold + 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 初期閾値を低くして、最初はとにかく発火させる
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重み: 接続されているなら1.0
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # ゲイン調整: 接続が多い初期段階でも信号が埋もれないように対数圧縮を強化
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 8.0 -> 10.0
        gain = 10.0 / torch.log1p(conn_count * 0.2)
        
        current = torch.matmul(x, w.t()) * gain.unsqueeze(0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # ノイズを減らし、信号の信頼度を上げる
        noise = torch.randn_like(current) * 0.1
        new_v_mem = v_mem.unsqueeze(0) * 0.8 + effective_current + noise
        spikes = (new_v_mem >= v_th.unsqueeze(0)).to(torch.float32)
        
        mean_spikes = spikes.mean(dim=0)
        
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        with torch.no_grad():
            self.adaptive_threshold.mul_(0.98) 
            target_activity = 0.20 # 目標発火率を上げて、より多くのニューロンを参加させる
            th_update = (mean_spikes - target_activity) * 0.1
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(2.0, 50.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                reward = torch.full((batch_size, 1), reward, device=pre_spikes.device)

            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            # 学習率を初期に非常に高く設定 (5.0 -> 10.0)
            # これにより、重要な入力への接続を一気に確立する
            lr = 10.0 * (1.0 - prof * 0.3)

            if reward.ndim == 1:
                reward = reward.unsqueeze(1) 
            
            if reward.shape[1] == self.out_features:
                 delta = torch.matmul(reward.t(), pre_spikes) / batch_size
                 delta *= lr
            else:
                 modulation = reward * post_spikes 
                 delta = torch.matmul(modulation.t(), pre_spikes) / batch_size
                 delta *= lr

            delta = delta.clamp(min=-10.0, max=20.0)
            
            self.states.add_(delta)
            
            # 構造恒常性
            # 目標接続率を少し高め(25%)に維持し、情報を失わないようにする
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.25 
            conn_error = target_conn - current_conn
            self.states.add_(conn_error * 2.0)
            
            self.states.mul_(0.999) # 忘却を少し強め、不要な結合を切る
            self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
