# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 選択的徐放成長版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値: ここを超えると接続がONになる
        # 高すぎると何も学習せず、低すぎると爆発する。
        self.threshold = 1.0
        
        # --- 初期化: 完全な静寂 ---
        # 重みの種（seeds）を非常に小さな値で初期化します。
        # ノイズレベルを閾値の1/10以下に抑え、偶然の接続を防ぎます。
        states = torch.randn(out_features, in_features) * 0.05
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 初期閾値: 最初は少し反応しやすくしておく
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 0.8))
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
        # リークを少し強めて(0.7)、古い情報を消しやすくする
        new_v_mem = v_mem.unsqueeze(0) * 0.7 + effective_current
        
        # 発火
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値調整 (Homeostasis)
        with torch.no_grad():
            target_activity = 0.15 
            th_update = (mean_spikes - target_activity) * 0.02
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(0.5, 5.0) 
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        選択的成長学習則 (Selective Growth Rule)
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # --- 決定的な修正: 学習率の大幅な抑制 ---
            # 以前は 2.0 でしたが、これを 0.02 に下げます。
            # これにより、1回のバッチで接続がいきなりONになることを防ぎ、
            # 「何度も繰り返し正しいと判断された接続」だけが閾値(1.0)を超えて成長します。
            lr = 0.02 * (1.0 - self.proficiency.item() * 0.5)
            
            # 相関の計算 (Delta Rule)
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 更新
            self.states.add_(delta * lr)
            
            # --- 構造的恒常性 (Decay) ---
            # 接続率が高すぎる場合、強力に減衰させて「剪定」する
            active_links = (self.states.abs() > self.threshold).float()
            conn_ratio = active_links.mean()
            
            # 目標接続率 20% (以前の80%は多すぎました)
            target_conn = 0.20 
            
            if conn_ratio > target_conn:
                # 目標を超えたら減衰を強くする (ペナルティ)
                decay = 0.99 
            else:
                # 目標以下なら自然減衰のみ (忘却)
                decay = 0.9999
            
            self.states.mul_(decay)
            
            # ノイズ (非常に小さく)
            self.states.add_(torch.randn_like(self.states) * 0.001)
            
            self.states.clamp_(-self.max_states, self.max_states)
            self.proficiency.add_(0.0005)
            self.proficiency.clamp_(0.0, 1.0)
