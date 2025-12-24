# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: タブラ・ラサ成長戦略版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値を低く設定し、成長しやすくする
        self.threshold = 1.0
        
        # --- 決定的な変更: タブラ・ラサ初期化 ---
        # 以前は大きな値で初期化していたため、間違った接続が消えませんでした。
        # 今回は「0.0」付近の小さなノイズで初期化し、初期状態では「ほぼ接続なし」にします。
        # 学習が進むにつれて、重要な接続だけが閾値(1.0)を超えて成長します。
        states = torch.randn(out_features, in_features) * 0.1
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        # 膜電位等はリセット
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 閾値も最初は低くして、弱い入力でも学習のきっかけを作れるようにする
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 0.5))
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重み: 閾値を超えたら1、下回ったら-1
        # 初期状態ではほぼ全て0になる
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
        
        # 膜電位更新 (リークあり)
        # 接続が少ない初期は減衰を弱めにして信号を保つ
        decay_factor = 0.8
        new_v_mem = v_mem.unsqueeze(0) * decay_factor + effective_current
        
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
            target_activity = 0.15 # 目標発火率
            th_update = (mean_spikes - target_activity) * 0.05
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(0.1, 10.0) # 下限を下げて反応しやすくする
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        成長型学習則
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 1. Delta Rule / Hebbian Update
            # reward (Error Signal) * Input
            # 正しい入力には結合を強め(正の方向へ成長)、間違いには弱める(負の方向へ)
            
            # 初期は学習率を高くして、一気に成長させる
            lr = 2.0 * (1.0 - self.proficiency.item() * 0.5)
            
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 更新
            self.states.add_(delta * lr)
            
            # 2. 忘却 (Decay) - 不要な結合を自然消滅させる
            # 0に向かって常に少しずつ減衰させる
            self.states.mul_(0.995)
            
            # 3. ノイズ (探索)
            # 停滞を防ぐための微小な揺らぎ
            self.states.add_(torch.randn_like(self.states) * 0.02)
            
            # クランプ
            self.states.clamp_(-self.max_states, self.max_states)
            
            # 熟練度
            self.proficiency.add_(0.001)
            self.proficiency.clamp_(0.0, 1.0)
