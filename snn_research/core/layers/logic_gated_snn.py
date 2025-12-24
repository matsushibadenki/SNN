# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: リザーバー対応・安定化版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100, trainable: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.trainable = trainable # 学習するかどうかを制御
        
        # 閾値を固定 (安定化のため)
        # 隠れ層(Reservoir)なら少し高め、出力層なら標準的
        self.threshold = 1.5 if not trainable else 0.5
        
        # 初期化
        # trainable=False (隠れ層) の場合は、最初からスパースで多様な接続を持たせる
        # trainable=True (出力層) の場合は、ゼロ付近からスタートして学習させる
        std_dev = 20.0 if not trainable else 0.1
        states = torch.randn(out_features, in_features) * std_dev
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 閾値は固定値を使うためバッファとしては持っておくが更新しない
        self.register_buffer('adaptive_threshold', torch.full((out_features,), self.threshold))
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 1.58bit (Ternary) Weights: -1, 0, 1
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
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位更新
        # リーク係数を固定 (0.9 = 情報を長く保持)
        new_v_mem = v_mem.unsqueeze(0) * 0.9 + effective_current
        
        # 発火判定 (固定閾値)
        # adaptive_thresholdは使わず、固定のself.threshold (またはそれに基づく値) を使う
        # ここでは発火しやすくするため、重み閾値とは別の「発火閾値」を設定
        firing_threshold = 1.0
        spikes = (new_v_mem >= firing_threshold).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        # 不応期
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        # リセット (ソフトリセット)
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # Homeostasis (閾値調整) は「行わない」。
        # これが学習の振動原因になるため、固定特性で学習させる。
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        デルタ則による学習 (Trainableな層のみ)
        """
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 学習率: 固定で少し高め
            lr = 0.5
            
            # Delta Rule: ΔW = lr * Error * Input
            # Errorは reward として渡される (Target - Output)
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.states.add_(delta * lr)
            
            # 減衰 (Weight Decay) - 爆発を防ぐ
            self.states.mul_(0.9999)
            
            # ノイズ
            self.states.add_(torch.randn_like(self.states) * 0.01)
            
            self.states.clamp_(-self.max_states, self.max_states)
            self.proficiency.add_(0.001)
