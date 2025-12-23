# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (再配線・再起動版)
# 修正内容: シナプス再配線(Rewiring)を導入し、無反応状態からの自律復帰を実現する。

import torch
import torch.nn as nn
from typing import cast

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 内部状態
        self.register_buffer('synapse_states', torch.randint(
            self.threshold + 5, self.threshold + 15, (out_features, in_features)
        ).float())
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0))
        
        # 統計情報: 過去の発火頻度を追跡
        self.register_buffer('firing_history', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    @property
    def v_mem(self) -> torch.Tensor:
        return cast(torch.Tensor, self.membrane_potential)

    @property
    def v_th(self) -> torch.Tensor:
        return cast(torch.Tensor, self.adaptive_threshold)

    def get_ternary_weights(self) -> torch.Tensor:
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        
        # 自動ゲイン制御 (AGC): 入力が少なすぎる場合は強調
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        input_density = x.mean()
        if input_density < 0.1:
            x = x * (0.1 / (input_density + 1e-6))

        # 電流計算
        current = torch.matmul(x, w.t()).view(-1)
        
        # 確率的ノイズ
        noise = torch.randn_like(current) * 0.5
        potential = self.v_mem + current + noise
        
        # 発火判定
        spikes = (potential >= self.v_th).to(torch.float32)
        
        # リセット
        self.v_mem.copy_(potential * (1.0 - spikes) * 0.3)
        
        # 履歴の更新
        self.firing_history.copy_(self.firing_history * 0.9 + spikes * 0.1)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, surprise: float = 1.0) -> None:
        """能動的再配線を含む学習則"""
        with torch.no_grad():
            # 1. 驚き(Surprise)に基づく配線組み換え
            # 誤差が大きい場合、一部の重みをランダムに初期化（探索）
            if surprise > 0.05:
                # 誤差に比例した確率で配線をシャッフル
                shuffle_mask = torch.rand_like(self.states) < (surprise * 0.1)
                self.states[shuffle_mask] = torch.randint(
                    self.threshold - 5, self.threshold + 15, (shuffle_mask.sum(),)
                ).float().to(self.states.device)

            # 2. 標準的な可塑性
            correlation = torch.outer(post_spikes, pre_spikes)
            # 強化 (LTP)
            self.states.add_(correlation * 1.5)
            # 抑圧 (LTD)
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.states[depression_mask] -= 0.8

            # 3. シナプス再配線 (Active Rewiring)
            # 全く発火していないニューロン（firing_historyが低い）の結合を強制復活
            dead_neurons = self.firing_history < 0.01
            if dead_neurons.any():
                self.states[dead_neurons] += 1.0 # 結合をInclude方向へ誘導
                self.v_th[dead_neurons] *= 0.9 # 閾値を下げて発火しやすくする

            self.states.clamp_(1, self.max_states)
