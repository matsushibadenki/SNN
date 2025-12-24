# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: ハイブリッド精度・堅牢化版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化 (-1, 0, 1)、リカレント接続（今回はFFとして使用）
          - 'readout': 学習可能、連続値重み (Float)、高精度分類用
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        # モードに応じた初期化
        if self.mode == 'readout':
            # 読み出し層: 学習するのでゼロ付近からスタート
            std_dev = 0.01
            self.threshold = 1.0 
            trainable = True
        else:
            # リザーバー層: 固定、スパース、強めの結合
            std_dev = 5.0
            self.threshold = 1.0
            trainable = False
            
        # 重み行列 (States)
        states = torch.randn(out_features, in_features) * std_dev
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        # 学習フラグ
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_effective_weights(self) -> torch.Tensor:
        """
        モードに応じて重みを返す
        """
        if self.mode == 'readout':
            # 連続値重み (Continuous Weights)
            return self.states
        else:
            # 3値量子化重み (Ternary Weights)
            w = torch.zeros_like(self.states)
            w[self.states > self.threshold] = 1.0
            w[self.states < -self.threshold] = -1.0
            return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass
        修正点: バッチ間の状態共有（不正な平均化）を廃止し、サンプル独立性を確保。
        """
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算: (Batch, Out)
        current = torch.matmul(x, w.t())
        
        # 膜電位 (Membrane Potential)
        # 今回のタスクは静的パターンのワンショット認識であるため、
        # 前の時刻の状態を引き継ぐ必要はない（ステートレスとして扱う）。
        # これによりバッチシャッフル時のデータリークを防ぐ。
        v_mem = current 
        
        # 発火判定 (Heaviside step function)
        # Reservoirモードの場合は閾値処理でスパース性を生む
        if self.mode == 'reservoir':
            # ノイズ耐性を高めるため、リザーバー層では少し高めの閾値または活性化制御を行っても良いが
            # ここでは標準的な閾値処理を行う
            spikes = (v_mem >= 1.0).float()
        else:
            # Readoutモード
            spikes = (v_mem >= 1.0).float()
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        デルタ則による学習 (Readout層のみ有効)
        堅牢性向上: 重みの爆発を防ぐための正規化を強化
        """
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 学習率設定
            lr = 0.05
            
            # Delta Rule: ΔW = lr * Error * Input
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 更新
            self.states.add_(delta * lr)
            
            # Weight Decay (L2 Regularization) - 過学習抑制
            self.states.mul_(0.9995)
            
            # クランプ (数値安定性確保)
            self.states.clamp_(-self.max_states, self.max_states)
