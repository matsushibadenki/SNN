# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: ハイブリッド精度・LSM対応版)

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
            self.threshold = 0.0 # 使用しないが定義
            trainable = True
        else:
            # リザーバー層: 固定、スパース、強めの結合
            std_dev = 5.0
            self.threshold = 1.0
            trainable = False
            
        states = torch.randn(out_features, in_features) * std_dev
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        # 膜電位
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        
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
            # これにより微細な境界決定が可能になる
            return self.states
        else:
            # 3値量子化重み (Ternary Weights)
            # ノイズに強いロジックゲートとして機能
            w = torch.zeros_like(self.states)
            w[self.states > self.threshold] = 1.0
            w[self.states < -self.threshold] = -1.0
            return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(x, w.t())
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        
        # 膜電位更新
        if self.mode == 'readout':
            # 読み出し層: 積分器として動作 (Leaky Integrator)
            # スパイクではなく、蓄積された電位そのもの（またはソフトマックス）を分類に使うのが一般的だが
            # ここではSNNの枠組みを守り、発火させる。ただし閾値は固定。
            
            # 減衰係数 0.8
            new_v_mem = v_mem.unsqueeze(0) * 0.8 + current
            
            # 発火判定 (単純な閾値 1.0)
            spikes = (new_v_mem >= 1.0).float()
            
            # リセット (減算リセット)
            v_mem_next = new_v_mem.mean(dim=0) - spikes.mean(dim=0) * 1.0
            
        else:
            # リザーバー層: 従来のLogicGated動作
            new_v_mem = v_mem.unsqueeze(0) * 0.5 + current # 減衰速め
            spikes = (new_v_mem >= 1.0).float()
            v_mem_next = new_v_mem.mean(dim=0) * (1.0 - spikes.mean(dim=0)) # ハードリセット

        self.membrane_potential.copy_(v_mem_next)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        デルタ則による学習 (Readout層のみ有効)
        """
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 学習率: 連続値なので小さめで安定させる
            lr = 0.05
            
            # Delta Rule: ΔW = lr * Error * Input
            # Error = reward (Target - Output)
            # pre_spikes = Input
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            self.states.add_(delta * lr)
            
            # Weight Decay (L2 Regularization)
            self.states.mul_(0.9999)
            
            # クランプ (発散防止)
            self.states.clamp_(-self.max_states, self.max_states)
