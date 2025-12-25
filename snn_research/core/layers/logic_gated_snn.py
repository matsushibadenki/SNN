# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 超・疎行列化 & 記憶保持強化)

import torch
import torch.nn as nn
import math
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化 (-1, 0, 1)。初期化時に計算して固定化（高速化）。
          - 'readout': 学習可能、連続値重み (Float)。Momentum学習則適用。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        # モードに応じた初期化
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            self.threshold = 1.0 
            trainable = True
            # 学習可能な状態変数
            states = torch.randn(out_features, in_features) * std_dev
            self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
            # Momentumバッファ
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
        else:
            # リザーバー層: 入力次元数に応じたスケーリング
            # 【修正】1.5 -> 1.0: さらに分散を小さくし、発火率を20%前後に抑制。
            # ノイズに埋もれた微弱なシグナルのみに反応する「静かな専門家」を作ることでSN比を改善する。
            std_dev = 1.0 / math.sqrt(in_features)
            self.threshold = 1.0
            trainable = False
            
            # 初期状態生成と量子化の事前実行（高速化）
            raw_states = torch.randn(out_features, in_features) * std_dev
            effective_w = self._quantize_weights(raw_states)
            self.register_buffer('frozen_weight', effective_w)
            
            # ダミー登録（互換性のため）
            self.register_buffer('synapse_states', torch.zeros(1))
            self.register_buffer('momentum_buffer', torch.zeros(1))
            
        # 膜電位モニタリング用バッファ
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def _quantize_weights(self, x: torch.Tensor) -> torch.Tensor:
        """3値量子化 (-1, 0, 1) * 0.5"""
        w = torch.zeros_like(x)
        threshold_val = 0.1 
        w[x > threshold_val] = 1.0
        w[x < -threshold_val] = -1.0
        return w * 0.5

    def get_effective_weights(self) -> torch.Tensor:
        if self.mode == 'readout':
            return self.states
        else:
            # 高速化: 事前に計算された固定重みを返す
            return self.frozen_weight

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算
        current = torch.matmul(x, w.t())
        
        # 膜電位 (Stateless)
        v_mem = current 
        
        # モニタリング用に保存
        # Mypyエラー回避のため、torch関数を使用し、明示的にTensorとして扱う
        if self.training or not self.training:
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        # 発火判定
        spikes = (v_mem >= self.threshold).float()
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # ハイパーパラメータ
            # 【修正】0.05 -> 0.025: 学習率を下げて、急激な重み変動による「微弱パターンの忘却」を防ぐ。
            lr = 0.025
            momentum = 0.9
            
            # Delta Rule
            delta = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # Momentum Update
            self.momentum_buffer.mul_(momentum).add_(delta)
            
            # 重み更新
            self.states.add_(self.momentum_buffer * lr)
            
            # 正則化 (Weight Decay equivalent)
            # 【修正】0.9999 -> 削除: 減衰を撤廃。
            # 限界領域(0.45)の記憶は非常に繊細なため、減衰させずに全て保持する戦略に切り替える。
            # Clampで爆発だけ防ぐ。
            self.states.clamp_(-self.max_states, self.max_states)
