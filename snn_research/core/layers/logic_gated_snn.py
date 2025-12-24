# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 構造安定化版・1.58ビットロジックゲートレイヤー
# 目的: シナプス正規化と保護付き剪定を導入し、Conn 100%への張り付きを回避して高精度を維持する。

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
        
        # 初期状態: 20%程度のスパース性で初期化
        states = torch.randn(out_features, in_features) * 5.0 + (self.threshold - 10)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 論理積和の実行
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレースの更新（相関学習の基礎）
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス: ニューロンごとの活動バランスを維持
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(2.0, 25.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 正規化と習熟度保護を伴う学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            # 学習率: 習熟するほど小さくし、安定化させる
            lr = 12.0 * (1.0 - prof * 0.8)
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.clamp(torch.tensor(reward), -1.0, 1.0).item()
            
            if modulation > 0:
                # 成功時: 寄与したパスを強化
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時: 寄与したパスを減衰（完全削除はしない）
                self.states.sub_(trace * abs(modulation) * lr * 0.3)
            
            # シナプス正規化と保護付き剪定
            conn_weights = self.get_ternary_weights()
            conn_count = conn_weights.sum(dim=1)
            
            # ターゲット密度: 20% (習熟度に応じて微調整)
            target_count = self.in_features * 0.2
            
            # 接続数が過剰なニューロンに対し、トレース（貢献）の低い順に削る
            excess_mask = (conn_count > target_count).float().view(-1, 1)
            # 習熟度が高いほど、既存の接続を保護する（減衰を小さくする）
            decay_rate = 0.5 * (1.0 - prof)
            
            # 剪定ロジック: 貢献度が低く、かつ接続が過剰な場合に働く
            self.states.sub_(excess_mask * (1.0 - trace / 5.0) * decay_rate)
            
            # 下限値を底上げし、完全死滅を防止
            if conn_count.mean() < target_count * 0.5:
                self.states.add_(0.2)

            self.states.clamp_(1, self.max_states)
