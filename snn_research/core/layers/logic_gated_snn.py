# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (自己再生・競合版)
# 目的: 全消滅(0%)を物理的に回避し、情報の「代謝」を回すことで高精度な認識を実現する。

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
        
        # 初期状態: 閾値付近にバラつかせ、一部が接続された状態からスタート
        states = torch.randn(out_features, in_features) * 5.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        
        # 修正1: 発火疲労(不応期)バッファ
        self.register_buffer('refractory_period', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        # 修正2: 活動依存の動的抑制
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 疲労しているニューロンは入力を受け付けにくくする
        v_mem.mul_(0.5).add_(current * (1.0 - self.refractory_period * 0.5))
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # トレース更新 (因果関係の保持)
            self.eligibility_trace.mul_(0.8).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 疲労の蓄積と回復
            self.refractory_period.add_(spikes).sub_(0.1).clamp_(0, 1.0)
        
        # リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes))
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 生存本能(自己再生)を組み込んだ学習則 """
        with torch.no_grad():
            trace = cast(torch.Tensor, self.eligibility_trace)
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 報酬のスケーリング
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 修正3: 積極的な配線強化と代謝
            if modulation > 0:
                # 成功時: 活躍した配線を強力に固定
                self.states.add_(trace * modulation * 25.0)
            else:
                # 失敗時または無報酬時: 適度に弱体化 (情報の代謝)
                self.states.sub_(0.1)

            # 修正4: 強力な自己再生メカニズム (全消滅の阻止)
            # 目標密度 10.0% を維持するように、ランダムに配線をスプラウトさせる
            target_rate = 0.10
            if conn_rate < target_rate:
                # 密度が足りないほど、発芽率を上げる
                sprout_prob = (target_rate - conn_rate) * 0.05
                sprout_mask = torch.rand_like(self.states) < sprout_prob
                self.states[sprout_mask] = float(self.threshold + 5)
            
            # 恒常的な閾値調整 (発火頻度の平準化)
            self.adaptive_threshold.add_((post_spikes - 0.05) * 0.5)
            self.adaptive_threshold.clamp_(1.0, 10.0)

            self.states.clamp_(1, self.max_states)
