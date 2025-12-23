# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (資源制限・スパース強制版)
# 目的: Conn: 100% への飽和を物理的に阻止し、認識に必要な「情報の隙間」を強制確保する。

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
        
        # 初期状態: 完全に疎な状態から開始
        self.register_buffer('synapse_states', torch.full((out_features, in_features), float(self.threshold - 5)))
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0)) # 閾値を高く
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # デジタル累積
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 修正1: リークを極限まで速める (0.8 -> 0.3)
        v_mem.mul_(0.3).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: トレースの減衰を速め、直近の因果のみを評価
            self.eligibility_trace.mul_(0.5).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 2.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes))
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 資源制限（Resource Constraint）を伴う学習則 """
        with torch.no_grad():
            trace = cast(torch.Tensor, self.eligibility_trace)
            conn_rate = float(self.get_ternary_weights().mean().item())
            
            # 修正3: 飽和に対する指数的なブレーキ
            # 10%を超えると急激に、100%付近では破壊的な負の圧力をかける
            saturation_penalty = torch.exp(torch.tensor(conn_rate * 10.0)).item() if conn_rate > 0.1 else 1.0
            
            # 報酬による更新 (報酬が正の時も、飽和ペナルティで相殺する)
            modulation = torch.tanh(torch.tensor(reward)).item()
            if modulation > 0:
                self.states.add_(trace * modulation * 5.0)
            else:
                self.states.sub_(trace * abs(modulation) * 2.0)
            
            # 修正4: 強制的なスパース化圧力 (基礎代謝)
            self.states.sub_(0.1 * saturation_penalty)
            
            # 全消滅防止 (最低限の生存)
            if conn_rate < 0.01:
                self.states.add_(0.5)

            self.states.clamp_(1, self.max_states)
