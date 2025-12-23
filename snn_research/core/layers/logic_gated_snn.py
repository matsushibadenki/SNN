# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (粘性・安定化版)
# 目的: Conn の激しい振動を抑制し、学習した回路を長期記憶として固定する。

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
        
        # 初期状態: 完全にランダムではなく、15% 程度をあらかじめ接続
        states = torch.full((out_features, in_features), float(self.threshold - 2))
        mask = torch.rand_like(states) < 0.15
        states[mask] = float(self.threshold + 2)
        self.register_buffer('synapse_states', states)
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0)) # 閾値を高めに設定
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        self.target_conn_rate = 0.12 # 目標 12%

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 修正1: リーク時定数を生物学的に適正化 (0.8 -> 0.9) し、情報の蓄積を促す
        v_mem.mul_(0.9).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 修正2: トレースの減衰を遅くし、因果関係をより深く刻む
            self.eligibility_trace.mul_(0.95).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火頻度のホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.05)
            self.adaptive_threshold.clamp_(1.0, 10.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 粘性と構造的摩擦を伴う学習則 """
        with torch.no_grad():
            # 習熟度の平滑化 (Acc 30% 以上でプルーニングを加速)
            is_good = 1.0 if reward > 2.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_good * 0.005)
            
            conn_rate = float(self.get_ternary_weights().mean().item())
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward / 10.0)).item()
            
            # 修正3: 重みの更新に「摩擦」を導入 (一気に値を飛ばさない)
            update_step = 2.0 # 以前より歩幅を小さく
            
            if modulation > 0:
                self.states.add_(trace * modulation * update_step)
            else:
                self.states.sub_(trace * abs(modulation) * update_step * 0.5)
            
            # 修正4: 密度の動的平衡 (急激な 100% への暴走を抑える)
            if conn_rate > self.target_conn_rate:
                # 目標を超えているときは、習熟度に応じて「不要な配線」を削る
                decay = 0.1 * (1.0 + self.proficiency.item() * 5.0)
                self.states.sub_(decay)
            elif conn_rate < 0.05:
                # 5% を切った時だけ、慎重に「発芽」させる
                revive_mask = torch.rand_like(self.states) < 0.001
                self.states[revive_mask] += 5.0

            self.states.clamp_(1, self.max_states)
