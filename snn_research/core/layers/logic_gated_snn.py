# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: ポテンシャル蓄積型・1.58ビットロジックゲートレイヤー
# 目的: 構造の急変（壊滅的忘却）を防ぎ、洗練されたロジック形成により精度 90% 以上を達成する。

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
        
        # 初期状態: 均一な分布ではなく、中心付近に集中させて学習の余地を作る
        states = torch.normal(self.threshold - 5, 2.0, (out_features, in_features))
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # ステートを閾値で二値化したものが「論理ゲート」として機能
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力スパイクと重みの論理積和（電流入力）
        current = torch.matmul(x, w.t()).view(-1)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        # 非線形リーキー積分: 小さな信号を維持し、大きな信号で発火を促す
        v_mem.mul_(0.92).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 減衰を遅くし、時間的相関を捉えやすくする
            self.eligibility_trace.mul_(0.98).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 10.0)
            
            # 発火率のホメオスタシス: ニューロンの公平な活用
            self.adaptive_threshold.add_((spikes - 0.1) * 0.05)
            self.adaptive_threshold.clamp_(1.5, 20.0)
        
        # 発火後の不応期リセット
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.05)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 確率的保護とポテンシャル蓄積を伴う学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            # 学習ゲインの最適化
            modulation = torch.tanh(torch.tensor(reward / 5.0)).item()
            lr = 5.0 * (1.0 - prof * 0.7)
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            if modulation > 0:
                # 成功時: 寄与したポテンシャルを向上
                # 確率的に一部のシナプスを保護（更新しない）ことで構造の急変を防ぐ
                mask = (torch.rand_like(self.states) > 0.1).float()
                self.states.add_(trace * modulation * lr * mask)
            else:
                # 失敗時: 寄与したポテンシャルを慎重に減少
                self.states.sub_(trace * abs(modulation) * lr * 0.2)
            
            # 密度の緩やかな制御（ハードな減算ではなく、ポテンシャルの「勾配」を作る）
            conn_weights = self.get_ternary_weights()
            current_conn = float(conn_weights.mean().item())
            target_conn = 0.25 # 理想的なロジック密度
            
            # 密度超過時、貢献度の低い(traceが小さい)シナプスを微弱に減衰
            if current_conn > target_conn:
                decay = (1.0 - trace / 10.0) * 0.2
                self.states.sub_(decay)
            # 密度不足時、全体を微増
            elif current_conn < target_conn - 0.1:
                self.states.add_(0.1)

            self.states.clamp_(1, self.max_states)
