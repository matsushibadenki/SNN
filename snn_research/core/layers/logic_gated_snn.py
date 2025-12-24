# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 競争的排他型・1.58ビットロジックゲートレイヤー
# 目的: 接続総量にハードキャップを設け、100%飽和を物理的に回避。90%超の精度を奪還する。

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
        
        # 初期状態: 極めてスパースな状態から開始し、必要な接続だけを「拾わせる」
        states = torch.full((out_features, in_features), self.threshold - 20.0)
        self.register_buffer('synapse_states', states)
        
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
        
        # 電流入力: 接続数に応じてスケーリングし、全結合時の暴走を防ぐ
        conn_sum = w.sum(dim=1).clamp(min=1.0)
        current = torch.matmul(x, w.t()).view(-1) / torch.sqrt(conn_sum)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.8).add_(current) # 時定数を短くし、情報の鮮度を優先
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレース: 短期的な集中力を高める
            self.eligibility_trace.mul_(0.85).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス
            self.adaptive_threshold.add_((spikes - 0.05) * 0.2)
            self.adaptive_threshold.clamp_(2.0, 15.0)
        
        # 強力なリセット（ハードリセット）
        self.membrane_potential.copy_(v_mem * (1.0 - spikes))
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 競争的排他による接続飽和の絶対阻止 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.5 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            # 報酬スケーリング: 失敗の影響を強める（100%飽和への恐怖）
            modulation = torch.tanh(torch.tensor(reward)).item()
            lr = 10.0 * (1.0 - prof * 0.5)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時は「今の接続全て」に疑いをかける
                self.states.sub_(trace * abs(modulation) * lr * 1.5)
            
            # --- 競争的排他 (Competitive Exclusion) ---
            # 各ニューロンが維持できる接続数は in_features の 15% までとする
            max_conn_per_neuron = int(self.in_features * 0.15)
            w = self.get_ternary_weights()
            conn_counts = w.sum(dim=1)
            
            for i in range(self.out_features):
                if conn_counts[i] > max_conn_per_neuron:
                    # 接続過剰な場合、貢献度(trace)が低い接続を強制的に閾値以下へ叩き落とす
                    # traceが最小のインデックスを特定
                    excess = int(conn_counts[i] - max_conn_per_neuron)
                    # 既に接続されている(w=1)の中でtraceが低いものを見つける
                    valid_synapses = (w[i] > 0)
                    _, indices = torch.sort(trace[i] * valid_synapses.float(), descending=False)
                    # 下位から削る
                    kill_indices = indices[:excess]
                    self.states[i, kill_indices] = self.threshold - 10.0

            # 最小密度の維持（全死滅防止）
            if w.mean() < 0.02:
                self.states.add_(2.0)

            self.states.clamp_(1, self.max_states)
