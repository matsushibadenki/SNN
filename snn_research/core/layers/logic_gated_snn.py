# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 弾性的接続制御型・1.58ビットロジックゲートレイヤー
# 目的: 接続枠を25%に最適化し、待機状態(Potential Relief)を導入することで学習の停滞を打破する。

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
        
        # 初期状態: 閾値付近に分布させ、最初の数ステップで接続が形成されやすくする
        states = torch.normal(self.threshold - 5.0, 3.0, (out_features, in_features))
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 4.0))
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
        
        # 入力電流の計算（接続密度による正規化をマイルドに）
        conn_sum = w.sum(dim=1).clamp(min=5.0)
        current = torch.matmul(x, w.t()).view(-1) * (15.0 / torch.sqrt(conn_sum))
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_mem.mul_(0.85).add_(current)
        
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        spikes = (v_mem >= v_th).to(torch.float32)
        
        with torch.no_grad():
            # 相関トレースの蓄積
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # ホメオスタシス（ターゲット発火率 10%）
            self.adaptive_threshold.add_((spikes - 0.1) * 0.1)
            self.adaptive_threshold.clamp_(2.0, 15.0)
        
        self.membrane_potential.copy_(v_mem * (1.0 - spikes) * 0.1)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 弾性的接続制御による学習の再点火 """
        with torch.no_grad():
            # 報酬判定（コア側の微小報酬をブースト）
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率: 初期は高く、習熟するほど安定へ
            lr = 15.0 * (1.0 - prof * 0.6)
            
            if modulation > 0:
                # 成功時: 寄与したパスを大幅に強化
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時: 寄与したパスを減衰
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # --- 弾性的接続制御 (Elastic Connectivity Control) ---
            # 維持可能な接続枠を 25% に設定
            target_conn_per_neuron = int(self.in_features * 0.25)
            w = self.get_ternary_weights()
            conn_counts = w.sum(dim=1)
            
            for i in range(self.out_features):
                if conn_counts[i] > target_conn_per_neuron:
                    # 枠を超えた場合のみ、最も貢献度の低い接続を「待機状態（閾値直下）」に移動
                    excess = int(conn_counts[i] - target_conn_per_neuron)
                    valid_synapses = (w[i] > 0)
                    _, indices = torch.sort(trace[i] * valid_synapses.float(), descending=False)
                    kill_indices = indices[:excess]
                    # 完全削除(1)ではなく、復帰可能な閾値直下(threshold-1)へ
                    self.states[i, kill_indices] = self.threshold - 1.0
                
                elif conn_counts[i] < target_conn_per_neuron * 0.5:
                    # 接続が少なすぎる場合、トレースがあるものを優先的に活性化
                    deficit = int(target_conn_per_neuron * 0.5 - conn_counts[i])
                    inactive_synapses = (w[i] == 0)
                    _, indices = torch.sort(trace[i] * inactive_synapses.float(), descending=True)
                    boost_indices = indices[:deficit]
                    self.states[i, boost_indices] += 2.0

            self.states.clamp_(1, self.max_states)
