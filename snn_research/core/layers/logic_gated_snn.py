# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (自己鎮静・不応期版)
# 目的: 過剰興奮（てんかん状態）を鎮め、不応期によって健全なスパイク列を形成する。

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
        
        # 初期状態: やや抑制的な初期分布からスタート
        states = torch.randn(out_features, in_features) * 5.0 + (self.threshold - 2.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 10.0)) # 初期閾値を高めに
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        
        # 不応期カウンタ (Refractory Counter)
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力ゲインの適正化: 接続数が少ない時は強く、多い時は弱く（平方根則）
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # 前回の 25.0 は強すぎたため 12.0 に抑制
        gain = 12.0 / torch.sqrt(conn_count / 10.0 + 1.0)
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        # 不応期処理: カウントが残っているニューロンは入力を無視
        is_refractory = (ref_count > 0).float()
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位更新
        v_mem.mul_(0.9).add_(effective_current)
        
        # 発火判定
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 不応期リセット: 発火したら 3ステップ休む
        new_refractory = (ref_count - 1.0).clamp(0) + spikes * 3.0
        self.refractory_count.copy_(new_refractory)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
            
            # ホメオスタシス: 発火したら閾値を上げ、発火しないなら下げる
            # 上限を撤廃し、過剰入力にも耐えられるようにする
            self.adaptive_threshold.add_((spikes - 0.05) * 1.0)
            self.adaptive_threshold.mul_(0.995) # 緩やかな減衰
            self.adaptive_threshold.clamp_(5.0, 100.0) # 下限は守るが上限は広く
        
        # リセット: 発火したニューロンは閾値を引いてリセット（Soft Reset）
        # これにより情報の完全消失を防ぐ
        v_mem_reset = v_mem - (spikes * v_th)
        self.membrane_potential.copy_(v_mem_reset)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ バランス調整型学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            # 学習率: 少し落ち着かせる
            lr = 2.0 * (1.0 - prof * 0.5)
            
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時の罰則
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # 密度制御: 20-30% をターゲットに緩やかに誘導
            conn_rates = float(self.get_ternary_weights().mean().item())
            target_density = 0.25
            density_error = conn_rates - target_density
            
            # 全体的なフィードバック圧
            self.states.sub_(density_error * 0.5)
            
            # ランダムノイズ（代謝）
            self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
