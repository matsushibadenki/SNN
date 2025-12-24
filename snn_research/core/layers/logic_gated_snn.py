# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (構造恒常性版)
# 目的: 報酬の有無に関わらず、接続率を物理的に15%〜20%の範囲に強制収束させる。

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
        
        # 初期状態: 少し高めの分散で再スタート
        states = torch.randn(out_features, in_features) * 10.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 閾値を少し下げて、発火しやすくする
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 6.0))
        self.register_buffer('eligibility_trace', torch.zeros(out_features, in_features))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        return (self.states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # ゲイン調整: 低接続時の感度を維持しつつ、過剰入力はlogで抑える
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # logスケールを採用し、Conn 3%でもConn 20%でも極端な差が出ないようにする
        gain = 8.0 / torch.log1p(conn_count * 0.5) 
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        # 不応期
        is_refractory = (ref_count > 0).float()
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位にわずかなノイズを加え、確率的発火を促す
        noise = torch.randn_like(v_mem) * 0.1
        v_mem.mul_(0.9).add_(effective_current + noise)
        
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 不応期セット (2ステップ)
        new_refractory = (ref_count - 1.0).clamp(0) + spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
            
            # ホメオスタシス: ターゲット範囲内 (10-25) に収める
            self.adaptive_threshold.add_((spikes - 0.05) * 0.5)
            self.adaptive_threshold.clamp_(3.0, 25.0)
        
        v_mem_reset = v_mem - (spikes * v_th)
        self.membrane_potential.copy_(v_mem_reset)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 構造恒常性（HSP）を取り入れた学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            lr = 2.0 * (1.0 - prof * 0.5)
            
            # 1. 報酬学習
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                self.states.sub_(trace * abs(modulation) * lr * 0.5)
            
            # 2. 強力な構造恒常性 (Homeostatic Structural Plasticity)
            # 報酬に関係なく、接続率を物理的に操作する
            current_conn = (self.states > self.threshold).float().mean(dim=1)
            target_conn = 0.15 # ターゲット 15%
            
            # ターゲットとの差分
            conn_error = target_conn - current_conn
            
            # 全シナプスに対して、不足していれば加算、超過していれば減算を行うバイアス
            # 係数を大きくし (2.0)、Conn 3% から強制的に引き上げる
            structural_bias = conn_error.unsqueeze(1) * 2.0
            self.states.add_(structural_bias)
            
            # ランダムな揺らぎ（探索）
            self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
