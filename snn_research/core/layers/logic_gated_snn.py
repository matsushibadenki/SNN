# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (自律調整・感度最適化版)
# 目的: 閾値の高止まりと接続不足を解消し、適切な発火率と接続密度（15%）を自律的に維持する。

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
        
        # 初期状態: やや広めの分散で探索を促進
        states = torch.randn(out_features, in_features) * 15.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        # 初期閾値を低めに設定し、早期の活動開始を促す
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 5.0))
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
        
        # ゲイン調整: 接続数に対する感度を調整
        conn_count = w.sum(dim=1).clamp(min=1.0)
        # ベースゲインを上げ (8.0 -> 12.0)、対数抑制を維持
        # Connが低い(5%)時でも十分な電流を確保
        gain = 12.0 / torch.log1p(conn_count * 0.3)
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float()
        effective_current = current * (1.0 - is_refractory)
        
        # ノイズ注入: デッドロック回避
        noise = torch.randn_like(v_mem) * 0.2
        v_mem.mul_(0.9).add_(effective_current + noise)
        
        spikes = (v_mem >= v_th).to(torch.float32)
        
        # 不応期: 2ステップ
        new_refractory = (ref_count - 1.0).clamp(0) + spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        with torch.no_grad():
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 3.0)
            
            # ホメオスタシス修正:
            # 1. 上昇はマイルドに (0.5 -> 0.2)
            # 2. 自然減衰（忘却）を強化 (x0.99) して高止まりを防ぐ
            # 3. 上限を解放しつつ、下限も確保
            self.adaptive_threshold.add_((spikes * 0.5)) # 発火時のみ上昇
            self.adaptive_threshold.mul_(0.99) # 常時減衰
            self.adaptive_threshold.clamp_(3.0, 50.0)
        
        # ソフトリセット: 発火しても電位を完全には消さない
        v_mem_reset = v_mem - (spikes * v_th)
        self.membrane_potential.copy_(v_mem_reset)
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float = 0.0) -> None:
        """ 強力な構造回復力を持つ学習則 """
        with torch.no_grad():
            is_success = 1.0 if reward > 0.1 else 0.0
            self.proficiency.copy_(self.proficiency * 0.995 + is_success * 0.005)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            modulation = torch.tanh(torch.tensor(reward)).item()
            
            lr = 2.0 * (1.0 - prof * 0.5)
            
            # 1. 報酬学習 (STDP)
            if modulation > 0:
                self.states.add_(trace * modulation * lr)
            else:
                # 失敗時のペナルティを少し軽減 (0.5 -> 0.3) し、接続維持を優先
                self.states.sub_(trace * abs(modulation) * lr * 0.3)
            
            # 2. 構造恒常性 (HSP) の強化
            current_conn = (self.states > self.threshold).float().mean(dim=1)
            target_conn = 0.20 # ターゲットを20%に設定
            
            # ターゲット不足時の「強制注入」を強化 (係数 2.0 -> 4.0)
            # これにより、学習で切断されても即座に再生する
            conn_error = target_conn - current_conn
            structural_bias = conn_error.unsqueeze(1) * 4.0
            
            self.states.add_(structural_bias)
            
            # 代謝ノイズ
            self.states.add_(torch.randn_like(self.states) * 0.05)

            self.states.clamp_(1, self.max_states)
