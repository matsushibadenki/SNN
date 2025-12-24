# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: 次元チェック強化版)
# 修正内容: update_plasticityでスカラTensorが渡された際のIndexErrorを修正

import torch
import torch.nn as nn
from typing import cast, Union

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 初期状態: 分散を大きくして多様な特徴を持たせる
        states = torch.randn(out_features, in_features) * 20.0 + (self.threshold - 5.0)
        self.register_buffer('synapse_states', states.clamp(1, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 10.0))
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
        
        # ゲイン調整
        conn_count = w.sum(dim=1).clamp(min=1.0)
        gain = 15.0 / torch.log1p(conn_count * 0.5)
        
        current = torch.matmul(x, w.t()).view(-1) * gain
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float()
        effective_current = current * (1.0 - is_refractory)
        
        noise = torch.randn_like(v_mem) * 0.5
        v_mem.mul_(0.8).add_(effective_current + noise)
        
        spikes = (v_mem >= v_th).to(torch.float32)
        
        new_refractory = (ref_count - 1.0).clamp(0) + spikes * 3.0
        self.refractory_count.copy_(new_refractory)
        
        with torch.no_grad():
            # プリ・ポストの両方の活動をトレースに記録
            self.eligibility_trace.mul_(0.9).add_(torch.outer(spikes, x.view(-1)))
            self.eligibility_trace.clamp_(0, 5.0)
            
            # 閾値ホメオスタシス
            target_activity = 0.1
            th_update = (spikes - target_activity) * 0.5
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(5.0, 40.0)
        
        v_mem_reset = v_mem * (1.0 - spikes)
        self.membrane_potential.copy_(v_mem_reset)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor] = 0.0) -> None:
        """ 
        ベクトル報酬に対応した可塑性更新則 
        修正: rewardがスカラTensor(0次元)の場合の処理を追加し、IndexErrorを防止
        """
        with torch.no_grad():
            if isinstance(reward, torch.Tensor):
                avg_reward = reward.mean().item()
            else:
                avg_reward = reward
                
            is_success = 1.0 if avg_reward > 0.0 else 0.0
            self.proficiency.copy_(self.proficiency * 0.99 + is_success * 0.01)
            prof = float(self.proficiency.item())
            
            trace = cast(torch.Tensor, self.eligibility_trace)
            
            # Modulationの計算 (ここを修正)
            # 次元数が1の場合（ベクトル）のみ unsqueeze する
            if isinstance(reward, torch.Tensor) and reward.ndim == 1:
                # ベクトル報酬: (out_features,) -> (out_features, 1)
                modulation = reward.unsqueeze(1)
            else:
                # スカラ (float) または 0-d Tensor の場合はそのまま
                modulation = reward
            
            lr = 2.0 * (1.0 - prof * 0.8)
            
            # 重み更新
            delta = trace * modulation * lr
            self.states.add_(delta)
            
            # 構造恒常性
            current_conn = (self.states > self.threshold).float().mean(dim=1, keepdim=True)
            target_conn = 0.20
            conn_error = target_conn - current_conn
            
            self.states.add_(conn_error * 2.0)
            
            # 忘却とノイズ
            self.states.mul_(0.9995)
            self.states.add_(torch.randn_like(self.states) * 0.02)

            self.states.clamp_(1, self.max_states)
