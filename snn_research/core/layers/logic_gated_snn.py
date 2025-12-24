# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: サロゲート勾配・臨界初期化版)

import torch
import torch.nn as nn
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        
        # 閾値設定 (比較的小さくしてダイナミクスを出しやすくする)
        self.threshold = 10.0
        
        # --- 決定的な修正: 臨界初期化 (Critical Initialization) ---
        # 重みの実体である `states` を、閾値の「ギリギリ手前」と「ギリギリ奥」に配置する。
        # これにより、少しの学習ですぐに接続が ON/OFF 切り替わる「感度の高い」状態を作る。
        # 中心: threshold, 分散: 小さめ(5.0)
        # 初期接続率: 約50%
        states = torch.randn(out_features, in_features) * 5.0 + self.threshold
        self.register_buffer('synapse_states', states.clamp(-max_states, max_states))
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 1.0)) # 初期閾値は低く
        self.register_buffer('refractory_count', torch.zeros(out_features))
        self.register_buffer('proficiency', torch.zeros(1))

    @property
    def states(self) -> torch.Tensor:
        return cast(torch.Tensor, self.synapse_states)

    def get_ternary_weights(self) -> torch.Tensor:
        # 重み: 閾値を超えたら1、負の閾値を下回ったら-1
        # この「階段関数」が学習の壁になるため、学習時はサロゲート勾配を使う
        w = torch.zeros_like(self.states)
        w[self.states > self.threshold] = 1.0
        w[self.states < -self.threshold] = -1.0
        return w

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_ternary_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 入力電流
        current = torch.matmul(x, w.t())
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        ref_count = cast(torch.Tensor, self.refractory_count)
        
        is_refractory = (ref_count > 0).float().unsqueeze(0)
        effective_current = current * (1.0 - is_refractory)
        
        # 膜電位更新 (リークあり)
        new_v_mem = v_mem.unsqueeze(0) * 0.5 + effective_current
        
        # 発火
        spikes = (new_v_mem >= v_th.unsqueeze(0)).float()
        
        # 状態更新
        mean_spikes = spikes.mean(dim=0)
        new_refractory = (ref_count - 1.0).clamp(0) + mean_spikes * 2.0
        self.refractory_count.copy_(new_refractory)
        
        v_mem_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
        self.membrane_potential.copy_(v_mem_next)
        
        # 閾値恒常性: 発火しすぎたら閾値を上げ、しなさすぎたら下げる
        with torch.no_grad():
            target_activity = 0.2
            th_update = (mean_spikes - target_activity) * 0.05
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(0.5, 20.0)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor]) -> None:
        """
        サロゲート勾配を用いた強力な学習則
        """
        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1).expand(-1, self.out_features)
            
            # 1. 勾配の計算 (Delta Rule)
            # ΔW_ideal = Error * Input
            # ここでは reward を Error Signal として扱う (正解なら+誤差, 不正解なら-誤差)
            # Feedback Alignmentにより、出力層の誤差が隠れ層に適切に分配されている前提
            raw_grad = torch.matmul(reward.t(), pre_spikes) / batch_size
            
            # 2. サロゲート勾配 (Surrogate Gradient) - 最重要修正
            # 重み(states)が閾値(threshold)に近いほど、勾配を通過させる。
            # 遠い場合は「どうせ変化しない」ので勾配を消す。これにより学習を集中させる。
            # 導関数近似: f'(x) ≈ 1 - |x - threshold| (ただし正規化)
            
            dist_from_th_pos = (self.states - self.threshold).abs()
            dist_from_th_neg = (self.states + self.threshold).abs()
            
            # 閾値付近の幅 5.0 以内にある重みだけを更新対象にする
            # これがないと、遠くにある重みがノイズで勝手に動いてしまう
            surrogate_scale = 5.0
            grad_mask_pos = torch.relu(1.0 - dist_from_th_pos / surrogate_scale)
            grad_mask_neg = torch.relu(1.0 - dist_from_th_neg / surrogate_scale)
            grad_mask = torch.max(grad_mask_pos, grad_mask_neg)
            
            # 学習率: 非常に高く設定 (閾値付近を一気に飛び越えさせるため)
            lr = 20.0 * (1.0 - self.proficiency.item() * 0.8)
            
            # 最終的な更新量
            delta = raw_grad * grad_mask * lr
            
            self.states.add_(delta)
            
            # 3. 構造恒常性 (Sparsity Control)
            # 接続率が目標を超えたら、全体を減衰させる（L2正則化に近い動き）
            active_links = (self.states.abs() > self.threshold).float()
            conn_ratio = active_links.mean()
            target_conn = 0.5
            
            if conn_ratio > target_conn:
                # 接続過多なら減衰を強く
                decay = 0.005 
            else:
                # 接続不足なら減衰なし（あるいは負の減衰＝増強）
                decay = 0.0
            
            # 原点（0）に向かって減衰させる（閾値を下回らせて接続を切る）
            self.states.sub_(self.states * decay)
            
            # ノイズ（焼きなまし）
            self.states.add_(torch.randn_like(self.states) * 0.1)
            
            self.states.clamp_(-self.max_states, self.max_states)
            
            # 熟練度
            self.proficiency.add_(0.0002)
            self.proficiency.clamp_(0.0, 1.0)
