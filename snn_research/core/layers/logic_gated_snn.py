# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Lateral Inhibition & Robust Contrast)
# 内容: 側抑制(Lateral Inhibition)と適応型コントラストによるS/N比最大化、高速化実装

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファの型ヒントを明示
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor
    adaptive_threshold: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        """
        mode: 
          - 'reservoir': 固定重み、3値量子化。
          - 'readout': 学習可能、連続値重み。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            # Initial Threshold
            self.base_threshold = 0.1
            trainable = True
            
            # 直交初期化 (Orthogonal Initialization)
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            # クランプ範囲 [-20, 20]
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 適応閾値 (各ニューロンごとに独立)
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * self.base_threshold)
        else:
            # リザーバー層 (std=3.0)
            std_dev = 3.0 / math.sqrt(in_features)
            self.base_threshold = 1.0
            trainable = False
            
            if out_features >= in_features:
                w = torch.empty(out_features, in_features)
                nn.init.orthogonal_(w, gain=1.0)
                mask = (torch.rand_like(w) > 0.7).float()
                raw_states = w * mask * (std_dev * 4.0)
            else:
                raw_states = torch.randn(out_features, in_features) * std_dev

            effective_w = self._quantize_weights(raw_states)
            self.register_buffer('frozen_weight', effective_w)
            self.register_buffer('synapse_states', torch.zeros(1))
            self.register_buffer('momentum_buffer', torch.zeros(1))
            self.register_buffer('adaptive_threshold', torch.ones(1))
            
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.trainable = trainable

    @property
    def states(self) -> torch.Tensor:
        return self.synapse_states

    def _quantize_weights(self, x: torch.Tensor) -> torch.Tensor:
        """3値量子化 (-1, 0, 1) * 0.5"""
        w = torch.zeros_like(x)
        threshold_val = 0.01 
        w[x > threshold_val] = 1.0
        w[x < -threshold_val] = -1.0
        return w * 0.5

    def get_effective_weights(self) -> torch.Tensor:
        if self.mode == 'readout':
            return self.states
        else:
            return self.frozen_weight

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        w = self.get_effective_weights()
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # --- Optimized Cosine Similarity Logic ---
        if self.mode == 'readout':
            # F.normalize を使用して高速化かつ安定化 (L2 Norm)
            # eps=1e-8 でゼロ除算防止
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # コサイン類似度計算: Shape [Batch, Out]
            cosine_sim = torch.matmul(x_norm, w_norm.t())
            
            # --- Lateral Inhibition (側抑制) ---
            # バッチ内の平均的な応答(コモンモードノイズ)を除去し、
            # 「他よりも突出して似ている」信号のみを強調する。
            # 高ノイズ環境(0.45)では、正解クラスも不正解クラスも相関が低くなるため、
            # 平均を引くことで相対的な差を浮き彫りにする。
            inhibition = cosine_sim.mean(dim=1, keepdim=True)
            centered_sim = cosine_sim - inhibition
            
            # --- Robust Cubic Contrast Enhancement ---
            # 3乗則は強力だが、信号が微弱すぎると消失するリスクがある。
            # 線形項(linear term)と3乗項(cubic term)をブレンドすることで、
            # 微小な信号差を維持しつつ、強い信号をブーストする。
            # Scale x50.0 で数値を扱いやすい範囲にする。
            # sign()を掛けることで負の相関（逆パターン）も区別する。
            sim_abs = centered_sim.abs()
            # 線形項 + 3乗項。
            # 微小領域では線形が支配的になり、消失を防ぐ。
            # 大信号領域では3乗が支配的になり、WTAを加速する。
            v_mem = centered_sim.sign() * (sim_abs * 1.0 + sim_abs.pow(3) * 10.0) * 50.0
            
        else:
            # リザーバー層は線形変換のみ
            v_mem = torch.matmul(x, w.t())
        
        # --- Thresholding Logic ---
        if self.mode == 'readout':
            # 適応型閾値
            adaptive_th = self.adaptive_threshold.unsqueeze(0)
            
            # 相対的閾値 (Batch-wise Adaptive)
            # Lateral Inhibitionにより値がゼロ中心に分布しているため、
            # 正の最大値に対する相対比率で閾値を決める。
            batch_max_v, _ = v_mem.max(dim=1, keepdim=True)
            # 最大値が低い(全体的に自信がない)場合は、閾値を下げてでも拾いに行く
            relative_th = batch_max_v * 0.25 
            
            # 最終的な閾値の決定
            # adaptive_th と relative_th の小さい方を採用するが、
            # ノイズフロア(0.0付近)を拾わないように下限(0.01)を設定。
            effective_threshold = torch.min(adaptive_th, relative_th)
            effective_threshold = effective_threshold.clamp(min=0.01)

            spikes = (v_mem >= effective_threshold).float()
            
            # 学習中の閾値更新 (Homeostasis)
            if self.training:
                with torch.no_grad():
                    # 発火率の目標値を維持するように閾値を調整
                    fire_rate = spikes.mean(dim=0)
                    target_rate = 0.1 # One-hotに近い状態を目指す
                    # 閾値更新速度
                    delta = 0.02 * (fire_rate - target_rate)
                    self.adaptive_threshold.add_(delta)
                    self.adaptive_threshold.clamp_(0.01, 30.0)
        else:
            spikes = (v_mem >= self.base_threshold).float()
        
        # --- Fallback & Sharpening ---
        if self.mode == 'readout':
            # 少なくとも1つは発火させる（デッドニューロン防止）
            has_spike = spikes.sum(dim=1) > 0
            if not has_spike.all():
                no_spike_mask = ~has_spike
                # 発火しなかったサンプルについては、膜電位最大のものを強制発火
                _, max_indices = v_mem[no_spike_mask].max(dim=1)
                spikes[no_spike_mask, max_indices] = 1.0
            
            # 推論時(eval)は Sharpening (Argmax) を適用して
            # 微弱なマルチ発火を抑制し、明確な予測を行う
            if not self.training:
                final_spikes = torch.zeros_like(spikes)
                _, max_idx = v_mem.max(dim=1)
                final_spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
                spikes = final_spikes

        # 膜電位の統計記録 (デバッグ用)
        with torch.no_grad():
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            if isinstance(reward, float):
                reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
            elif reward.dim() == 1:
                reward_tensor = reward.unsqueeze(1).expand(-1, self.out_features)
            else:
                reward_tensor = reward
            
            # デルタ則による更新
            # (Target - Output) * Input に相当する成分が含まれる
            delta = torch.matmul(reward_tensor.t(), pre_spikes) / batch_size
            
            # モメンタム更新
            # ノイズ環境下では勾配が振動しやすいため、高めのモメンタム(0.99)で安定化
            momentum_factor = 0.99
            self.momentum_buffer.mul_(momentum_factor).add_(delta)
            
            # 重み更新
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Weight Normalization & Constraints ---
            
            # 1. Centering (平均除去)
            # 各ニューロンの重みベクトルを中心化し、バイアスを防ぐ
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Chaos Injection (Reduced)
            # 極小確率で微小ノイズを加え、局所解脱出を促す
            if random.random() < 0.0001: 
                noise = torch.randn_like(self.states) * 0.01 * learning_rate
                self.states.add_(noise)
            
            # 3. Norm Scaling (球面射影)
            # 重みベクトルの長さを一定に保つ (Cosine類似度ベースの学習に必須)
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            # 目標ノルムは入力次元の平方根付近が安定的
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # 値の爆発を防ぐClamp
            self.states.clamp_(-20.0, 20.0)
