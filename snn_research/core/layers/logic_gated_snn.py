# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: EMA & Decorrelation)
# 内容: EMAによるロバストなプロトタイプ学習、重み直交化、バイポーラ信号処理

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import cast, Union, Optional

class LogicGatedSNN(nn.Module):
    # バッファ型ヒント
    membrane_potential: torch.Tensor
    synapse_states: torch.Tensor
    frozen_weight: torch.Tensor
    momentum_buffer: torch.Tensor
    adaptive_threshold: torch.Tensor

    def __init__(self, in_features: int, out_features: int, max_states: int = 100, mode: str = 'reservoir') -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.mode = mode
        
        if self.mode == 'readout':
            # 読み出し層
            std_dev = 0.05
            trainable = True
            
            # 直交初期化により初期の分離能を確保
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 温度パラメータ: 初期値は低めに設定し、学習とともに上昇させる
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 20.0)
        else:
            # リザーバー層 (固定)
            std_dev = 3.0 / math.sqrt(in_features)
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
        
        if self.mode == 'readout':
            # --- Bipolar Transformation ---
            # 0/1 の入力を -1/1 に変換。ノイズ成分(0.5)を0.0にする効果がある。
            x_bipolar = (x - 0.5) * 2.0
            
            # Normalization
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # Cosine Similarity
            # 高ノイズ下でも、プロトタイプが正しく形成されていれば信号が検出できる
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # Temperature Scaling
            temperature = self.adaptive_threshold.mean()
            scaled_sim = cosine_sim * temperature
            
            if self.training:
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                spikes = torch.zeros_like(scaled_sim)
                _, max_idx = scaled_sim.max(dim=1)
                spikes.scatter_(1, max_idx.unsqueeze(1), 1.0)
            
            v_mem = scaled_sim
            
        else:
            v_mem = torch.matmul(x, w.t())
            spikes = (v_mem >= 1.0).float()
        
        with torch.no_grad():
            v_mean = torch.mean(v_mem, dim=0).detach()
            self.membrane_potential.copy_(v_mean)

        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: Union[float, torch.Tensor], learning_rate: float = 0.02) -> None:
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            # Input Bipolar Transform
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                target_onehot = reward
                
                # --- EMA Prototype Update ---
                # 各クラスに対応する入力ベクトルの平均(Centroid)を計算
                # centroid[c] = sum(target[b, c] * input[b]) / sum(target[b, c])
                
                # バッチ内の各クラスの出現回数
                class_counts = target_onehot.sum(dim=0).unsqueeze(1) + 1e-8
                
                # クラスごとの入力の合計
                class_sums = torch.matmul(target_onehot.t(), pre_spikes_bipolar)
                
                # クラスごとの平均ベクトル (Batch Centroids)
                batch_centroids = class_sums / class_counts
                
                # 正規化して方向ベクトルにする
                batch_centroids = F.normalize(batch_centroids, p=2, dim=1)
                
                # 現在の重みとの差分 (Delta)
                # w_new = w_old + lr * (centroid - w_old)
                # これにより重みはCentroidに向かって指数移動平均で近づく
                delta = batch_centroids - F.normalize(self.states, p=2, dim=1)
                
                # ターゲットが存在したクラスのみ更新するマスク
                update_mask = (target_onehot.sum(dim=0).unsqueeze(1) > 0).float()
                delta = delta * update_mask

            else:
                # Fallback (Legacy)
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size

            # 重み更新 (Momentumは使わず、純粋なEMA的な挙動にするため直接加算)
            self.states.add_(delta * learning_rate)
            
            # --- Constraints & Regularization ---
            
            # 1. Centering (平均除去)
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Decorrelation (直交化) - 非常に重要
            # 重みベクトル同士が似てしまうのを防ぎ、分離能を維持する
            # w = w - beta * (w @ w.T) @ w  (Gram-Schmidt的な効果)
            if random.random() < 0.1: # 毎回やると重いので時々実行
                w_norm = F.normalize(self.states, p=2, dim=1)
                gram_matrix = torch.matmul(w_norm, w_norm.t())
                # 対角成分(自己相関)を0にして、他との相関だけを残す
                eye = torch.eye(self.out_features, device=self.states.device)
                gram_matrix = gram_matrix * (1.0 - eye)
                # 相関がある方向成分を引く
                decorrelation_delta = torch.matmul(gram_matrix, self.states)
                self.states.sub_(decorrelation_delta * 0.05)
            
            # 3. Norm Scaling
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # Clamp
            self.states.clamp_(-20.0, 20.0)
            
            # Temperature Auto-tuning
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.2
                # エントロピーが高い(迷っている) -> 温度を上げて差を強調
                temp_delta = 0.1 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                # 温度が高すぎると不安定になるので上限を設定
                self.adaptive_threshold.clamp_(10.0, 60.0)
