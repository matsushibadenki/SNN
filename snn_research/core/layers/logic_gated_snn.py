# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Prototype Aggregation)
# 内容: 教師ありヘブ学習(平均化)によるノイズ除去、バイポーラ信号処理、適応型温度

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
            
            # 初期重み
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 温度パラメータ (初期値は適度に)
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 20.0)
        else:
            # リザーバー層
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
            # 0/1 -> -1/1
            x_bipolar = (x - 0.5) * 2.0
            
            # 正規化: ベクトルの向きだけを比較するため
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # Cosine Similarity
            # 高ノイズ下でも、プロトタイプ学習が成功していれば、
            # 正解クラスとの類似度がわずかに他より高くなる
            cosine_sim = torch.matmul(x_norm, w_norm.t()) 
            
            # Adaptive Temperature Scaling
            temperature = self.adaptive_threshold.mean()
            
            # スケーリング
            scaled_sim = cosine_sim * temperature
            
            if self.training:
                # 学習中はSoftmaxで確率的勾配を流す
                spikes = F.softmax(scaled_sim, dim=1)
            else:
                # 推論時はHard Argmaxで最も確信度の高いものを選ぶ
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
        """
        Supervised Hebbian Averaging (Prototype Aggregation)
        reward引数にTarget One-Hotを受け取ることを想定。
        """
        if not self.trainable:
            return

        with torch.no_grad():
            batch_size = pre_spikes.size(0)
            
            # Input Bipolar Transform
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            # rewardがTarget One-Hotであるか確認
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                target_onehot = reward
                
                # --- Prototype Aggregation Logic ---
                # 正解クラスの重みを、現在の入力ベクトル(の平均)に近づける。
                # w_new = w_old + lr * (x - w_old)  <-- 移動平均の式
                # これを展開すると: delta = x - w_old (ただしターゲットのみ)
                # 簡略化: delta = Target^T * Input
                # これを正規化後に加算することで、重みは入力の平均方向へ向く
                
                # Positive Phase (Attraction): 正解クラスを入力へ引き寄せる
                pos_delta = torch.matmul(target_onehot.t(), pre_spikes_bipolar) / batch_size
                
                # Negative Phase (Repulsion): 不正解クラスを入力から遠ざける（マージン最大化）
                # ただし、高ノイズ時は「遠ざける」操作がノイズ学習になるリスクがあるため、
                # Attraction（正解への収束）を主成分とする。
                # Repulsionは弱めに設定、または正解との乖離（エラー）に基づく場合のみ適用。
                
                # 純粋なHebbian Aggregationを採用
                delta = pos_delta

            else:
                # Fallback for legacy calls
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size

            # Momentum Update
            momentum_factor = 0.95
            self.momentum_buffer.mul_(momentum_factor).add_(delta)
            
            # Weight Update
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Constraints & Normalization ---
            
            # 1. Centering: 重み自体のバイアスを除去
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # 2. Norm Scaling: 重みベクトルを単位球面上に配置
            # これにより、重みは「方向（プロトタイプ）」のみを表現するようになる
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # Clamp
            self.states.clamp_(-20.0, 20.0)
            
            # Temperature Auto-tuning
            if self.mode == 'readout':
                # エントロピー制御: 確信度が高まるにつれて温度を下げていく（尖らせる）のではなく、
                # 逆にCosine類似度の値が安定してくるため、温度を上げてSoftmaxをシャープにする。
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                target_entropy = 0.1 # 非常に低いエントロピー（One-hotに近い状態）を目指す
                
                # エントロピーが高すぎる(迷っている) -> 温度を上げる(差を強調する)
                # エントロピーが低い(確信している) -> 温度を維持
                temp_delta = 0.5 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                self.adaptive_threshold.clamp_(10.0, 150.0) # 上限を高めに設定
