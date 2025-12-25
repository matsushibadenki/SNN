# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Final: Bipolar Signal Processing)
# 内容: バイポーラ変換(-1/+1)によるノイズキャンセリング、デルタ則、温度スケーリング

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
            
            # 重みもバイポーラ的な挙動を期待して初期化
            states = torch.empty(out_features, in_features)
            nn.init.orthogonal_(states, gain=1.0)
            states = states * std_dev
            
            self.register_buffer('synapse_states', states.clamp(-20, 20))
            self.register_buffer('momentum_buffer', torch.zeros_like(states))
            # 温度パラメータ
            self.register_buffer('adaptive_threshold', torch.ones(out_features) * 30.0)
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
            # 0/1 のスパイク入力を -1/+1 に変換
            # ノイズ(0.5付近)は 0.0 にマップされ、ドット積で相殺されるようになる
            x_bipolar = (x - 0.5) * 2.0
            
            # Normalization (Bipolar空間での正規化)
            x_norm = F.normalize(x_bipolar, p=2, dim=1, eps=1e-8)
            w_norm = F.normalize(w, p=2, dim=1, eps=1e-8)
            
            # Cosine Similarity (Noise Cancellation Active)
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
            
            # 入力もバイポーラ化して学習に使う
            # これにより「抑制(-1)」と「興奮(+1)」が正しく重みに反映される
            pre_spikes_bipolar = (pre_spikes - 0.5) * 2.0
            
            if isinstance(reward, torch.Tensor) and reward.shape == post_spikes.shape:
                error_signal = reward
                # Delta Rule with Bipolar Input
                # Error(batch, out) * Input(batch, in) -> Delta(out, in)
                delta = torch.matmul(error_signal.t(), pre_spikes_bipolar) / batch_size
            else:
                # Fallback (Legacy)
                if isinstance(reward, float):
                    reward_tensor = torch.full((batch_size, self.out_features), reward, device=pre_spikes.device)
                else:
                    reward_tensor = reward
                delta = torch.matmul(reward_tensor.t() * post_spikes.t(), pre_spikes_bipolar) / batch_size

            # Momentum
            momentum_factor = 0.95
            self.momentum_buffer.mul_(momentum_factor).add_(delta)
            
            # Update
            self.states.add_(self.momentum_buffer * learning_rate)
            
            # --- Constraints ---
            # バイポーラ学習ではCenteringは自然に行われる傾向があるが、明示的に行うと安定する
            mean_weight = self.states.mean(dim=1, keepdim=True)
            self.states.sub_(mean_weight)
            
            # Norm Scaling
            norm = self.states.norm(p=2, dim=1, keepdim=True)
            target_norm = math.sqrt(self.in_features)
            scale_factor = target_norm / (norm + 1e-8)
            self.states.mul_(scale_factor)
            
            # Clamp
            self.states.clamp_(-20.0, 20.0)
            
            # Temperature Auto-tuning
            if self.mode == 'readout':
                entropy = -(post_spikes * (post_spikes + 1e-8).log()).sum(dim=1).mean()
                # 少しエントロピー低め(確信度高め)を目指す
                target_entropy = 0.3 
                temp_delta = 0.2 * (entropy - target_entropy)
                self.adaptive_threshold.add_(temp_delta)
                self.adaptive_threshold.clamp_(10.0, 100.0)
