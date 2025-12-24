# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 統合最適化版・1.58ビットロジックゲートレイヤー (Fix: サロゲート勾配実装版)

import torch
import torch.nn as nn
from typing import cast, Union

# --- サロゲート勾配関数 ---
class SurrogateSpike(torch.autograd.Function):
    """
    Forward: 閾値を超えたら1、それ以外0 (Heaviside step function)
    Backward: Sigmoidの導関数で勾配を近似 (Surrogate Gradient)
    """
    scale = 10.0 # 勾配の鋭さ

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Sigmoid derivative approximation: σ(x) * (1 - σ(x))
        # inputは (v_mem - v_th)
        sigmoid_grad = (torch.sigmoid(SurrogateSpike.scale * input) * (1 - torch.sigmoid(SurrogateSpike.scale * input)) * SurrogateSpike.scale)
        return grad_output * sigmoid_grad

def surrogate_spike(input):
    return SurrogateSpike.apply(input)


class LogicGatedSNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states * 0.5
        
        # パラメータとして登録 (requires_grad=Trueになる)
        # 初期値は少し分散を持たせる
        self.synapse_states = nn.Parameter(torch.randn(out_features, in_features) * 5.0)
        
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('adaptive_threshold', torch.full((out_features,), 3.0))
        self.register_buffer('proficiency', torch.zeros(1))
        self.register_buffer('refractory_count', torch.zeros(out_features))

    def get_differentiable_weights(self) -> torch.Tensor:
        # 学習用: 重みを微分可能なSigmoidで近似 [0, 1]
        # steepnessを高くすることでバイナリに近づける
        return torch.sigmoid((self.synapse_states) * 1.0)

    def get_ternary_weights(self) -> torch.Tensor:
        # 推論用: 完全なバイナリ重み
        return (self.synapse_states > 0).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        # 学習中か推論中かで重みの生成方法を変える
        if self.training:
            w = self.get_differentiable_weights()
        else:
            w = self.get_ternary_weights()
            
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # ゲイン調整
        # w.sum(dim=1) は勾配を切る(detach)のが一般的だが、今回は構造学習も含めるため残す
        conn_count = w.sum(dim=1).clamp(min=1.0)
        gain = 10.0 / torch.log1p(conn_count * 0.2)
        
        # 電流計算 (Batch Matrix Multiplication)
        # x: (B, In), w: (Out, In) -> (B, Out)
        current = torch.matmul(x, w.t()) * gain.unsqueeze(0)
        
        v_mem = cast(torch.Tensor, self.membrane_potential)
        v_th = cast(torch.Tensor, self.adaptive_threshold)
        
        # ノイズ
        noise = torch.randn_like(current) * 0.1 if self.training else 0.0
        
        # 膜電位の積分 (Leaky Integrate)
        # 時系列を厳密に展開せず、即時応答モデルとして近似
        # v_memはバッチ方向にはブロードキャストされるが、更新は「バッチ平均」で行う必要がある
        # ここではBackpropを通すために、v_memバッファは「前の状態」として使い、
        # 計算グラフ上では新しいTensorを作る
        
        new_v_mem = v_mem.unsqueeze(0) * 0.5 + current + noise
        
        # 発火 (Surrogate Gradient)
        # v_mem - v_th > 0 なら発火
        spikes = surrogate_spike(new_v_mem - v_th.unsqueeze(0))
        
        # 状態更新 (no_grad)
        with torch.no_grad():
            # バッチ平均で内部状態を更新
            mean_spikes = spikes.mean(dim=0)
            
            # 膜電位リセット (Soft Reset)
            v_next = new_v_mem.mean(dim=0) * (1.0 - mean_spikes)
            self.membrane_potential.copy_(v_next)
            
            # 閾値ホメオスタシス
            target_activity = 0.15
            th_update = (mean_spikes - target_activity) * 0.05
            self.adaptive_threshold.add_(th_update)
            self.adaptive_threshold.clamp_(1.0, 20.0)
            
            # 熟練度更新
            if self.training:
                self.proficiency.fill_(min(1.0, self.proficiency.item() + 0.0001))

        return spikes

    def update_plasticity(self, *args, **kwargs) -> None:
        # Backpropモードでは手動更新は行わない
        pass
