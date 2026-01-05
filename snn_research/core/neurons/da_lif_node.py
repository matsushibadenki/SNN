# ファイルパス: snn_research/core/neurons/da_lif_node.py
# Title: DA-LIF (Dual Adaptive Leaky Integrate-and-Fire) Neuron
# Description:
#   ROADMAP Phase 3 Step 1 実装。
#   膜電位減衰 (tau_m) と入力電流減衰 (tau_s) の2つの時定数を学習可能にしたニューロンモデル。
#   修正: 'v' の多重登録によるAssertionErrorを回避するため、hasattrチェックを追加。
#   修正: 型アノテーションを __init__ 内または型ヒントとして適切に配置。

import torch
import torch.nn as nn
from typing import Callable, Union
from spikingjelly.activation_based import neuron, surrogate

class DualAdaptiveLIFNode(neuron.BaseNode):
    def __init__(
        self,
        tau_m_init: float = 2.0,
        tau_s_init: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Callable = surrogate.ATan(),
        detach_reset: bool = False,
        step_mode: str = 'm',
        backend: str = 'torch',
        store_v_seq: bool = False
    ):
        """
        Args:
            tau_m_init (float): 膜電位時定数の初期値
            tau_s_init (float): シナプス入力電流時定数の初期値
            v_threshold (float): 発火閾値
            v_reset (float): リセット電位
            surrogate_function (Callable): サロゲート勾配関数
            detach_reset (bool): リセット時の勾配を切るか
            step_mode (str): 'm' (multistep) or 's' (singlestep)
            backend (str): バックエンド ('torch')
            store_v_seq (bool): 膜電位系列を保存するか
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        # パラメータを対数領域で定義（正の値であることを保証するため）
        # sigmoidを通して (0, 1) の減衰係数に変換する
        self.w_m = nn.Parameter(torch.as_tensor(tau_m_init).log())
        self.w_s = nn.Parameter(torch.as_tensor(tau_s_init).log())

        # 型ヒント (mypy用)
        self.v: Union[float, torch.Tensor]
        self.i_syn: Union[float, torch.Tensor]

        # 状態変数: 膜電位(v) と シナプス電流(i_syn)
        # 親クラスですでに登録されている場合はスキップする
        if not hasattr(self, 'v'):
            self.register_memory('v', 0.)
        
        if not hasattr(self, 'i_syn'):
            self.register_memory('i_syn', 0.)

    @property
    def supported_backends(self):
        return ('torch',)

    def extra_repr(self):
        return super().extra_repr() + f', tau_m_init={self.w_m.data.exp().item():.2f}, tau_s_init={self.w_s.data.exp().item():.2f}'

    def forward(self, x: torch.Tensor):
        return super().forward(x)

    def neuronal_charge(self, x: torch.Tensor):
        """
        シングルステップの積分処理
        I[t] = I[t-1] * decay_s + x[t]
        V[t] = V[t-1] * decay_m + I[t]
        """
        decay_m = torch.sigmoid(self.w_m)
        decay_s = torch.sigmoid(self.w_s)

        if self.v_reset is None:
            pass

        self.i_syn = self.i_syn * decay_s + x
        self.v = self.v * decay_m + self.i_syn

    def neuronal_fire(self):
        """
        発火判定と膜電位リセット
        """
        spike = self.surrogate_function(self.v - self.v_threshold)
        
        # リセット処理
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1. - spike) * self.v + spike * self.v_reset

        return spike

    def forward_step(self, x: torch.Tensor):
        """
        SpikingJellyの標準インターフェース (Single Step)
        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        return spike

    def forward_m(self, x_seq: torch.Tensor):
        """
        Multi-step forward (Batch, Time, ...)
        ループを展開して計算
        """
        T = x_seq.shape[1]
        y_seq = []
        
        decay_m = torch.sigmoid(self.w_m)
        decay_s = torch.sigmoid(self.w_s)
        
        if self.v is None:
            self.v = torch.zeros_like(x_seq[:, 0])
        if self.i_syn is None:
            self.i_syn = torch.zeros_like(x_seq[:, 0])

        for t in range(T):
            x = x_seq[:, t]
            
            self.i_syn = self.i_syn * decay_s + x
            self.v = self.v * decay_m + self.i_syn
            
            spike = self.surrogate_function(self.v - self.v_threshold)
            
            if self.v_reset is None:
                self.v = self.v - spike * self.v_threshold
            else:
                self.v = (1. - spike) * self.v + spike * self.v_reset
            
            y_seq.append(spike)
            
        return torch.stack(y_seq, dim=1)