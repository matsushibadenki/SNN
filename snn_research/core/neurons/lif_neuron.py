# snn_research/core/neurons/lif_neuron.py
# Title: Advanced LIF Neuron (Phase 2 Optimized)
# Description:
#   Stepモード(T=1, RNN的利用)とMulti-stepモード(T>1, Transformer的利用)をサポート。
#   Phase 2目標達成のため、以下の改良を実施:
#   1. 減衰計算を指数関数的減衰 (exp(-dt/tau)) に変更し、学習安定性を向上。
#   2. メモリ割り当てを最適化し、推論レイテンシを削減。
#   3. 学習可能な時定数 (Parametric LIF) への拡張性を確保。

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..surrogates import surrogate_factory


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with Spatiotemporal Support

    Args:
        tau_mem (float): 膜電位の時定数 (初期値)
        v_threshold (float): 発火閾値
        v_reset (float): リセット電位
        v_rest (float): 静止膜電位
        dt (float): シミュレーション時間刻み
        detach_reset (bool): リセット時の勾配を切断するかどうか (Surrogate Gradientに必須)
        step_mode (str): 's' (single-step) or 'm' (multi-step)
        surrogate_name (str): サロゲート関数の種類 ('atan', 'sigmoid', 'piecewise', 'fast_sigmoid')
        surrogate_alpha (float): サロゲート関数の鋭さパラメータ
        trainable (bool): tau_mem を学習可能にするか (PLIF化, 学習安定性向上に寄与)
    """

    def __init__(self,
                 features: int = 0,  # 互換性維持のための引数
                 tau_mem: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 detach_reset: bool = True,
                 step_mode: str = 's',
                 surrogate_name: str = 'atan',
                 surrogate_alpha: float = 2.0,
                 trainable: bool = False):
        super().__init__()
        
        # パラメータ設定
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.is_stateful = True

        # 時定数の設定 (学習安定性向上のため exp(-dt/tau) を使用)
        # tau_mem < dt の場合の発散を防ぐため、clampなどで保護することを推奨するが、
        # ここでは初期値に対して安全な変換を行う。
        self.trainable = trainable
        if self.trainable:
            # log空間で学習させることで tau > 0 を保証する実装も一般的だが、
            # ここではシンプルに Parameter 化する
            self.tau_mem = nn.Parameter(torch.tensor(float(tau_mem)))
        else:
            self.register_buffer('tau_mem', torch.tensor(float(tau_mem)))

        # サロゲート関数の生成
        self.surrogate_function = surrogate_factory(
            surrogate_name, surrogate_alpha)

        # 内部状態
        self.membrane_potential: Optional[torch.Tensor] = None
        self.spikes: Optional[torch.Tensor] = None

    @property
    def decay_factor(self) -> torch.Tensor:
        """
        減衰係数 beta = exp(-dt / tau) を計算して返す。
        tau が学習可能な場合、Forward ごとに計算グラフに組み込まれる。
        """
        # ゼロ除算防止のための微小値加算
        return torch.exp(-self.dt / (self.tau_mem + 1e-6))

    def set_stateful(self, stateful: bool = True):
        """
        状態保持モードの設定 (SpikingJelly等とのAPI互換性のため)
        """
        self.is_stateful = stateful

    def reset(self):
        """内部状態のリセット"""
        self.membrane_potential = None
        self.spikes = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass

        Input:
            step_mode='s': (Batch, Features...) -> T=1 ネイティブ学習に最適
            step_mode='m': (Time, Batch, Features...)
        Output:
            (spikes, membrane_potential)
        """
        if self.step_mode == 's':
            return self._forward_step(x)
        elif self.step_mode == 'm':
            return self._forward_multistep(x)
        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

    def _forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一タイムステップの処理 (高速化)
        T=1 ネイティブ学習および低レイテンシ推論用
        """
        # 状態の初期化
        if self.membrane_potential is None:
            # torch.full_like はコストがかかるため、初回のみ実行し形状チェックを行う
            self.membrane_potential = torch.full_like(x, self.v_rest)
        elif self.membrane_potential.shape != x.shape:
            # バッチサイズ変更時などの対応
            self.membrane_potential = torch.full_like(x, self.v_rest)

        mem_prev = self.membrane_potential
        
        # 減衰係数の取得 (計算グラフ対応)
        beta = self.decay_factor

        # 積分 (Integrate)
        # V[t] = V[t-1] * beta + X[t]
        # (入力 x は電流*抵抗*係数などを含む前提、あるいは x * (1-beta) などのスケーリングは層構成に依存)
        # ここでは標準的な LIF: V = V * beta + input とする
        mem_next = mem_prev * beta + x

        # 発火 (Fire)
        # サロゲート関数を通してスパイク生成 (勾配はサロゲート、値は0/1)
        spike = self.surrogate_function(mem_next - self.v_threshold)

        # リセット (Reset)
        if self.detach_reset:
            spike_for_reset = spike.detach()
        else:
            spike_for_reset = spike

        # Hard Reset: 発火したら v_reset に戻す
        # V[t] = V[t] * (1 - S[t]) + V_reset * S[t]
        mem_next = mem_next * (1.0 - spike_for_reset) + \
            self.v_reset * spike_for_reset

        # 状態更新
        self.membrane_potential = mem_next
        self.spikes = spike

        return spike, self.membrane_potential

    def _forward_multistep(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        複数タイムステップの一括処理 (Time, Batch, ...)
        学習時のBPTT展開用
        """
        T = x_seq.shape[0]
        spike_seq = []
        mem_seq = []

        # 初期状態
        if self.membrane_potential is None:
            # x_seq[0] と同じデバイス・型で初期化
            mem = torch.zeros_like(x_seq[0]) + self.v_rest
        else:
            mem = self.membrane_potential

        beta = self.decay_factor

        # 時間軸ループ
        for t in range(T):
            x = x_seq[t]

            # Integrate
            mem = mem * beta + x

            # Fire
            spike = self.surrogate_function(mem - self.v_threshold)

            # Reset
            if self.detach_reset:
                spike_reset = spike.detach()
            else:
                spike_reset = spike

            mem = mem * (1.0 - spike_reset) + self.v_reset * spike_reset

            spike_seq.append(spike)
            mem_seq.append(mem)

        # 最終状態を保存
        self.membrane_potential = mem

        # Stackして返す (Time, Batch, ...)
        return torch.stack(spike_seq, dim=0), torch.stack(mem_seq, dim=0)