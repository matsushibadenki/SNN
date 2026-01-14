# snn_research/core/neurons/lif_neuron.py
# Title: Advanced LIF Neuron (Phase 2 Optimized)
# Description:
#   Stepモード(T=1, RNN的利用)とMulti-stepモード(T>1, Transformer的利用)をサポート。
#   Phase 2目標達成のため、以下の改良を実施:
#   1. 減衰計算を指数関数的減衰 (exp(-dt/tau)) に変更し、学習安定性を向上。
#   2. メモリ割り当てを最適化し、推論レイテンシを削減。
#   3. 学習可能な時定数 (Parametric LIF) への拡張性を確保。
#   4. Soft Resetモードの追加による情報損失の低減 (Phase 2 Accuracy対策)。

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..surrogates import surrogate_factory


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with Spatiotemporal Support

    Args:
        features (int): 特徴量次元数 (互換性用)
        tau_mem (float): 膜電位の時定数 (初期値)
        v_threshold (float): 発火閾値
        v_reset (float): リセット電位 (Hard Reset時のみ使用)
        v_rest (float): 静止膜電位
        dt (float): シミュレーション時間刻み
        detach_reset (bool): リセット時の勾配を切断するかどうか (Surrogate Gradientに必須)
        step_mode (str): 's' (single-step) or 'm' (multi-step)
        reset_mode (str): 'hard' (v_resetに強制リセット) or 'soft' (閾値を減算、情報保存重視)
        surrogate_name (str): サロゲート関数の種類 ('atan', 'sigmoid', 'piecewise', 'fast_sigmoid', 'gaussian')
        surrogate_alpha (float): サロゲート関数の鋭さパラメータ
        trainable (bool): tau_mem を学習可能にするか (PLIF化, 学習安定性向上に寄与)
    """

    def __init__(self,
                 features: int = 0,
                 tau_mem: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 detach_reset: bool = True,
                 step_mode: str = 's',
                 reset_mode: str = 'hard',
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
        self.reset_mode = reset_mode
        self.is_stateful = True

        # 時定数の設定 (学習安定性向上のため exp(-dt/tau) を使用)
        self.trainable = trainable
        if self.trainable:
            # 学習可能パラメータとして定義
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
        
        Stability Fix (Phase 2):
        学習中に tau が負または0になると発散するため、abs() + epsilon で保護する。
        """
        # ゼロ除算・負値防止のための保護
        safe_tau = self.tau_mem.abs() + 1e-6
        return torch.exp(-self.dt / safe_tau)

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

    def _apply_reset(self, mem: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        """
        リセットメカニズムの適用 (Hard vs Soft)
        """
        if self.detach_reset:
            spike_for_reset = spike.detach()
        else:
            spike_for_reset = spike

        if self.reset_mode == 'soft':
            # Soft Reset: 閾値を引く (情報の損失を防ぐ、Deep SNN推奨)
            # V[t] = V[t] - V_th * S[t]
            return mem - self.v_threshold * spike_for_reset
        else:
            # Hard Reset: リセット電位に戻す (生物学的、標準的)
            # V[t] = V[t] * (1 - S[t]) + V_reset * S[t]
            return mem * (1.0 - spike_for_reset) + self.v_reset * spike_for_reset

    def _forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一タイムステップの処理 (高速化)
        T=1 ネイティブ学習および低レイテンシ推論用
        """
        # 状態の初期化
        if self.membrane_potential is None:
            self.membrane_potential = torch.full_like(x, self.v_rest)
        elif self.membrane_potential.shape != x.shape:
            # バッチサイズ変更時などの対応
            self.membrane_potential = torch.full_like(x, self.v_rest)

        mem_prev = self.membrane_potential
        
        # 減衰係数の取得 (計算グラフ対応)
        beta = self.decay_factor

        # 積分 (Integrate)
        # V[t] = V[t-1] * beta + X[t]
        mem_next = mem_prev * beta + x

        # 発火 (Fire)
        # サロゲート関数を通してスパイク生成 (勾配はサロゲート、値は0/1)
        spike = self.surrogate_function(mem_next - self.v_threshold)

        # リセット (Reset)
        mem_next = self._apply_reset(mem_next, spike)

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
            mem = self._apply_reset(mem, spike)

            spike_seq.append(spike)
            mem_seq.append(mem)

        # 最終状態を保存
        self.membrane_potential = mem

        # Stackして返す (Time, Batch, ...)
        return torch.stack(spike_seq, dim=0), torch.stack(mem_seq, dim=0)