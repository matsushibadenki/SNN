# ファイルパス: snn_research/core/neurons/bif_neuron.py
# Title: Bistable Integrate-and-Fire (BIF) ニューロンモデル (完全版)
# Description: 
#   双安定性を持ち、確率的な挙動を示す積分発火ニューロン。
#   修正点:
#   - forwardメソッドの戻り値を (spike, mem) のタプルに変更し、
#     AdaptiveLIFNeuron など他のニューロンクラスとの互換性を確保。
#   - 省略されていた simulate_fluctuations などのメソッドを含む完全なコード。

import torch
from typing import Optional, Callable, Tuple, Dict, Any, Union
import logging

# 代理勾配関数 (学習に必要)
from spikingjelly.activation_based import surrogate, base  # type: ignore

# 追加ライブラリの安全なインポート（分析用ライブラリがない環境への配慮）
try:
    import numpy as np
    import matplotlib.pyplot as plt
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    np = None # type: ignore
    plt = None # type: ignore
    _VISUALIZATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BistableIFNeuron(base.MemoryModule):
    """
    双安定積分発火ニューロン (Bistable Integrate-and-Fire Neuron)。
    三次関数的な非線形項を持つことで、膜電位が静止電位(v_rest)と閾値付近(v_th_high)の
    2つの安定点を持つように設計されたニューロンモデルです。

    追加メソッド:
      - simulate_fluctuations(...): このインスタンス状態を汚さずに時系列シミュレーションを行う。
    """

    # 状態変数の型定義
    membrane_potential: Optional[torch.Tensor]
    spikes: Optional[torch.Tensor]

    def __init__(self,
                 features: int,
                 v_threshold_high: float = 1.0,
                 v_reset: float = 0.6,
                 tau_mem: float = 10.0,
                 bistable_strength: float = 0.25,
                 v_rest: float = 0.0,
                 unstable_equilibrium_offset: float = 0.5,
                 surrogate_function: Callable = surrogate.ATan(alpha=2.0)
                ):
        super().__init__()
        self.features = features
        self.v_th_high = v_threshold_high
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.dt = 1.0  # タイムステップ幅

        self.bistable_strength = bistable_strength
        self.v_rest = v_rest
        self.unstable_equilibrium = self.v_rest + unstable_equilibrium_offset
        self.surrogate_function = surrogate_function

        # MemoryModuleの状態フラグ（デフォルトはFalse: 各ステップでリセット）
        self.stateful = False

        # 状態変数は forward 時に動的に初期化するため、ここでは None に設定
        self.membrane_potential = None
        self.spikes = None

        # 統計用バッファ
        self.register_buffer("total_spikes", torch.tensor(0.0))

        logging.info("BistableIFNeuron: Initialized with cubic dynamics.")
        # 詳細なデバッグログはDEBUGレベルに
        logging.debug(f"  - Params: v_reset={v_reset}, bistable_k={bistable_strength}, v_unstable={self.unstable_equilibrium}")

    def set_stateful(self, stateful: bool) -> None:
        """時系列データの処理モードを設定します。"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self) -> None:
        """
        状態変数をリセットします。
        MemoryModuleの仕様に従い、内部状態をクリアします。
        """
        super().reset()
        self.membrane_potential = None
        self.spikes = None

        if hasattr(self, 'total_spikes'):
            self.total_spikes.zero_()

    def _bif_dynamics(self, v: torch.Tensor, input_current: torch.Tensor) -> torch.Tensor:
        """
        BIFニューロンの膜電位更新式（三次非線形項を含む）。
        dv/dt = -leak + cubic_non_linearity + Input
        """
        # 漏れ項
        leak = (v - self.v_rest) / self.tau_mem

        # 三次非線形項: (v - v_rest)(v - v_unstable)(v_th - v)
        non_linear = self.bistable_strength * (v - self.v_rest) * (v - self.unstable_equilibrium) * (self.v_th_high - v)

        dv = (-leak + non_linear + input_current) * self.dt
        return v + dv

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1タイムステップ分の処理を実行します。

        Args:
            input_current (torch.Tensor): 入力電流テンソル (Batch, Features, ...)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (スパイク出力, 膜電位)
            ※ 以前はスパイクのみでしたが、互換性のためタプルに変更しました。
        """
        # --- 状態の初期化チェック ---
        if not self.stateful or self.membrane_potential is None or self.membrane_potential.shape != input_current.shape:
            # 初期化時に微小なノイズを加えることでダイナミクスを活性化
            noise = torch.randn_like(input_current) * 0.05
            self.membrane_potential = (noise + self.unstable_equilibrium).to(input_current.device)

        # --- BIFの膜電位更新 ---
        # self.membrane_potential は None でないことが保証されている
        current_mem = self.membrane_potential if self.membrane_potential is not None else torch.zeros_like(input_current)
        next_potential = self._bif_dynamics(current_mem, input_current)

        # --- スパイク判定 (代理勾配を使用) ---
        spike = self.surrogate_function(next_potential - self.v_th_high)

        # --- リセット処理 (Hard Reset) ---
        reset_potential = torch.where(spike > 0.5, torch.full_like(next_potential, self.v_reset), next_potential)

        # --- 状態更新 ---
        self.membrane_potential = reset_potential
        self.spikes = spike  # 学習観測用

        # 統計情報の更新（勾配計算不要）
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        # 修正: AdaptiveLIFNeuron等と同様に (spike, mem) のタプルを返す
        return spike, self.membrane_potential

    def simulate_fluctuations(self,
                              input_sequence: Optional[torch.Tensor] = None,
                              n_steps: int = 1000,
                              batch_shape: Tuple[int, ...] = (1, 1),
                              device: Optional[torch.device] = None,
                              noise_scale: float = 0.05,
                              initial_perturb: float = 0.05,
                              commit: bool = False,
                              return_dict: bool = True
                              ) -> Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
        """
        このインスタンス単体でゆらぎの時系列シミュレーションを行うヘルパー。
        外部ライブラリ(numpy, matplotlib)がない環境ではImportErrorの代わりに警告を出して終了します。
        """
        if not _VISUALIZATION_AVAILABLE:
            logging.warning("simulate_fluctuations: numpy or matplotlib not found. Skipping simulation.")
            return {} if return_dict else (torch.tensor([]), torch.tensor([]))

        # --- 入力テンソル準備 ---
        if device is None:
            # パラメータがあればそのデバイスを使用、なければCPU
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        if input_sequence is not None:
            inp = input_sequence.to(device)
            if inp.dim() < 1:
                raise ValueError("input_sequence must have time dimension as first axis")
            T = inp.shape[0]
        else:
            T = n_steps
            inp_shape = (T, ) + batch_shape
            inp = torch.zeros(*inp_shape, device=device)

        # --- 初期膜電位のローカルコピー（self を汚さない）---
        state_shape = inp.shape[1:]
        # 初期は不安定平衡付近に微小ノイズを加える
        v0 = (torch.ones(*state_shape, device=device) * self.unstable_equilibrium) + torch.randn(*state_shape, device=device) * initial_perturb

        v = v0.clone()
        spikes_hist = []
        potentials = []

        # シミュレーションループ（勾配不要）
        with torch.no_grad():
            for t in range(T):
                current = inp[t]
                # ダイナミクス計算
                next_v = self._bif_dynamics(v, current)
                # 膜電位更新時のゆらぎ（ノイズ）を付加
                next_v = next_v + (noise_scale * torch.randn_like(next_v))
                # スパイク判定は surrogate を用いず閾値比較（シミュレーション用には決定論的にする）
                spike = (next_v >= self.v_th_high).float()
                # リセット
                v = torch.where(spike > 0.5, torch.full_like(next_v, self.v_reset), next_v)
                
                potentials.append(v.cpu().clone())
                spikes_hist.append(spike.cpu().clone())

        potentials_t = torch.stack(potentials, dim=0)  # (T, *state_shape)
        spikes_t = torch.stack(spikes_hist, dim=0)    # (T, *state_shape)

        # 統計情報（ダイナミックレンジ・std・推定ビット数など）
        # numpy計算のためにfloat変換
        pot_flat = potentials_t[int(T*0.5):].flatten().double()  # 後半を安定領域とみなす
        
        if pot_flat.numel() > 0:
            rng = float(pot_flat.max().item() - pot_flat.min().item())
            std = float(torch.std(pot_flat).item())
            estimated_levels = max(1.0, (rng / (2.0 * std))) if std > 0 else 1.0
            estimated_bits = float(np.log2(estimated_levels)) if estimated_levels > 0 else 0.0
        else:
            rng, std, estimated_levels, estimated_bits = 0.0, 0.0, 1.0, 0.0

        result = {
            "membrane_potentials": potentials_t,
            "spikes": spikes_t,
            "params": {
                "noise_scale": noise_scale,
                "initial_perturb": initial_perturb,
                "v_th_high": self.v_th_high,
                "v_reset": self.v_reset,
                "bistable_strength": self.bistable_strength
            },
            "stats": {
                "range": rng,
                "std": std,
                "estimated_levels": estimated_levels,
                "estimated_bits": estimated_bits,
                "total_spikes": float(spikes_t.sum().item())
            }
        }

        # commit オプション: 最終状態をインスタンスに反映したいときのみ書き込む
        if commit:
            self.membrane_potential = potentials_t[-1].to(device)
            self.spikes = spikes_t[-1].to(device)
            with torch.no_grad():
                self.total_spikes += torch.tensor(float(spikes_t.sum().item()), device=self.total_spikes.device)

        if return_dict:
            return result
        else:
            return potentials_t, spikes_t

# ------------------------------
# 単体実行用デモ（モジュールインポート時には実行されない）
# ------------------------------
if __name__ == "__main__":
    # 簡易デモ: ノイズを段階的に変えてプロット
    if not _VISUALIZATION_AVAILABLE:
        print("Error: numpy and matplotlib are required for this demo.")
    else:
        import os
        
        print("Running BIF fluctuation demo...")
        neuron = BistableIFNeuron(features=1)
        T = 2000
        batch_shape = (1, 1)

        noise_scales = [0.005, 0.02, 0.08, 0.2]
        fig, axes = plt.subplots(len(noise_scales), 1, figsize=(10, 3*len(noise_scales)), sharex=True)

        summaries = []
        for i, ns in enumerate(noise_scales):
            # commit=False なので neuron の内部状態は変わらない
            res = neuron.simulate_fluctuations(n_steps=T, batch_shape=batch_shape, noise_scale=ns, commit=False)
            # 型チェック回避のためのキャスト
            if isinstance(res, dict):
                pot = res["membrane_potentials"].squeeze().numpy()
                stats = res['stats']
                
                # 時系列プロット
                axes[i].plot(pot)
                axes[i].set_title(f"noise={ns} | est_bits={stats['estimated_bits']:.3f} | spikes={stats['total_spikes']:.0f}")
                axes[i].grid(True)
                summaries.append((ns, stats))

        plt.xlabel("time step")
        plt.tight_layout()
        out_png = os.path.join(os.getcwd(), "bif_fluctuation_demo.png")
        try:
            plt.savefig(out_png, dpi=150)
            print("Saved demo plot to:", out_png)
        except Exception as e:
            print(f"Failed to save plot: {e}")