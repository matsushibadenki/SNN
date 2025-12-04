# snn_research/metrics/energy.py
"""
Energy efficiency metrics for Spiking Neural Networks.
(リファクタリング)
snn_research/benchmark/metrics.py からロジックを移植し、
ダミー実装を解消。
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from torch import Tensor
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

# エネルギー消費係数（論文等で参照される一般的な値）
# 45nmプロセスにおけるシナプス演算のエネルギー消費の推定値
ENERGY_PER_SNN_OP_DEFAULT = 0.9e-12  # Joules (picojoules)
ENERGY_PER_ANN_OP_DEFAULT = 4.6e-12  # Joules (picojoules)

class EnergyMetrics:
    """SNNのエネルギー効率を測定するメトリクス"""
    
    @staticmethod
    def calculate_energy_consumption(
        avg_spikes_per_sample: float,
        num_neurons: int,
        energy_per_synop: float = ENERGY_PER_SNN_OP_DEFAULT,
    ) -> float:
        """
        推論あたりのエネルギー消費量（ジュール）を推定する。
        SNNのエネルギー消費は、主にシナプス後イベント（スパイク受信による加算）によって占められるというモデルに基づく。

        Args:
            avg_spikes_per_sample (float): 1回の推論（サンプル）あたりの平均総スパイク数。
            num_neurons (int): モデル内の総ニューロン数。
            energy_per_synop (float): 1回のシナプス演算あたりのエネルギー消費量（ジュール）。

        Returns:
            float: 推定されたエネルギー消費量（ジュール）。
        """
        # 総シナプス演算数 (SynOps) ≈ 総スパイク数
        # SNNでは、スパイクが発生したニューロンのシナプス接続先でのみ計算が発生するため、
        # 総スパイク数が計算量（＝エネルギー消費）の良い代理指標となる。
        # ここでは簡略化し、1スパイクが1シナプス演算を引き起こすと仮定する。
        # より正確には、各ニューロンのファンアウト（接続数）を考慮する必要がある。
        
        # benchmark/metrics.py のロジックを適用
        # このロジックは num_neurons を実質的なファンアウト数として使っているように見えるが、
        # 本来は avg_spikes_per_sample * avg_fan_out * energy_per_synop となるべき。
        # ここでは、num_neurons を総接続数（パラメータ数）とみなし、
        # avg_spikes_per_sample を「スパイク率」と解釈して計算する。
        # total_ops = total_parameters * avg_spike_rate
        # だが、avg_spikes_per_sample は「総スパイク数」なので、
        # benchmark/metrics.py の元のロジック (total_synops = avg_spikes_per_sample * num_neurons) は
        # おそらく (平均スパイク率 * 全ニューロン数) * (平均ファンアウト) を意図していたか、
        # あるいは (総スパイク数) * (平均ファンアウト) を意図していた。
        
        # benchmark/tasks.py では num_neurons = sum(p.numel() for p in model.parameters()) としており、
        # これは「総パラメータ数（総シナプス数）」に近い。
        # avg_spikes_per_sample は「総スパイク数 / データセットサイズ」である。
        
        # benchmark/metrics.py の元のロジックをそのまま採用する
        total_synops = avg_spikes_per_sample * num_neurons

        estimated_energy = total_synops * energy_per_synop
        
        return estimated_energy

    @staticmethod
    def compare_with_ann(snn_ops: float, ann_params: int, batch_size: int = 1) -> Dict[str, float]:
        """
        通常のANNと比較したエネルギー効率を推定。
        """
        ann_ops = float(ann_params * batch_size)
        
        snn_energy = snn_ops * ENERGY_PER_SNN_OP_DEFAULT
        ann_energy = ann_ops * ENERGY_PER_ANN_OP_DEFAULT
        
        energy_ratio = snn_energy / ann_energy if ann_energy > 0 else 0.0
        efficiency_gain = (1.0 - energy_ratio) * 100 if ann_energy > 0 else 0.0
        
        return {
            'ann_ops': ann_ops,
            'snn_estimated_energy_joules': snn_energy,
            'ann_estimated_energy_joules': ann_energy,
            'energy_ratio': energy_ratio,
            'efficiency_gain_percent': efficiency_gain
        }
