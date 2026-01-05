# ファイルパス: snn_research/metrics/energy.py
# 日本語タイトル: SNNエネルギー効率メトリクス (詳細版)
# 機能説明:
#   SNNのエネルギー消費を、スパイクコストと膜電位維持コスト(Leak)に分けて詳細に推定する。
#   45nmプロセスを想定した物理パラメータを使用。

from typing import Dict

# エネルギー消費係数 (Joules/Op) - 45nmプロセス推定
ENERGY_PER_SNN_SYNOP = 0.9e-12  # スパイク受信時の加算 (0.9 pJ)
ENERGY_PER_NEURON_UPDATE = 0.1e-12 # 膜電位のリーク更新 (0.1 pJ)
ENERGY_PER_ANN_MAC = 4.6e-12    # ANNの積和演算 (4.6 pJ)

class EnergyMetrics:
    """SNNのエネルギー効率を測定する詳細メトリクス"""
    
    @staticmethod
    def calculate_energy_consumption(
        total_spikes: float,
        num_neurons: int,
        time_steps: int,
        num_synapses: int = 0, # オプション: 接続数が分かればより正確
    ) -> float:
        """
        推論あたりのエネルギー消費量（ジュール）を詳細に推定する。
        
        Total Energy = (Spike Energy) + (Leak Energy)
        - Spike Energy: 総スパイク数 * 平均ファンアウト * シナプス演算コスト
        - Leak Energy:  総ニューロン数 * タイムステップ * 状態更新コスト

        Args:
            total_spikes (float): 推論全体での総スパイク数。
            num_neurons (int): モデル内の総ニューロン数。
            time_steps (int): シミュレーションステップ数。
            num_synapses (int): 総シナプス数（接続数）。指定がない場合は平均ファンアウト100と仮定。

        Returns:
            float: 推定されたエネルギー消費量（ジュール）。
        """
        # 平均ファンアウトの推定
        if num_synapses > 0 and num_neurons > 0:
            avg_fan_out = num_synapses / num_neurons
        else:
            avg_fan_out = 100.0 # デフォルト
            
        # 1. スパイクによる動的エネルギー (Synaptic Operations)
        # スパイク1つにつき、接続先の数だけ加算が発生する
        synops_energy = total_spikes * avg_fan_out * ENERGY_PER_SNN_SYNOP
        
        # 2. 膜電位維持による静的エネルギー (Neuron Updates)
        # スパイクがなくても、毎ステップ全ニューロンでリーク計算などが発生する
        leak_energy = num_neurons * time_steps * ENERGY_PER_NEURON_UPDATE
        
        total_energy = synops_energy + leak_energy
        
        return total_energy

    @staticmethod
    def compare_with_ann(
        snn_energy: float, 
        ann_params: int, 
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        通常のANNと比較したエネルギー効率を推定。
        ANNは全結合と仮定し、パラメータ数 * 1回の積和演算コストとする。
        """
        # ANNの総演算数 (MACs) ≈ 総パラメータ数 (全結合の場合)
        ann_ops = float(ann_params * batch_size)
        
        ann_energy = ann_ops * ENERGY_PER_ANN_MAC
        
        energy_ratio = snn_energy / ann_energy if ann_energy > 0 else 0.0
        efficiency_gain = (1.0 - energy_ratio) * 100 if ann_energy > 0 else 0.0
        
        return {
            'snn_energy_joules': snn_energy,
            'ann_energy_joules': ann_energy,
            'energy_ratio': energy_ratio,
            'efficiency_gain_percent': efficiency_gain
        }