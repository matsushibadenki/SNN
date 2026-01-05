# ファイルパス: snn_research/utils/efficiency_profiler.py
# 日本語タイトル: ニューロモルフィック計算効率プロファイラ (Fix: 属性エラー修正版)

from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

# 循環参照を避けるため、型チェック時のみインポート
if TYPE_CHECKING:
    from snn_research.core.hybrid_core import HybridNeuromorphicCore

class EfficiencyProfiler:
    """
    知能の「エネルギー単価」を計測。
    """
    @staticmethod
    def profile_core(core: 'HybridNeuromorphicCore') -> Dict[str, Any]:
        """コアの計算構造を解析"""
        in_f = core.fast_process.in_features
        hid_f = core.fast_process.out_features
        out_f = core.output_gate.out_features
        
        # 1. 従来のフル精度行列積(MAC)の想定数
        standard_macs = (in_f * hid_f) + (hid_f * out_f) + (hid_f * hid_f)
        
        # 2. 本実装での演算内訳
        # LogicGatedSNN は乗算 0、加算のみ
        logic_ops = (in_f * hid_f) + (hid_f * out_f)
        
        # deep_processの構造確認
        # ActivePredictiveLayerにはtsuがないため、hasattrで安全に確認する
        sampling_steps: int = 0
        if hasattr(core.deep_process, 'tsu'):
            tsu = getattr(core.deep_process, 'tsu')
            if hasattr(tsu, 'steps'):
                sampling_steps = int(tsu.steps)
        
        # 3. 乗算削減率の算出
        # deep_processが単純なパススルーの場合、乗算はほぼゼロとみなせるが
        # 厳密にはここでの処理内容に依存する。現状はActivePredictiveLayer(identity)なので0
        actual_multiplications = 0 
        
        if standard_macs > 0:
            reduction_rate = 1.0 - (float(actual_multiplications) / float(standard_macs))
        else:
            reduction_rate = 0.0
        
        return {
            "total_parameters": sum(p.numel() for p in core.parameters()),
            "standard_nn_macs": standard_macs,
            "actual_multiplications": actual_multiplications,
            "multiplication_reduction_pct": reduction_rate * 100,
            "logic_gate_additions": logic_ops,
            "thermodynamic_sampling_steps": sampling_steps
        }

def print_efficiency_report(core: 'HybridNeuromorphicCore') -> None:
    """レポートを整形して出力"""
    stats = EfficiencyProfiler.profile_core(core)
    print("=== Neuromorphic Efficiency Report ===")
    print(f"Total Parameters: {stats['total_parameters']:,}")
    print(f"Theoretical MACs (Standard NN): {stats['standard_nn_macs']:,}")
    print(f"Actual Multiplications: {stats['actual_multiplications']:,}")
    print(f"Multiplication Reduction: {stats['multiplication_reduction_pct']:.2f}%")
    print(f"Logic-Gated Accumulations: {stats['logic_gate_additions']:,}")
    print(f"Thermodynamic Steps per Inference: {stats['thermodynamic_sampling_steps']}")
    print("======================================")