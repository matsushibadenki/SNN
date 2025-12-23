# ファイルパス: snn_research/utils/efficiency_profiler.py
# 日本語タイトル: ニューロモルフィック計算効率プロファイラ
# 目的: 従来の行列演算(MAC)と比較して、本プロジェクトの実装がどれだけ乗算を削減したかを算出する。

import torch
import torch.nn as nn
from typing import Dict, Any

class EfficiencyProfiler:
    """
    知能の「エネルギー単価」を計測。
    標準的なディープラーニングモデルとの比較を行う。
    """
    @staticmethod
    def profile_core(core: HybridNeuromorphicCore) -> Dict[str, Any]:
        """コアの計算構造を解析"""
        in_f = core.fast_process.in_features
        hid_f = core.fast_process.out_features
        out_f = core.output_gate.out_features
        
        # 1. 従来のフル精度行列積(MAC)の数
        # Layer1: in * hid, Layer2: hid * out, Deep: hid * hid (予測モデル)
        standard_macs = (in_f * hid_f) + (hid_f * out_f) + (hid_f * hid_f)
        
        # 2. 本実装での演算内訳
        # LogicGatedSNN は乗算 0、加算のみ
        logic_ops = (in_f * hid_f) + (hid_f * out_f)
        # ThermodynamicLayer はサンプリングステップ数に依存した演算
        sampling_ops = hid_f * hid_f * core.deep_process.tsu.steps
        
        # 3. 乗算削減率
        # ActivePredictiveLayerの内部予測のみに乗算が残る(改良の余地)
        multiplications = hid_f * hid_f 
        reduction_rate = 1.0 - (multiplications / standard_macs)
        
        return {
            "total_parameters": sum(p.numel() for p in core.parameters()),
            "standard_nn_macs": standard_macs,
            "actual_multiplications": multiplications,
            "multiplication_reduction_pct": reduction_rate * 100,
            "logic_gate_additions": logic_ops,
            "thermodynamic_sampling_steps": core.deep_process.tsu.steps
        }

def print_efficiency_report(core: HybridNeuromorphicCore) -> None:
    stats = EfficiencyProfiler.profile_core(core)
    print("=== Neuromorphic Efficiency Report ===")
    print(f"Total Parameters: {stats['total_parameters']:,}")
    print(f"Theoretical MACs (Standard NN): {stats['standard_nn_macs']:,}")
    print(f"Actual Multiplications: {stats['actual_multiplications']:,}")
    print(f"Multiplication Reduction: {stats['multiplication_reduction_pct']:.2f}%")
    print(f"Logic-Gated Accumulations: {stats['logic_gate_additions']:,}")
    print("======================================")