# snn_research/hardware/profiles.py
# (新規作成)
# Title: ニューロモーフィック・ハードウェアプロファイル
# Description: ROADMAPフェーズ4「ハードウェア展開」に基づき、
#              特定のハードウェアの物理特性（エネルギー消費など）を定義する。

from typing import Dict, Any

# Intel Loihiなど、45nmプロセスチップの一般的な推定値に基づく
# 参考文献: doc/SNN開発：人工脳アーキテクチャの概念設計.md
loihi_profile: Dict[str, Any] = {
    "name": "Generic Neuromorphic (Loihi-like)",
    "technology_node_nm": 45,
    # 1回のシナプス演算あたりのエネルギー消費量 (ジュール)
    "energy_per_synop": 0.9e-12, # 0.9 pJ
    # 比較対象としての同世代ANNアクセラレータの推定値
    "ann_energy_per_op": 4.6e-12, # 4.6 pJ
}

# デフォルトプロファイル
default_profile = loihi_profile

def get_hardware_profile(name: str = "default") -> Dict[str, Any]:
    """
    指定されたハードウェアプロファイルを返す。
    """
    if name == "default" or name == "loihi":
        return loihi_profile
    else:
        raise ValueError(f"Unknown hardware profile: {name}")