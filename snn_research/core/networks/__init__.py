# ファイルパス: snn_research/core/networks/__init__.py
# タイトル: ネットワークモジュール初期化
# 機能説明: 
#   主要なネットワークアーキテクチャをエクスポートする。
#   v2.6追加: Oscillatory Neural Network (ONN) のエクスポート。

from .abstract_snn_network import AbstractSNNNetwork
from .sequential_snn_network import SequentialSNNNetwork
from .bio_pc_network import BioPCNetwork
from .sequential_pc_network import SequentialPCNetwork
from .liquid_association_cortex import LiquidAssociationCortex

# --- Oscillatory / Phase-based Networks ---
from .oscillatory_network import OscillatoryNeuronGroup, HopfieldONN

__all__ = [
    "AbstractSNNNetwork",
    "SequentialSNNNetwork",
    "BioPCNetwork",
    "SequentialPCNetwork",
    "LiquidAssociationCortex",
    # New
    "OscillatoryNeuronGroup",
    "HopfieldONN",
]