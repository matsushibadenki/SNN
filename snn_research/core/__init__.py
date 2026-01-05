# ファイルパス: snn_research/core/__init__.py
# (更新)
# Title: SNNコアパッケージ初期化
# Description: コアコンポーネントのエクスポート

# Base classes
try:
    from .base import BaseSNNLayer, BaseNeuron  # type: ignore[attr-defined]
except ImportError:
    # 定義されていない場合のフォールバック
    class DummyBaseSNNLayer:
        pass  # type: ignore[no-redef]
    BaseSNNLayer = DummyBaseSNNLayer

    class DummyBaseNeuron:
        pass  # type: ignore[no-redef]
    BaseNeuron = DummyBaseNeuron

# Network
try:
    from .network import SNNNetwork  # type: ignore[attr-defined]
except ImportError:
    class DummySNNNetwork:
        pass  # type: ignore[no-redef]
    SNNNetwork = DummySNNNetwork

# Core & Attention
from .snn_core import SNNCore
from .attention import SpikeDrivenSelfAttention
# --- ▼ 追加 ▼ ---
from .cortical_column import CorticalColumn
from .ensemble_scal import EnsembleSCAL, AdaptiveEnsembleSCAL, BootstrapEnsembleSCAL
# --- ▲ 追加 ▲ ---
from .adaptive_neuron_selector import AdaptiveNeuronSelector
from .adaptive_attention_selector import AdaptiveAttentionModule

__all__ = [
    "BaseSNNLayer",
    "BaseNeuron",
    "SNNNetwork",
    "SNNCore",
    "SpikeDrivenSelfAttention",
    # --- ▼ 追加 ▼ ---
    "CorticalColumn",
    "EnsembleSCAL",
    "AdaptiveEnsembleSCAL",
    "BootstrapEnsembleSCAL",
    # --- ▲ 追加 ▲ ---
    "AdaptiveNeuronSelector",
    "AdaptiveAttentionModule"
]
