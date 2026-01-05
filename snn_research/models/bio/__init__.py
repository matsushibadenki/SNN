# ファイルパス: snn_research/models/bio/__init__.py
# Title: 生物学的モデルパッケージ初期化 (修正版)
# Description:
#   snn_research.models.bio パッケージへのエイリアスとして機能します。
#   VisualCortex, SimpleRSNN (TemporalSNN) を公開。

from snn_research.models.bio.lif_neuron_legacy import BioLIFNeuron
from snn_research.models.bio.simple_network import BioSNN
from snn_research.models.bio.visual_cortex import VisualCortex
from snn_research.models.bio.temporal_snn import SimpleRSNN

__all__ = [
    "BioLIFNeuron", 
    "BioSNN",
    # --- ▼ 追加 ▼ ---
    "VisualCortex",
    "SimpleRSNN"
    # --- ▲ 追加 ▲ ---
]