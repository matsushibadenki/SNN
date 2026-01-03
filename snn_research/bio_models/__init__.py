# ファイルパス: snn_research/bio_models/__init__.py
# Title: 生物学的モデルパッケージ初期化 (修正版)
# Description:
#   snn_research.models.bio パッケージへのエイリアスとして機能します。
#   mypyエラー [import-not-found] を修正。

from snn_research.models.bio.lif_neuron_legacy import BioLIFNeuron
from snn_research.models.bio.simple_network import BioSNN

__all__ = ["BioLIFNeuron", "BioSNN"]