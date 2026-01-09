# ファイルパス: snn_research/hybrid/__init__.py
# Title: ハイブリッドモデル関連モジュール
# Description: ANNとSNNを組み合わせたハイブリッドモデルに関連する
#              コンポーネント（アダプタ層など）を格納します。
# mypy --strict 準拠。

from .adapter import AnalogToSpikes, SpikesToAnalog
# UnifiedSensoryProjector と 互換用 MultimodalProjector をインポート
from .multimodal_projector import UnifiedSensoryProjector, MultimodalProjector
from typing import List

__all__: List[str] = [
    "AnalogToSpikes",
    "SpikesToAnalog",
    "UnifiedSensoryProjector",
    "MultimodalProjector"
]
