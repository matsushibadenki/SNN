# ファイルパス: snn_research/hybrid/__init__.py
# (更新)
# Title: ハイブリッドモデル関連モジュール
# Description: ANNとSNNを組み合わせたハイブリッドモデルに関連する
#              コンポーネント（アダプタ層など）を格納します。
# mypy --strict 準拠。

from .adapter import AnalogToSpikes, SpikesToAnalog
# --- ▼ 追加 ▼ ---
from .multimodal_projector import MultimodalProjector
# --- ▲ 追加 ▲ ---
from typing import List

__all__: List[str] = [
    "AnalogToSpikes",
    "SpikesToAnalog",
    # --- ▼ 追加 ▼ ---
    "MultimodalProjector"
    # --- ▲ 追加 ▲ ---
]