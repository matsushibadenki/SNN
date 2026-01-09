# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# Title: Global Workspace (Consciousness Stream v2)
# Description:
# - Phase 6.2対応: クオリアベクトルによる意識内容の保持。
# - Thalamusへのトップダウン注意信号の生成。

from typing import Dict, Any, List, Callable, Optional, Deque
from collections import deque
import torch
import logging

logger = logging.getLogger(__name__)

class GlobalWorkspace:
    """
    認知モジュール間の中央情報交換ハブ。
    意識の「座」として機能し、クオリアを放送する。
    """

    def __init__(self, capacity: int = 7, model_registry: Optional[Any] = None, **kwargs):
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: List[Callable[[str, Any], None]] = []
        
        # 意識の内容 (Current Conscious Content)
        self.conscious_broadcast_content: Optional[Any] = None
        self.current_qualia_vector: Optional[torch.Tensor] = None
        
        # 意識の流れる履歴 (Stream of Consciousness)
        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(maxlen=capacity)

        self.model_registry = model_registry
        logger.info(f"🧠 Global Workspace initialized (Stream Capacity: {capacity}).")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """放送の受信登録。"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        """ 
        モジュールから情報をワークスペースへアップロード（ボトムアップ）。
        """
        self.blackboard[source] = {"data": data, "salience": salience}

    def get_information(self, source: str) -> Any:
        info = self.blackboard.get(source)
        return info['data'] if info else None

    def get_context(self) -> Dict[str, Any]:
        """現在のコンテキスト状態を取得。"""
        return {
            "blackboard_snapshot": {k: v.get("data") for k, v in self.blackboard.items()},
            "current_conscious_content": self.conscious_broadcast_content,
            "has_qualia": self.current_qualia_vector is not None
        }

    def conscious_broadcast_cycle(self, qualia_vector: Optional[torch.Tensor] = None):
        """
        意識のブロードキャストサイクル (Ignition)。
        最も顕著な情報を選択し、クオリアとして統合して全モジュールへ放送する。
        """
        if not self.blackboard and qualia_vector is None:
            return

        # 1. Winner-Take-All: 最も重要な入力を選択
        winner_source = "internal"
        winner_content = None
        
        if self.blackboard:
            winner_source = max(self.blackboard.items(), key=lambda x: x[1]["salience"])[0]
            winner_content = self.blackboard[winner_source]["data"]

        # 2. 状態更新
        self.conscious_broadcast_content = winner_content
        self.current_qualia_vector = qualia_vector # Synthesizerから来たクオリア
        
        # 3. 意識のストリームへの記録
        entry = {
            "source": winner_source, 
            "content": winner_content,
            "qualia_phi": qualia_vector.std().item() if qualia_vector is not None else 0.0
        }
        self.stream_of_consciousness.append(entry)

        # 4. ブロードキャスト (Top-down transmission)
        # 全サブスクライバー（各皮質領域）へ信号を送る
        for callback in self.subscribers:
            callback(winner_source, winner_content)

        # 黒板のクリア (次の瞬間のために)
        self.blackboard.clear()