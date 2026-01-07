# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: グローバル・ワークスペース (Fix: get_context added)
# 修正: ArtificialBrainが必要とする get_context メソッドを追加。

from typing import Dict, Any, List, Callable, Optional, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """認知モジュール間の中央情報交換ハブ。"""

    def __init__(self, capacity: int = 7, model_registry: Optional[Any] = None, **kwargs):
        """
        Args:
            capacity: 意識のストリームの容量
            model_registry: モデルレジストリ（DIコンテナから注入される場合があるため追加）
            **kwargs: その他の予期せぬ引数を吸収
        """
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.conscious_broadcast_content: Optional[Any] = None
        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(
            maxlen=capacity)

        # 依存関係の保持（必要に応じて使用）
        self.model_registry = model_registry

        logger.info(f"Global Workspace initialized (Capacity: {capacity}).")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """放送の受信登録。"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        """ 情報をワークスペースへアップロード。"""
        self.blackboard[source] = {"data": data, "salience": salience}

    def get_information(self, source: str) -> Any:
        """ 特定ソースの情報を取得。"""
        info = self.blackboard.get(source)
        return info['data'] if info else None

    def get_context(self) -> Dict[str, Any]:
        """
        現在のワークスペースの状態（コンテキスト）を取得する。
        ArtificialBrainやReasoningEngineが参照するために使用。
        """
        return {
            "blackboard_snapshot": {k: v.get("data") for k, v in self.blackboard.items()},
            "current_conscious_content": self.conscious_broadcast_content,
            "recent_stream": list(self.stream_of_consciousness)
        }

    def conscious_broadcast_cycle(self):
        """最も顕著な情報を選択し、全モジュールへ放送。"""
        if not self.blackboard:
            return

        # 顕著性に基づく勝者選択 (Winner-Take-All)
        winner = max(self.blackboard.items(),
                     key=lambda x: x[1]["salience"])[0]
        content = self.blackboard[winner]["data"]

        self.conscious_broadcast_content = content
        self.stream_of_consciousness.append(
            {"source": winner, "content": content})

        for callback in self.subscribers:
            callback(winner, content)

        self.blackboard.clear()
