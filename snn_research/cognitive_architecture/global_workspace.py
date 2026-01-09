# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# Title: Global Workspace (Consciousness Stream v2.1)
# Description:
# - broadcast() メソッドを追加し、ArtificialBrainとのIF不整合を解消。
# - 複数の入力を受け取り、最も顕著なものを意識に昇らせるロジックを統合。

from typing import Dict, Any, List, Callable, Optional, Deque, Union
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
        self.stream_of_consciousness: Deque[Dict[str, Any]] = deque(
            maxlen=capacity)

        self.model_registry = model_registry
        logger.info(
            f"🧠 Global Workspace initialized (Stream Capacity: {capacity}).")

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

    def broadcast(self, inputs: List[Any], context: Optional[Any] = None) -> Any:
        """
        [ArtificialBrain互換用]
        複数の入力を一括で受け取り、意識の競合(Ignition)を実行して結果を返す高レベルAPI。

        Args:
            inputs: モジュールからの入力リスト (例: [VisualFeatures, Episode])
            context: 現在のコンテキスト (PFCのゴールなど)
        Returns:
            Any: 意識に昇ったコンテンツ
        """
        # 1. Upload inputs to blackboard
        # 本来は各モジュールが個別にupload_to_workspaceを呼ぶべきだが、
        # 簡易実装としてここでまとめて登録する。
        for i, item in enumerate(inputs):
            if item is None:
                continue

            # ソース名の推定 (簡易的)
            source_name = f"input_{i}"
            salience = 0.5  # Default

            if isinstance(item, dict):
                # 辞書なら中身から推測
                if "surprise" in item:
                    salience = float(item["surprise"])  # 驚きが高いほど顕著性高
                    source_name = "episodic_memory"
                elif "features" in item:
                    salience = 0.6  # 視覚は比較的強い
                    source_name = "visual_cortex"
            elif isinstance(item, torch.Tensor):
                source_name = "sensory_tensor"
                salience = 0.4

            self.upload_to_workspace(source_name, item, salience)

        # 2. Run Cycle
        self.conscious_broadcast_cycle()

        return self.conscious_broadcast_content

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
            # Salienceに基づいて勝者を選択
            winner_source = max(self.blackboard.items(),
                                key=lambda x: x[1]["salience"])[0]
            winner_content = self.blackboard[winner_source]["data"]

        # 2. 状態更新
        self.conscious_broadcast_content = winner_content
        self.current_qualia_vector = qualia_vector  # Synthesizerから来たクオリア

        # 3. 意識のストリームへの記録
        entry = {
            "source": winner_source,
            "content": str(winner_content)[:100],  # ログ用に短縮
            "qualia_phi": qualia_vector.std().item() if qualia_vector is not None else 0.0
        }
        self.stream_of_consciousness.append(entry)

        # 4. ブロードキャスト (Top-down transmission)
        # 全サブスクライバー（各皮質領域）へ信号を送る
        for callback in self.subscribers:
            try:
                callback(winner_source, winner_content)
            except Exception as e:
                logger.warning(f"Subscriber callback failed: {e}")

        # 黒板のクリア (次の瞬間のために)
        self.blackboard.clear()
