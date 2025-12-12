# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# Title: Global Workspace with Attention Mechanism v14.1 (Fix: ModelRegistry Injection)
# Description:
#   グローバルワークスペース理論 (GWT) に基づく意識の中枢。
#   各モジュールからのボトムアップな注意（Salience）を競合させ、
#   勝者となった情報をトップダウンに放送（Broadcast）する。
#   修正: __init__ に model_registry 引数を追加し、DIコンテナからの注入に対応。

from typing import Dict, Any, List, Callable, Optional, Tuple, TYPE_CHECKING
import operator
import logging

if TYPE_CHECKING:
    from snn_research.distillation.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class AttentionHub:
    """
    Winner-Take-All競合により、最も重要な情報を選択する注意メカニズム。
    """
    def __init__(self, inhibition_strength: float = 0.5, decay_rate: float = 0.1):
        """
        Args:
            inhibition_strength (float): 選択された情報源に対する一時的な抑制（Inhibition of Return）。
            decay_rate (float): 抑制の減衰率。
        """
        self.history: List[str] = []
        self.inhibition_scores: Dict[str, float] = {}
        self.inhibition_strength = inhibition_strength
        self.decay_rate = decay_rate

    def select_winner(self, salience_signals: Dict[str, float]) -> Optional[str]:
        """
        顕著性信号と抑制履歴に基づき、注意を向けるべき「勝者」を選択する。
        """
        if not salience_signals:
            return None

        # 抑制の減衰
        for key in list(self.inhibition_scores.keys()):
            self.inhibition_scores[key] *= (1.0 - self.decay_rate)
            if self.inhibition_scores[key] < 0.01:
                del self.inhibition_scores[key]

        # 実効顕著性の計算 (Salience - Inhibition)
        adjusted_signals: Dict[str, float] = {}
        for name, raw_salience in salience_signals.items():
            inhibition = self.inhibition_scores.get(name, 0.0)
            adjusted_signals[name] = max(0.0, raw_salience - inhibition)

        # Winner-Take-All
        if not adjusted_signals:
            return None
            
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        
        # 勝者には強い抑制をかけ、次は選ばれにくくする（注意の移動を促す）
        self.inhibition_scores[winner] = self.inhibition_scores.get(winner, 0.0) + self.inhibition_strength
        
        # 履歴更新
        self.history.append(winner)
        if len(self.history) > 20:
            self.history.pop(0)

        return winner


class GlobalWorkspace:
    """
    認知アーキテクチャ全体で情報を共有する中央情報ハブ。
    """
    def __init__(self, capacity: int = 7, model_registry: Optional["ModelRegistry"] = None):
        """
        Args:
            capacity (int): ワーキングメモリの容量（同時保持可能な情報の数）。
            model_registry (ModelRegistry, optional): モデルレジストリへの参照。
                                                      意識的プロセスがスキルを検索する場合などに使用。
        """
        self.blackboard: Dict[str, Any] = {} # 現在のサイクルでアップロードされた全情報
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.attention_hub = AttentionHub()
        
        # モデルレジストリの保持 (DIコンテナから注入される)
        self.model_registry = model_registry
        
        # 現在「意識」に上っている内容
        self.conscious_broadcast_content: Optional[Any] = None
        self.current_context_source: Optional[str] = None
        
        logger.info("✨ Global Workspace (Consciousness Hub) initialized.")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """モジュールがブロードキャストを受信するためのコールバックを登録"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        """
        各モジュールが情報をワークスペースにアップロードする。
        salience (0.0~1.0) はその情報の「重要度・緊急度」。
        """
        # logger.debug(f"[GW Upload] Source: {source}, Salience: {salience:.2f}")
        self.blackboard[source] = {"data": data, "salience": salience}

    def conscious_broadcast_cycle(self):
        """
        意識的情報処理サイクルを実行する。
        1. 顕著性マップの作成
        2. 注意の勝者決定
        3. 全システムへのブロードキャスト
        """
        if not self.blackboard:
            self.conscious_broadcast_content = None
            return

        # 1. 顕著性信号の収集
        salience_signals = {
            source: info["salience"]
            for source, info in self.blackboard.items()
        }

        # 2. 注意の勝者を選択
        winner = self.attention_hub.select_winner(salience_signals)

        if winner and winner in self.blackboard:
            # 3. ブロードキャスト
            winner_info = self.blackboard[winner]
            self.conscious_broadcast_content = winner_info['data']
            self.current_context_source = winner
            
            # 生データにメタデータを付与して送信
            broadcast_packet = self.conscious_broadcast_content
            if isinstance(broadcast_packet, dict):
                broadcast_packet = broadcast_packet.copy()
                broadcast_packet["source_module"] = winner

            logger.info(f"📡 CONSCIOUS BROADCAST: <{winner}> took the stage (Salience: {winner_info['salience']:.2f})")
            self._notify_subscribers(winner, broadcast_packet)
        else:
            self.conscious_broadcast_content = None

        # サイクル終了後にブラックボードをクリア（短期記憶/海馬などは別途保持するため）
        self.blackboard.clear()

    def _notify_subscribers(self, source: str, content: Any):
        """全サブスクライバーに同期的に通知"""
        for callback in self.subscribers:
            try:
                callback(source, content)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}", exc_info=True)

    def get_information(self, source: str) -> Any:
        """特定のソースからの直近の情報を取得する（デバッグ・補完用）"""
        # 注意: blackboardは毎サイクルクリアされるため、サイクル内でのみ有効
        info = self.blackboard.get(source)
        return info['data'] if info else None