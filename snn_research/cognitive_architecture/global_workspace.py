# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# (更新)
#
# Title: Global Workspace with Attention Mechanism
#
# Description:
# - mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
# - 改善点: ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、SpikeEncoderDecoderを導入。
# - 改善点 (v2): 設計図に基づき、注意機構(AttentionHub)を統合。
#              各モジュールからの誤差信号を競合させ、勝者となった情報を
#              システム全体にブロードキャストする「意識」の仕組みを実装。
# 修正点(v3): SpikeEncoderDecoderのAPI変更に伴い、メソッド呼び出しを修正。
# 改善点(v4): 「意識的認知サイクル」実装のため、salienceスコアに基づき情報を競合させ、
#              勝者となった情報をブロードキャストする機能を追加。
#              - broadcastをupload_to_workspaceに改名し、salience引数を追加。
#              - conscious_broadcast_cycleを改修し、最も顕著性の高い情報を選択・ブロードキャストするロジックを実装。


from typing import Dict, Any, List, Callable, Optional, Tuple
import random
import operator

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder

class AttentionHub:
    """
    Winner-Take-All競合により、最も重要な情報を選択する注意メカニズム。
    """
    def __init__(self, inhibition_strength: float = 0.5):
        """
        Args:
            inhibition_strength (float): 最近選択された情報源に対する抑制の強さ。
        """
        self.history: List[str] = []
        self.inhibition_strength = inhibition_strength

    def select_winner(self, salience_signals: Dict[str, float]) -> Optional[str]:
        """
        顕著性信号の大きさと過去の履歴に基づき、最も注意を向けるべき情報源（勝者）を選択する。

        Args:
            salience_signals (Dict[str, float]): 各モジュール名とその顕著性スコア。

        Returns:
            Optional[str]: 勝者となったモジュールの名前。
        """
        if not salience_signals:
            return None

        # 過去に選択された情報源に抑制をかける (Inhibition of Return)
        adjusted_signals: Dict[str, float] = {}
        for name, signal_strength in salience_signals.items():
            inhibition = self._get_inhibition_factor(name)
            adjusted_signals[name] = signal_strength * (1 - inhibition)
            if inhibition > 0:
                print(f"  - AttentionHub: '{name}' に抑制を適用 (抑制率: {inhibition:.2f})")

        # 最も顕著性が大きいモジュールを選択
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        print(f"🏆 AttentionHub: '{winner}' が注意を獲得しました (調整後顕著性: {adjusted_signals[winner]:.4f})。")

        # 履歴を更新
        self.history.append(winner)
        if len(self.history) > 10:  # 履歴の長さを制限
            self.history.pop(0)

        return winner

    def _get_inhibition_factor(self, module_name: str) -> float:
        """最近選択された頻度に基づいて抑制係数を計算する。"""
        recent_wins = self.history[-5:]  # 直近5回の履歴を参照
        win_count = recent_wins.count(module_name)
        return self.inhibition_strength * (win_count / 5)


class GlobalWorkspace:
    """
    注意機構を備え、認知アーキテクチャ全体で情報をスパイクパターンとして共有する中央情報ハブ。
    """
    def __init__(self, model_registry: ModelRegistry):
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: List[Callable] = []
        self.model_registry = model_registry
        self.encoder_decoder = SpikeEncoderDecoder()
        self.attention_hub = AttentionHub()
        self.conscious_broadcast_content: Optional[Any] = None

    def upload_to_workspace(self, source: str, data: Any, salience: float):
        """
        情報をブラックボードに書き込む（アップロードする）。
        """
        print(f"[GlobalWorkspace] '{source}' から情報を受信 (顕著性: {salience:.2f})...")
        self.blackboard[source] = {"data": data, "salience": salience}

    def conscious_broadcast_cycle(self):
        """
        意識的な情報処理サイクルを実行する。
        1. 全モジュールから顕著性信号を収集する。
        2. 注意機構が最も重要な情報（勝者）を選択する。
        3. 勝者の情報をシステム全体にブロードキャストする。
        """
        print("\n--- 意識的ブロードキャストサイクル開始 ---")
        if not self.blackboard:
            print("  - ブラックボードに情報がありません。")
            self.conscious_broadcast_content = None
            return
            
        # 1. 顕著性信号を収集
        salience_signals = {
            source: info["salience"]
            for source, info in self.blackboard.items()
        }
        print(f"  - 収集された顕著性信号: {salience_signals}")

        # 2. 注意を向ける勝者を選択
        winner = self.attention_hub.select_winner(salience_signals)

        if winner and winner in self.blackboard:
            # 3. 勝者の情報をデコードしてブロードキャスト
            winner_info = self.blackboard[winner]
            self.conscious_broadcast_content = winner_info['data']
            print(f"📡 意識的ブロードキャスト: '{winner}' からの情報を全システムに伝達します。")
            self._notify_subscribers(winner, self.conscious_broadcast_content)
        else:
            print("  - ブロードキャストするべき支配的な情報はありませんでした。")
            self.conscious_broadcast_content = None
        
        # サイクル終了後、ブラックボードをクリア
        self.blackboard.clear()
        print("--- 意識的ブロードキャストサイクル終了 ---\n")

    def subscribe(self, callback: Callable):
        """情報更新を購読するコールバックを登録する。"""
        self.subscribers.append(callback)

    def _notify_subscribers(self, source: str, conscious_info: Any):
        """全ての購読者に通知する。"""
        for callback in self.subscribers:
            try:
                callback(source, conscious_info)
            except Exception as e:
                print(f"Error notifying subscriber {callback.__name__} for '{source}': {e}")

    def get_information(self, source: str) -> Any:
        """
        ブラックボードから情報を取得する（デコードは不要）。
        """
        source_info = self.blackboard.get(source)
        return source_info['data'] if source_info else None

    def get_full_context(self) -> Dict[str, Any]:
        """
        現在のワークスペースの全コンテキストを取得する。
        """
        return {source: info['data'] for source, info in self.blackboard.items()}