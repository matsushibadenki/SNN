# ファイルパス: snn_research/cognitive_architecture/amygdala.py
# (更新)
#
# Title: Amygdala (扁桃体) モジュール
#
# Description:
# - mypyエラーを解消するため、Optional型を明示的にインポート・使用し、
#   メソッドに戻り値の型ヒントを追加。
# - 人工脳アーキテクチャにおける「価値評価層」の中核コンポーネント。
# - 入力された情報（テキスト）に対し、情動価（Valence: 快・不快）と
#   覚醒度（Arousal: 興奮・沈静）を評価し、スコアを算出する。
# - シンプルなキーワードベースの実装だが、将来的により高度な
#   学習済みモデルに置き換え可能なように設計されている。
#
# 改善点(v3):
# - 「意識的認知サイクル」実装のため、GlobalWorkspaceと連携。
# - 評価結果を直接返すのではなく、顕著性スコアと共にWorkspaceにアップロードする。

from typing import Dict, List, Tuple, Optional
from .global_workspace import GlobalWorkspace

class Amygdala:
    """
    テキスト情報から情動価と覚醒度を評価する扁桃体モジュール。
    """
    # クラス属性として型ヒントを定義
    emotion_lexicon: Dict[str, Tuple[float, float]]

    def __init__(self, workspace: GlobalWorkspace, emotion_lexicon: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            workspace (GlobalWorkspace): 情報をアップロードするための中央ハブ。
            emotion_lexicon (Optional[Dict[str, Tuple[float, float]]], optional):
                感情の辞書。キーは単語、値は(Valence, Arousal)のタプル。
                指定されない場合は、デフォルトの簡易辞書を使用する。
        """
        self.workspace = workspace
        if emotion_lexicon is None:
            self.emotion_lexicon = self._get_default_lexicon()
        else:
            self.emotion_lexicon = emotion_lexicon

    def _get_default_lexicon(self) -> Dict[str, Tuple[float, float]]:
        """
        デフォルトの簡易的な感情辞書を返す。
        Valence: [-1.0 (不快) ... 1.0 (快)]
        Arousal: [ 0.0 (沈静) ... 1.0 (興奮)]
        """
        return {
            # Positive / High Arousal
            "素晴らしい": (0.9, 0.8), "最高": (1.0, 0.9), "成功": (0.8, 0.7),
            "発見": (0.7, 0.8), "進化": (0.8, 0.7), "喜び": (0.9, 0.6),

            # Positive / Low Arousal
            "安定": (0.6, 0.2), "安全": (0.7, 0.1), "満足": (0.8, 0.3),
            "静か": (0.5, 0.1), "穏やか": (0.6, 0.2),

            # Negative / High Arousal
            "失敗": (-0.8, 0.8), "エラー": (-0.7, 0.7), "危険": (-0.9, 0.9),
            "攻撃": (-0.8, 0.8), "問題": (-0.6, 0.6), "不安": (-0.7, 0.7),

            # Negative / Low Arousal
            "停滞": (-0.5, 0.2), "退屈": (-0.4, 0.1), "悲しい": (-0.8, 0.3),
            "失望": (-0.7, 0.4),
        }

    def evaluate_and_upload(self, text: str) -> None:
        """
        入力テキストを評価し、情動価、覚醒度、および顕著性スコアをGlobalWorkspaceにアップロードする。
        """
        valence_scores: List[float] = []
        arousal_scores: List[float] = []

        for word, (v, a) in self.emotion_lexicon.items():
            if word in text:
                valence_scores.append(v)
                arousal_scores.append(a)

        if not valence_scores:
            avg_valence = 0.0
            avg_arousal = 0.1
        else:
            avg_valence = sum(valence_scores) / len(valence_scores)
            avg_arousal = sum(arousal_scores) / len(arousal_scores)
        
        # 顕著性スコアを計算（感情の強さ＝覚醒度と情動価の絶対値の組み合わせ）
        salience = avg_arousal + abs(avg_valence)
        
        emotion_data = {
            "type": "emotion",
            "valence": avg_valence,
            "arousal": avg_arousal
        }
        
        self.workspace.upload_to_workspace(
            source="amygdala",
            data=emotion_data,
            salience=salience
        )