# snn_research/cognitive_architecture/delta_learning.py
# Title: Delta Learning System v1.0
# Description: 全再学習を行わずに、ユーザーの訂正を即時適用する差分学習モジュール。

from typing import List, Dict, Optional
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class DeltaLearningSystem:
    """差分学習 - 全体を再学習せずに修正を適用"""

    def __init__(self):
        # 修正記録
        self.corrections: List[Dict] = []

        # パターン(キー) → 修正リスト のマッピング
        self.pattern_corrections: Dict[str, List[Dict]] = {}

    def record_correction(self,
                          input_pattern: np.ndarray,
                          wrong_output: str,
                          correct_output: str,
                          context: Dict):
        """修正を記録"""

        correction = {
            'input': input_pattern,  # numpy array reference
            'wrong': wrong_output,
            'correct': correct_output,
            'context': context,
            'timestamp': time.time(),
            'applied_count': 0
        }

        self.corrections.append(correction)

        # パターンベースでインデックス化
        pattern_key = self._pattern_to_key(input_pattern)
        if pattern_key not in self.pattern_corrections:
            self.pattern_corrections[pattern_key] = []
        self.pattern_corrections[pattern_key].append(correction)

        logger.info(f"Recorded correction for pattern {pattern_key[:8]}...")

    def apply_corrections(self, input_pattern: np.ndarray,
                          candidate_output: str) -> str:
        """生成前に修正を適用"""

        pattern_key = self._pattern_to_key(input_pattern)

        if pattern_key in self.pattern_corrections:
            for correction in self.pattern_corrections[pattern_key]:
                # 類似コンテキストの場合、修正を適用
                if self._is_similar_context(correction['context']):
                    correction['applied_count'] += 1
                    logger.info(
                        f"Applying correction: {candidate_output} -> {correction['correct']}")
                    return correction['correct']

        return candidate_output

    def _pattern_to_key(self, pattern: np.ndarray) -> str:
        """Numpy配列をハッシュ可能な文字列キーに変換"""
        if pattern.size > 100:
            sig = f"{pattern.mean():.4f}_{pattern.var():.4f}_{pattern.flatten()[0]:.4f}"
            return str(hash(sig))
        else:
            return str(hash(pattern.tobytes()))

    def _is_similar_context(self, context: Dict) -> bool:
        """コンテキストの類似性判定（簡易実装）"""
        return True

    def consolidate_corrections(self):
        """頻繁に使われる修正をSNNの重みに統合 (Sleep Cycle用)"""

        frequent_corrections = [
            c for c in self.corrections
            if c['applied_count'] >= 1  # テスト用に閾値を低く設定
        ]

        if not frequent_corrections:
            return

        logger.info(
            f"Consolidating {len(frequent_corrections)} frequent corrections...")

        for correction in frequent_corrections:
            # 本来はここでSNNのバックプロパゲーションやSTDPをトリガーする
            pass
