# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (能動学習・蒸留強化版)
# 目的: System 2 の思考トレースを System 1 (SNN) の重みへ蒸留し、記憶を固定化する。

import torch
import logging
from typing import Any, List, Dict
from snn_research.distill.thought_distiller import ThoughtDistiller

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    睡眠フェーズ中に動作し、日中の経験（思考トレース）を効率的なSNN重みへ変換する。
    """
    def __init__(self, system1_model: Any, config: Dict[str, Any]):
        self.model = system1_model
        self.config = config
        self.distiller = ThoughtDistiller(config)
        self.experience_buffer: List[Dict[str, Any]] = []

    def add_experience(self, trace: Dict[str, Any]):
        """日中の思考プロセス（CoTや外部検索結果）をバッファに追加。"""
        self.experience_buffer.append(trace)
        if len(self.experience_buffer) > 100:
            self.experience_buffer.pop(0)

    def consolidate_memory(self):
        """
        🌙 蒸留プロセスを実行。
        System 2 が導き出した「正解」を教師データとして System 1 をファインチューニングする。
        """
        if not self.experience_buffer:
            logger.info("No experiences to consolidate.")
            return

        logger.info(f"Consolidating {len(self.experience_buffer)} experiences via distillation...")
        
        try:
            # 経験バッファから学習データを生成
            for exp in self.experience_buffer:
                # BitSpikeMamba 等のモデルに対して、思考プロセスを蒸留学習
                self.distiller.distill_step(
                    student=self.model,
                    teacher_trace=exp.get('thought_trace'),
                    target_output=exp.get('final_answer')
                )
            
            # 学習完了後にバッファをクリア
            self.experience_buffer.clear()
            logger.info("Memory consolidation successful. System 1 updated.")
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    def clear_toxins(self, astrocyte: Any):
        """アストロサイトと連携し、蓄積された疲労物質をリセットする。"""
        if hasattr(astrocyte, 'replenish_energy'):
            astrocyte.replenish_energy(1000.0)
            logger.info("Metabolic waste cleared from Astrocyte Network.")
