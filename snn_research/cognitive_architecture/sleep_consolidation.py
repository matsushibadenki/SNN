# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (mypy型不整合修正版)
# 目的: self.distiller への None 代入による型エラーを解消し、堅牢な初期化を行う。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from snn_research.distill.thought_distiller import ThoughtDistiller

if TYPE_CHECKING:
    from snn_research.models.experimental.world_model_snn import SpikingWorldModel

logger = logging.getLogger(__name__)

class SleepConsolidator(nn.Module):
    def __init__(
        self, 
        memory_system: Any, 
        target_brain_model: Optional[nn.Module] = None, 
        world_model: Optional['SpikingWorldModel'] = None,
        device: str = 'cpu',
        consolidation_threshold: float = 0.6
    ):
        super().__init__()
        self.memory = memory_system
        self.brain_model = target_brain_model
        self.world_model = world_model
        self.device = device
        self.consolidation_threshold = consolidation_threshold
        
        # [Fix] self.distiller の型ヒントに Optional を追加して None 代入を許容する
        self.distiller: Optional[ThoughtDistiller] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        if self.brain_model is not None:
            # 実際のモデルが存在する場合のみインスタンス化
            self.distiller = ThoughtDistiller(student_model=self.brain_model, device=self.device)
            self.optimizer = torch.optim.Adam(self.brain_model.parameters(), lr=1e-4)
            
        self.criterion = nn.CrossEntropyLoss()
        self.experience_buffer: List[Dict[str, Any]] = []

        logger.info("🌙 Sleep Consolidator v2.2 (Mypy Assignment Fix) initialized.")

    def add_experience(self, trace: Dict[str, Any]) -> None:
        """日中の経験をバッファに追加。"""
        self.experience_buffer.append(trace)

    def perform_sleep_cycle(self, duration_cycles: int = 5) -> Dict[str, Any]:
        """既存のシミュレーション互換メソッド。"""
        stats: Dict[str, Any] = {"consolidated": 0, "dreams_replayed": 0, "loss_history": []}
        
        # メモリの固定化処理
        if hasattr(self.memory, 'short_term_memory'):
            # (ロジック略)
            pass
            
        return stats

    def consolidate_memory(self) -> None:
        """
        脳の睡眠サイクルから呼び出される主メソッド。
        バッファにある思考トレースを System 1 (student) へ蒸留する。
        """
        if self.distiller is not None and self.experience_buffer:
            logger.info(f"Distilling {len(self.experience_buffer)} experiences into System 1 weights...")
            for exp in self.experience_buffer:
                self.distiller.distill_step(
                    teacher_trace=exp.get('thought_trace'),
                    target_output=exp.get('final_answer')
                )
            self.experience_buffer.clear()
        else:
            logger.debug("No experiences to distill or distiller not initialized.")
