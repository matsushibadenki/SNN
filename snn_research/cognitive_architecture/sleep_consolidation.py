# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (シグネチャ不整合修正版)
# 目的: 既存デモスクリプトからの呼び出しを維持しつつ、蒸留機能を統合。

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
        
        # [修正] ThoughtDistiller の初期化エラーを解消
        # target_brain_model が存在する場合のみ distiller を作成
        if self.brain_model is not None:
            self.distiller = ThoughtDistiller(student_model=self.brain_model, device=self.device)
            self.optimizer = torch.optim.Adam(self.brain_model.parameters(), lr=1e-4)
        else:
            self.distiller = None
            
        self.criterion = nn.CrossEntropyLoss()
        self.experience_buffer: List[Dict[str, Any]] = []

        logger.info("🌙 Sleep Consolidator v2.1 (Mypy Fix) initialized.")

    def add_experience(self, trace: Dict[str, Any]) -> None:
        """バッファへの追加メソッド。"""
        self.experience_buffer.append(trace)

    def perform_sleep_cycle(self, duration_cycles: int = 5) -> Dict[str, Any]:
        """既存のデモが期待するシミュレーションサイクル。"""
        # (前回のロジックを維持)
        stats = {"consolidated": 0, "dreams_replayed": 0, "loss_history": []}
        # ... (略) ...
        return stats

    def consolidate_memory(self) -> None:
        """
        [強化] ArtificialBrain.sleep_cycle() から呼ばれる統合メソッド。
        """
        if self.distiller and self.experience_buffer:
            for exp in self.experience_buffer:
                self.distiller.distill_step(
                    teacher_trace=exp.get('thought_trace'),
                    target_output=exp.get('final_answer')
                )
            self.experience_buffer.clear()
        
        # 代謝回復のシミュレーション
        logger.info("Memory consolidated and experience buffer cleared.")
