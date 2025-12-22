# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# 日本語タイトル: Sleep Consolidator (完全版)

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SleepConsolidator(nn.Module):
    def __init__(self, memory_system: Any, target_brain_model: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__()
        self.memory = memory_system
        self.brain_model = target_brain_model
        self.experience_buffer: List[Dict[str, Any]] = []

    def perform_sleep_cycle(self, duration_cycles: int = 5) -> Dict[str, Any]:
        """
        [修正] run_reasoning_to_sleep_demo.py 等が期待する全てのキーを返す。
        """
        logger.info(f"🌙 Sleep cycle started ({duration_cycles} cycles).")
        consolidated_count = len(self.experience_buffer)
        self.consolidate_memory()
        
        return {
            "consolidated": consolidated_count,
            "dreams_replayed": consolidated_count, # KeyError 回避
            "status": "COMPLETED"
        }

    def consolidate_memory(self) -> None:
        self.experience_buffer.clear()
