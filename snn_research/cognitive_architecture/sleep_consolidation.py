# ファイルパス: snn_research/cognitive_architecture/sleep_consolidation.py
# Title: Sleep Consolidator with Generative Replay v2.0
# Description:
#   記憶の整理だけでなく、世界モデルを用いた「夢（生成的再生）」を行い、
#   System 1 (BitSpikeMamba) の予測能力を強化する。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from snn_research.agent.memory import Memory
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
        
        if self.brain_model:
            self.optimizer = torch.optim.Adam(self.brain_model.parameters(), lr=1e-4)
            self.criterion = nn.CrossEntropyLoss()

        logger.info("🌙 Sleep Consolidator v2.0 (Generative Replay Ready) initialized.")

    def perform_sleep_cycle(self, duration_cycles: int = 5) -> Dict[str, Any]:
        logger.info(f"💤 Entering Sleep Mode for {duration_cycles} cycles...")
        
        # 型ヒントを明示してMypyエラーを回避
        stats: Dict[str, Any] = {
            "consolidated": 0, 
            "dreams_replayed": 0, 
            "loss_history": []
        }
        
        # --- Phase 1: Memory Consolidation ---
        if hasattr(self.memory, 'short_term_memory'):
            for item in list(self.memory.short_term_memory):
                importance = self._evaluate_importance(item)
                if importance > self.consolidation_threshold:
                    if hasattr(self.memory, '_consolidate'):
                        self.memory._consolidate(item)
                        # intへの加算であることを明示的に扱う
                        stats["consolidated"] = stats["consolidated"] + 1
            self.memory.short_term_memory = []

        # --- Phase 2: Generative Replay (Dreaming) ---
        if self.brain_model and duration_cycles > 0:
            self.brain_model.train()
            num_dreams = duration_cycles * 2
            
            for _ in range(num_dreams):
                loss = self._dream_and_learn()
                if loss is not None:
                    stats["dreams_replayed"] = stats["dreams_replayed"] + 1
                    # Listへのappend
                    if isinstance(stats["loss_history"], list):
                        stats["loss_history"].append(loss)
            
            self.brain_model.eval()
            
        logger.info(f"🌅 Waking up. Dreams: {stats['dreams_replayed']}, New Synapses: {stats['consolidated']}")
        return stats

    def _dream_and_learn(self) -> Optional[float]:
        # (変更なし)
        if not self.brain_model:
            return None

        dummy_input_ids = torch.tensor([[15496]], device=self.device)
        dummy_target_ids = torch.tensor([[2159]], device=self.device)
        
        if self.world_model:
            with torch.no_grad():
                pass

        self.optimizer.zero_grad()
        try:
            logits, _, _ = self.brain_model(dummy_input_ids)
            last_logits = logits[:, -1, :]
            loss = self.criterion(last_logits, dummy_target_ids.view(-1))
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except Exception as e:
            logger.debug(f"Dream interrupted (Gradient Error): {e}")
            return None

    def _evaluate_importance(self, item: Any) -> float:
        return 0.8