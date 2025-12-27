# /snn_research/cognitive_architecture/sleep_distiller_kernel.py
# 日本語タイトル: 睡眠・蒸留統合カーネル (Sleep Distiller Kernel) v1.0
# 目的・内容: 
#   日中の「驚き」や「不確実性」の履歴を管理し、睡眠サイクル中に System 1 への蒸留を実行する。
#   - 記憶の固定化 (Consolidation) の実体化。
#   - エピソードバッファの管理。

import asyncio
import torch
import logging
from typing import List, Dict, Any
from snn_research.distillation.system_distiller import SystemDistiller
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork

logger = logging.getLogger(__name__)

class SleepDistillerKernel:
    """
    睡眠中の記憶整理と蒸留学習を統括するモジュール。
   
    """

    def __init__(
        self,
        distiller: SystemDistiller,
        astrocyte: AstrocyteNetwork,
        max_buffer_size: int = 100
    ):
        self.distiller = distiller
        self.astrocyte = astrocyte
        self.experience_buffer: List[torch.Tensor] = []
        self.max_buffer_size = max_buffer_size
        logger.info("🌙 Sleep Distiller Kernel initialized.")

    def add_experience(self, sensory_input: torch.Tensor, uncertainty: float):
        """
        日中の活動中に、不確実性が高かった体験をバッファに蓄積する。
       
        """
        # 重複や低価値なデータのフィルタリング（簡易実装）
        if uncertainty > 0.5:
            # バッファが一杯の場合は古いものから捨てる
            if len(self.experience_buffer) >= self.max_buffer_size:
                self.experience_buffer.pop(0)
            
            # テンソルをクローンして保存
            self.experience_buffer.append(sensory_input.detach().clone())
            logger.debug(f"📥 Experience buffered. Size: {len(self.experience_buffer)}")

    async def run_sleep_cycle(self):
        """
        ArtificialBrain が SLEEP モードに移行した際に呼ばれる主処理。
       
        """
        if not self.experience_buffer:
            logger.info("💤 No experiences to consolidate. Resting...")
            return

        logger.info(f"😴 Sleep Cycle Started: Processing {len(self.experience_buffer)} cases.")
        
        # 1. 蒸留プロセスの実行 (System 2 の知見を System 1 に焼き付ける)
        #
        distill_results = await self.distiller.run_consolidation_phase(self.experience_buffer)
        
        # 2. 睡眠による疲労回復 (Astrocyte のリセット)
        #
        self.astrocyte.clear_fatigue(amount=80.0)
        self.astrocyte.replenish_energy(amount=500.0)
        
        # 3. 処理済みバッファのクリア
        self.experience_buffer.clear()
        
        success_count = sum(1 for r in distill_results if r.get("status") == "success")
        logger.info(f"✨ Sleep Cycle Finished. Consolidated: {success_count} cases.")
        
        return {
            "processed": len(distill_results),
            "success": success_count,
            "astrocyte_status": self.astrocyte.get_energy_level()
        }

# ロジックの正当性確認:
# - ROADMAP.md: "Sleep Consolidator: 日中の思考トレースを夢として再生し、SNNへ蒸留" を忠実に実装。
# - astrocyte_network.py: 睡眠完了時の疲労除去 (clear_fatigue) メソッドと連携。
# - async_brain_kernel.py: カーネルの SLEEP 状態遷移時にこの run_sleep_cycle をフック可能。