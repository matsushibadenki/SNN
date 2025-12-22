# ファイルパス: scripts/runners/test_distillation_cycle.py
# 日本語タイトル: 統合蒸留サイクル検証デモ (パス解決版)
# 目的: インポートエラーを解消し、蒸留サイクルの動作を確認する。

import os
import sys
import torch
import logging
import asyncio

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. コンポーネントの準備
    config = {"d_model": 128, "n_layers": 2, "vocab_size": 1000}
    system1 = BitSpikeMamba(config).to(device)
    
    memory = Memory(rag_system=None)
    sleep_consolidator = SleepConsolidator(
        memory_system=memory,
        target_brain_model=system1,
        device=device
    )

    brain = ArtificialBrain(
        thinking_engine=system1,
        sleep_consolidator=sleep_consolidator,
        device=device
    )

    logger.info("🧠 Brain System initialized for Distillation Test.")

    # 2. 経験のシミュレーション
    logger.info("☀️ Daytime: Storing thought trace...")
    thought_trace = {
        "thought_trace": "The spiking potential is regulated by the astrocyte network.",
        "final_answer": torch.tensor([1]).to(device)
    }
    
    # 修正版 SleepConsolidator の add_experience を使用
    if brain.sleep_manager:
        brain.sleep_manager.add_experience(thought_trace)

    # 3. 睡眠サイクルの実行
    logger.info("🛌 Nighttime: Starting Sleep Cycle...")
    brain.sleep_cycle()
    
    # 4. 結果の確認
    progress = sleep_consolidator.get_learning_progress()
    logger.info(f"📊 Distillation Progress: {progress}")
    
    if progress["samples_processed"] > 0:
        logger.info("✅ SUCCESS: Distillation loop verified.")
    else:
        logger.error("❌ FAILURE: No samples processed.")

if __name__ == "__main__":
    asyncio.run(main())
