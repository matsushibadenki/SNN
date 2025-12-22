# ファイルパス: scripts/runners/test_distillation_cycle.py
# 日本語タイトル: 統合蒸留サイクル検証デモ
# 目的: 不確実性検知から蒸留学習までの一連のループを検証する。

import torch
import logging
import asyncio
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. コンポーネントの準備
    # System 1 モデル (学生)
    config = {"d_model": 128, "n_layers": 2, "vocab_size": 1000}
    system1 = BitSpikeMamba(config).to(device)
    
    # メモリと睡眠管理
    memory = Memory(rag_system=None)
    sleep_consolidator = SleepConsolidator(
        memory_system=memory,
        target_brain_model=system1,
        device=device
    )

    # 人工脳の構築
    brain = ArtificialBrain(
        thinking_engine=system1,
        sleep_consolidator=sleep_consolidator,
        device=device
    )

    logger.info("🧠 Brain System initialized for Distillation Test.")

    # 2. 日中の経験シミュレーション (不確実性の高い事象が発生)
    logger.info("☀️ Daytime: Processing high-uncertainty input...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 脳が「わからない」と判断し、System 2 の思考トレースを作成したと仮定
    thought_trace = {
        "thought_trace": "The object is a high-efficiency spiking neuron with 1.58-bit weights.",
        "final_answer": torch.tensor([42]).to(device) # カテゴリIDなど
    }
    
    # 経験をバッファに追加
    brain.sleep_manager.add_experience(thought_trace)
    logger.info("📝 Thought trace stored in experience buffer.")

    # 3. 睡眠サイクルの実行 (蒸留の開始)
    logger.info("🛌 Nighttime: Starting Sleep Cycle (Distillation)...")
    stats = brain.sleep_cycle()
    
    # 4. 結果の確認
    progress = sleep_consolidator.get_learning_progress()
    logger.info(f"📊 Distillation Stats: {progress}")
    
    if progress["samples_processed"] > 0:
        logger.info("✅ SUCCESS: System 2 knowledge has been distilled into System 1.")
    else:
        logger.error("❌ FAILURE: Distillation did not process any samples.")

if __name__ == "__main__":
    asyncio.run(main())