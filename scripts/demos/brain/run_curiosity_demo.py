# ファイルパス: scripts/runners/run_curiosity_demo.py

import sys
import os
import asyncio
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
# 修正: 正しいクラス名 IntrinsicMotivationSystem をインポート
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("CuriosityDemo")

class ExpressiveActuator:
    def process(self, command):
        logger.info(f"🤖 Action/Speech: {command}")

async def main():
    logger.info("==================================================")
    logger.info("   Brain v2.0: Active Inference (Curiosity) Demo ")
    logger.info("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    astrocyte = AstrocyteNetwork()
    # 修正: IntrinsicMotivationSystem を使用
    motivation_engine = IntrinsicMotivationSystem()
    
    mamba_config = {"d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"}
    thinking_engine = AsyncBitSpikeMambaAdapter(mamba_config, device=device)
    
    actuator = ExpressiveActuator()

    # Brain構築
    brain = AsyncArtificialBrain(
        modules={
            "curiosity": motivation_engine, 
            "system1": thinking_engine,
            "actuator": actuator
        },
        astrocyte=astrocyte,
        max_workers=4
    )

    # Brain Kernelの配線を動的に追加
    brain.bus.subscribe("PERCEPTION_DONE", 
        lambda event: asyncio.create_task(brain._run_module("curiosity", event.payload, "MOTIVATION_UPDATE"))
    )
    
    # MOTIVATION_UPDATE ハンドラ
    async def on_motivation_update(event):
        data = event.payload
        # 退屈レベルが高い場合
        if isinstance(data, dict) and data.get('boredom', 0.0) > 0.8:
            logger.info("⚠️ Brain Signal: I AM BORED! CHANGE THE TOPIC!")
            # ここでSystem 1に介入することも可能
    
    brain.bus.subscribe("MOTIVATION_UPDATE", on_motivation_update)

    await brain.start()
    brain.astrocyte.replenish_energy(100.0)
    
    # --- Scenario 1: Repetition (Boredom) ---
    logger.info("\n🔁 --- Test 1: Repetitive Input (Inducing Boredom) ---")
    logger.info("User keeps saying 'Hello'...")
    
    for i in range(5):
        await brain.receive_input("Hello")
        await asyncio.sleep(1.5) # 反応待ち
        
    # --- Scenario 2: Novelty (Curiosity) ---
    logger.info("\n✨ --- Test 2: Novel Input (Triggering Curiosity) ---")
    logger.info("User suddenly shows something new...")
    
    await brain.receive_input("Look at this shining star!")
    await asyncio.sleep(3.0)

    await brain.stop()
    logger.info(">>> Curiosity Demo Finished.")

if __name__ == "__main__":
    asyncio.run(main())