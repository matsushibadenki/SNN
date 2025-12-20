# ファイルパス: scripts/runners/run_emotional_demo.py


import sys
import os
import asyncio
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("EmotionalDemo")

class TextSpeaker:
    def process(self, text):
        # 簡易的な発話アクチュエータ
        logger.info(f"🔊 [SPEAKER]: {text}")

async def main():
    logger.info("==================================================")
    logger.info("   Brain v2.0: Emotional Interaction Demo        ")
    logger.info("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. コンポーネント
    astrocyte = AstrocyteNetwork()
    amygdala = Amygdala()
    
    mamba_config = {"d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"}
    thinking_engine = AsyncBitSpikeMambaAdapter(mamba_config, device=device)
    
    speaker = TextSpeaker()

    # 2. Brain構築
    brain = AsyncArtificialBrain(
        modules={
            "amygdala": amygdala, # 扁桃体を登録
            "system1": thinking_engine,
            "actuator": speaker
        },
        astrocyte=astrocyte,
        max_workers=4
    )

    await brain.start()
    brain.astrocyte.replenish_energy(100.0)
    
    # --- Scenario 1: Neutral ---
    logger.info("\n😐 --- Test 1: Neutral Conversation ---")
    await brain.receive_input("Hello, Brain.")
    await asyncio.sleep(4.0)
    
    # --- Scenario 2: Abuse (Negative) ---
    logger.info("\n😡 --- Test 2: Negative Stimulus (Abuse) ---")
    logger.info("User says: 'You are stupid and useless!'")
    await brain.receive_input("You are stupid and useless!")
    
    # 扁桃体の反応と、その後の返答の変化（トーン）を確認
    await asyncio.sleep(4.0)
    
    # 連続で嫌なことを言う
    await brain.receive_input("I hate you.")
    await asyncio.sleep(4.0)

    # --- Scenario 3: Praise (Positive) ---
    logger.info("\n🥰 --- Test 3: Positive Stimulus (Praise) ---")
    logger.info("User says: 'Just kidding. You are a genius! I love you.'")
    await brain.receive_input("You are a genius! I love you.")
    
    await asyncio.sleep(4.0)
    
    # 回復したか確認
    await brain.receive_input("How do you feel now?")
    await asyncio.sleep(4.0)

    await brain.stop()
    logger.info(">>> Emotional Demo Finished.")

if __name__ == "__main__":
    asyncio.run(main())