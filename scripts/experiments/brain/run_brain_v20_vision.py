# ファイルパス: scripts/runners/run_brain_v20_vision.py
# Title: Brain v20.2 Integration - Real Vision
# Description:
#   SimulatedPerceptionModule を AsyncVisionAdapter に置き換え、
#   本物のSpikingCNNを用いた「視覚認識」を含む統合デモを実行する。

import sys
import os
import asyncio
import logging
import torch

# プロジェクトルートの設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.models.adapters.async_vision_adapter import AsyncVisionAdapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
    force=True
)
logger = logging.getLogger("BrainV20.2-Vision")

# --- Mocks (視覚以外) ---
class SimulatedMotorCortex:
    async def process(self, command):
        logger.info(f"🤖 ACTUATOR: Received motor command: {command}")
        return True

class SimulatedReasoningEngine:
    async def process(self, data):
        logger.info(f"🤔 System 2: Analyzing complex visual data... {data}")
        return "Analysis Complete"

# --- Main Routine ---

async def main():
    logger.info("==================================================")
    logger.info("   Brain v20.2: Real Visual Cortex Integration    ")
    logger.info("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")

    # 1. コンポーネント構築
    astrocyte = AstrocyteNetwork()
    
    # 視覚野 (Real SpikingCNN)
    vision_config = {
        "architecture_type": "spiking_cnn",
        "input_channels": 3,
        "features": 128,
        "time_steps": 4,
        "layers": [32, 64, 128]
    }
    visual_cortex = AsyncVisionAdapter(config=vision_config, device=device)
    
    # 言語野 (System 1)
    mamba_config = {"d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"}
    language_area = AsyncBitSpikeMambaAdapter(mamba_config, device=device)

    # 2. Brain Kernel構築
    brain = AsyncArtificialBrain(
        modules={
            "visual_cortex": visual_cortex,   # ★ 本物を接続
            "language_area": language_area,
            "system1": language_area,         # 言語野と共有
            "reasoning_engine": SimulatedReasoningEngine(),
            "actuator": SimulatedMotorCortex()
        },
        astrocyte=astrocyte,
        web_crawler=None,
        distillation_manager=None,
        max_workers=4
    )

    # 3. 起動
    await brain.start()
    
    # --- Scenario: Visual Recognition ---
    logger.info("\n--- Scenario: Visual Input Processing ---")
    
    # ケース1: ランダムノイズ（未知の物体）を入力
    # 実際にはカメラ画像などを渡すが、ここではTensorを生成
    dummy_image = torch.randn(1, 3, 32, 32).to(device)
    
    logger.info("📸 Sending visual signal to Brain...")
    # Brainに画像データを直接注入 (通常はSensory Receptor経由だが短絡)
    await brain.receive_input(dummy_image)
    
    # 処理待ち
    await asyncio.sleep(2.0)
    
    logger.info("\n--- Scenario: Low Confidence Trigger ---")
    # 自信度が低くなるような入力をシミュレートし、System 2が呼ばれるか確認したいが、
    # ランダムノイズなら確率はバラつくため、運が良ければTrigger System 2が発動する
    
    await brain.stop()
    logger.info(">>> Vision Integration Test Finished.")

if __name__ == "__main__":
    asyncio.run(main())