# ファイルパス: scripts/runners/run_multimodal_brain.py
# Title: run_multimodal_brain
# Description:
#   「目（Industrial Eye）」、「脳（Mamba）」、「脊髄（Reflex）」をすべて繋げた、Brain v2.0の完成形に近いデモです。

from snn_research.modules.reflex_module import ReflexModule
from snn_research.models.adapters.async_vision_adapter import AsyncVisionAdapter
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
import sys
import os
import asyncio
import logging
import torch

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../..")))

# [Fix] Import correct class name

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("MultimodalDemo")


class RobotArm:
    def process(self, command):
        if "DEFECT" in str(command):
            logger.warning(
                f"🤖 REJECT ACTION: Removing defective product based on {command}")
        else:
            logger.info(f"🤖 Action: {command}")


async def main():
    logger.info("==================================================")
    logger.info("   Brain v2.0: Multimodal Integration Demo       ")
    logger.info("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. コンポーネント構築
    astrocyte = AstrocyteNetwork()

    # 視覚野 (Visual Cortex)
    # [Fix] Use correct class
    vision_adapter = AsyncVisionAdapter(
        config={'architecture_type': 'spiking_cnn', 'features': 128}, device=device)

    # 思考エンジン (System 2)
    mamba_config = {"d_model": 128, "d_state": 32,
                    "num_layers": 4, "tokenizer": "gpt2"}
    thinking_engine = AsyncBitSpikeMambaAdapter(mamba_config, device=device)

    # 反射モジュール (System 1 Fast)
    reflex = ReflexModule(input_dim=128, action_dim=10).to(device)  # ダミー入力用

    actuator = RobotArm()

    # 2. Brain Kernel構築
    brain = AsyncArtificialBrain(
        modules={
            "visual_cortex": vision_adapter,  # ここに視覚野を接続
            "system1": thinking_engine,
            "actuator": actuator
        },
        astrocyte=astrocyte,
        max_workers=4
    )
    brain.reflex_module = reflex  # 反射も接続

    await brain.start()
    brain.astrocyte.replenish_energy(100.0)

    # --- Scenario 1: 正常な製品を見る ---
    logger.info("\n📦 --- Test 1: Inspecting Normal Product ---")
    # ダミーのDVS入力 (Batch, Time, Channels, Height, Width)
    # ランダムノイズ（正常パターンと仮定）
    normal_input = torch.randn(1, 8, 2, 128, 128).to(device)

    # カーネルに送る (SENSORY_INPUT -> visual_cortex -> PERCEPTION_DONE)
    await brain.receive_input(normal_input)
    await asyncio.sleep(2.0)

    # --- Scenario 2: 欠陥品を見る (Defect) ---
    logger.info("\n⚠️ --- Test 2: Inspecting Defective Product ---")
    # 欠陥パターン（ここではモデルが未学習なので、クラス1が出るとは限りませんが、
    # Adapter側でロジックを確認済みと仮定、もしくは強制的に異常値を注入）

    defect_input = torch.randn(1, 8, 2, 128, 128).to(device) + 2.0  # 輝度が高い＝異常？

    await brain.receive_input(defect_input)

    # 脳が「欠陥」を認識し、除去コマンドを出すのを待つ
    await asyncio.sleep(4.0)

    # --- Scenario 3: 会話 ---
    logger.info("\n🗣️ --- Test 3: Verbal Report ---")
    await brain.receive_input("Report status.")
    await asyncio.sleep(3.0)

    await brain.stop()
    logger.info(">>> Multimodal Demo Finished.")

if __name__ == "__main__":
    asyncio.run(main())
