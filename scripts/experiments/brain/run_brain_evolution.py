# ファイルパス: scripts/runners/run_brain_evolution.py
# Title: Self-Evolving Brain Demo (Integration)
# Description:
#   Brain v20.3: 完全統合デモ。
#   1. Brainに未知の質問をする (System 1 fails)
#   2. System 2 (Reasoning) が起動し、答えを導き出す
#   3. AsyncDistillationManager がその思考過程を System 1 にバックグラウンドで教える
#   4. 同じ質問をすると、System 1 が即答する (Evolution)

from snn_research.distillation.async_manager import AsyncDistillationManager
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
import sys
import os
import asyncio
import logging

# パス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../..')))


# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
    force=True
)
logger = logging.getLogger("BrainEvolution")

# --- Mocks ---


class MockReasoningEngine:
    async def process(self, data):
        # 質問から答えを導き出すふりをする
        question = str(data)
        logger.info(f"🤔 System 2: Deep reasoning on '{question}'...")
        await asyncio.sleep(1.0)  # 思考時間

        thought = "Step 1: Parse numbers. Step 2: Calculate sum. Step 3: Verify."
        answer = "42" if "plus" in question else "Unknown"

        return {
            "input": question,
            "thought": thought,
            "result": answer,
            "metadata": {"source": "System2_Reasoning"}
        }


class MockActuator:
    async def process(self, command):
        # System 1と2で出力形式が違うので吸収
        msg = ""
        if isinstance(command, dict):
            if "thought" in command:  # System 1 output
                msg = f"Fast Response: {command['thought']} (Conf: {command.get('confidence', 0.0):.2f})"
            elif "result" in command:  # System 2 output
                msg = f"Slow Response: {command['result']} (via Logic)"
        else:
            msg = str(command)

        logger.info(f"🤖 ACTUATOR: {msg}")

# --- Main ---


async def main():
    logger.info("==================================================")
    logger.info("   Brain v20.3: Self-Evolving Architecture        ")
    logger.info("==================================================")

    device = "cpu"  # デモ用にはCPUで十分
    astrocyte = AstrocyteNetwork()

    # 1. System 1 (Student)
    mamba_config = {"d_model": 64, "d_state": 16, "vocab_size": 1000}
    system1 = AsyncBitSpikeMambaAdapter(mamba_config, device=device)

    # 2. System 2 (Teacher)
    system2 = MockReasoningEngine()

    # 3. Distillation Manager
    # System 1のモデル実体を渡して学習可能にする
    distiller = AsyncDistillationManager(system1.model)
    await distiller.start_worker()

    # 4. Brain Kernel
    brain = AsyncArtificialBrain(
        modules={
            "system1": system1,
            "reasoning_engine": system2,
            "actuator": MockActuator()
        },
        astrocyte=astrocyte,
        distillation_manager=distiller,  # カーネルに登録
        max_workers=2
    )

    # イベントバスの配線: THOUGHT_COMPLETE -> Distiller
    # カーネルの標準配線には含まれていないため、ここで手動購読
    async def on_thought_complete(event):
        # Reasoning Engineが答えを出したら、それを学習キューに送る
        await distiller.schedule_learning(event.payload)

    # ACTION_COMMANDとして出てくるイベントをフック
    brain.bus.subscribe("ACTION_COMMAND", on_thought_complete)

    await brain.start()

    # --- Scenario ---

    logger.info("\n--- Step 1: Encounter Unknown Problem ---")
    question = "What is 15 plus 27?"

    # 意図的にSystem 1が失敗（自信なし）するように、メタデータでSystem 2をトリガー
    # (本来はMetaCognitiveSNNが自動判定するが、ここでは明示的に指示)
    {
        "text": question,
        "trigger_system2": True  # これによりSystem 2へルーティングされる
    }

    # 入力をコンシャス・ブロードキャストとして直接発行（知覚プロセスをスキップ）
    from snn_research.cognitive_architecture.async_brain_kernel import BrainEvent
    await brain.bus.publish(BrainEvent(
        event_type="CONSCIOUS_BROADCAST",
        source="sensory",
        payload=question,
        metadata={"trigger_system2": True}  # 強制的にSystem 2へ
    ))

    # System 2が思考し、蒸留が走るのを待つ
    logger.info("⏳ Waiting for reasoning and consolidation...")
    await asyncio.sleep(5.0)

    logger.info("\n--- Step 2: Verify Evolution (Ask Same Question) ---")
    # 今度はSystem 2をトリガーせず、System 1に任せる
    await brain.bus.publish(BrainEvent(
        event_type="CONSCIOUS_BROADCAST",
        source="sensory",
        payload=question,
        metadata={"trigger_system2": False}  # System 1で処理
    ))

    await asyncio.sleep(2.0)

    await brain.stop()
    logger.info(">>> Evolution Demo Finished.")

if __name__ == "__main__":
    asyncio.run(main())
