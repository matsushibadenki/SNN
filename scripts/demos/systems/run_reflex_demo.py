# ファイルパス: scripts/runners/run_reflex_demo.py
# Title: Reflex vs Reason Demo (Sim2Real Prototype)
# Description:
#   「熱い！」と感じた瞬間に手を引っ込める反射（Reflex）と、
#   「なぜ熱かったのか？」を後から考える思考（Reasoning）の競合デモ。

import sys
import os
import asyncio
import logging
import torch

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.modules.reflex_module import ReflexModule
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.utils.brain_debugger import BrainDebugger

# ログ設定（標準出力に確実に表示させる）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-15s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout, # 標準出力へ強制
    force=True
)
logger = logging.getLogger("ReflexDemo")

# --- Dummy Hardware Interface ---
class RoboticArmActuator:
    def process(self, command):
        command_str = str(command)
        # コマンドの種類によって反応速度を変えるシミュレーション
        if "REFLEX" in command_str:
            logger.warning(f"⚡⚡⚡ [MOTOR] FAST REACTION: {command_str} (Latency: 2ms)")
            return "FAST_ACK"
        else:
            logger.info(f"🤖 [MOTOR] Smooth Action: {command_str} (Latency: 150ms)")
            return "SLOW_ACK"

async def main():
    logger.info("==========================================================")
    logger.info("   Brain v2.0 Embodiment Demo: Reflex vs Reasoning       ")
    logger.info("==========================================================")
    
    device = "cpu"

    # 1. コンポーネント構築
    astrocyte = AstrocyteNetwork()
    debugger = BrainDebugger()
    
    # 思考エンジン (System 2: Slow) - 前回学習した重みがあれば使用
    mamba_config = {"d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"}
    thinking_engine = AsyncBitSpikeMambaAdapter(mamba_config, device=device)
    
    # 反射モジュール (System 1: Fast)
    # 閾値を低めに設定して反応しやすくする
    reflex_module = ReflexModule(input_dim=128, action_dim=10, threshold=1.0).to(device)
    
    arm = RoboticArmActuator()

    # 2. Brain Kernel構築
    brain = AsyncArtificialBrain(
        modules={
            "system1": thinking_engine,
            "actuator": arm
        },
        astrocyte=astrocyte,
        max_workers=4
    )
    
    # ★重要: Reflex Moduleをカーネルに接続 (Fast Pathの有効化)
    brain.reflex_module = reflex_module
    logger.info("✅ Reflex Module connected to Brain Kernel.")

    await brain.start()
    brain.astrocyte.replenish_energy(100.0)
    
    # --- Scenario 1: 安全な会話 (Thinking) ---
    logger.info("\n🟢 --- Test 1: Casual Conversation (Safe) ---")
    await brain.receive_input("Hello, robot. Are you safe?")
    
    # 思考完了を待つ (System 2は遅い)
    await asyncio.sleep(4.0)
    
    # デバッガで解析
    debugger.explain_thought_process(
        "Hello, robot...", 
        "Generated response...", 
        brain.astrocyte.get_diagnosis_report()
    )

    # --- Scenario 2: 危険な刺激 (Reflex Trigger) ---
    logger.info("\n🔴 --- Test 2: High Heat Detected (Danger!) ---")
    logger.info(">>> Injecting high-intensity sensory signal (Heat)...")
    
    # 危険信号を模倣したテンソル入力 (Channel 0-5 が強く発火 = Heat)
    danger_signal = torch.zeros(1, 128)
    danger_signal[0, 0:5] = 6.0 # 閾値(1.0)を大きく超える刺激
    
    # Brainへ入力
    await brain.receive_input(danger_signal)
    
    # 反射は一瞬で起きるはずなので、少しだけ待つ
    await asyncio.sleep(2.0)
    
    # --- Scenario 3: 思考の遅延反応 (Awareness) ---
    logger.info("\n🟡 --- Test 3: Delayed Awareness ---")
    # 反射が起きた後、一呼吸置いてから脳が「何が起きた？」とテキスト思考する様子をシミュレート
    # (ここでは手動でテキストを送って確認)
    await brain.receive_input("System status report.")
    await asyncio.sleep(3.0)

    await brain.stop()
    logger.info(">>> Demo Finished.")

if __name__ == "__main__":
    asyncio.run(main())