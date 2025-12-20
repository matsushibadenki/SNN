# ファイルパス: scripts/runners/run_sleep_learning_demo.py
# 日本語タイトル: Brain v2.0 Sleep & Learn Demo
# 目的: 統合された「活動→睡眠→成長」サイクルを実証する。

import sys
import os
import asyncio
import logging
import torch

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.agent.memory import Memory # 仮定: 既存モジュール
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter
from snn_research.models.experimental.world_model_snn import SpikingWorldModel

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-15s | %(message)s')
logger = logging.getLogger("SleepDemo")

# ダミーメモリクラス (依存関係解決用)
class DummyMemory:
    def __init__(self):
        self.short_term_memory = ["Experience A", "Experience B", "Important Event C"]
        self.long_term_memory = []
    def _consolidate(self, item):
        self.long_term_memory.append(item)

async def main():
    logger.info("=== Brain v2.0 Life Cycle Demo: Awake & Sleep ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. コンポーネント構築
    astrocyte = AstrocyteNetwork()
    memory = DummyMemory()
    
    # モデル (System 1)
    mamba_config = {"d_model": 128, "d_state": 32, "num_layers": 4, "tokenizer": "gpt2"}
    thinking_adapter = AsyncBitSpikeMambaAdapter(mamba_config, device=device)
    
    # 世界モデル (Dreaming用)
    world_model = SpikingWorldModel(vocab_size=50257, d_model=128).to(device)
    
    # 睡眠管理者 (ここでSystem 1とWorld Modelを接続)
    sleep_manager = SleepConsolidator(
        memory_system=memory,
        target_brain_model=thinking_adapter.model, # アダプターの中身(nn.Module)を渡す
        world_model=world_model,
        device=device
    )

    # Brain Kernel
    # 注意: ここでは簡易的にSleepManagerを外部から制御する形にします
    # (Brain Kernelに直接組み込むにはkernelの修正が必要ですが、今回はRunnerで制御)
    
    # 2. 日中フェーズ (Daytime)
    logger.info("\n☀️ DAYTIME: Brain is active.")
    astrocyte.replenish_energy(100.0)
    
    # 思考活動
    response = thinking_adapter.process("Hello, who are you?")
    logger.info(f"Brain: {response}")
    
    # 活動によるエネルギー消費シミュレーション
    logger.info("...Working hard (Thinking)...")
    astrocyte.request_resource("prefrontal", 30.0)
    astrocyte.request_resource("motor", 40.0)
    
    status = astrocyte.get_diagnosis_report()
    logger.info(f"Status before sleep: Energy={status['metrics']['current_energy']:.1f}, Fatigue={status['metrics']['fatigue_index']:.1f}")

    # 3. 夜間フェーズ (Nighttime / Sleep)
    logger.info("\n🌙 NIGHTTIME: Initiating Sleep Cycle...")
    
    # 睡眠実行 (夢を見る)
    sleep_stats = sleep_manager.perform_sleep_cycle(duration_cycles=5)
    
    # アストロサイト回復
    astrocyte.replenish_energy(500.0)
    astrocyte.clear_fatigue(100.0)
    
    logger.info(f"💤 Sleep Report: {sleep_stats}")
    
    # 4. 翌朝フェーズ (Morning)
    logger.info("\n🌅 MORNING: Brain woke up refreshed and smarter.")
    status = astrocyte.get_diagnosis_report()
    logger.info(f"Status after sleep: Energy={status['metrics']['current_energy']:.1f}")
    
    # 学習効果の確認 (同じ質問を投げてみる - 重みが更新されているため、結果が変わる可能性がある)
    response_new = thinking_adapter.process("Hello, who are you?")
    logger.info(f"Brain (Day 2): {response_new}")

if __name__ == "__main__":
    asyncio.run(main())