# ファイルパス: scripts/runners/run_sleep_learning_demo.py
# 修正: インポートエラー修正 (AsyncBrainKernel -> AsyncArtificialBrain)

import sys
import os
import logging
import asyncio

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# 修正: 正しいクラス名をインポート
from snn_research.cognitive_architecture.async_brain_kernel import AsyncArtificialBrain
from snn_research.models.adapters.async_mamba_adapter import AsyncBitSpikeMambaAdapter

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("SleepLearningDemo")

async def run_sleep_cycle(brain: AsyncArtificialBrain, mamba_adapter: AsyncBitSpikeMambaAdapter):
    """
    睡眠学習サイクルのデモを実行する
    1. 覚醒状態での活動（エネルギー消費）
    2. 疲労蓄積によるパフォーマンス低下
    3. 睡眠状態への移行
    4. 記憶の整理（Distillation）
    5. 覚醒とエネルギー回復確認
    """
    logger.info(">>> Starting Sleep Learning Cycle Demo...")
    
    # Brain起動
    await brain.start()
    
    # 1. 覚醒状態での活動シミュレーション
    logger.info("\n--- Phase 1: Awake & Active ---")
    inputs = ["Input_A", "Input_B", "Input_C"]
    
    for i, inp in enumerate(inputs):
        logger.info(f"Processing input {i+1}: {inp}")
        # Mambaアダプター経由で入力 (awaitを追加して警告を解消)
        await mamba_adapter.process(inp)
        
        # 脳の活動を少し進める
        await asyncio.sleep(0.5)
        
        # エネルギー消費をシミュレート
        if hasattr(brain, "astrocyte_network"):
             brain.astrocyte_network.consume_energy(50.0) # type: ignore

    # ステータス確認
    status = brain.get_status()
    # KeyError修正: get()を使って安全に取得
    current_energy = status['metrics'].get('current_energy', 1000.0)
    fatigue_index = status['metrics'].get('fatigue_index', 0.0)
    
    logger.info(f"Status before sleep: Energy={current_energy:.1f}, Fatigue={fatigue_index:.1f}")

    # 2. 睡眠モードへ移行
    logger.info("\n--- Phase 2: Entering Sleep Mode ---")
    # 強制的に睡眠モードへ
    await brain.set_mode("sleep")
    
    # 睡眠中の処理を待機（本来は自律的だが、デモ用に時間をとる）
    logger.info("Sleeping... (Consolidating Memories)")
    await asyncio.sleep(2.0)
    
    # 3. 睡眠中のステータス確認
    sleep_status = brain.get_status()
    logger.info(f"Sleep Status: Mode={sleep_status['mode']}")
    
    # 4. 覚醒
    logger.info("\n--- Phase 3: Waking Up ---")
    await brain.set_mode("active")
    
    # 回復確認
    final_status = brain.get_status()
    recovered_energy = final_status['metrics'].get('current_energy', 1000.0)
    final_fatigue = final_status['metrics'].get('fatigue_index', 0.0)
    
    logger.info(f"Status after sleep: Energy={recovered_energy:.1f}, Fatigue={final_fatigue:.1f}")
    
    if recovered_energy > current_energy:
        logger.info("SUCCESS: Energy recovered during sleep.")
    else:
        logger.warning("WARNING: Energy did not recover significantly.")

    await brain.stop()
    logger.info(">>> Sleep Learning Demo Finished.")

async def main():
    # コンポーネントの初期化
    brain = AsyncArtificialBrain()
    mamba_adapter = AsyncBitSpikeMambaAdapter(model_path="dummy_path", vocab_size=100)
    
    # アダプターを脳に接続
    brain.connect_adapter(mamba_adapter)
    
    try:
        await run_sleep_cycle(brain, mamba_adapter)
    except Exception as e:
        logger.error(f"An error occurred during the demo: {e}", exc_info=True)
    finally:
        # 終了処理
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")