# scripts/runners/run_reasoning_to_sleep_demo.py
# 多段階推論(System 2)から睡眠統合(Distillation)への自律学習デモ
# 目的: 難解な問題に対してReasoningEngineで論理推論を行い、その思考トレースを
#       SleepConsolidatorを通じてBitSpikeMamba(System 1)へ蒸留・学習させる。

import asyncio
import logging
import torch
from pathlib import Path

# プロジェクト内モジュールのインポート
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.core.snn_core import SNNCore
from snn_research.utils.brain_debugger import BrainDebugger

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReasoningSleepDemo")

async def run_reasoning_to_sleep_demo():
    logger.info("=== Brain v20: Reasoning to Sleep Integration Demo ===")

    # 1. コンポーネントの初期化
    # System 1: 高速・省エネな直感モデル (BitSpikeMamba)
    model_config = {
        "d_model": 256,
        "n_layers": 4,
        "vocab_size": 1000,
        "dt_rank": "auto"
    }
    system1_brain = BitSpikeMamba(**model_config)
    
    # System 2: 深い思考と検証を行うエンジン
    # RAGやコード実行ツールを内部で保持
    reasoning_engine = ReasoningEngine(model=system1_brain)
    
    # 記憶の定着を担う睡眠コンソリデータ
    sleep_consolidator = SleepConsolidator(target_model=system1_brain)
    
    debugger = BrainDebugger()

    # 2. 難解な課題の入力 (System 1では即答できない問題を想定)
    complex_query = "13番目の素数に5を足して、その結果を2倍にした数値は何ですか？"
    logger.info(f"User Input: {complex_query}")

    # 3. System 2 による多段階推論 (Chain-of-Thought + Verification)
    # 内部で <think> タグを用いた推論とコード実行による自己検証が行われる
    logger.info("System 2 (Reasoning Engine) is thinking...")
    reasoning_result = await reasoning_engine.process_query(complex_query)
    
    print("\n--- Reasoning Output ---")
    print(f"Final Answer: {reasoning_result['answer']}")
    print(f"Thought Trace (CoT): \n{reasoning_result['thought_trace']}")
    print("------------------------\n")

    # 4. 思考プロセスの記録 (日中の体験としてバッファに蓄積)
    # 成功した推論トレースは将来の「直感」の種になる
    experience = {
        "query": complex_query,
        "thought": reasoning_result['thought_trace'],
        "answer": reasoning_result['answer'],
        "success": True
    }
    sleep_consolidator.add_to_daily_buffer(experience)
    logger.info("Experience stored in daily buffer for consolidation.")

    # 5. 睡眠フェーズ (Sleep Consolidation)
    # 起きていた時の「深い思考」を「直感(SNNの重み)」へ蒸留する
    logger.info("Entering Sleep Phase: Consolidating thoughts into System 1 weights...")
    
    # 思考トレースを教師データとしてモデルを微調整
    consolidation_stats = await sleep_consolidator.run_consolidation_cycle(epochs=3)
    
    logger.info(f"Consolidation complete. Loss improved: {consolidation_stats['loss_history']}")
    logger.info("Astrocyte: Energy restored. Fatigue cleared.")

    # 6. 学習後の動作確認 (同じような問いに対して直感で答えられるか)
    logger.info("Post-learning check: Testing System 1 (Intuition) response...")
    # 実際には学習が収束するにはより多くのデータが必要だが、流れをデモ
    test_input = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        output = system1_brain(test_input)
    
    logger.info("Demo finished successfully.")
    debugger.report_brain_state(system1_brain)

if __name__ == "__main__":
    # 非同期ループでの実行
    try:
        asyncio.run(run_reasoning_to_sleep_demo())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")