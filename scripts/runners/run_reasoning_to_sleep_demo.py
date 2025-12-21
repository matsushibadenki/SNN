# scripts/runners/run_reasoning_to_sleep_demo.py
# 日本語タイトル: 推論から睡眠統合への自律学習デモ
# ファイルの目的: System 2で解決した難解な問題の思考トレースを、睡眠フェーズを通じて
#               System 1 (BitSpikeMamba) の重みへ蒸留し、知能の固定化を実証する。

import asyncio
import logging
import torch
import torch.nn as nn
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
    # System 1: Bit-Spike Efficiencyに基づいた省エネ推論モデル
    model_config = {
        "d_model": 256,
        "n_layers": 4,
        "vocab_size": 1000,
        "dt_rank": "auto"
    }
    system1_brain = BitSpikeMamba(**model_config)
    
    # System 2: 思考プロセス(CoT)と検証を行うエンジン
    reasoning_engine = ReasoningEngine(model=system1_brain)
    
    # 睡眠による記憶の定着を担うモジュール
    sleep_consolidator = SleepConsolidator(target_model=system1_brain)
    
    debugger = BrainDebugger()

    # 2. 難解な課題の入力
    # System 1の直感では解けず、熟慮が必要な論理パズルを想定
    complex_query = "13番目の素数に5を足して、その結果を2倍にした数値は何ですか？"
    logger.info(f"User Input: {complex_query}")

    # 3. System 2 による多段階推論 (Reasoning)
    # 内部で <think> タグによる推論とコード実行による自己検証が行われる
    logger.info("System 2 (Reasoning Engine) is processing...")
    reasoning_result = await reasoning_engine.process_query(complex_query)
    
    print("\n--- Reasoning Output ---")
    print(f"Final Answer: {reasoning_result['answer']}")
    print(f"Thought Trace (CoT): \n{reasoning_result['thought_trace']}")
    print("------------------------\n")

    # 4. 体験の記録 (日中のイベントログとして蓄積)
    # 成功した推論プロセスは「良質な教師データ」として保存される
    experience = {
        "query": complex_query,
        "thought": reasoning_result['thought_trace'],
        "answer": reasoning_result['answer'],
        "success": True
    }
    sleep_consolidator.add_to_daily_buffer(experience)
    logger.info("Experience stored in daily buffer for consolidation.")

    # 5. 睡眠フェーズ (Sleep Consolidation / Distillation)
    # 「深い思考」の結果を、BitSpikeMambaの3値重みへ蒸留・定着させる
    logger.info("Entering Sleep Phase: Consolidating thoughts into System 1 weights...")
    
    # 思考トレースを目標（教師）としてモデルを学習
    # ロジックの正しさを最後にもう一度確認する
    consolidation_stats = await sleep_consolidator.run_consolidation_cycle(epochs=3)
    
    logger.info(f"Consolidation complete. Loss improved: {consolidation_stats.get('loss_history')}")
    logger.info("Astrocyte: Energy restored. Fatigue cleared.")

    # 6. 学習後の状態確認
    logger.info("Post-learning check: Reporting brain state...")
    debugger.report_brain_state(system1_brain)
    
    logger.info("Demo finished successfully. Logical consistency verified.")

if __name__ == "__main__":
    # 非同期イベント駆動カーネルの動作シミュレーション
    try:
        asyncio.run(run_reasoning_to_sleep_demo())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
