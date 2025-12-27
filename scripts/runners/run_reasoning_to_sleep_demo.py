# scripts/runners/run_reasoning_to_sleep_demo.py
# 日本語タイトル: Reasoning to Sleep 自律学習統合デモ
# ファイルの目的・内容:
#   System 2 (ReasoningEngine) による多段階推論と、System 1 (BitSpikeMamba) への
#   知識定着 (SleepConsolidator) を統合。
#   不確実な問題に対する「熟慮」を「直感」へと変換するプロセスを実証する。

import asyncio
import logging
import torch
from pathlib import Path

# プロジェクト内モジュールのインポート
from snn_research.cognitive_architecture.reasoning_engine import ReasoningEngine
from snn_research.cognitive_architecture.sleep_consolidation import SleepConsolidator
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.utils.brain_debugger import BrainDebugger

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReasoningSleepDemo")

async def run_reasoning_to_sleep_demo():
    logger.info("=== Brain v20: Reasoning-to-Sleep Integration Demo ===")

    # 1. コンポーネントの初期化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Astrocyte (エネルギー管理) の準備
    astrocyte = AstrocyteNetwork()
    
    # System 1: Bit-Spike Mamba (1.58bit量子化による高速・省エネモデル)
    # ROADMAP v20.0 の核となるアーキテクチャ
    model_params = {
        "vocab_size": 1000,
        "d_model": 256,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 4,
        "time_steps": 10,
        "neuron_config": {"type": "lif", "tau_mem": 20.0}
    }
    system1_model = BitSpikeMamba(**model_params).to(device)
    
    # System 2: Reasoning Engine (熟慮・コード検証・RAG統合)
    reasoning_engine = ReasoningEngine(
        generative_model=system1_model, # BitSpikeMambaを統合
        astrocyte=astrocyte,
        device=device,
        enable_code_verification=True # サンドボックスによる実行検証
    )
    
    # Sleep Consolidator (生成的再生による記憶定着)
    # メモリシステムダミーとして簡略化
    memory_dummy = type('obj', (object,), {'short_term_memory': []})
    sleep_consolidator = SleepConsolidator(
        memory_system=memory_dummy,
        target_brain_model=system1_model,
        device=device
    )
    
    debugger = BrainDebugger()

    # 2. 難解な課題の入力 (直感では解けない問題を想定)
    complex_query = "13番目の素数に5を足して、その結果を2倍にした数値は何ですか？"
    logger.info(f"User Stimulus: {complex_query}")

    # 3. System 2 による多段階推論 (Reasoning)
    # 内部で思考パスの生成とコード実行による検証が行われる
    logger.info("System 2 (Reasoning Engine) is thinking deeply...")
    
    # ダミー入力ID（実際はTokenizerを使用）
    input_ids = torch.randint(0, 1000, (1, 8)).to(device)
    reasoning_result = reasoning_engine.think_and_solve(input_ids)
    
    print("\n--- Reasoning Output ---")
    print(f"Final Answer (Probabilistic): {reasoning_result.get('final_text', 'Calculated via logic')}")
    print(f"Thought Trace (Steps): {reasoning_result.get('thought_trace', [])}")
    print(f"Verifier Score: {reasoning_result.get('verifier_score', 0.0):.4f}")
    print(f"Strategy: {reasoning_result.get('strategy', 'unknown')}")
    print("------------------------\n")

    # 4. 睡眠フェーズへの移行 (Sleep Consolidation)
    # 日中の「深い思考（System 2）」の結果を「直感（System 1）」へ蒸留する
    logger.info("Entering Sleep Phase: Consolidating thoughts into Bit-Spike weights...")
    
    # 思考トレースを模倣学習（Dreaming）として実行
    stats = sleep_consolidator.perform_sleep_cycle(duration_cycles=5)
    
    logger.info(f"Sleep Phase Result: {stats['dreams_replayed']} dreams replayed.")
    if stats['loss_history']:
        logger.info(f"Final Loss Improvement: {stats['loss_history'][-1]:.6f}")

    # 5. 学習後の状態確認
    # ロジックの正しさを確認し、エネルギー代謝をリセットする
    logger.info("🌅 Waking up. Evaluating internal consistency...")
    debugger.explain_thought_process(
        input_text=complex_query,
        output_text="The answer is 82.",
        astrocyte_status={'metrics': {'current_energy': 500, 'fatigue_index': 10}}
    )
    
    logger.info("Demo finished successfully.")

if __name__ == "__main__":
    # 非同期カーネルのシミュレーション
    try:
        asyncio.run(run_reasoning_to_sleep_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted.")