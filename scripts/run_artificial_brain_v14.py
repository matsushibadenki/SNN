# ファイルパス: scripts/run_artificial_brain_v14.py
# Title: SNN Artificial Brain v14.0 Runner
# Description:
#   v14.0の全機能を統合した人工脳の実行スクリプト。
#   覚醒と睡眠のサイクルを回しながら、ユーザーとの対話を通じて進化する様子をシミュレートする。
#   CLI引数でモデル設定やモードを切り替え可能。

import sys
import os
import argparse
import logging
import asyncio
from pathlib import Path
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.containers import BrainContainer

# ロギング設定 (フォーマットをシンプルに)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BrainRunner")
# 外部ライブラリのログを抑制
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def interactive_session(brain):
    """対話モードのメインループ"""
    print("\n" + "="*60)
    print("🧠 Artificial Brain v14.0 (Interactive Mode)")
    print("   - Type your message to talk to the brain.")
    print("   - Type 'sleep' to force a sleep cycle.")
    print("   - Type 'status' to see internal state.")
    print("   - Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Shutting down...")
                break
                
            if user_input.lower() == "sleep":
                brain.sleep_and_dream()
                continue
                
            if user_input.lower() == "status":
                print(f"  [Status] Energy: {brain.energy_level:.1f}%, Fatigue: {brain.fatigue_level:.1f}")
                print(f"  [Memory] WM Items: {len(brain.hippocampus.working_memory)}")
                continue

            # 認知サイクルの実行
            brain.run_cognitive_cycle(user_input)
            
            # 応答の表示 (Actuatorのログを見るのが正だが、簡易的に)
            # 実際にはActuatorが発話するが、ここでは直近の行動を表示
            action = brain.basal_ganglia.selected_action
            if action:
                print(f"Brain: (Action) {action.get('action')}")
            else:
                print("Brain: ... (Listening / Thinking)")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            logger.error(f"Error in loop: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Run SNN Artificial Brain v14")
    parser.add_argument("--config", type=str, default="configs/experiments/brain_v14_config.yaml", help="Path to experiment config")
    parser.add_argument("--base_config", type=str, default="configs/templates/base_config.yaml", help="Base config")
    parser.add_argument("--mode", type=str, choices=["interactive", "demo"], default="interactive", help="Run mode")
    args = parser.parse_args()
    
    # 1. コンテナと設定のロード
    container = BrainContainer()
    
    # 設定ファイルのロード
    if os.path.exists(args.base_config):
        container.config.from_yaml(args.base_config)
    
    if os.path.exists(args.config):
        container.config.from_yaml(args.config)
    else:
        # コンフィグがない場合はデフォルト値を生成して使用（フォールバック）
        container.config.from_dict({
            "model": {"architecture_type": "predictive_coding", "d_model": 64, "time_steps": 16},
            "training": {"biologically_plausible": {"neuron": {"type": "lif"}}}
        })

    # 2. RAGシステムのセットアップ
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        rag_system.setup_vector_store()

    # 3. 人工脳の構築
    try:
        brain = container.artificial_brain()
    except Exception as e:
        logger.error(f"Failed to build Artificial Brain: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. モード実行
    if args.mode == "interactive":
        interactive_session(brain)
    elif args.mode == "demo":
        logger.info("Running Demo Sequence...")
        inputs = [
            "Hello, who are you?",
            "What is a Spiking Neural Network?",
            "I feel happy today.",
            "sleep", # 強制睡眠
            "Tell me about SNNs again."
        ]
        for inp in inputs:
            if inp == "sleep":
                brain.sleep_and_dream()
            else:
                logger.info(f"Input: {inp}")
                brain.run_cognitive_cycle(inp)
            import time
            time.sleep(1)

if __name__ == "__main__":
    main()
