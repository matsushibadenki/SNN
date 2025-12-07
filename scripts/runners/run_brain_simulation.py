# ファイルパス: scripts/runners/run_brain_simulation.py
# Title: SNN Artificial Brain v14.0 Runner
# Description:
#   v14.0の全機能を統合した人工脳の実行スクリプト。
#   覚醒と睡眠のサイクルを回しながら、ユーザーとの対話を通じて進化する様子をシミュレートする。
#   CLI引数でモデル設定やモードを切り替え可能。
#   修正: ヘルスチェックのバリデーションキーワード "認知サイクル完了" を出力するように修正。
#   修正(v2): プロジェクトルートのパス解決を "../" から "../../" に修正し、appモジュールのインポートエラーを解消。

import sys
import os
import argparse
import logging
import asyncio
from pathlib import Path
from omegaconf import OmegaConf

# プロジェクトルートの設定
# --- 修正: 2階層上 (../../) を指定してプロジェクトルートを正しく取得 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Run SNN Artificial Brain Simulation")
    parser.add_argument("--prompt", type=str, help="人工脳への単一の入力テキスト。指定しない場合はデモを実行します。")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイルのパス。")
    parser.add_argument("--config", type=str, default="configs/experiments/brain_v14_config.yaml", help="Path to experiment config")
    parser.add_argument("--base_config", type=str, default="configs/templates/base_config.yaml", help="Base config")
    parser.add_argument("--mode", type=str, choices=["interactive", "demo"], default="interactive", help="Run mode (only used if prompt is not provided)")
    args = parser.parse_args()
    
    # 1. コンテナと設定のロード
    container = BrainContainer()
    
    # 設定ファイルのロード (OmegaConf経由で安全にマージ)
    try:
        base_cfg = OmegaConf.load(args.base_config) if os.path.exists(args.base_config) else OmegaConf.create()
        # model_configのロード
        if os.path.exists(args.model_config):
            model_cfg = OmegaConf.load(args.model_config)
            base_cfg = OmegaConf.merge(base_cfg, model_cfg)
        
        # 実験設定のロード
        if os.path.exists(args.config):
            exp_cfg = OmegaConf.load(args.config)
            base_cfg = OmegaConf.merge(base_cfg, exp_cfg)

        # デフォルト設定のフォールバック
        if not base_cfg.get("model"):
             base_cfg.model = {"architecture_type": "predictive_coding", "d_model": 64, "time_steps": 16}
        if not base_cfg.get("training"):
             base_cfg.training = {"biologically_plausible": {"neuron": {"type": "lif"}}}

        # DIコンテナに適用
        container.config.from_dict(OmegaConf.to_container(base_cfg, resolve=True))

    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        sys.exit(1)

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
        sys.exit(2)

    # 4. 実行
    if args.prompt:
        # 単一の入力で実行 (ヘルスチェック用)
        logger.info(f"--- Running single cognitive cycle for input: '{args.prompt}' ---")
        brain.run_cognitive_cycle(args.prompt)
        logger.info("認知サイクル完了") # ヘルスチェック通過用キーワード
    else:
        # モード実行
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
            logger.info("認知サイクル完了")

if __name__ == "__main__":
    main()
