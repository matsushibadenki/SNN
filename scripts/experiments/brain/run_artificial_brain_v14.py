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
import time
import traceback
from omegaconf import OmegaConf

# プロジェクトルートの設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True, stream=sys.stdout)
logger = logging.getLogger("BrainRunner")
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
            
            # 応答の表示
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
    try:
        print("[Debug] Script started.", flush=True)
        
        parser = argparse.ArgumentParser(description="Run SNN Artificial Brain Simulation")
        parser.add_argument("--prompt", type=str, help="人工脳への単一の入力テキスト。指定しない場合はデモを実行します。")
        parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイルのパス。")
        parser.add_argument("--config", type=str, default="configs/experiments/brain_v14_config.yaml", help="Path to experiment config")
        parser.add_argument("--base_config", type=str, default="configs/templates/base_config.yaml", help="Base config")
        parser.add_argument("--mode", type=str, choices=["interactive", "demo"], default="interactive", help="Run mode (only used if prompt is not provided)")
        args = parser.parse_args()
        
        print("[Debug] Importing BrainContainer...", flush=True)
        try:
            from app.containers import BrainContainer
        except ImportError as e:
            print(f"Error importing app.containers: {e}", flush=True)
            sys.exit(1)

        # 1. コンテナと設定のロード
        print("[Debug] Loading Config...", flush=True)
        container = BrainContainer()
        
        try:
            base_cfg = OmegaConf.load(args.base_config) if os.path.exists(args.base_config) else OmegaConf.create()
            if os.path.exists(args.model_config):
                model_cfg = OmegaConf.load(args.model_config)
                base_cfg = OmegaConf.merge(base_cfg, model_cfg)
            if os.path.exists(args.config):
                exp_cfg = OmegaConf.load(args.config)
                base_cfg = OmegaConf.merge(base_cfg, exp_cfg)

            if not base_cfg.get("model"):
                 base_cfg.model = {"architecture_type": "predictive_coding", "d_model": 64, "time_steps": 16}
            if not base_cfg.get("training"):
                 base_cfg.training = {"biologically_plausible": {"neuron": {"type": "lif"}}}

            container.config.from_dict(OmegaConf.to_container(base_cfg, resolve=True))

        except Exception as e:
            logger.error(f"Config loading failed: {e}")
            sys.exit(1)

        # 2. RAGシステムのセットアップ
        print("[Debug] Setting up RAG...", flush=True)
        rag_system = container.agent_container.rag_system()
        if not rag_system.vector_store:
            rag_system.setup_vector_store()

        # 3. 人工脳の構築
        print("[Debug] Building Artificial Brain...", flush=True)
        try:
            brain = container.artificial_brain()
        except Exception as e:
            logger.error(f"Failed to build Artificial Brain: {e}")
            traceback.print_exc()
            sys.exit(2)

        # 4. 実行
        if args.prompt:
            print(f"--- Running single cognitive cycle for input: '{args.prompt}' ---", flush=True)
            if not hasattr(brain, 'run_cognitive_cycle'):
                 print("[Error] Brain object does not have 'run_cognitive_cycle' method!", flush=True)
                 sys.exit(1)
            
            brain.run_cognitive_cycle(args.prompt)
            print("認知サイクル完了", flush=True) 
        else:
            if args.mode == "interactive":
                interactive_session(brain)
            elif args.mode == "demo":
                logger.info("Running Demo Sequence...")
                inputs = ["Hello.", "sleep"]
                for inp in inputs:
                    if inp == "sleep":
                        brain.sleep_and_dream()
                    else:
                        logger.info(f"Input: {inp}")
                        brain.run_cognitive_cycle(inp)
                    time.sleep(1)
                print("認知サイクル完了", flush=True)

    except Exception as e:
        print(f"\n[Fatal Error] An unhandled exception occurred: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()