# ファイルパス: scripts/runners/run_life_form.py

import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_life_form.py
# Title: Digital Life Form Launcher (Phase 5)
# Description:
#   DIコンテナを使用してデジタル生命体（DigitalLifeForm）を初期化し、
#   指定された時間だけ自律的に活動させる。

import time
import argparse
import sys
import logging
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import BrainContainer

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    デジタル生命体を起動し、指定時間（または無限に）活動させる。
    """
    parser = argparse.ArgumentParser(description="Digital Life Form Orchestrator (Phase 5)")
    parser.add_argument("--duration", type=int, default=60, help="実行時間（秒）。0を指定すると無限に実行します。")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル")
    args = parser.parse_args()
    
    logger.info("🏗️ Initializing Digital Life Form environment...")
    
    # 1. DIコンテナの初期化
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml(args.model_config)
    
    # 2. RAGシステム（知識ベース）の準備
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        logger.info("📚 Setting up RAG Vector Store for the first time...")
        rag_system.setup_vector_store()
    
    # 3. デジタル生命体のインスタンス化
    life_form = container.digital_life_form()
    
    logger.info(f"🧬 Digital Life Form initialized. Starting for {args.duration if args.duration > 0 else 'infinite'} seconds.")

    try:
        life_form.start()
        
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            logger.info("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received. Shutting down.")
    finally:
        life_form.stop()
        logger.info("✅ DigitalLifeForm has been deactivated safely.")

if __name__ == "__main__":
    main()