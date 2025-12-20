# ファイルパス: scripts/runners/run_life_form.py
# Title: Digital Life Form Launcher (Asyncio Fixed)
# Description:
#   DIコンテナを使用してデジタル生命体（DigitalLifeForm）を初期化し、
#   指定された時間だけ自律的に活動させる。
#   修正: 同期的な time.sleep ループを廃止し、asyncio イベントループを使用して
#   非同期エージェントが正しく動作するように修正。

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# ------------------------------------------------------------------------------
# プロジェクトルートディレクトリをsys.pathに追加
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

from app.containers import BrainContainer

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_life_form_async(duration: int, model_config: str):
    """
    デジタル生命体の非同期実行ループ。
    """
    logger.info("🏗️ Initializing Digital Life Form environment (Async Mode)...")
    
    # 1. DIコンテナの初期化
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml(model_config)
    
    # 2. RAGシステム（知識ベース）の準備
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        logger.info("📚 Setting up RAG Vector Store for the first time...")
        rag_system.setup_vector_store()
    
    # 3. デジタル生命体のインスタンス化
    life_form = container.digital_life_form()
    
    logger.info(f"🧬 Digital Life Form initialized. Starting for {duration if duration > 0 else 'infinite'} seconds.")

    # 4. エージェントの起動
    # DigitalLifeForm.start() は内部で asyncio.create_task を呼ぶことを想定
    life_form.start()
    
    try:
        if duration > 0:
            # 指定時間だけ待機（この間、バックグラウンドでエージェントが活動する）
            await asyncio.sleep(duration)
            logger.info(f"⏰ Duration ({duration}s) expired.")
        else:
            # 無限ループ待機
            logger.info("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(3600) # 1時間ごとに起床（実質寝ているだけ）
                
    except asyncio.CancelledError:
        logger.info("🛑 Task cancelled.")
    finally:
        life_form.stop()
        logger.info("✅ DigitalLifeForm has been deactivated safely.")

def main():
    parser = argparse.ArgumentParser(description="Digital Life Form Orchestrator (Phase 5)")
    parser.add_argument("--duration", type=int, default=60, help="実行時間（秒）。0を指定すると無限に実行します。")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル")
    args = parser.parse_args()

    try:
        # asyncioランタイムで実行
        asyncio.run(run_life_form_async(args.duration, args.model_config))
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()