#snn_research/cli/main.py

import click
import logging
import sys
from .core_commands import register_core_commands
from .model_commands import register_model_commands
from .app_commands import register_app_commands

# ロガーの取得
logger = logging.getLogger("snn_cli")

# --- メインCLIグループ ---
@click.group()
def cli():
    """SNNプロジェクト 統合CLIツール (Integrated v17.1)"""
    pass

def main():
    """エントリーポイント関数"""
    # 各コマンドグループを登録
    register_core_commands(cli)
    register_model_commands(cli)
    register_app_commands(cli)
    
    # CLI実行
    try:
        cli()
    except Exception as e:
        logger.error(f"❌ 致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()