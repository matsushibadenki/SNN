# ファイルパス: snn_research/cli/main.py
# 日本語タイトル: SNNプロジェクト統合CLI (Integrated v17.2)
# 目的: プロジェクトの全機能を統括するコマンドラインインターフェース。

import click
import logging
import sys
from .core_commands import register_core_commands
from .model_commands import register_model_commands
from .app_commands import register_app_commands

# 新規コマンドのインポート
from .recipe_commands import recipe_cli
from .demo_commands import demo_cli

# ロガーの初期設定
logger = logging.getLogger("snn_cli")


def setup_logging(debug: bool = False):
    """ロギングの詳細設定"""
    log_level = logging.DEBUG if debug else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # ルートロガーの設定
    logging.basicConfig(
        level=log_level,
        format=format_str,
        stream=sys.stdout
    )

    # 外部ライブラリのログ抑制
    if not debug:
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


@click.group()
@click.option('--debug', is_flag=True, help='デバッグモードを有効にし、詳細なログを出力します。')
def cli(debug):
    """SNNプロジェクト 統合CLIツール (Integrated v17.2)"""
    setup_logging(debug)
    if debug:
        logger.debug("Debug mode enabled.")


def main():
    """エントリーポイント関数"""
    # 各コマンドグループを登録
    register_core_commands(cli)
    register_model_commands(cli)
    register_app_commands(cli)
    
    # --- 新規追加 ---
    cli.add_command(recipe_cli)
    cli.add_command(demo_cli)
    # ----------------

    # CLI実行
    try:
        cli()
    except SystemExit:
        raise
    except Exception as e:
        is_debug = '--debug' in sys.argv

        if is_debug:
            logger.exception(f"❌ 致命的なエラーが発生しました (詳細): {e}")
        else:
            logger.error(f"❌ エラーが発生しました: {e}")
            logger.error("詳細を見るには --debug オプションを付けて実行してください。")

        sys.exit(1)


if __name__ == '__main__':
    main()