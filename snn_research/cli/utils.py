#snn_research/cli/utils.py

import sys
import os
import subprocess
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("snn_cli")

def find_python_executable():
    """実行中のPythonインタプリタのパスを返す"""
    return sys.executable

def verify_path_exists(path, description, suggestion=None):
    """
    パスの存在を確認し、存在しない場合は親切なエラーメッセージを出して終了する。
    """
    if path and not os.path.exists(path):
        logger.error(f"❌ {description} が見つかりません: {path}")
        if suggestion:
            logger.info(f"💡 ヒント: {suggestion}")
        sys.exit(1)
    return True

def run_script(script_path, args, capture_output=False):
    """指定されたPythonスクリプトをサブプロセスとして実行する。"""
    python_exec = find_python_executable()
    
    # スクリプトパスの解決: CLI実行場所(プロジェクトルート想定)からの相対パス
    if not os.path.exists(script_path):
        logger.error(f"❌ スクリプトが見つかりません: {script_path}")
        logger.error("プロジェクトのルートディレクトリから実行しているか確認してください。")
        sys.exit(1)

    command = [python_exec, script_path] + args
    logger.info(f"🚀 実行中: {' '.join(command)}")
    
    try:
        if capture_output:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(command, check=True, text=True)
        logger.info(f"✅ スクリプト {os.path.basename(script_path)} が正常に完了しました。")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ スクリプト実行中にエラーが発生しました: {script_path}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"❌ 予期せぬエラーが発生しました: {e}")
        sys.exit(1)

def run_external_command(command_list, capture_output=False):
    """指定された外部コマンドをサブプロセスとして実行する。"""
    logger.info(f"🚀 実行中: {' '.join(command_list)}")
    try:
        if capture_output:
            result = subprocess.run(command_list, check=False, text=True, capture_output=True)
        else:
            result = subprocess.run(command_list, text=True)
        return result
    except FileNotFoundError:
        logger.error(f"❌ コマンドが見つかりません: {command_list[0]}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ コマンド {command_list[0]} 実行中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)