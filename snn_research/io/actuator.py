# ファイルパス: snn_research/io/actuator.py
# (新規作成)
#
# Title: Actuator (アクチュエータ)
#
# Description:
# - 人工脳アーキテクチャの「出力層」を担うコンポーネント。
# - MotorCortexから受け取った最終的な実行コマンドを、
#   物理的なアクションとして表現する。
# - この実装では、受け取ったコマンドをコンソールに出力することで
#   アクションの実行をシミュレートする。

from typing import List, Dict, Any

class Actuator:
    """
    MotorCortexからのコマンドを受け取り、最終的なアクションを実行するモジュール。
    """
    def __init__(self, actuator_name: str):
        """
        Args:
            actuator_name (str): アクチュエータの名称 (例: 'robot_arm', 'speaker')。
        """
        self.actuator_name = actuator_name
        print(f"🤖 アクチュエータ '{self.actuator_name}' が初期化されました。")

    def execute(self, command_log: str):
        """
        単一のコマンドを実行する。
        実際のハードウェア制御では、ここで物理的な動作がトリガーされる。

        Args:
            command_log (str): MotorCortexから送られてきた実行ログ文字列。
        """
        # ここでは、コマンドをコンソールに出力することで実行をシミュレート
        print(f"⚡️ [{self.actuator_name}] 実行: {command_log}")

    def run_command_sequence(self, command_logs: List[str]):
        """
        一連のコマンドシーケンスを順番に実行する。

        Args:
            command_logs (List[str]): MotorCortexによって生成された実行ログのリスト。
        """
        print(f"▶️ [{self.actuator_name}] コマンドシーケンスの実行を開始...")
        if not command_logs:
            print("  - 実行すべきコマンドがありません。")
            return

        for log in command_logs:
            self.execute(log)
        
        print(f"⏹️ [{self.actuator_name}] コマンドシーケンスの実行が完了しました。")