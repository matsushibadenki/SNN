# ファイルパス: snn_research/cognitive_architecture/motor_cortex.py
# (修正)
#
# Title: Motor Cortex (運動野) モジュール
#
# Description:
# - mypyエラーを解消するため、Optional型を明示的にインポート・使用するよう修正。
# - 人工脳アーキテクチャの「運動層」の最終出力を担うコンポーネント。
# - 小脳から受け取った一連の精密な運動コマンドを、
#   実際のアクチュエータを駆動するための具体的な出力信号に変換する。
# - これにより、抽象的な行動計画が物理的なアクションとして結実する。

from typing import List, Dict, Any, Optional

class MotorCortex:
    actuators: List[str]

    def __init__(self, actuators: Optional[List[str]] = None):
        if actuators is None:
            self.actuators = ['output_alpha', 'output_beta']
        else:
            self.actuators = actuators
        print("🧠 運動野モジュールが初期化されました。")

    def execute_commands(self, motor_commands: List[Dict[str, Any]]) -> List[str]:
        """
        コマンドシーケンスを実行し、実行結果ログを返す。
        （ハードウェアAPIへのフックポイントとして機能）
        """
        execution_log: List[str] = []
        if not motor_commands:
            return execution_log

        print("🦾 運動野: コマンドシーケンスの実行を開始...")

        for command_data in motor_commands:
            timestamp = command_data.get('timestamp')
            command = command_data.get('command')
            target_actuator = self.actuators[0] # 簡易的に割り当て

            # 実際のハードウェア制御APIをここに記述する
            # ex: hardware_api.send(target_actuator, command)
            
            log_entry = f"[T={timestamp:.2f}s] ACTUATOR<{target_actuator}>: EXECUTE '{command}'"
            print(f"  - {log_entry}")
            execution_log.append(log_entry)

        print("✅ 運動野: 全コマンドの実行が完了しました。")
        return execution_log