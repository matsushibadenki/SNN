# ファイルパス: snn_research/cognitive_architecture/cerebellum.py
# (新規作成)
#
# Title: Cerebellum (小脳) モジュール
#
# Description:
# - 人工脳アーキテクチャの「運動層」に属し、運動制御を担うコンポーネント。
# - 大脳基底核から受け取った抽象的な行動計画を、より細かく、
#   タイミングが調整された一連の運動コマンドに変換する。
# - 運動学習と精密なタイミング制御の基盤となる。

from typing import Dict, Any, List, Optional

class Cerebellum:
    """
    行動計画を精密な運動コマンドのシーケンスに変換する小脳モジュール。
    """
    def __init__(self, time_resolution: float = 0.1):
        """
        Args:
            time_resolution (float): 生成される運動コマンドの時間分解能（秒）。
        """
        self.time_resolution = time_resolution
        print("🧠 小脳モジュールが初期化されました。")

    def refine_action_plan(self, selected_action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        単一の選択された行動を、一連のサブコマンドに分解・精緻化する。

        Args:
            selected_action (Dict[str, Any]):
                BasalGangliaによって選択された行動。
                例: {'action': 'A', 'value': 0.9, 'duration': 1.0}

        Returns:
            List[Dict[str, Any]]:
                タイムスタンプとサブコマンドを含む辞書のリスト。
                例: [{'timestamp': 0.0, 'command': 'A_start'},
                     {'timestamp': 0.5, 'command': 'A_mid'},
                     {'timestamp': 1.0, 'command': 'A_end'}]
        """
        action_name = selected_action.get("action")
        duration = selected_action.get("duration", 1.0) # デフォルトの持続時間を1秒とする

        if not action_name:
            return []

        print(f"🔬 小脳: 行動 '{action_name}' を精密化しています (持続時間: {duration}s)...")

        motor_commands: List[Dict[str, Any]] = []
        num_steps = int(duration / self.time_resolution)

        # 簡単な例として、行動を「開始」「中間」「終了」の3段階に分解する
        if num_steps > 0:
            # 開始コマンド
            motor_commands.append({
                "timestamp": 0.0,
                "command": f"{action_name}_start"
            })

            # 中間コマンド (もしあれば)
            if num_steps > 2:
                mid_time = (num_steps // 2) * self.time_resolution
                motor_commands.append({
                    "timestamp": round(mid_time, 2),
                    "command": f"{action_name}_mid"
                })

            # 終了コマンド
            end_time = (num_steps - 1) * self.time_resolution
            motor_commands.append({
                "timestamp": round(end_time, 2),
                "command": f"{action_name}_end"
            })

        return motor_commands