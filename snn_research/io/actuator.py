# ファイルパス: snn_research/io/actuator.py
# タイトル: アクチュエータ制御モジュール (Fix: execute accepts kwargs)
# 修正: executeメソッドが action_id などの追加引数を許容するように変更。

from typing import List, Dict, Any, Union
import json
import time

# ROS2ライブラリのインポート試行
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from geometry_msgs.msg import Twist  # type: ignore
    from std_msgs.msg import String  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

    class DummyNode:
        pass  # type: ignore
    Node = DummyNode


class Actuator(Node if ROS2_AVAILABLE else object):  # type: ignore
    """
    MotorCortexからのコマンドを受け取り、シミュレーションまたは実機(ROS2)で実行するモジュール。
    """

    def __init__(self, actuator_name: str, mode: str = "auto"):
        self.actuator_name = actuator_name
        self.mode = mode

        # モードの決定
        if self.mode == "auto":
            self.use_ros = ROS2_AVAILABLE
        elif self.mode == "ros2":
            if not ROS2_AVAILABLE:
                print(
                    "⚠️ [Actuator] ROS2モードが要求されましたが、ライブラリが見つかりません。シミュレーションモードに切り替えます。")
                self.use_ros = False
            else:
                self.use_ros = True
        else:
            self.use_ros = False

        # ROS2の初期化
        if self.use_ros:
            try:
                if not rclpy.ok():
                    rclpy.init()
                super().__init__(f'snn_actuator_{actuator_name}')

                # パブリッシャーの設定
                self.vel_publisher = self.create_publisher(
                    Twist, f'/{actuator_name}/cmd_vel', 10)
                self.log_publisher = self.create_publisher(
                    String, f'/{actuator_name}/snn_log', 10)

                print(
                    f"🤖 [Actuator] ROS2 Node initialized: /snn_actuator_{actuator_name}")
            except Exception as e:
                print(f"⚠️ [Actuator] ROS2初期化中にエラーが発生しました: {e}")
                self.use_ros = False

        if not self.use_ros:
            print(
                f"🖥️ [Actuator] Simulation Mode initialized for '{self.actuator_name}'")

    def execute(self, command: Any, **kwargs: Any):
        """
        単一のコマンドを実行する。
        Args:
            command: 実行するコマンド
            **kwargs: action_id 等の追加メタデータを受け取る
        """
        # メタ情報のログ出力などが必要であればkwargsを使用
        action_id = kwargs.get("action_id")
        if action_id:
            # ROSログやデバッグ出力にIDを含める等の処理が可能
            pass

        # コマンドの正規化
        cmd_str = ""
        cmd_dict = {}

        if isinstance(command, str):
            cmd_str = command
            # JSON形式の文字列ならパースを試みる
            if command.strip().startswith("{"):
                try:
                    cmd_dict = json.loads(command)
                except Exception:
                    cmd_dict = {"action": "raw_command", "content": command}
            else:
                cmd_dict = {"action": "raw_command", "content": command}
        elif isinstance(command, dict):
            cmd_dict = command
            cmd_str = json.dumps(command)

        # 実行ロジック
        if self.use_ros:
            self._execute_ros(cmd_dict, cmd_str)
        else:
            self._execute_sim(cmd_dict, cmd_str)

    def _execute_ros(self, cmd_dict: Dict, cmd_str: str):
        """ROS2経由でのコマンド実行"""
        action = cmd_dict.get("action", "unknown")

        # ログトピックへのPublish
        msg_log = String()
        msg_log.data = f"Executing: {cmd_str}"
        self.log_publisher.publish(msg_log)

        # 移動コマンドの処理
        if action in ["move", "navigate"]:
            twist = Twist()
            params = cmd_dict.get("params", {})
            twist.linear.x = float(params.get("linear_x", 0.0))
            twist.angular.z = float(params.get("angular_z", 0.0))

            content = cmd_dict.get("content", "")
            if content == "move_forward":
                twist.linear.x = 0.2
            elif content == "turn_left":
                twist.angular.z = 0.5
            elif content == "stop":
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            self.vel_publisher.publish(twist)
            print(
                f"⚡️ [ROS2] Published cmd_vel: linear={twist.linear.x}, angular={twist.angular.z}")
        else:
            print(f"⚡️ [ROS2] Generic command handled: {action}")

    def _execute_sim(self, cmd_dict: Dict, cmd_str: str):
        """シミュレーションモードでのコマンド実行"""
        print(f"⚡️ [SIM] Actuator '{self.actuator_name}' executing: {cmd_str}")

    def run_command_sequence(self, command_logs: List[Union[str, Dict[str, Any]]]):
        """一連のコマンドシーケンスを順番に実行する。"""
        print(f"▶️ [{self.actuator_name}] コマンドシーケンスの実行を開始...")
        if not command_logs:
            print("  - 実行すべきコマンドがありません。")
            return

        for log in command_logs:
            self.execute(log)
            if self.use_ros:
                time.sleep(0.1)

        print(f"⏹️ [{self.actuator_name}] コマンドシーケンスの実行が完了しました。")

    def __del__(self):
        """終了処理"""
        if self.use_ros and rclpy.ok():
            self.destroy_node()
            print("💤 [Actuator] ROS2 Node destroyed.")
