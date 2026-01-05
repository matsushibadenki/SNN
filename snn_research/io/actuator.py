# ファイルパス: snn_research/io/actuator.py
# タイトル: アクチュエータ制御モジュール (ROS2/Simulation Hybrid)
#
# 目的:
# - 人工脳 (MotorCortex) からの抽象的な行動コマンドを物理的な動作に変換する。
# - Phase 10 (Embodiment) に向けたROS2連携の実装。
# - ROS2環境がない場合は自動的にシミュレーションモードで動作し、コンソールに出力する。

# mypy fix: Added Union to imports
from typing import List, Dict, Any, Union
import json
import time

# ROS2ライブラリのインポート試行 (環境にない場合はMockとして振る舞う)
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from geometry_msgs.msg import Twist  # type: ignore # 一般的な移動ロボット用
    from std_msgs.msg import String  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # 型ヒント用のダミークラス

    class DummyNode:
        pass  # type: ignore

    Node = DummyNode


class Actuator(Node if ROS2_AVAILABLE else object):  # type: ignore
    """
    MotorCortexからのコマンドを受け取り、シミュレーションまたは実機(ROS2)で実行するモジュール。

    Features:
    - ROS2が利用可能な場合、自動的にNodeとして初期化され、トピックをpublishする。
    - 利用不可の場合、標準出力を用いたシミュレーションモードで動作する。
    """

    def __init__(self, actuator_name: str, mode: str = "auto"):
        """
        Args:
            actuator_name (str): アクチュエータの識別名 (例: 'turtlebot', 'robot_arm').
            mode (str): 'auto', 'ros2', 'simulation'. 'auto'は環境に応じて切り替え。
        """
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

                # パブリッシャーの設定 (汎用的なcmd_velとlog出力)
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

    def execute(self, command: Any):
        """
        単一のコマンドを実行する。
        文字列または辞書形式のコマンドを受け付ける。

        Args:
            command (Union[str, Dict]): 実行するコマンド。
                例: "move_forward"
                例: {"action": "move", "params": {"x": 1.0, "z": 0.5}}
        """
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

        # 移動コマンドの処理 (Twistメッセージへの変換)
        if action in ["move", "navigate"]:
            twist = Twist()
            params = cmd_dict.get("params", {})
            # 単純なマッピング例
            twist.linear.x = float(params.get("linear_x", 0.0))
            twist.angular.z = float(params.get("angular_z", 0.0))

            # "move_forward" などの簡易文字列コマンド対応
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
        # ここに将来的に物理シミュレータ(PyBullet/Mujoco)との連携コードを追加可能

    # mypy fix: Updated type hint to allow Dict (fixes artificial_brain.py error)
    def run_command_sequence(self, command_logs: List[Union[str, Dict[str, Any]]]):
        """
        一連のコマンドシーケンスを順番に実行する。

        Args:
            command_logs (List[Union[str, Dict]]): MotorCortexによって生成された実行ログのリスト。
        """
        print(f"▶️ [{self.actuator_name}] コマンドシーケンスの実行を開始...")
        if not command_logs:
            print("  - 実行すべきコマンドがありません。")
            return

        for log in command_logs:
            self.execute(log)
            # 連続実行時のウェイト（実機の場合は重要）
            if self.use_ros:
                time.sleep(0.1)

        print(f"⏹️ [{self.actuator_name}] コマンドシーケンスの実行が完了しました。")

    def __del__(self):
        """終了処理"""
        if self.use_ros and rclpy.ok():
            self.destroy_node()
            # rclpy.shutdown() はグローバルな影響があるため、アプリのメイン側で呼ぶのが安全
            print("💤 [Actuator] ROS2 Node destroyed.")
