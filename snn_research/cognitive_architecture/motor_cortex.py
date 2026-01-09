# ファイルパス: snn_research/cognitive_architecture/motor_cortex.py
# 日本語タイトル: 運動野モジュール v2.4 (Reflex Integration)
# 目的: ReflexModuleを統合し、感覚入力から直接アクションを生成するパスを追加。

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Union

# Import ReflexModule (Circular import avoidance if needed, but here simple import)
from snn_research.modules.reflex_module import ReflexModule

logger = logging.getLogger(__name__)


class MotorCortex(nn.Module):
    """
    前頭前野(PFC)からの計画を実行可能な運動指令に変換する。
    """

    def __init__(self, actuators: Optional[List[str]] = None, device: str = 'cpu'):
        super().__init__()
        self.actuators = actuators or ["voice_synthesizer", "robotic_arm"]
        self.device = device

        # 簡易的なアクションマッピング
        self.action_space = {
            "wait": 0,
            "speak": 1,
            "move": 2,
            "observe": 3,
            "sleep": 4
        }

        # 反射モジュール (Spinal Cord equivalent)
        # 入力128次元、アクション5種類と仮定
        self.reflex_module = ReflexModule(
            input_dim=128, action_dim=5, threshold=2.0).to(device)
        self.reflex_enabled = False

        logger.info(
            f"🦾 Motor Cortex initialized (Actuators: {self.actuators}, Device: {self.device}).")

    def forward(self, x):
        # PyTorch Moduleとしての互換性
        return x

    def generate_command(self, plan: Union[Dict[str, Any], str, Any]) -> Dict[str, Any]:
        """
        [ArtificialBrain Interface]
        PFCからの計画や意識内容を受け取り、具体的なアクションコマンドを生成する。
        """
        command: Dict[str, Any] = {
            "action_type": "wait",
            "parameters": {},
            "target_actuator": None
        }

        # 入力が辞書の場合 (PFC Plan)
        if isinstance(plan, dict):
            directive = plan.get("directive", "monitor")
            target = plan.get("target")

            if directive == "process_language":
                command["action_type"] = "speak"
                command["parameters"] = {"text": f"Processing: {target}"}
                command["target_actuator"] = "voice_synthesizer"

            elif directive == "inspect_visual":
                command["action_type"] = "move"
                command["parameters"] = {
                    "direction": "focus", "target": target}
                command["target_actuator"] = "camera_gimbal"

            elif directive == "sleep":
                command["action_type"] = "sleep"
                command["target_actuator"] = "system"

        # 入力が文字列の場合 (Simple String)
        elif isinstance(plan, str):
            if "hello" in plan.lower():
                command["action_type"] = "speak"
                command["parameters"] = {"text": "Hello."}
            elif "sleep" in plan.lower():
                command["action_type"] = "sleep"

        return command

    def execute_commands(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        [Legacy/Batch Interface] 複数のコマンドを実行し、実行ログを返す。
        """
        results = []
        for cmd in commands:
            command_str = cmd.get('command', str(cmd))
            log_entry = f"Executed: {command_str}"
            logger.info(f"🦾 {log_entry}")
            results.append(log_entry)
        return results

    def generate_spiking_signal(self, sensory_input: torch.Tensor) -> Optional[int]:
        """
        [New] 感覚入力に対して、反射モジュールを用いて即座にスパイク信号（アクションID）を生成する。
        ReflexがトリガーされなければNoneを返す。
        """
        if not self.reflex_enabled:
            return None

        sensory_input = sensory_input.to(self.device)
        action_id, confidence = self.reflex_module(sensory_input)

        if action_id is not None:
            logger.info(
                f"⚡ Reflex Action Triggered: ID={action_id} (Conf: {confidence:.2f})")
            return action_id

        return None
