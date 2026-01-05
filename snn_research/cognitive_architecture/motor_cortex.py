# ファイルパス: snn_research/cognitive_architecture/motor_cortex.py
# Title: Motor Cortex (運動野) インターフェース統合版
# Description:
#   - 人工脳アーキテクチャの運動出力を担う。
#   - ArtificialBrainからの直接呼び出し(generate_signal)に対応。
#   - 既存のコマンドシーケンス実行機能(execute_commands)も維持。

from typing import List, Dict, Any, Optional
import torch
import logging
from snn_research.modules.reflex_module import ReflexModule

logger = logging.getLogger(__name__)


class MotorCortex:
    actuators: List[str]

    def __init__(self, actuators: Optional[List[str]] = None, device: str = 'cpu'):
        """
        Args:
            actuators: 制御対象のアクチュエータリスト。
            device: 計算デバイス。
        """
        if actuators is None:
            self.actuators = ['output_alpha', 'output_beta']
        else:
            self.actuators = actuators

        self.device = device

        # Reflex Module (Phase 4.1 Integration)
        # 入力次元などはサンプルのため固定値だが、本来はconfigから読み込む
        self.reflex_module = ReflexModule(
            input_dim=128, action_dim=10, threshold=2.0).to(device)
        self.reflex_enabled = True

        logger.info("🧠 運動野モジュール (Phase 4 Enhanced) が初期化されました。")

    def generate_signal(self, action: Any) -> List[str]:
        """
        選択された行動を具体的な運動信号（ログ文字列）に変換する。
        ArtificialBrain の run_cognitive_cycle から呼び出される。

        Args:
            action: 選択された行動（文字列やIDなど）。

        Returns:
            List[str]: 生成された実行ログ。
        """
        # 単一の行動をコマンド形式にラップして既存の execute_commands を再利用
        command_packet = [{
            'timestamp': 0.0,
            'command': str(action)
        }]
        return self.execute_commands(command_packet)

    def generate_spiking_signal(self, sensory_input: torch.Tensor) -> Optional[int]:
        """
        [Phase 4.1 New] 
        感覚入力から脊髄反射レベルでの高速なアクション生成を試みる。

        Args:
            sensory_input: 感覚入力テンソル (Batch, Dim)

        Returns:
            action_id: 反射がトリガーされた場合はアクションID、そうでない場合はNone
        """
        if not self.reflex_enabled:
            return None

        with torch.no_grad():
            action, confidence = self.reflex_module(
                sensory_input.to(self.device))

        if action is not None:
            logger.info(
                f"⚡ Reflex Triggered! Action: {action} (Conf: {confidence:.2f})")
            return action
        return None

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
            timestamp = command_data.get('timestamp', 0.0)
            command = command_data.get('command', 'IDLE')
            # 登録されている最初のアクチュエータを使用
            target_actuator = self.actuators[0] if self.actuators else "unknown"

            # 実際のハードウェア制御APIをここに記述可能

            log_entry = f"[T={timestamp:.2f}s] ACTUATOR<{target_actuator}>: EXECUTE '{command}'"
            print(f"  - {log_entry}")
            execution_log.append(log_entry)

        print("✅ 運動野: 全コマンドの実行が完了しました。")
        return execution_log
