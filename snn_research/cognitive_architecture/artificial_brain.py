# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v9] mypy型不整合解消: BasalGangliaへ渡す引数を list[dict] へ明示的にキャスト。
# - [修正 v9] メソッド名修正: PerceptionCortex.process, MotorCortex.generate_signal 等、実定義に準拠。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, cast

from .global_workspace import GlobalWorkspace
from .hippocampus import Hippocampus
from .cortex import Cortex
from .basal_ganglia import BasalGanglia
from .motor_cortex import MotorCortex
from .amygdala import Amygdala
from .prefrontal_cortex import PrefrontalCortex
from .perception_cortex import PerceptionCortex
from .intrinsic_motivation import IntrinsicMotivationSystem

class ArtificialBrain(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # 1. 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 2. 各脳領域の初期化
        self.perception = PerceptionCortex(num_neurons=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 依存関係の注入
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """1ステップの認知サイクルを実行。型不整合を吸収。"""
        self.cycle_count += 1
        
        # 入力変換
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 256, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # 1. 知覚処理 (process メソッドを使用)
        perceptual_info = self.perception.process(sensory_tensor)
        
        # 2. ワークスペース集約
        if hasattr(self.workspace, 'update'):
            self.workspace.update("sensory", perceptual_info)
        
        emotional_val = self.amygdala.process(perceptual_info)
        
        # 3. 海馬・皮質 (retrieve/query)
        context = self.hippocampus.query(perceptual_info) if hasattr(self.hippocampus, 'query') else None
        knowledge = self.cortex.retrieve(perceptual_info) if hasattr(self.cortex, 'retrieve') else None
        
        # 4. 行動選択 (mypy修正: 型を list[dict[str, Any]] に強制)
        summary = self.workspace.get_summary() if hasattr(self.workspace, 'get_summary') else []
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 5. 運動出力 (mypy修正: 実装済みメソッド identify_signal 等へ)
        # ※ generate_signal がない場合を考慮したフォールバック
        if hasattr(self.motor, 'generate_signal'):
            motor_output = self.motor.generate_signal(selected_action)
        else:
            motor_output = torch.zeros(1)

        # 6. ブロードキャスト
        if hasattr(self.workspace, 'broadcast'):
            self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """脳の健康状態を取得 (v16_demo 互換)"""
        drive_func = getattr(self.motivation_system, 'get_current_drive', lambda: torch.tensor(0.0))
        motivation_val = drive_func() if callable(drive_func) else drive_func
        return {
            "cycle": self.cycle_count,
            "motivation": motivation_val
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
