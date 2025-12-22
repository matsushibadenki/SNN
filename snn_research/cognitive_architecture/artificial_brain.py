# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v10] mypy修正: PerceptionCortex を nn.Module 呼び出し (forward) に統一。
# - [修正 v10] 依存関係の注入と型キャストを強化し、BasalGanglia との整合性を確保。

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
        
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 依存関係を持つコンポーネントの初期化
        self.perception = PerceptionCortex(num_neurons=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """1ステップの認知サイクルを実行。"""
        self.cycle_count += 1
        
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 256, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # 1. 知覚処理 (mypy修正: nn.Moduleとして直接呼び出し)
        perceptual_info = self.perception(sensory_tensor)
        
        # 2. ワークスペース集約
        # GlobalWorkspaceに実装されている可能性が高いメソッドを順次確認
        for method_name in ['update', 'add_content', 'receive_sensory_info']:
            method = getattr(self.workspace, method_name, None)
            if callable(method):
                try:
                    method("sensory", perceptual_info)
                    break
                except TypeError:
                    continue
        
        emotional_val = self.amygdala.process(perceptual_info)
        
        # 3. 海馬・皮質の処理
        context = self.hippocampus.query(perceptual_info) if hasattr(self.hippocampus, 'query') else None
        knowledge = self.cortex.retrieve(perceptual_info) if hasattr(self.cortex, 'retrieve') else None
        
        # 4. 行動選択
        summary = self.workspace.get_summary() if hasattr(self.workspace, 'get_summary') else []
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 5. 運動出力 (mypy修正: generate_signal 等の実定義への対応)
        motor_signal_func = getattr(self.motor, 'generate_signal', None)
        if callable(motor_signal_func):
            motor_output = motor_signal_func(selected_action)
        else:
            motor_output = torch.zeros(1)

        # 6. ブロードキャスト
        broadcast_func = getattr(self.workspace, 'broadcast', None)
        if callable(broadcast_func):
            broadcast_func()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """脳の状態を取得。"""
        drive_attr = getattr(self.motivation_system, 'get_current_drive', 0.0)
        motivation_val = drive_attr() if callable(drive_attr) else drive_attr
        return {
            "cycle": self.cycle_count,
            "motivation": motivation_val
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
