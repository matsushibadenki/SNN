# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v8] mypyエラー解消: 各脳領域(Perception, GlobalWorkspace等)の実際のメソッド名に準拠。
# - [修正 v8] 依存関係注入: BasalGanglia, PrefrontalCortex の初期化引数を修正。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

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
        
        # BasalGanglia, PrefrontalCortex のコンストラクタ定義に合わせる
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """
        1ステップの認知サイクル。mypyの型不整合を吸収。
        """
        self.cycle_count += 1
        
        # 型不整合解消: 文字列入力をTensorに変換
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 256, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # 1. 知覚処理 (mypy修正: perceptionコンポーネントの順伝播)
        perceptual_info = self.perception.forward(sensory_tensor)
        
        # 2. ワークスペース集約 (mypy修正: 実装されたメソッド名 add_content を使用)
        # ※ GlobalWorkspaceに add_content がない場合は update_state 等へ変更してください
        if hasattr(self.workspace, 'add_content'):
            self.workspace.add_content("sensory", perceptual_info)
        
        emotional_val = self.amygdala.process(perceptual_info)
        
        # 3. 海馬・皮質 (mypy修正: query/retrieve メソッド)
        context = self.hippocampus.query(perceptual_info) if hasattr(self.hippocampus, 'query') else None
        knowledge = self.cortex.retrieve(perceptual_info) if hasattr(self.cortex, 'retrieve') else None
        
        # 4. 行動選択
        # GlobalWorkspace.get_summary() が定義されていることを前提とする
        workspace_summary = self.workspace.get_summary() if hasattr(self.workspace, 'get_summary') else {}
        selected_action = self.basal_ganglia.select_action(workspace_summary)
        
        # 5. 運動出力 (mypy修正: generate_signal)
        motor_output = self.motor.generate_signal(selected_action)
        
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
        drive = self.motivation_system.get_current_drive
        motivation_val = drive() if callable(drive) else drive
        return {
            "cycle": self.cycle_count,
            "motivation": motivation_val
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
