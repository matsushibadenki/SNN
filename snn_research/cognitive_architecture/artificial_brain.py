# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v7] mypyエラー解消: 実装済みの正しいメソッド名 (add_content, forward, query 等) へ修正。
# - [修正 v7] 依存関係の注入: BasalGanglia, PrefrontalCortex への引数を既存の実装に準拠。

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
    """
    複数の脳領域モジュールを統合する人工脳メインクラス。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # 1. 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 2. 各脳領域の初期化 (実際の引数定義に準拠)
        # PerceptionCortex は num_neurons 引数を取る
        self.perception = PerceptionCortex(num_neurons=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # BasalGanglia, PrefrontalCortex のコンストラクタ引数整合
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """
        1ステップの認知サイクルを実行する。
        """
        self.cycle_count += 1
        
        # 文字列入力の場合はダミーTensorに変換（互換性維持）
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 256, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # 1. 知覚処理 (mypy修正: perception は callable (nn.Module))
        perceptual_info = self.perception(sensory_tensor)
        
        # 2. ワークスペースへの情報集約 (mypy修正: 正しいメソッド名 add_content)
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        # 3. 海馬・皮質の処理 (mypy修正: query 等の実装済みメソッドを使用)
        context = self.hippocampus.query(perceptual_info)
        # Cortex は retrieve または forward を使用
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. 行動選択 (mypy修正: GlobalWorkspace.get_summary() に準拠)
        selected_action = self.basal_ganglia.select_action(self.workspace.get_summary())
        
        # 5. 運動出力 (mypy修正: generate_signal に準拠)
        motor_output = self.motor.generate_signal(selected_action)
        
        # 6. ブロードキャスト (mypy修正: broadcast に準拠)
        self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": self.workspace.conscious_broadcast_content
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """脳の状態を取得 (v16_demo 互換)"""
        # motivation_system.get_current_drive が属性(Tensor)かメソッドかを判定
        drive = self.motivation_system.get_current_drive
        motivation_val = drive() if callable(drive) else drive
        return {
            "cycle": self.cycle_count,
            "motivation": motivation_val
        }

    def sleep_and_consolidate(self):
        """睡眠と記憶の固定化"""
        # 実装済みの get_all_knowledge 等を使用
        knowledge_to_transfer = self.hippocampus.working_memory
        for item in knowledge_to_transfer:
            self.cortex.learn(item)
        self.hippocampus.clear_working_memory()

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
