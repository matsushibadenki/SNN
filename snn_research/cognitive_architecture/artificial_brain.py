# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v5] mypyエラー解消: cortex, basal_ganglia, motor, amygdala 等の属性を明示的に初期化。
# - [修正 v5] 外部テスト (test_artificial_brain.py) およびダッシュボード (app/dashboard.py) との整合性を確保。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .global_workspace import GlobalWorkspace
from .hippocampus import Hippocampus
from .cortex import Cortex
from .basal_ganglia import BasalGanglia
from .motor_cortex import MotorCortex
from .amygdala import Amygdala
from .prefrontal_cortex import PrefrontalCortex
from .perception_cortex import PerceptionCortex

class ArtificialBrain(nn.Module):
    """
    複数の脳領域モジュールを統合する人工脳メインクラス。
    Global Workspace (GWT) を通じて情報のブロードキャストと意識的処理を模倣する。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # 主要コンポーネントの初期化
        # mypyエラーを回避し、テストやダッシュボードからのアクセスを可能にする
        self.workspace = GlobalWorkspace()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        self.basal_ganglia = BasalGanglia()
        self.motor = MotorCortex()
        self.amygdala = Amygdala()
        self.prefrontal_cortex = PrefrontalCortex()
        self.perception = PerceptionCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        1ステップの認知サイクルを実行する。
        知覚 -> ワークスペース集約 -> 意思決定 -> 行動生成
        """
        self.cycle_count += 1
        
        # 1. 知覚処理
        perceptual_info = self.perception(sensory_input)
        
        # 2. ワークスペースへの情報集約
        self.workspace.add_content("sensory", perceptual_info)
        self.workspace.add_content("emotional", self.amygdala.process(perceptual_info))
        
        # 3. 皮質および海馬による文脈参照
        context = self.hippocampus.query(perceptual_info)
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. 基底核による行動選択 (Action Selection)
        selected_action = self.basal_ganglia.select_action(self.workspace.get_summary())
        
        # 5. 運動出力の生成
        motor_output = self.motor.generate_signal(selected_action)
        
        # 6. 意識的ブロードキャスト (GWT)
        self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": selected_action,
            "motor_output": motor_output,
            "broadcasted": self.workspace.conscious_broadcast_content
        }

    def sleep_and_consolidate(self):
        """
        睡眠フェーズ: 海馬から皮質への記憶の固定化を行う。
        """
        knowledge_to_transfer = self.hippocampus.working_memory
        for item in knowledge_to_transfer:
            self.cortex.learn(item)
        self.hippocampus.clear_working_memory()
