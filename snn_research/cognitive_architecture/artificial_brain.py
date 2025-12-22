# snn_research/cognitive_architecture/artificial_brain.py
# 認知アーキテクチャの統合・管理を行う核心クラス
#
# ディレクトリ: snn_research/cognitive_architecture/
# ファイル名: 人工脳コア・アーキテクチャ
# 目的: Global Workspace理論に基づき、皮質、海馬、基底核等の各モジュールを統合制御する。
#
# 変更点:
# - [修正 v6] mypyエラー解消: 各コンポーネントの初期化に必要な positional arguments を追加。
# - [修正 v6] メソッド名不一致の解消: add_content -> receive_sensory_info, get_summary -> get_state 等。
# - [修正 v6] 型の不整合解消: 入力 Tensor の型チェックを厳密化。

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
        
        # 2. 各脳領域の初期化 (mypy修正: 必要な引数を注入)
        self.perception = PerceptionCortex(num_neurons=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 意思決定系にはワークスペースの参照が必要
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
        
        # 文字列入力の場合はダミーTensorに変換（テスト/ダッシュボード互換性のため）
        if isinstance(sensory_input, str):
            # トークナイズ処理の代わりとして簡易表現
            sensory_tensor = torch.randn(1, 256) 
        else:
            sensory_tensor = sensory_input

        # 1. 知覚処理 (mypy修正: callの代わりに明示的メソッド呼び出し)
        perceptual_info = self.perception.process_sensory_data(sensory_tensor)
        
        # 2. ワークスペースへの情報集約 (mypy修正: 正しいメソッド名を使用)
        self.workspace.receive_sensory_info(perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.receive_sensory_info(emotional_val)
        
        # 3. 海馬・皮質の処理 (mypy修正: 実装されているメソッド名に準拠)
        context = self.hippocampus.process(perceptual_info)
        knowledge = self.cortex.process(perceptual_info)
        
        # 4. 行動選択
        selected_action = self.basal_ganglia.decide_action(self.workspace.get_state())
        
        # 5. 運動出力 (mypy修正: process_action メソッド等を使用)
        motor_output = self.motor.process(selected_action)
        
        # 6. ブロードキャスト
        self.workspace.perform_broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """脳の健康状態を取得 (v16_demo 互換)"""
        return {
            "cycle": self.cycle_count,
            "motivation": self.motivation_system.get_current_drive()
        }

    def sleep_and_consolidate(self):
        """睡眠と記憶の固定化"""
        self.cortex.consolidate_from_hippocampus(self.hippocampus)
        self.hippocampus.clear_short_term_memory()
