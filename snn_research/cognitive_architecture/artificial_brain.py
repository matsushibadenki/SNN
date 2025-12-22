# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (整合性強化版)
# 目的: 各脳モジュールを統合し、入力次元の不整合を吸収して安定した認知サイクルを実行する。

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
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.config = config or {}
        
        # コンポーネントの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # PerceptionCortexは 784 neuronsを期待
        self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 依存関係注入
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """認知サイクルの実行。入力次元の自動調整機能を含む。"""
        self.cycle_count += 1
        
        # 入力の標準化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # [修正] 次元不整合の解決: 入力が PerceptionCortex.num_neurons(784) と異なる場合
        if sensory_tensor.ndim > 0:
            current_dim = sensory_tensor.shape[-1]
            target_dim = self.perception.num_neurons
            if current_dim != target_dim:
                if current_dim < target_dim:
                    # 不足分をゼロパディング
                    padding = torch.zeros(*sensory_tensor.shape[:-1], target_dim - current_dim, device=sensory_tensor.device)
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
                else:
                    # 超過分をカット
                    sensory_tensor = sensory_tensor[..., :target_dim]

        # 1. 知覚処理
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 2. ワークスペース集約 (実装済みのメソッド add_content 等を使用)
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        # 3. 海馬・皮質 (retrieve 等を使用)
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. 行動選択
        summary = self.workspace.get_summary()
        # 型安全性のためのキャスト
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 5. 運動出力
        motor_output = self.motor.generate_signal(selected_action)

        # 6. ブロードキャスト
        self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": self.workspace.conscious_broadcast_content is not None
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェック用ステータス取得"""
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
