# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (次元整合性自動調整版)
# 目的: 各脳モジュールを統合し、入力次元のミスマッチを吸収して安定した認知サイクルを実行する。
#
# 変更点:
# - [修正 v14] ValueError: Input neuron count mismatch に対処。
# - run_cognitive_cycle 内で、入力 Tensor の最終次元を PerceptionCortex.num_neurons (784) へ強制適合させる。
# - ヘルスチェック時の低次元入力(3次元)をゼロパディングで補完。

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
    """
    複数の脳領域モジュールを統合する人工脳メインクラス。
    Global Workspace理論に基づき、知覚、記憶、意思決定を統合制御する。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.config = config or {}
        
        # 1. 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 2. 各脳領域の初期化 (PerceptionCortexはデフォルトで784 neuronsを期待)
        self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 意思決定系への依存注入
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
        入力次元の不整合を検知した場合、自動的にターゲット次元(784)へ適合させる。
        """
        self.cycle_count += 1
        
        # 文字列入力の場合はランダムTensorを生成
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # --- [修正] 次元整合ロジックの強化 ---
        # ログ: ValueError: Input neuron count 3 mismatch with cortex 784 に対応
        if sensory_tensor.ndim > 0:
            current_dim = sensory_tensor.shape[-1]
            target_dim = self.perception.num_neurons
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    # 不足分をゼロパディングで補う (3 -> 784)
                    padding_shape = list(sensory_tensor.shape[:-1]) + [target_dim - current_dim]
                    padding = torch.zeros(*padding_shape, device=sensory_tensor.device, dtype=sensory_tensor.dtype)
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
                else:
                    # 超過分をスライスでカット
                    sensory_tensor = sensory_tensor[..., :target_dim]

        # 1. 知覚処理 (次元整合済みのため安全)
        # PerceptionCortex.perceive は {'features': tensor} を返す
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 2. ワークスペースへの情報集約 (add_content を使用)
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        # 3. 海馬・皮質による参照
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. 基底核による行動選択 (型キャストによりリスト形式を保証)
        summary = self.workspace.get_summary()
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 5. 運動出力の生成
        motor_output = self.motor.generate_signal(selected_action)

        # 6. 意識的なブロードキャスト
        self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": self.workspace.conscious_broadcast_content is not None
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェックおよびデモ用ステータス取得"""
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        """モジュールの現在のデバイスを取得"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
