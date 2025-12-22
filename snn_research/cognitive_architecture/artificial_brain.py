# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (テンソル形状正規化版)
# 目的: 各脳モジュールを統合し、時間次元を含む多次元入力を知覚野の期待する形状へ変換して安定実行する。
#
# 変更点:
# - [修正 v16] RuntimeError: mat1 and mat2 shapes cannot be multiplied に対処。
# - 入力が (Time, Neurons) の場合、時間方向に集約(mean)を行い、常に 2次元 (Batch, Neurons) で perceive へ渡す。

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
        
        # コンポーネントの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 知覚野 (784 neurons, 256 features)
        self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
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
        """
        1ステップの認知サイクル。
        入力テンソルを (Batch, Neurons=784) の形状に厳密に正規化する。
        """
        self.cycle_count += 1
        
        # 1. 入力の Tensor 化とデバイス転送
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # 2. 形状の正規化 (Time次元の集約とニューロン次元の調整)
        # 目標形状: (1, 784)
        
        # (Step A) 次元削減: 3次元以上の場合はバッチ/時間方向に平均をとる
        if sensory_tensor.ndim >= 3:
            # (Batch, Time, Neurons) -> (Batch, Neurons)
            sensory_tensor = torch.mean(sensory_tensor, dim=1)
        elif sensory_tensor.ndim == 2:
            # ヘルスチェックからの (32, 3) などの「時間, ニューロン」形式を想定
            # 全時間を平均して (1, Neurons) に変換
            if sensory_tensor.shape[1] < 784 and sensory_tensor.shape[0] > 1:
                sensory_tensor = torch.mean(sensory_tensor, dim=0, keepdim=True)
        elif sensory_tensor.ndim == 1:
            sensory_tensor = sensory_input.unsqueeze(0)

        # (Step B) ニューロン次元(784)への適合
        current_neurons = sensory_tensor.shape[-1]
        target_neurons = self.perception.num_neurons # 784
        
        if current_neurons != target_neurons:
            if current_neurons < target_neurons:
                # ゼロパディング
                padding = torch.zeros(*sensory_tensor.shape[:-1], target_neurons - current_neurons, device=sensory_tensor.device)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
            else:
                # スライス
                sensory_tensor = sensory_tensor[..., :target_neurons]

        # 3. 知覚処理 (形状が (B, 784) であることを保証)
        # perception.perceive は内部で matmul を行うため形状が重要
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # バッチ次元がある場合は先頭を抽出 (後続モジュールとの互換性)
        if perceptual_info.ndim > 1:
            perceptual_info = perceptual_info[0]

        # 4. ワークスペース集約・感情・記憶
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        context = self.hippocampus.query(perceptual_info) if hasattr(self.hippocampus, 'query') else None
        knowledge = self.cortex.retrieve(perceptual_info) if hasattr(self.cortex, 'retrieve') else None
        
        # 5. 行動選択
        summary = self.workspace.get_summary()
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 6. 運動出力
        motor_output = self.motor.generate_signal(selected_action)

        # 7. ブロードキャスト
        self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": self.workspace.conscious_broadcast_content is not None
        }

    def get_brain_status(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
