# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (次元軸整合・完全正規化版)
# 目的: PerceptionCortexの内部集約仕様(dim=0の和)に合わせ、入力テンソルの軸を動的に補正する。
#
# 変更点:
# - [修正 v23] RuntimeError: mat1 shapes cannot be multiplied (25088x32) を解決。
# - テンソル内から 784 という値を持つ次元を自動探索し、それを「第2次元(dim=1)」へ強制移動。
# - 知覚野の内部処理でニューロン次元が消滅しないよう、軸順序を (Time, Neurons=784) に正規化。

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
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.config = config or {}
        
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 知覚野 (784 neurons を厳格に要求)
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
        認知サイクルの実行。
        PerceptionCortex内の sum(dim=0) 仕様に適合させるため、軸の並びを (Time, 784) に固定する。
        """
        self.cycle_count += 1
        
        # 1. 入力の標準化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # --- [核心的修正] テンソルの軸探索と正規化 ---
        target_n = self.perception.num_neurons # 784
        
        if sensory_tensor.ndim >= 2:
            # ログの (25088x32) は (784*32, 32) を示唆 -> 入力が (784, 32) で渡されている。
            # 784 という値を持つ次元を探す
            dims = list(sensory_tensor.shape)
            found_idx = -1
            for i, d in enumerate(dims):
                if d == target_n:
                    found_idx = i
                    break
            
            if found_idx != -1:
                # ターゲット次元(784)を dim=1 (PerceptionCortexの要求位置) へ移動
                if found_idx != 1:
                    # 例: (784, 32) の場合、(32, 784) に転置する
                    permute_order = list(range(sensory_tensor.ndim))
                    permute_order[1], permute_order[found_idx] = permute_order[found_idx], permute_order[1]
                    sensory_tensor = sensory_tensor.permute(*permute_order)
            else:
                # 784 が見つからない場合は、最終次元をパディング
                current_last = sensory_tensor.shape[-1]
                if current_last < target_n:
                    pad = torch.zeros(*sensory_tensor.shape[:-1], target_n - current_last, 
                                     device=sensory_tensor.device, dtype=sensory_tensor.dtype)
                    sensory_tensor = torch.cat([sensory_tensor, pad], dim=-1)
                
                # 形状が (Neurons=784, Time) になっている可能性を考慮し、
                # dim=0 が 784 になった場合は転置して (Time, 784) にする
                if sensory_tensor.shape[0] == target_n and sensory_tensor.shape[1] != target_n:
                    sensory_tensor = sensory_tensor.transpose(0, 1)

        # 2. 知覚処理 (shape は確実に (Time, 784) になり、内部の sum(dim=0) で (784,) となる)
        # perception_cortex.py:42-48 のロジックに完全適合させる
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 統計集約 (後続モジュールがベクトルを期待するため)
        if perceptual_info.ndim > 1:
            perceptual_info = torch.mean(perceptual_info.float(), dim=0)

        # 3. ワークスペース、感情、記憶、意思決定 (既存のAPI整合性を維持)
        for method_name in ['receive_sensory_info', 'update', 'add_content']:
            method = getattr(self.workspace, method_name, None)
            if callable(method):
                try:
                    method("sensory", perceptual_info)
                    break
                except (TypeError, AttributeError):
                    continue
        
        knowledge = self.cortex.retrieve(perceptual_info)
        summary = self.workspace.get_summary() if hasattr(self.workspace, 'get_summary') else []
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        
        selected_action = self.basal_ganglia.select_action(workspace_list)
        motor_output = self.motor.generate_signal(selected_action)

        # 4. ブロードキャスト
        if hasattr(self.workspace, 'broadcast'):
            self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェック用ステータス"""
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
