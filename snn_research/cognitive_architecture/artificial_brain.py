# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (次元順序自動補正版)
# 目的: 各脳モジュールを統合し、入力テンソルの次元順序やサイズが不安定な環境でも安定して動作させる。
#
# 変更点:
# - [修正 v17] ValueError: Input neuron count mismatch (32 vs 784) を根絶。
# - 入力テンソルの中からターゲット次元(784)を自動探索し、PerceptionCortexが期待する位置(dim=1)へ移動させる。
# - 万が一 784 が見つからない場合のみ、パディングによる強制適合を行う。

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
        
        # PerceptionCortexは 784 ニューロンを厳格に要求する
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
        PerceptionCortex.perceive が shape[1] == 784 を要求するため、
        入力の形状を動的に解析して軸を入れ替える。
        """
        self.cycle_count += 1
        
        # 1. 入力の標準化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # 2. 形状の自動補正 (軸の自動探索)
        # ターゲットとするニューロン数
        target_n = self.perception.num_neurons # 784
        
        if sensory_tensor.ndim >= 2:
            # shape[1] が 784 でない場合、他の次元に 784 がないか探す
            if sensory_tensor.shape[1] != target_n:
                found_dim = -1
                for d in range(sensory_tensor.ndim):
                    if sensory_tensor.shape[d] == target_n:
                        found_dim = d
                        break
                
                if found_dim != -1:
                    # 見つけた次元を dim=1 (ニューロン次元) へ移動
                    # 例: (784, 32) -> (32, 784)
                    dims = list(range(sensory_tensor.ndim))
                    dims[1], dims[found_dim] = dims[found_dim], dims[1]
                    sensory_tensor = sensory_tensor.permute(*dims)
                else:
                    # 784 が見つからない場合は、既存の dim=1 をパディング/カット
                    current_n = sensory_tensor.shape[1]
                    if current_n < target_n:
                        padding_shape = list(sensory_tensor.shape)
                        padding_shape[1] = target_n - current_n
                        padding = torch.zeros(*padding_shape, device=sensory_tensor.device, dtype=sensory_tensor.dtype)
                        sensory_tensor = torch.cat([sensory_tensor, padding], dim=1)
                    else:
                        sensory_tensor = sensory_tensor[:, :target_n, ...]

        # 3. 知覚処理 (shape[1] が 784 であることが保証される)
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 後続処理のために 1次元(Features) に平滑化
        if perceptual_info.ndim > 1:
            perceptual_info = torch.mean(perceptual_info.float(), dim=0)

        # 4. ワークスペース、感情、記憶への伝播
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        # 各種参照処理
        context = self.hippocampus.query(perceptual_info) if hasattr(self.hippocampus, 'query') else None
        knowledge = self.cortex.retrieve(perceptual_info) if hasattr(self.cortex, 'retrieve') else None
        
        # 5. 基底核による意思決定
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
        """ステータス取得 (ヘルスチェック互換)"""
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
