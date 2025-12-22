# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (多次元完全整合版)
# 目的: 入力テンソルの次元数(1D/2D/3D)を問わず、最後の次元をニューロン数として整合させる。

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
        
        # 1. 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 2. 各脳領域の初期化 (PerceptionCortexは 784 neuronsを期待)
        self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 3. 意思決定系の初期化
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
        device = self.get_device()
        
        # 入力の Tensor 化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=device) 
        else:
            sensory_tensor = sensory_input.to(device)

        # --- [修正 v22] 最後の次元をニューロン数として正規化 ---
        if sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0) # (N,) -> (1, N)

        target_n = self.perception.num_neurons # 784
        current_n = sensory_tensor.shape[-1]
        
        if current_n != target_n:
            if current_n < target_n:
                # パディング: 最後の次元(dim=-1)を target_n に合わせる
                diff = target_n - current_n
                pad_shape = list(sensory_tensor.shape)
                pad_shape[-1] = diff
                padding = torch.zeros(*pad_shape, device=device, dtype=sensory_tensor.dtype)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
            else:
                # スライス: 最後の次元を超過分カット
                sensory_tensor = sensory_tensor[..., :target_n]

        # 1. 知覚処理
        perception_result = self.perception.perceive(sensory_tensor)
        
        # perceptual_info の集約 (BatchやTimeを潰してベクトル化)
        raw_features = perception_result.get("features")
        if raw_features is not None:
            # 全次元を平均して (feature_dim,) のベクトルにする
            # 理由: 後続の amygdala や cortex は単一ベクトルを期待しているため
            while raw_features.ndim > 1:
                raw_features = torch.mean(raw_features.float(), dim=0)
            perceptual_info = raw_features
        else:
            perceptual_info = torch.zeros(256, device=device)

        # 2. ワークスペース集約
        for method_name in ['receive_sensory_info', 'update', 'add_content']:
            method = getattr(self.workspace, method_name, None)
            if callable(method):
                try:
                    method("sensory", perceptual_info)
                    break
                except (TypeError, AttributeError):
                    continue
        
        # 3. 感情・記憶・意思決定
        emotional_val = self.amygdala.process(perceptual_info)
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
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
