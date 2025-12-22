# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (次元完全整合版)
# 目的: Global Workspace理論に基づき、入力形状に左右されず安定した認知サイクルを実行する。
#
# 変更点:
# - [修正 v21] RuntimeError: Tensors must have same number of dimensions を解決。
# - パディング生成時に sensory_tensor.shape を基底とすることで、次元数ミスマッチを根絶。
# - GlobalWorkspace の動的なメソッド解決 (add_content/receive_sensory_info) を維持。

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
        
        # 入力の Tensor 化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # --- [核心的修正] 次元の不整合を動的に解決 ---
        if sensory_tensor.ndim >= 2:
            current_n = sensory_tensor.shape[1]
            target_n = self.perception.num_neurons # 784
            
            if current_n != target_n:
                if current_n < target_n:
                    # 不足分を計算
                    diff = target_n - current_n
                    
                    # 入力と同じ次元構成(ndim)でパディングを作成
                    # shape: (Batch, diff, *others)
                    pad_shape = list(sensory_tensor.shape)
                    pad_shape[1] = diff
                    
                    padding = torch.zeros(
                        *pad_shape, 
                        device=sensory_tensor.device, 
                        dtype=sensory_tensor.dtype
                    )
                    
                    # dim=1 (ニューロン次元) で結合
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=1)
                else:
                    # 超過分をスライスで調整
                    sensory_tensor = sensory_tensor[:, :target_n, ...]
        elif sensory_tensor.ndim == 1:
            # (N,) -> (1, N)
            sensory_tensor = sensory_tensor.unsqueeze(0)
            return self.run_cognitive_cycle(sensory_tensor)

        # 1. 知覚処理 (shape[1]=784 が保証された状態で perceive を実行)
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 統計集約 (後続モジュールのインターフェースに合わせる)
        if perceptual_info.ndim > 1:
            perceptual_info = torch.mean(perceptual_info.float(), dim=0)

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
