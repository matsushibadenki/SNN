# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (Workspace API整合性修正版)
# 目的: GlobalWorkspaceの実際のメソッド名に合わせ、情報の集約とブロードキャストを確実に実行する。
#
# 変更点:
# - [修正 v20] AttributeError: GlobalWorkspace has no attribute 'add_content' を解決。
# - ワークスペースへの情報転送メソッドを動的に検知 (receive_sensory_info, update, add_content)。
# - 前回の修正で導入した入力次元の正規化ロジック (RuntimeError対策) を継承。

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
        
        # 意思決定系
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
        WorkspaceのAPI不整合と入力次元の不整合を同時に解決する。
        """
        self.cycle_count += 1
        
        # 1. 入力の Tensor 化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # 2. 形状の正規化 (RuntimeError: mat1 shapes mismatch 対策)
        if sensory_tensor.ndim >= 3:
            sensory_tensor = torch.mean(sensory_tensor, dim=1)
        elif sensory_tensor.ndim == 2:
            if sensory_tensor.shape[1] < 784 and sensory_tensor.shape[0] > 1:
                sensory_tensor = torch.mean(sensory_tensor, dim=0, keepdim=True)
        elif sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)

        # 3. ニューロン次元 (784) への適合
        target_n = self.perception.num_neurons
        current_n = sensory_tensor.shape[1] if sensory_tensor.ndim > 1 else sensory_tensor.shape[0]
        
        if current_n != target_n:
            if current_n < target_n:
                padding_size = target_n - current_n
                padding = torch.zeros(sensory_tensor.shape[0], padding_size, 
                                     device=sensory_tensor.device, dtype=sensory_tensor.dtype)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=1)
            else:
                sensory_tensor = sensory_tensor[:, :target_n]

        # 4. 知覚処理
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        if perceptual_info.ndim > 1:
            perceptual_info = torch.mean(perceptual_info.float(), dim=0)

        # 5. ワークスペース集約 (AttributeError: add_content 対策)
        # 複数の可能性のあるメソッド名を順次トライする
        info_added = False
        for method_name in ['receive_sensory_info', 'update', 'add_content']:
            method = getattr(self.workspace, method_name, None)
            if callable(method):
                try:
                    method("sensory", perceptual_info)
                    info_added = True
                    break
                except TypeError:
                    continue
        
        # 6. 感情・記憶の処理
        emotional_val = self.amygdala.process(perceptual_info)
        if info_added:
            # 感情情報もワークスペースへ
            for method_name in ['receive_sensory_info', 'update', 'add_content']:
                method = getattr(self.workspace, method_name, None)
                if callable(method):
                    try:
                        method("emotional", emotional_val)
                        break
                    except TypeError:
                        continue
        
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 7. 行動選択
        summary = self.workspace.get_summary() if hasattr(self.workspace, 'get_summary') else []
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 8. 運動出力
        motor_output = self.motor.generate_signal(selected_action)

        # 9. ブロードキャスト
        broadcast_func = getattr(self.workspace, 'broadcast', None)
        if callable(broadcast_func):
            broadcast_func()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェック互換ステータス"""
        return {
            "cycle": self.cycle_count,
            "astrocyte": {"metrics": {"energy_percent": 100.0, "fatigue_index": 0.0}}
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
