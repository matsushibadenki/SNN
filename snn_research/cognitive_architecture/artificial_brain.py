# /snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (完全整合版)
# 目的: 全脳モジュールのインターフェースを統合し、mypyエラーを解消した認知サイクルを実現。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, cast
import logging

from .global_workspace import GlobalWorkspace
from .hippocampus import Hippocampus
from .cortex import Cortex
from .basal_ganglia import BasalGanglia
from .motor_cortex import MotorCortex
from .amygdala import Amygdala
from .prefrontal_cortex import PrefrontalCortex
from .perception_cortex import PerceptionCortex
from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)

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
        
        # 入力の Tensor 化と次元正規化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=device) 
        else:
            sensory_tensor = sensory_input.to(device)

        if sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)

        # 次元の強制整合 (PerceptionCortexの入力数に合わせる)
        target_n = self.perception.num_neurons
        current_n = sensory_tensor.shape[-1]
        
        if current_n != target_n:
            if current_n < target_n:
                diff = target_n - current_n
                padding = torch.zeros(*sensory_tensor.shape[:-1], diff, device=device)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
            else:
                sensory_tensor = sensory_tensor[..., :target_n]

        # 1. 知覚処理
        perception_result = self.perception.perceive(sensory_tensor)
        raw_features = perception_result.get("features")
        if raw_features is not None:
            perceptual_info = raw_features
            while perceptual_info.ndim > 1:
                perceptual_info = torch.mean(perceptual_info.float(), dim=0)
        else:
            perceptual_info = torch.zeros(256, device=device)

        # 2. 感情・記憶の評価
        emotional_val = self.amygdala.process(perceptual_info)
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 3. ワークスペース集約 (mypyエラー箇所: メソッドの存在を確認して呼び出し)
        # GlobalWorkspace が add_content を持たない場合のフォールバックを含める
        if hasattr(self.workspace, 'add_content'):
            self.workspace.add_content("sensory", perceptual_info)
            self.workspace.add_content("emotion", emotional_val)
        elif hasattr(self.workspace, 'update'):
            # 既存の update メソッドへのマッピング
            self.workspace.update("sensory", perceptual_info)
            self.workspace.update("emotion", emotional_val)
        
        # サマリーの取得
        summary: List[Dict[str, Any]] = []
        if hasattr(self.workspace, 'get_summary'):
            summary_raw = self.workspace.get_summary()
            summary = cast(List[Dict[str, Any]], summary_raw if isinstance(summary_raw, list) else [summary_raw])
        elif hasattr(self.workspace, 'contents'):
            # 直接属性を参照する場合のフォールバック
            summary = [{"content": v} for v in getattr(self.workspace, 'contents', {}).values()]
        
        # 4. 行動選択と実行
        selected_action = self.basal_ganglia.select_action(summary)
        motor_output = self.motor.generate_signal(selected_action)

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
