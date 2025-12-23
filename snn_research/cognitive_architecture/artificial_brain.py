# /snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (高精度統合版)
# 目的: 全脳モジュールのインターフェースを高度に統合し、動的なリソース配分と認知サイクルを実現する。

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
    GWT(Global Workspace Theory)に基づき、情報の放送と意思決定を制御する。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.config = config or {}
        
        # 1. 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 2. 感覚・認知処理系の初期化 (入力次元 784: MNIST等に準拠)
        self.perception = PerceptionCortex(num_neurons=784, feature_dim=256)
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.cortex = Cortex()
        
        # 3. 実行・制御系の初期化
        self.basal_ganglia = BasalGanglia(workspace=self.workspace)
        self.prefrontal_cortex = PrefrontalCortex(
            workspace=self.workspace, 
            motivation_system=self.motivation_system
        )
        self.motor = MotorCortex()

        self.cycle_count = 0
        self.energy_budget = self.config.get("energy_budget", 100.0)

    def run_cognitive_cycle(self, sensory_input: Union[torch.Tensor, str]) -> Dict[str, Any]:
        """
        1ステップの認知サイクルを実行する（知覚 -> 思考 -> 行動）。
        """
        self.cycle_count += 1
        device = self.get_device()
        
        # 入力の Tensor 化と次元整合性の確保
        if isinstance(sensory_input, str):
            # 文字列入力の場合はランダムな潜在表現から生成（デモ用）
            sensory_tensor = torch.randn(1, 784, device=device) 
        else:
            sensory_tensor = sensory_input.to(device)

        if sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)

        # 次元合わせ (784 neurons)
        target_n = self.perception.num_neurons
        current_n = sensory_tensor.shape[-1]
        
        if current_n != target_n:
            if current_n < target_n:
                diff = target_n - current_n
                padding = torch.zeros(*sensory_tensor.shape[:-1], diff, device=device)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=-1)
            else:
                sensory_tensor = sensory_tensor[..., :target_n]

        # --- 認知ステップ ---

        # 1. 知覚 (Perception): 外部刺激から特徴を抽出
        perception_result = self.perception.perceive(sensory_tensor)
        raw_features = perception_result.get("features")
        
        if raw_features is not None:
            # 高次元出力を256次元の特徴ベクトルに集約
            perceptual_info = raw_features
            while perceptual_info.ndim > 1:
                perceptual_info = torch.mean(perceptual_info.float(), dim=0)
        else:
            perceptual_info = torch.zeros(256, device=device)

        # 2. 感情評価 (Amygdala): 刺激の情動的価値を計算
        emotional_val = self.amygdala.process(perceptual_info)
        
        # 3. 記憶検索 (Cortex & Hippocampus): 過去の知識を呼び出す
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. ワークスペース集約 (Broadcasting準備)
        # 感覚情報と情動情報をワークスペースに統合
        self.workspace.add_content("sensory", perceptual_info)
        self.workspace.add_content("emotion", emotional_val)
        
        summary = self.workspace.get_summary()
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        
        # 5. 意思決定 (Basal Ganglia): 競合する行動候補から一つを選択
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 6. 運動出力 (Motor Cortex): 行動を具体的な信号に変換
        motor_output = self.motor.generate_signal(selected_action)

        # 7. ブロードキャスト: 選択された情報を全モジュールへ同期
        if hasattr(self.workspace, 'broadcast'):
            self.workspace.broadcast()
        
        return {
            "cycle": self.cycle_count,
            "action": str(selected_action),
            "motor_output": motor_output,
            "emotional_state": emotional_val.tolist() if isinstance(emotional_val, torch.Tensor) else emotional_val,
            "broadcasted": True
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """ヘルスチェックおよびモニタリング用ステータス取得"""
        return {
            "cycle": self.cycle_count,
            "energy_level": self.energy_budget,
            "astrocyte": {
                "metrics": {
                    "energy_percent": self.energy_budget, 
                    "fatigue_index": max(0.0, 100.0 - self.energy_budget)
                }
            },
            "workspace_occupancy": len(self.workspace.get_summary()) if hasattr(self.workspace, 'get_summary') else 0
        }

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
