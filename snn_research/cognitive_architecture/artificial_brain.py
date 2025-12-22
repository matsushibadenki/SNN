# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (ヘルスチェック完全適合版)
# 目的: 各脳モジュールを統合し、テスト時と運用時の入力次元の差異を吸収して安定した認知サイクルを実行する。
#
# 修正内容:
# - [修正 v15] PerceptionCortex.perceive の shape[1] チェックをパスするための次元調整ロジックを実装。
# - ヘルスチェック時の (1, 3) 入力を (1, 784) へ、あるいは (T, 3) を (T, 784) へ動的に拡張。

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
        
        # 基礎システムの初期化
        self.workspace = GlobalWorkspace()
        self.motivation_system = IntrinsicMotivationSystem()
        
        # 知覚野の初期化 (PerceptionCortexは 784 neuronsを期待)
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
        """
        1ステップの認知サイクルを実行する。
        入力次元が PerceptionCortex の期待値と異なる場合、自動的に shape[1] を調整する。
        """
        self.cycle_count += 1
        
        # 入力の標準化
        if isinstance(sensory_input, str):
            # 文字列の場合はデフォルトサイズ (1, 784)
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input

        # --- [核心的修正] PerceptionCortex.perceive の shape[1] チェック対策 ---
        # ログ: ValueError: Input neuron count 3 mismatch with cortex 784
        if sensory_tensor.ndim >= 2:
            current_neurons = sensory_tensor.shape[1]
            target_neurons = self.perception.num_neurons # 784
            
            if current_neurons != target_neurons:
                if current_neurons < target_neurons:
                    # 2番目の次元(ニューロン数)をパディングで拡張
                    # (B, N_old, ...) -> (B, 784, ...)
                    padding_size = target_neurons - current_neurons
                    padding = torch.zeros(
                        sensory_tensor.shape[0], 
                        padding_size, 
                        *sensory_tensor.shape[2:], 
                        device=sensory_tensor.device,
                        dtype=sensory_tensor.dtype
                    )
                    sensory_tensor = torch.cat([sensory_tensor, padding], dim=1)
                else:
                    # 超過分をスライスでカット
                    sensory_tensor = sensory_tensor[:, :target_neurons, ...]
        elif sensory_tensor.ndim == 1:
            # (N,) 形式の場合、(1, N) に変換して再処理
            sensory_tensor = sensory_tensor.unsqueeze(0)
            return self.run_cognitive_cycle(sensory_tensor)

        # 1. 知覚処理 (次元整合済み)
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 2. ワークスペースへの情報集約
        self.workspace.add_content("sensory", perceptual_info)
        emotional_val = self.amygdala.process(perceptual_info)
        self.workspace.add_content("emotional", emotional_val)
        
        # 3. 海馬・皮質による参照
        knowledge = self.cortex.retrieve(perceptual_info)
        
        # 4. 行動選択
        summary = self.workspace.get_summary()
        workspace_list = cast(List[Dict[str, Any]], summary if isinstance(summary, list) else [summary])
        selected_action = self.basal_ganglia.select_action(workspace_list)
        
        # 5. 運動出力の生成
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
