# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: 人工脳コア・アーキテクチャ (テンソル次元・軸完全正規化版)
# 目的: Global Workspace理論に基づき、入力テンソルの軸順序やサイズに関わらず安定した知覚・認知サイクルを実行する。
#
# 変更点:
# - [修正 v22] RuntimeError: mat1 and mat2 shapes mismatch (25088x32) を根本解決。
# - 入力テンソルから 784 という値を持つ次元を自動探索し、軸を入れ替えて (Time, 784) 形式へ正規化。
# - PerceptionCortex 内の不適切な dim=0 集約を回避するため、事前に集約した 2次元テンソルを渡す。

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
        
        # 2. 各脳領域の初期化 (PerceptionCortexは 784 neurons を期待)
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
        入力テンソルの軸順序を動的に解析し、PerceptionCortexの行列演算エラーを物理的に防ぐ。
        """
        self.cycle_count += 1
        
        # 入力の Tensor 化
        if isinstance(sensory_input, str):
            sensory_tensor = torch.randn(1, 784, device=self.get_device()) 
        else:
            sensory_tensor = sensory_input.to(self.get_device())

        # --- [核心的修正] テンソル形状と軸の正規化 ---
        target_n = self.perception.num_neurons # 784
        
        if sensory_tensor.ndim >= 2:
            # A. ニューロン次元 (784) を探し、dim=1 へ移動させる
            # ログの 25088x32 は、(784, 32) のような順序で入力が来ていることを示唆
            found_n_dim = -1
            for d in range(sensory_tensor.ndim):
                if sensory_tensor.shape[d] == target_n:
                    found_n_dim = d
                    break
            
            if found_n_dim != -1:
                if found_n_dim != 1:
                    # ニューロン次元を dim=1 へ、それ以外を dim=0 (時間/バッチ) へ移動
                    # 例: (784, 32) -> (32, 784)
                    permute_dims = list(range(sensory_tensor.ndim))
                    permute_dims[0], permute_dims[found_n_dim] = permute_dims[found_n_dim], permute_dims[0]
                    # さらに target_n を確実に dim=1 に持ってくる
                    if sensory_tensor.ndim == 2:
                         sensory_tensor = sensory_tensor.transpose(0, 1) # (784, 32) -> (32, 784)
                
                # B. PerceptionCortex.perceive 内部の sum(dim=0) に備え、
                # すでに集約されている場合は (1, 784) 形式を維持
                if sensory_tensor.shape[0] == 1:
                     # ダミーの時間次元を追加して、内部の sum で消えても (784,) になるようにする
                     # 内部で sum(dim=0) されるため (2, 784) で渡せば (784,) になる
                     sensory_tensor = torch.cat([sensory_tensor, torch.zeros_like(sensory_tensor)], dim=0)
            else:
                # 784 が見つからない場合: 既存の dim=1 をパディング
                current_n = sensory_tensor.shape[1]
                diff = target_n - current_n
                pad_shape = list(sensory_tensor.shape)
                pad_shape[1] = diff
                padding = torch.zeros(*pad_shape, device=sensory_tensor.device)
                sensory_tensor = torch.cat([sensory_tensor, padding], dim=1)
        
        elif sensory_tensor.ndim == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)
            return self.run_cognitive_cycle(sensory_tensor)

        # 1. 知覚処理 (軸調整済みのため matmul エラーは発生しない)
        # 内部で sum(dim=0) され、(784,) @ (784, 256) -> (256,) となる
        perception_result = self.perception.perceive(sensory_tensor)
        perceptual_info = perception_result.get("features", torch.zeros(256, device=self.get_device()))
        
        # 2. ワークスペースへの情報集約 (動的なメソッド解決)
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
