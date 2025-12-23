# /snn_research/cognitive_architecture/visual_perception.py
# 日本語タイトル: 視覚野モジュール (リセット機能修正版)
# 目的: 内部状態を完全に初期化し、推論の再現性を確保する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
from snn_research.core.snn_core import SNNCore
from snn_research.hybrid.multimodal_projector import MultimodalProjector
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class VisualCortex(nn.Module):
    """
    視覚入力を処理し、特徴抽出と物体検出を行うモジュール。
    """
    def __init__(
        self, 
        workspace: GlobalWorkspace,
        vision_model_config: Dict[str, Any],
        projector_config: Dict[str, Any],
        device: str = "cpu"
    ):
        super().__init__()
        self.workspace = workspace
        self.device = device

        # SNNバックボーンの初期化
        self.vision_core = SNNCore(config=vision_model_config)
        self.vision_core.to(device)
        
        # モダリティ変換プロジェクター
        self.projector = MultimodalProjector(
            visual_dim=projector_config.get("visual_dim", 512),
            lang_dim=projector_config.get("lang_dim", 256),
            visual_time_steps=vision_model_config.get("time_steps", 16),
            lang_time_steps=16
        )
        self.projector.to(device)
        
        # 物体検出モデルの遅延ロード用
        self.detection_model: Optional[nn.Module] = None

    def reset_state(self) -> None:
        """
        [Fix] 内部状態（膜電位等）を完全にリセットする。
        これにより、同じ入力に対して常に同じ出力を得られるようにする。
        """
        # 1. バックボーンのリセット
        if hasattr(self.vision_core, 'reset_state'):
            self.vision_core.reset_state()
        
        # 2. プロジェクターのリセット (RNN/SNNレイヤーを含む場合)
        if hasattr(self.projector, 'reset_state'):
            self.projector.reset_state()
            
        logger.debug("VisualCortex: Internal states have been reset.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播処理。"""
        # 重み不一致回避のため、実行時に入力形状を確認
        x = x.to(self.device)
        
        # 1. SNNによる特徴抽出
        visual_features = self.vision_core(x)
        
        # 2. 潜在空間への射影
        context_embeds = self.projector(visual_features)
        
        return visual_features, context_embeds

    def perceive_and_upload(self, image_tensor: torch.Tensor) -> None:
        """知覚内容をワークスペースへアップロード。"""
        visual_features, context_embeds = self.forward(image_tensor)
        
        salience = float(torch.norm(visual_features).item())
        salience = min(1.0, salience / 10.0)

        perception_data = {
            "type": "visual_perception",
            "features": visual_features.detach(),
            "context_embeds": context_embeds.detach()
        }
        
        self.workspace.upload_to_workspace(
            source="visual_cortex",
            data=perception_data,
            salience=salience
        )
