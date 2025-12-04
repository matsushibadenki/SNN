# ファイルパス: snn_research/cognitive_architecture/visual_cortex.py
# (修正: mypy エラー解消)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast
import logging
import random
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights # type: ignore
from snn_research.core.snn_core import SNNCore
from snn_research.hybrid.multimodal_projector import MultimodalProjector
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class VisualCortex(nn.Module):
    vision_core: nn.Module 

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

        self.vision_core = SNNCore(config=vision_model_config)
        self.vision_core.to(device)
        
        self.projector = MultimodalProjector(
            visual_dim=projector_config.get("visual_dim", 512),
            lang_dim=projector_config.get("lang_dim", 256),
            visual_time_steps=vision_model_config.get("time_steps", 16),
            lang_time_steps=16,
            use_bitnet=projector_config.get("use_bitnet", False)
        )
        self.projector.to(device)
        
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detection_model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.8)
        self.detection_model.to(device)
        self.detection_model.eval()
        self.detection_classes = weights.meta["categories"]

        print("👁️ 視覚野 (Visual Cortex) が初期化されました (Faster R-CNN搭載)。")

    def process_image(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.vision_core.eval()
        self.projector.eval()
        image_tensor = image_tensor.to(self.device)
        
        outputs = self.vision_core(image_tensor)
        
        if isinstance(outputs, tuple):
            visual_features = outputs[0]
        else:
            visual_features = outputs

        context_embeds = self.projector(visual_features)
        return visual_features, context_embeds

    def detect_objects(self, image_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        self.detection_model.eval()
        image_tensor = image_tensor.to(self.device)
        
        detected_objects = []
        with torch.no_grad():
            predictions = self.detection_model(image_tensor)
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5: 
                    label_name = self.detection_classes[label]
                    detected_objects.append({
                        "label": label_name,
                        "bbox": box.tolist(), 
                        "confidence": float(score)
                    })
                    
        return detected_objects

    def perceive_and_upload(self, image_tensor: torch.Tensor) -> None:
        print("👁️ 視覚野: 画像を処理中...")
        visual_features, context_embeds = self.process_image(image_tensor)
        
        detected_objects = self.detect_objects(image_tensor)
        if detected_objects:
            obj_summaries = [f"{obj['label']} at {['{:.2f}'.format(v) for v in obj['bbox']]}" for obj in detected_objects]
            print(f"  - 物体を検出: {', '.join(obj_summaries)}")
        
        salience = torch.norm(visual_features).item()
        salience = min(1.0, salience / 10.0)

        perception_data = {
            "type": "visual_perception",
            "features": visual_features.detach().cpu(),
            "context_embeds": context_embeds.detach(),
            "detected_objects": detected_objects, 
            "description": f"Visual input processed. Detected: {[o['label'] for o in detected_objects]}"
        }
        
        self.workspace.upload_to_workspace(
            source="visual_cortex",
            data=perception_data,
            salience=salience
        )
        print(f"  - 視覚コンテキストと空間情報を生成し、Workspaceにアップロードしました。")