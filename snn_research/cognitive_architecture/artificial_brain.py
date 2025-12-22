# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (属性定義修正版)
# 目的: 実行時の AttributeError 防止とデモ用属性の追加。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from torchvision import transforms

logger = logging.getLogger(__name__)

class ArtificialBrain:
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 属性の明示的な初期化 (None を許容)
        self.astrocyte = kwargs.get('astrocyte_network')
        self.visual = kwargs.get('visual_cortex')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.guardrail = kwargs.get('ethical_guardrail')
        
        self.state = "AWAKE"
        self.cycle_count = 0

        # デモ用の画像変換プロセッサ
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("ArtificialBrain Kernel v20.1 initialized.")

    def calculate_uncertainty(self, result: Any) -> float:
        """不確実性の計算"""
        if isinstance(result, (list, tuple)): result = result[0][0] # 最初のレイヤーの最初のステップ
        if not isinstance(result, torch.Tensor): return 0.5
        probs = torch.softmax(result.float().mean(dim=1), dim=-1) # Time平均
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        return min(1.0, float(entropy / 2.3))

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        self.cycle_count += 1
        uncertainty = 0.0
        
        if self.visual is not None:
            res = self.visual(raw_input)
            uncertainty = self.calculate_uncertainty(res)

        # 属性チェック付きで実行
        if hasattr(self, 'astrocyte') and self.astrocyte:
            if hasattr(self.astrocyte, 'accumulate_fatigue'):
                self.astrocyte.accumulate_fatigue(0.1)

        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": self.get_status()["astrocyte"]
        }

    def get_status(self) -> Dict[str, Any]:
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        return {
            "state": self.state,
            "astrocyte": {"status": "NORMAL", "fatigue": fatigue}
        }
