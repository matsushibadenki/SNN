# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (完全互換修正版)
# 目的: get_brain_status の実装と初期化属性の網羅。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from torchvision import transforms

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ。
    [修正] get_brain_status を明示的に実装し、デモスクリプトとの互換性を確保。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 主要コンポーネント・バインディング
        self.astrocyte = kwargs.get('astrocyte_network')
        self.visual = kwargs.get('visual_cortex')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.workspace = kwargs.get('global_workspace')
        
        self.state = "AWAKE"
        self.cycle_count = 0

        # 画像変換プロセッサ (空間認識デモ等で使用)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("ArtificialBrain Kernel v20.1 initialized.")

    def get_brain_status(self) -> Dict[str, Any]:
        """
        [復元] run_sleep_cycle_demo.py 等から直接呼び出されるメソッド。
        """
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """統合診断レポート。"""
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        
        astro_status = "NORMAL"
        if fatigue > 50.0: astro_status = "WARNING"
        if fatigue > 80.0: astro_status = "CRITICAL"

        return {
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": astro_status,
                "energy_percent": (energy / 1000.0) * 100.0,
                "fatigue": fatigue,
                "metrics": {"energy_level": energy}
            }
        }

    def calculate_uncertainty(self, result: Any) -> float:
        """エントロピーに基づく不確実性推定。"""
        # 戻り値が (states, errors) のタプルの場合の考慮
        if isinstance(result, (tuple, list)) and len(result) > 0:
            val = result[0]
            if isinstance(val, list): val = val[0]
        else:
            val = result

        if not isinstance(val, torch.Tensor):
            return 0.5
            
        with torch.no_grad():
            logits = val.float()
            # Time次元がある場合は平均
            if logits.dim() == 3: logits = logits.mean(dim=1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            return min(1.0, float(entropy / 2.3))

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """知覚 -> 代謝 -> 報告のサイクル。"""
        self.cycle_count += 1
        uncertainty = 0.0

        if self.visual is not None and hasattr(self.visual, 'forward'):
            try:
                res = self.visual(raw_input)
                uncertainty = self.calculate_uncertainty(res)
            except Exception as e:
                logger.error(f"Perception error: {e}")

        # 代謝計算
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            self.astrocyte.accumulate_fatigue(0.2)

        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": self.get_status()["astrocyte"]
        }
