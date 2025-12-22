# ディレクトリパス: snn_research/cognitive_architecture/
# ファイルパス: artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (v20.5 完全統合版)
# 目的: 認知サイクル、睡眠サイクル、および高度な診断メトリクスの提供。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from torchvision import transforms

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ。
    [修正] sleep_cycle の実装と、診断レポート (get_status) のメトリクス構造を修正。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 主要コンポーネント・バインディング
        self.astrocyte = kwargs.get('astrocyte_network')
        self.visual = kwargs.get('visual_cortex')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.workspace = kwargs.get('global_workspace')
        
        # 領野コンポーネント
        self.pfc = kwargs.get('prefrontal_cortex')
        self.hippocampus = kwargs.get('hippocampus')

        self.state = "AWAKE"
        self.cycle_count = 0

        # 画像変換プロセッサ (空間認識デモ等で使用)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info("ArtificialBrain Kernel v20.1 initialized.")

    def get_brain_status(self) -> Dict[str, Any]:
        """デモスクリプト互換用エイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """
        統合診断レポート。
        [修正] energy_percent を含め、デモスクリプトの KeyError を防止。
        """
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
                "energy_percent": (energy / 1000.0) * 100.0, # 追加
                "fatigue": fatigue,
                "metrics": {
                    "energy_level": energy,
                    "energy_percent": (energy / 1000.0) * 100.0 # 二重に保持して安全性を確保
                }
            }
        }

    def sleep_cycle(self) -> None:
        """
        🌙 睡眠サイクル。
        記憶の固定化（Consolidation）とアストロサイトのリセット。
        """
        if self.state == "SLEEPING":
            return

        logger.info(f"🛌 Cycle {self.cycle_count}: Entering Sleep state...")
        self.state = "SLEEPING"
        
        try:
            # 記憶固定化
            if self.sleep_manager and hasattr(self.sleep_manager, 'perform_sleep_cycle'):
                self.sleep_manager.perform_sleep_cycle(duration_cycles=5)
            elif self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
                self.sleep_manager.consolidate_memory()
                
            # エネルギー回復
            if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
                self.astrocyte.replenish_energy(1000.0)
                
        finally:
            self.state = "AWAKE"
            logger.info("☀️ Brain restored to AWAKE state.")

    def calculate_uncertainty(self, result: Any) -> float:
        """エントロピーに基づく不確実性推定。"""
        if isinstance(result, (tuple, list)) and len(result) > 0:
            val = result[0]
            if isinstance(val, list): val = val[0]
        else:
            val = result

        if not isinstance(val, torch.Tensor):
            return 0.5
            
        with torch.no_grad():
            logits = val.float()
            if logits.dim() == 3: logits = logits.mean(dim=1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            return min(1.0, float(entropy / 2.3))

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """認知サイクルの実行。"""
        self.cycle_count += 1
        uncertainty = 0.0

        if self.visual is not None and hasattr(self.visual, 'forward'):
            try:
                res = self.visual(raw_input)
                uncertainty = self.calculate_uncertainty(res)
            except Exception as e:
                logger.error(f"Perception failed: {e}")

        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            self.astrocyte.accumulate_fatigue(0.2)

        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": self.get_status()["astrocyte"]
        }
