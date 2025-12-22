# ディレクトリパス: snn_research/cognitive_architecture/
# ファイルパス: artificial_brain.py
# 日本語タイトル: 人工脳カーネル (統合機能維持版)
# 目的: 全脳領野の統制、疲労代謝管理、睡眠による記憶固定化の実行。

import logging
import torch
from typing import Dict, Any, Optional
from torchvision import transforms

logger = logging.getLogger(__name__)

class ArtificialBrain:
    def __init__(self, **kwargs: Any):
        # [2025-12-16] 既存の全領野バインディングを削除しない
        self.config = kwargs.get('config', {})
        self.astrocyte = kwargs.get('astrocyte_network')
        self.visual = kwargs.get('visual_cortex')
        self.sleep_manager = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.workspace = kwargs.get('global_workspace')
        
        self.state = "AWAKE"
        self.cycle_count = 0
        
        # 空間認識・知覚デモ用トランスフォーム
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        logger.info("ArtificialBrain Kernel v20.5: All cognitive fields mapped.")

    def get_brain_status(self) -> Dict[str, Any]:
        """[機能維持] デモスクリプトからの直接参照用。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """[ロジック確認] アストロサイトの状態を階層構造で返却。"""
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        
        # energy_percent キーはデモ完走に必須
        energy_pct = (energy / 1000.0) * 100.0
        
        return {
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": "CRITICAL" if fatigue > 80 else "NORMAL",
                "energy_percent": energy_pct,
                "fatigue": fatigue,
                "metrics": {"energy_level": energy, "energy_percent": energy_pct}
            }
        }

    def sleep_cycle(self) -> None:
        """[機能維持] 睡眠による代謝リセットと記憶固定化を連動。"""
        if self.state == "SLEEPING": return
        self.state = "SLEEPING"
        logger.info("🛌 Sleep state: Starting memory consolidation...")
        
        try:
            # 記憶固定化 (Consolidatorの呼び出し)
            if self.sleep_manager:
                if hasattr(self.sleep_manager, 'perform_sleep_cycle'):
                    self.sleep_manager.perform_sleep_cycle(duration_cycles=5)
                elif hasattr(self.sleep_manager, 'consolidate_memory'):
                    self.sleep_manager.consolidate_memory()
            
            # 代謝リセット (Astrocyteの呼び出し)
            if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
                self.astrocyte.replenish_energy(1000.0)
        finally:
            self.state = "AWAKE"
            logger.info("☀️ Awake: Brain restored.")

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        self.cycle_count += 1
        # [2025-12-03] 既存の機能（知覚→疲労蓄積）を維持
        if self.visual:
            self.visual(raw_input)
        
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            self.astrocyte.accumulate_fatigue(0.2)
            
        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "astrocyte": self.get_status()["astrocyte"]
        }
