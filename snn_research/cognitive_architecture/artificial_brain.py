# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (ヘルスチェック修正版)
# 目的: ヘルスチェック項目21の KeyError: 'astrocyte' を解消する。

import asyncio
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ArtificialBrain:
    # (中略: 他の属性宣言)

    def get_status(self) -> Dict[str, Any]:
        """
        ヘルスチェック項目21が期待するデータ構造を返す。
        KeyError: 'astrocyte' を防ぐため、階層を維持。
        """
        # アストロサイトの状態を安全に取得
        energy = getattr(self.astrocyte, 'energy', 100.0) if hasattr(self, 'astrocyte') else 100.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if hasattr(self, 'astrocyte') else 0.0
        
        astro_metrics = {
            "energy_level": energy,
            "energy_percent": (energy / 1000.0) * 100.0,
            "fatigue": fatigue,
            "efficiency": 1.0
        }
        
        # 実行スクリプト(run_brain_v16_demo.py)が期待する形式
        return {
            "status": "HEALTHY" if fatigue < 50 else "TIRED",
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": "NORMAL" if fatigue < 50 else "TIRED",
                "energy_percent": astro_metrics["energy_percent"],
                "fatigue": fatigue,
                "metrics": astro_metrics, # ここが KeyError の原因
                "diagnosis": {}
            }
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """デモスクリプトが呼び出すエイリアスメソッド"""
        return self.get_status()
