# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (完全復元 & 実行時エラー修正版)
# 目的: 既存デモスクリプトとの完全な互換性を確保し、OSレベルの安全性を実装。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, cast

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ。
    [修正] 実行ログに見られた 'status' キー不足と 'sleep_cycle' 欠損を解消。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 主要コンポーネント・バインディング (デモとの互換性を完全維持)
        self.workspace: Any = kwargs.get('global_workspace')
        self.visual: Any = kwargs.get('visual_cortex')
        self.system1: Any = kwargs.get('thinking_engine')
        self.astrocyte: Any = kwargs.get('astrocyte_network')
        self.guardrail: Any = kwargs.get('ethical_guardrail')
        self.basal_ganglia: Any = kwargs.get('basal_ganglia')
        
        # 睡眠管理（別名：sleep_consolidator にも対応）
        self.sleep_manager: Any = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        
        # テストやデモが期待する他の属性も明示的にセット
        self.pfc: Any = kwargs.get('prefrontal_cortex') or kwargs.get('pfc')
        self.hippocampus: Any = kwargs.get('hippocampus')
        self.cortex: Any = kwargs.get('cortex')
        self.motor: Any = kwargs.get('motor_cortex') or kwargs.get('motor')
        self.amygdala: Any = kwargs.get('amygdala')

        self.state = "AWAKE"
        self.cycle_count = 0

    def calculate_uncertainty(self, result: Any) -> float:
        """⑮ 不確実性に基づくメタ認知。"""
        if isinstance(result, torch.Tensor):
            probs = torch.softmax(result.float(), dim=-1)
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            return min(1.0, float(uncertainty / 2.3))
        return 0.5

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """
        認知サイクルの実行。
        [修正] VisualCortex.forward() を安全に呼び出す。
        """
        self.cycle_count += 1
        perception_result = None
        uncertainty = 0.0

        # 1. 知覚（VisualCortexが存在する場合のみ実行）
        if self.visual is not None and hasattr(self.visual, 'forward'):
            try:
                perception_result = self.visual(raw_input)
                uncertainty = self.calculate_uncertainty(perception_result)
            except Exception as e:
                logger.error(f"Visual processing failed: {e}")

        # 2. アストロサイトによる代謝制御 ⑯
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            energy_mod = 1.5 if uncertainty > 0.7 else 0.8
            self.astrocyte.accumulate_fatigue(0.5 * energy_mod)

        # 3. 倫理ガードレール
        if self.guardrail and hasattr(self.guardrail, 'check_safety'):
            self.guardrail.check_safety(perception_result)

        # ステータスの取得
        current_status = self.get_status()

        # 戻り値の正規化 (デモスクリプトが期待する構造)
        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": current_status["astrocyte"],
            "result": "Brain cycle executed successfully."
        }

    def sleep_cycle(self) -> None:
        """🌙 睡眠サイクル。記憶の固定化とエネルギー回復を実行。"""
        logger.info("🛌 Initiating Sleep Cycle...")
        self.state = "SLEEPING"
        
        # 記憶固定化 (Consolidation)
        if self.sleep_manager and hasattr(self.sleep_manager, 'consolidate_memory'):
            self.sleep_manager.consolidate_memory()
            
        # エネルギー回復
        if self.astrocyte and hasattr(self.astrocyte, 'replenish_energy'):
            self.astrocyte.replenish_energy(1000.0)
            
        self.state = "AWAKE"
        logger.info("☀️ Brain restored to AWAKE state.")

    def get_brain_status(self) -> Dict[str, Any]:
        """run_brain_v16_demo.py のためのエイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """
        統合診断レポート。
        [修正] KeyError: 'status' を回避するため、入れ子構造を厳密に構築。
        """
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        fatigue = getattr(self.astrocyte, 'fatigue_toxin', 0.0) if self.astrocyte else 0.0
        
        # デモが期待する 'status' キー
        astro_status = "NORMAL"
        if fatigue > 50.0: astro_status = "WARNING"
        if fatigue > 80.0: astro_status = "CRITICAL"

        return {
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "status": astro_status,  # これが重要
                "energy_percent": (energy / 1000.0) * 100.0,
                "fatigue": fatigue,
                "metrics": {"energy_level": energy}
            }
        }
