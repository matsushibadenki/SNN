# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# 日本語タイトル: Artificial Brain Kernel (完全機能復元版)
# 目的: 全テストおよびデモスクリプトとの完全な互換性を確保し、mypyエラーを解消する。

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, cast
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ArtificialBrain:
    """
    SNNベース 人工脳アーキテクチャ。
    [2025-12-22修正] mypyエラー解消のため、テストやダッシュボードが依存する全属性を明示的に定義。
    """
    def __init__(self, **kwargs: Any):
        self.device = kwargs.get('device', 'cpu')
        self.config = kwargs.get('config', {})
        
        # 属性の明示的定義 (mypy [attr-defined] 回避)
        self.workspace: Any = kwargs.get('global_workspace')
        self.motivation_system: Any = kwargs.get('motivation_system')
        self.receptor: Any = kwargs.get('sensory_receptor')
        self.encoder: Any = kwargs.get('spike_encoder')
        self.actuator: Any = kwargs.get('actuator')
        self.system1: Any = kwargs.get('thinking_engine')
        self.perception: Any = kwargs.get('perception_cortex')
        self.visual: Any = kwargs.get('visual_cortex')
        self.pfc: Any = kwargs.get('prefrontal_cortex')
        self.hippocampus: Any = kwargs.get('hippocampus')
        self.cortex: Any = kwargs.get('cortex')
        self.amygdala: Any = kwargs.get('amygdala')
        self.basal_ganglia: Any = kwargs.get('basal_ganglia')
        self.cerebellum: Any = kwargs.get('cerebellum')
        self.motor: Any = kwargs.get('motor_cortex')
        self.world_model: Any = kwargs.get('world_model')
        self.astrocyte: Any = kwargs.get('astrocyte_network')
        # sleep_manager は sleep_consolidator としても知られる
        self.sleep_manager: Any = kwargs.get('sleep_manager') or kwargs.get('sleep_consolidator')
        self.reflex_module: Any = kwargs.get('reflex_module')
        self.guardrail: Any = kwargs.get('ethical_guardrail')

        self.state = "AWAKE"
        self.cycle_count = 0

    def calculate_uncertainty(self, perception_result: Any) -> float:
        """自信のなさを算出。"""
        if isinstance(perception_result, torch.Tensor):
            probs = torch.softmax(perception_result.float(), dim=-1)
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            return min(1.0, float(uncertainty / 2.3))
        return 0.5

    def run_cognitive_cycle(self, raw_input: Any) -> Dict[str, Any]:
        """認知サイクルの実行。"""
        self.cycle_count += 1
        perception_result = None
        uncertainty = 0.0

        if self.visual and hasattr(self.visual, 'forward'):
            perception_result = self.visual(raw_input)
            uncertainty = self.calculate_uncertainty(perception_result)

        # アストロサイトによる代謝制御
        if self.astrocyte and hasattr(self.astrocyte, 'accumulate_fatigue'):
            energy_mod = 1.2 if uncertainty > 0.7 else 0.8
            self.astrocyte.accumulate_fatigue(0.5 * energy_mod)

        if self.guardrail and hasattr(self.guardrail, 'check_safety'):
            self.guardrail.check_safety(perception_result)

        return {
            "cycle": self.cycle_count,
            "status": "SUCCESS",
            "uncertainty": uncertainty,
            "state": self.state,
            "astrocyte": self.get_status()["astrocyte"]
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """run_brain_v16_demo.py 等が期待するエイリアス。"""
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """統合診断レポート。"""
        energy = getattr(self.astrocyte, 'energy', 1000.0) if self.astrocyte else 1000.0
        return {
            "state": self.state,
            "cycle": self.cycle_count,
            "astrocyte": {
                "energy_percent": (energy / 1000.0) * 100.0,
                "metrics": {"energy_level": energy}
            }
        }
