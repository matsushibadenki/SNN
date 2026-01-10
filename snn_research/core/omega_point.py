# ファイルパス: snn_research/core/omega_point.py
# 日本語タイトル: Omega Point Controller (Import Fix)
# 修正内容: randomモジュールのインポートを追加し、エラーを解消。

import asyncio
import logging
import time
import random  # Added missing import
from typing import cast, Any, Dict

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.evolution.recursive_improver import RecursiveImprover
from snn_research.safety.ethical_guardrail import EthicalGuardrail

logger = logging.getLogger(__name__)


class OmegaPointSystem:
    """
    オメガ・ポイント・システム。
    人工脳の進化と統合を管理する最上位コントローラ。
    """

    def __init__(self, base_brain: ArtificialBrain, os_kernel: NeuromorphicOS):
        self.brain = base_brain
        self.os = os_kernel

        # コンフィグ取得の安全化
        base_config: Dict[str, Any] = {}
        if hasattr(self.brain, "config"):
            base_config = cast(Dict[str, Any], self.brain.config)
        elif hasattr(self.brain, "model_config"):
            base_config = cast(Dict[str, Any], self.brain.model_config)

        self.improver = RecursiveImprover(base_config=base_config)
        self.system_guardrail = EthicalGuardrail(safety_threshold=0.95)
        self.iteration_count = 0
        self.is_active = False

        logger.info("🌌 Omega Point System initialized. Awaiting ignition.")

    async def ignite_singularity(self, target_metric_score: float = 100.0):
        """シンギュラリティ・ループの開始"""
        logger.info("🚀 IGNITION: Initiating Recursive Self-Improvement Loop...")
        self.is_active = True
        self.os.is_running = True

        start_time = time.time()

        try:
            while self.is_active:
                self.iteration_count += 1
                brain_instance = cast(ArtificialBrain, self.brain)

                # 1. Status Check
                brain_status = brain_instance.get_brain_status()
                # 安全な辞書アクセス
                astrocyte = brain_status.get("astrocyte", {})
                if isinstance(astrocyte, dict):
                    metrics = astrocyte.get("metrics", {})
                    if isinstance(metrics, dict):
                        fatigue = metrics.get("fatigue_index", 0)
                        if isinstance(fatigue, (int, float)) and fatigue > 90:
                            logger.warning(
                                "⚠️ High fatigue. Forcing sleep cycle...")
                            await self.os.sys_sleep()
                            continue

                # 2. Self-Improvement Cycle
                print(
                    f"   [Cycle {self.iteration_count}] Spawning candidates...", end="", flush=True)
                candidates = self.improver.spawn_generation(pop_size=2)
                print(" Done. Evaluating...", end="", flush=True)

                def evaluate_brain(candidate: Any) -> float:
                    """
                    候補モデルの評価関数。
                    ここではPerceptionテスト、またはランダムなスコアを使用。
                    """
                    # Mypyエラー修正: randomを使用
                    return random.uniform(0.0, 100.0)

                best_candidate, score = self.improver.evaluate_and_select(
                    candidates, evaluate_brain)
                print(f" Done. Best Score: {score:.2f}")

                # 3. Upgrade Logic (Simulation)
                if score > target_metric_score:
                    logger.info(
                        "🏆 Target Performance Reached! Singularity Achieved.")
                    self.is_active = False

                if self.iteration_count >= 5:
                    logger.info("🛑 Simulation Limit Reached (Demo Mode).")
                    self.is_active = False

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("User interrupted Singularity loop.")
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🏁 Simulation finished. Time: {elapsed:.2f}s")
