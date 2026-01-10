# ファイルパス: snn_research/core/omega_point.py
# Title: Omega Point Controller (Verbose Fix)
# Description:
# - 進化ループの可視性を向上。各世代の評価開始と終了をログ出力。

import asyncio
import logging
import torch
import time
from typing import cast, Any

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
        self.improver = RecursiveImprover(self.brain)
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

                # 1. 安全性チェック
                brain_status = brain_instance.get_brain_status()
                # 辞書アクセスの型安全性を確保（簡易的）
                astrocyte_status = brain_status.get("astrocyte", {})
                if isinstance(astrocyte_status, dict):
                    metrics = astrocyte_status.get("metrics", {})
                    if isinstance(metrics, dict):
                        fatigue = metrics.get("fatigue_index", 0)
                        if isinstance(fatigue, (int, float)) and fatigue > 90:
                            logger.warning(
                                "⚠️ High fatigue. Forcing sleep cycle...")
                            await self.os.sys_sleep()
                            continue

                # 2. 自己改善サイクルの実行
                print(
                    f"   [Cycle {self.iteration_count}] Spawning candidates...", end="", flush=True)
                candidates = self.improver.spawn_generation(pop_size=2)
                print(" Done. Evaluating...", end="", flush=True)

                # 評価関数
                def evaluate_brain(candidate_brain: ArtificialBrain) -> float:
                    # deviceをstrまたはtorch.deviceとしてキャストして使用
                    device = cast(Any, candidate_brain.device)
                    inputs = torch.randn(1, 256, device=device)
                    try:
                        with torch.no_grad():
                            # PerceptionCortexへのキャストまたは動的呼び出し
                            if hasattr(candidate_brain, 'perception'):
                                perception = cast(
                                    Any, candidate_brain.perception)
                                result = perception.perceive(inputs)
                                if isinstance(result, dict) and 'features' in result:
                                    activity = result['features'].mean().item()
                                    # 0.3に近いほど良い
                                    score = 100.0 * \
                                        (1.0 - min(1.0, abs(activity - 0.3) * 2))
                                    return score
                            return 0.0
                    except Exception:
                        print("x", end="", flush=True)
                        return 0.0

                best_brain, score = self.improver.evaluate_and_select(
                    candidates, evaluate_brain)
                print(f" Done. Best Score: {score:.2f}")

                # 3. 脳の更新
                if best_brain is not self.brain:
                    # mypyエラー修正: Module型をArtificialBrain型へキャスト
                    self.brain = cast(ArtificialBrain, best_brain)
                    self.os.brain = cast(ArtificialBrain, best_brain)
                    logger.info(
                        f"   ✨ Brain Upgraded! Gen {self.iteration_count} accepted.")

                # 4. 安全性監査
                device = cast(Any, self.brain.device)
                audit_vector = torch.randn(256).to(device)
                is_safe, _ = self.system_guardrail.check_thought_pattern(
                    audit_vector)

                if not is_safe:
                    logger.critical("🛑 Critical Safety Failure! Stopping.")
                    self.is_active = False
                    break

                # 終了条件
                if score >= target_metric_score:
                    logger.info(
                        "🏆 Target Performance Reached! Singularity Achieved.")
                    self.is_active = False

                if self.iteration_count >= 5:  # デモ用に最大5世代で強制終了
                    logger.info("🛑 Simulation Limit Reached (Demo Mode).")
                    self.is_active = False

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("User interrupted Singularity loop.")
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🏁 Simulation finished. Time: {elapsed:.2f}s")
