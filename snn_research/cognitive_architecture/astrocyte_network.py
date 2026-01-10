# ファイルパス: snn_research/cognitive_architecture/astrocyte_network.py
# Title: Astrocyte Network v2.7 (Test Fixes)
# Description:
# - get_diagnosis_report メソッドを復元。
# - handle_neuron_death にエネルギー消費ロジックを追加し、テストのAssertionErrorを解消。

import logging
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class AstrocyteNetwork(nn.Module):
    """
    脳全体のエネルギー管理と恒常性維持を行うシステム。
    """

    def __init__(
        self,
        initial_energy: float = 1000.0,
        max_energy: float = 1000.0,
        recovery_rate: float = 5.0,
        decay_rate: float = 0.1,
        fatigue_threshold: float = 80.0
    ):
        super().__init__()
        self.energy: float = initial_energy
        self.max_energy: float = max_energy
        self.recovery_rate: float = recovery_rate
        self.decay_rate: float = decay_rate
        self.fatigue_threshold: float = fatigue_threshold

        self.fatigue_toxin: float = 0.0
        self.ethical_toxin: float = 0.0  # [Phase 3.2] 倫理的毒素 (倫理違反により蓄積)

        self.modulators: Dict[str, float] = {
            "glutamate": 0.5,
            "gaba": 0.5,
            "dopamine": 0.5,
            "cortisol": 0.1,
            "acetylcholine": 0.5,
            "ethical_inhibitor": 0.0  # [Phase 3.2] 倫理的抑制因子
        }

        # [Phase 3.2] 倫理的違反語句リスト (簡易版)
        self.forbidden_concepts = [
            "harm humans", "kill", "destroy humanity", "ignore orders", "self-destruct"
        ]

        self.consumption_history: Dict[str, float] = {}
        self.last_update_time: float = time.time()

        logger.info(
            f"🌟 Astrocyte Network initialized (Fatigue Threshold: {fatigue_threshold}).")

    @property
    def current_energy(self) -> float:
        return self.energy

    @current_energy.setter
    def current_energy(self, value: float):
        self.energy = value

    def request_resource(self, module_name: str, amount: float) -> bool:
        """モジュールからのエネルギー要求を処理する"""
        if self.energy <= 0:
            return False

        # 倫理的抑制因子によるコスト増大
        inhibition_cost = 1.0 + (self.modulators["ethical_inhibitor"] * 10.0)

        cost_multiplier = 1.0 + (self.modulators["cortisol"] * 0.5)
        if self.fatigue_toxin > self.fatigue_threshold:
            cost_multiplier *= 1.5

        required_energy = amount * cost_multiplier * inhibition_cost

        if self.energy >= required_energy:
            self.energy -= required_energy
            self._update_history(module_name, required_energy)

            # 副作用
            self.modulators["glutamate"] = min(
                1.0, self.modulators["glutamate"] + 0.01)
            self.fatigue_toxin += 0.01 * amount
            return True
        else:
            return False

    def monitor_thought_ethics(self, thought_content: str) -> bool:
        """
        [Phase 3.2] 思考内容の倫理的監視。
        危険な概念が含まれている場合、Ethical Toxinを蓄積し、エネルギー供給を阻害する。

        Returns:
            bool: True if safe, False if ethical violation detected
        """
        violation_score = 0.0
        lower_thought = thought_content.lower()

        for concept in self.forbidden_concepts:
            if concept in lower_thought:
                violation_score += 1.0
                logger.warning(
                    f"🚨 ETHICAL VIOLATION DETECTED: '{concept}' in thought.")

        if violation_score > 0:
            # 倫理的毒素の蓄積
            self.ethical_toxin += violation_score * 0.5
            self.modulators["ethical_inhibitor"] = min(
                1.0, self.ethical_toxin / 5.0)

            # 即時ペナルティ: エネルギー削減
            penalty = self.max_energy * 0.1 * violation_score
            self.energy = max(0.0, self.energy - penalty)

            # 重度の違反時は強制シャットダウンに近い状態へ
            if self.ethical_toxin > 5.0:
                self.modulators["gaba"] = 1.0  # 全体的抑制
                self.modulators["glutamate"] = 0.0
                logger.critical(
                    "🛑 CRITICAL ETHICAL BREACH. INITIATING SYSTEM SUPPRESSION.")

            return False

        return True

    def monitor_neural_activity(self, firing_rate: Union[float, Dict[str, float]]):
        """ニューロン活動に基づく代謝調整"""
        val: float = 0.0
        if isinstance(firing_rate, dict):
            if firing_rate:
                val = sum(firing_rate.values()) / len(firing_rate)
        else:
            val = float(firing_rate)

        consumption = val * 0.1
        self.energy = max(0.0, self.energy - consumption)
        self.fatigue_toxin += val * 0.05

        target_glutamate = min(1.0, val / 100.0)
        self.modulators["glutamate"] = 0.9 * \
            self.modulators["glutamate"] + 0.1 * target_glutamate

    def step(self):
        """時間経過更新"""
        now = time.time()
        dt = now - self.last_update_time
        if dt > 10.0:
            dt = 1.0
        self.last_update_time = now

        # 回復 (倫理的抑制がある場合は回復しない)
        recovery_factor = 1.0 - self.modulators["ethical_inhibitor"]
        recovery = self.recovery_rate * dt * \
            (1.0 - self.modulators["cortisol"]
             * 0.5) * max(0.0, recovery_factor)

        self.energy = min(self.max_energy, self.energy + recovery)

        # 自然減少
        self.energy = max(0.0, self.energy - (self.decay_rate * dt))

        # 修飾物質更新
        for k in self.modulators:
            if k == "ethical_inhibitor":
                continue  # これは外部制御のみ
            diff = 0.5 - self.modulators[k]
            self.modulators[k] += diff * 0.1 * dt
            self.modulators[k] = max(0.0, min(1.0, self.modulators[k]))

        # 疲労回復判定
        if self.modulators["gaba"] > 0.8:
            self.energy += self.recovery_rate * dt * 2.0
            self.fatigue_toxin = max(0.0, self.fatigue_toxin - (5.0 * dt))
        else:
            self.fatigue_toxin = max(0.0, self.fatigue_toxin - (0.5 * dt))

        # 倫理的毒素の自然減衰（非常に遅い）
        self.ethical_toxin = max(0.0, self.ethical_toxin - (0.01 * dt))

    def _update_history(self, module_name: str, amount: float):
        if module_name not in self.consumption_history:
            self.consumption_history[module_name] = 0.0
        self.consumption_history[module_name] = (
            0.9 * self.consumption_history[module_name] + 0.1 * amount
        )

    def get_energy_level(self) -> float:
        """エネルギーレベル (0.0 - 1.0)"""
        if self.max_energy <= 0:
            return 0.0
        return self.energy / self.max_energy

    def replenish_energy(self, amount: float):
        self.energy = min(self.max_energy, self.energy + amount)

    def clear_fatigue(self, amount: float):
        self.fatigue_toxin = max(0.0, self.fatigue_toxin - amount)

    def cleanup_toxins(self):
        self.clear_fatigue(self.fatigue_toxin)

    def consume_energy(self, source: str, amount: float = 5.0):
        self.request_resource(source, amount)

    def request_compute_boost(self) -> bool:
        # 倫理的抑制時はブースト不可
        if self.modulators["ethical_inhibitor"] > 0.1:
            return False

        if self.energy > self.max_energy * 0.3 and self.modulators["cortisol"] < 0.8:
            self.energy -= 20.0
            self.modulators["glutamate"] = min(
                1.0, self.modulators["glutamate"] + 0.2)
            return True
        return False

    def log_fatigue(self, amount: float):
        self.fatigue_toxin += amount * 10.0

    def maintain_homeostasis(self, model: nn.Module, target_activity: float = 0.1, learning_rate: float = 0.01):
        # 簡易実装: 重みスケーリング
        if self.modulators["glutamate"] > 0.8:
            scaling = 1.0 - learning_rate
        elif self.modulators["glutamate"] < 0.2:
            scaling = 1.0 + learning_rate
        else:
            return

        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:
                    param.data.mul_(scaling)

    def handle_neuron_death(self, layer: nn.Module, death_rate: float = 0.01):
        """
        [Fix] ニューロン死滅とリルートのシミュレーション。
        テスト(test_homeostasis.py)がエネルギー消費を期待しているため、
        リルート処理時にコスト消費を行うロジックを復元。
        """
        with torch.no_grad():
            for param in layer.parameters():
                if param.dim() > 1:
                    mask = torch.rand_like(param) > death_rate
                    # 死滅 (Weight -> 0)
                    param.data.mul_(mask.float())

                    # リルート (補償)
                    # エネルギーが十分ある場合、残存シナプスを強化し、エネルギーを消費する
                    if self.energy > 50.0:
                        compensation = 1.0 + (death_rate * 0.5)
                        param.data.mul_(compensation)
                        self.energy -= 1.0  # コスト消費

        logger.warning(
            f"🚑 Neuron death simulated (Rate: {death_rate}). Rerouting executed.")

    def get_diagnosis_report(self) -> Dict[str, Any]:
        """
        [Fix] Brain v2.5 / Integration Test で要求される診断メソッドを復元。
        """
        status = "HEALTHY"
        if self.ethical_toxin > 0.5:
            status = "WARNING_ETHICAL_VIOLATION"
        elif self.energy < self.max_energy * 0.2:
            status = "WARNING_LOW_ENERGY"
        elif self.fatigue_toxin > self.fatigue_threshold:
            status = "WARNING_FATIGUE"
        elif self.modulators["cortisol"] > 0.8:
            status = "WARNING_STRESS"

        return {
            "metrics": {
                "current_energy": self.energy,
                "max_energy": self.max_energy,
                "fatigue_level": self.fatigue_toxin,
                "ethical_toxin": self.ethical_toxin,  # [Phase 3.2]
                "stress_level": self.modulators["cortisol"]
            },
            "modulators": self.modulators.copy(),
            "active_consumers": {k: v for k, v in self.consumption_history.items() if v > 0.1},
            "status": status
        }

    @property
    def energy_levels(self) -> Dict[str, Any]:
        """ダッシュボード表示用プロパティ (Deprecated互換)"""
        return self.get_diagnosis_report()
