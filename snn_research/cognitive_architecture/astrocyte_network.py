# ファイルパス: snn_research/cognitive_architecture/astrocyte_network.py
# Title: Astrocyte Network v2.5 (Fully Implemented)
# Description:
#   Brain v2.5 / Runner v2.5 が要求する全メソッドを実装。
#   - cleanup_toxins: clear_fatigueのエイリアス
#   - consume_energy, request_compute_boost, log_fatigue: 追加実装

import logging
import time
import numpy as np
import torch
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class AstrocyteNetwork:
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
        self.energy = initial_energy
        self.max_energy = max_energy
        self.recovery_rate = recovery_rate  # 安静時の回復量
        self.decay_rate = decay_rate       # 自然減衰
        self.fatigue_threshold = fatigue_threshold

        # 疲労毒素 (0.0 - 100.0+)
        self.fatigue_toxin = 0.0

        # 化学物質濃度 (Modulators)
        self.modulators: Dict[str, float] = {
            "glutamate": 0.5,   # 興奮性 (Excitatory) - 活動レベル
            "gaba": 0.5,        # 抑制性 (Inhibitory) - 鎮静レベル
            "dopamine": 0.5,    # 報酬・動機 (Reward/Motivation)
            "cortisol": 0.1,    # ストレス (Stress)
            "acetylcholine": 0.5  # 注意 (Attention)
        }

        # モジュールごとの消費履歴 (Heatmap用)
        self.consumption_history: Dict[str, float] = {}
        self.last_update_time = time.time()

        logger.info(
            f"🌟 Astrocyte Network initialized (Fatigue Threshold: {fatigue_threshold}).")

    # --- Properties for Backward Compatibility ---
    @property
    def current_energy(self) -> float:
        return self.energy

    @current_energy.setter
    def current_energy(self, value: float):
        self.energy = value

    # --- Core Methods ---

    def request_resource(self, module_name: str, amount: float) -> bool:
        """
        モジュールからのエネルギー要求を処理する。
        """
        # 1. 基本チェック
        if self.energy <= 0:
            logger.warning(
                f"⚠️ Energy Depleted! Denying request from {module_name}")
            return False

        # 2. ストレス/疲労によるコスト補正
        cost_multiplier = 1.0 + (self.modulators["cortisol"] * 0.5)

        if self.fatigue_toxin > self.fatigue_threshold:
            cost_multiplier *= 1.5

        # アセチルコリン（注意）が低いとSystem 2系の要求を確率的に却下（集中力低下のシミュレート）
        if module_name in ["prefrontal_cortex", "reasoning_engine", "planner"] and self.modulators["acetylcholine"] < 0.2:
            if np.random.random() < 0.5:
                return False

        required_energy = amount * cost_multiplier

        # 3. 承認判定
        if self.energy >= required_energy:
            self.energy -= required_energy
            self._update_history(module_name, required_energy)

            # 活動に応じた神経伝達物質の変動
            self.modulators["glutamate"] = min(
                1.0, self.modulators["glutamate"] + 0.01)
            self.fatigue_toxin += 0.01 * amount
            return True
        else:
            return False

    def monitor_neural_activity(self, firing_rate: Union[float, Dict[str, float]]):
        """
        ニューロンの全体発火率を監視し、代謝を調整する。
        """
        if isinstance(firing_rate, dict):
            if not firing_rate:
                val = 0.0
            else:
                val = sum(firing_rate.values()) / len(firing_rate)
        else:
            val = firing_rate

        # 発火率が高いとエネルギー消費増、疲労蓄積
        consumption = val * 0.1
        self.energy = max(0.0, self.energy - consumption)
        self.fatigue_toxin += val * 0.05

        # グルタミン酸濃度調整
        target_glutamate = min(1.0, val / 100.0)
        self.modulators["glutamate"] = 0.9 * \
            self.modulators["glutamate"] + 0.1 * target_glutamate

    def step(self):
        """時間経過による恒常性維持サイクル"""
        now = time.time()
        dt = now - self.last_update_time
        if dt > 10.0:
            dt = 1.0
        self.last_update_time = now

        # 1. エネルギー回復
        recovery = self.recovery_rate * dt * \
            (1.0 - self.modulators["cortisol"] * 0.5)
        self.energy = min(self.max_energy, self.energy + recovery)

        # 2. 自然代謝
        self.energy = max(0.0, self.energy - (self.decay_rate * dt))

        # 3. 化学物質の崩壊・相互作用
        self._update_modulators(dt)

        # 4. 疲労の蓄積と解消
        if self.modulators["glutamate"] > 0.8:  # 活動過多
            self.modulators["gaba"] += 0.05 * dt

        if self.modulators["gaba"] > 0.8:  # 休息モード
            self.energy += self.recovery_rate * dt * 2.0
            self.fatigue_toxin = max(0.0, self.fatigue_toxin - (5.0 * dt))
        else:
            self.fatigue_toxin = max(0.0, self.fatigue_toxin - (0.5 * dt))

    def _update_modulators(self, dt: float):
        """神経修飾物質の自然減衰と相互作用"""
        for k in self.modulators:
            diff = 0.5 - self.modulators[k]
            self.modulators[k] += diff * 0.1 * dt
            self.modulators[k] = max(0.0, min(1.0, self.modulators[k]))

    def _update_history(self, module_name: str, amount: float):
        """消費履歴の更新"""
        if module_name not in self.consumption_history:
            self.consumption_history[module_name] = 0.0
        self.consumption_history[module_name] = (
            0.9 * self.consumption_history[module_name] + 0.1 * amount
        )

    # --- Maintenance / Diagnostics APIs ---

    def get_energy_level(self) -> float:
        """現在のエネルギーレベル（0.0 - 1.0）"""
        return self.energy / self.max_energy

    def replenish_energy(self, amount: float):
        """外部からのエネルギー補充（食事・充電）"""
        self.energy = min(self.max_energy, self.energy + amount)
        logger.info(
            f"🔋 Energy replenished by {amount}. Current: {self.energy:.1f}")

    def clear_fatigue(self, amount: float):
        """疲労の強制除去（睡眠完了時など）"""
        self.fatigue_toxin = max(0.0, self.fatigue_toxin - amount)
        logger.info(
            f"✨ Fatigue cleared by {amount}. Current: {self.fatigue_toxin:.1f}")

    def cleanup_toxins(self):
        """疲労物質を完全に除去する（エイリアス）"""
        self.clear_fatigue(self.fatigue_toxin)

    # --- New Methods for SurpriseGatedBrain Compatibility ---

    def consume_energy(self, source: str, amount: float = 5.0):
        """特定のソースによるエネルギー消費を強制的に記録する"""
        self.request_resource(source, amount)

    def request_compute_boost(self) -> bool:
        """
        System 2 などの高負荷処理のためのブースト要求。
        コストが高いが、許可されればリソースを割り当てる。
        """
        # エネルギー残量が十分かつ、ストレスが高すぎない場合
        if self.energy > self.max_energy * 0.3 and self.modulators["cortisol"] < 0.8:
            # ブーストコストの消費
            self.energy -= 20.0
            # 興奮レベル上昇
            self.modulators["glutamate"] = min(
                1.0, self.modulators["glutamate"] + 0.2)
            return True
        return False

    def log_fatigue(self, amount: float):
        """
        疲労を直接蓄積させる（推論負荷などによる）。
        amount: 0.0 - 1.0 (相対値)
        """
        self.fatigue_toxin += amount * 10.0  # スケール調整

    # --- Phase 4.2: Self-Repair & Homeostasis ---

    def maintain_homeostasis(self, model: torch.nn.Module, target_activity: float = 0.1, learning_rate: float = 0.01):
        """
        [Phase 4.2 New] シナプススケーリングによる恒常性維持。
        平均発火率が目標値から乖離した場合、重みを全体的にスケーリングして調整する。
        """
        # ※ 実際の実装ではモデルのフックや勾配操作が必要だが、ここではシミュレーションロジック
        if self.modulators["glutamate"] > 0.8:  # 過活動
            scaling_factor = 1.0 - learning_rate
        elif self.modulators["glutamate"] < 0.2:  # 低活動
            scaling_factor = 1.0 + learning_rate
        else:
            return

        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:  # 重み行列のみ対象 (バイアス除く)
                    param.data.mul_(scaling_factor)

        logger.debug(
            f"⚖️ Homeostasis applied. Scaling factor: {scaling_factor:.3f}")

    def handle_neuron_death(self, layer: torch.nn.Module, death_rate: float = 0.01):
        """
        [Phase 4.2 New] ニューロン死滅とリルート（再配線）のシミュレーション。
        一部の重みを0にし（死滅）、隣接ニューロンの重みを強化して補償する（リルート）。
        """
        with torch.no_grad():
            for param in layer.parameters():
                if param.dim() > 1:  # 重み行列
                    mask = torch.rand_like(param) > death_rate
                    # 死滅したニューロン(False)の重みを0にする
                    param.data.mul_(mask.float())

                    # リルート: 生き残った結合を少し強化して補填 (アストロサイトによる支援)
                    if self.energy > 50.0:
                        compensation = 1.0 + (death_rate * 0.5)
                        param.data.mul_(compensation)
                        self.energy -= 1.0  # コスト消費

        logger.warning(
            f"🚑 Neuron death simulated (Rate: {death_rate}). Rerouting executed.")

    # ------------------------------------------------------

    def get_diagnosis_report(self) -> Dict[str, Any]:
        """Brain v2.5 / Health Check API用の診断レポート"""
        status = "HEALTHY"
        if self.energy < self.max_energy * 0.2:
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
