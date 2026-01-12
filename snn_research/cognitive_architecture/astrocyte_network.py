# snn_research/cognitive_architecture/astrocyte_network.py
# Title: Optimized Astrocyte Network v2.6 (Mypy Fix)
# Description:
#   神経活動のモニタリングとエネルギー代謝制御。
#   - handle_neuron_death メソッド内の layer.weight の型キャストを追加し、mypyエラーを解消。

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, cast, Union

logger = logging.getLogger(__name__)


class AstrocyteNetwork(nn.Module):
    """
    アストロサイト（グリア細胞）ネットワーク。
    ニューロンの代謝を支え、活動過多による疲労（枯渇）や、
    睡眠による老廃物除去（Glymphatic System）をシミュレートする。
    """

    # 型ヒントの明示 (内部バッファ)
    current_energy: torch.Tensor
    _fatigue_toxin: torch.Tensor
    _glutamate_level: torch.Tensor

    def __init__(
        self,
        initial_energy: float = 1000.0,
        max_energy: float = 1000.0,
        decay_rate: float = 0.01,
        recovery_rate: float = 0.5,
        fatigue_threshold: float = 80.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.max_energy = max_energy
        self.decay_rate = decay_rate
        self.recovery_rate = recovery_rate
        self.fatigue_threshold = fatigue_threshold
        self.device = device

        # 状態変数をBufferとして登録 (内部名はアンダースコア付き)
        self.register_buffer('current_energy', torch.tensor(initial_energy, dtype=torch.float32))
        self.register_buffer('_fatigue_toxin', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('_glutamate_level', torch.tensor(0.5, dtype=torch.float32))

        # 領域ごとの活動履歴
        self.activity_history: Dict[str, float] = {}
        
        # 互換性: 神経修飾物質の辞書
        self.modulators: Dict[str, float] = {
            "dopamine": 0.0,
            "serotonin": 0.0,
            "gaba": 0.0,
            "glutamate": 0.0,
            "noradrenaline": 0.0
        }

    # --- Properties for External Access (Float Interface) ---

    @property
    def energy(self) -> float:
        """Current energy level (float)."""
        return self.current_energy.item()

    @energy.setter
    def energy(self, value: float):
        self.current_energy.fill_(value)

    @property
    def fatigue_toxin(self) -> float:
        """Fatigue toxin level (float). Accesses internal tensor."""
        return self._fatigue_toxin.item()

    @fatigue_toxin.setter
    def fatigue_toxin(self, value: float):
        """Allows assigning a float directly (updates internal tensor)."""
        self._fatigue_toxin.fill_(value)

    @property
    def glutamate_level(self) -> float:
        """Glutamate level (float). Accesses internal tensor."""
        return self._glutamate_level.item()

    @glutamate_level.setter
    def glutamate_level(self, value: float):
        self._glutamate_level.fill_(value)

    # --- Core Methods ---

    def forward(self, x):
        return x

    def step(self):
        """時間ステップごとの自然代謝と回復"""
        # Tensor演算として型を保証 (内部バッファを使用)
        recovery = (self.max_energy - self.current_energy) * 0.001
        self.current_energy.add_(recovery)

        decay = self._fatigue_toxin * 0.005
        self._fatigue_toxin.sub_(decay)
        self._fatigue_toxin.clamp_(min=0.0)
        
        # 外部からの辞書更新への簡易同期
        if self.modulators["glutamate"] != 0.0:
             pass

    def monitor_neural_activity(self, firing_rate: float, region: str = "global"):
        """ニューロン活動の監視と状態更新"""
        cost = (firing_rate * 10.0) ** 1.5
        self.consume_energy(region, cost)

        target_glutamate = 1.0 - firing_rate
        
        # 内部Tensorを使用して演算
        new_val = self._glutamate_level * 0.9 + target_glutamate * 0.1
        self._glutamate_level.copy_(new_val)
        
        # 辞書も更新 (互換性維持)
        self.modulators["glutamate"] = self._glutamate_level.item()

    def consume_energy(self, source: str, amount: float):
        current_val = self.current_energy.item()
        actual_consume = min(amount, current_val)
        
        self.current_energy.sub_(actual_consume)
        
        toxin_buildup = actual_consume * 0.5
        self._fatigue_toxin.add_(toxin_buildup)

        self.activity_history[source] = amount

    def request_resource(self, source: str, amount: float) -> bool:
        if self.current_energy.item() > amount:
            self.consume_energy(source, amount)
            return True
        return False

    def replenish_energy(self, amount: float):
        self.current_energy.add_(amount)
        self.current_energy.clamp_(max=self.max_energy)

    def clear_fatigue(self, amount: float):
        self._fatigue_toxin.sub_(amount)
        self._fatigue_toxin.clamp_(min=0.0)

    def log_fatigue(self, amount: float):
        self._fatigue_toxin.add_(amount)

    def get_energy_level(self) -> float:
        return (self.current_energy / self.max_energy).item()

    def get_modulation_factor(self) -> float:
        energy_ratio = self.get_energy_level()
        fatigue_val = self._fatigue_toxin.item()
        fatigue_ratio = min(1.0, fatigue_val / 100.0)

        modulation = energy_ratio * (1.0 - fatigue_ratio * 0.8)
        return max(0.1, modulation)

    def get_diagnosis_report(self) -> Dict[str, Any]:
        """
        診断レポートを返す。
        """
        glutamate_val = max(self._glutamate_level.item(), self.modulators.get("glutamate", 0.0))
        
        return {
            "metrics": {
                "current_energy": self.current_energy.item(),
                "fatigue_level": self._fatigue_toxin.item(),
                "glutamate_balance": glutamate_val
            },
            "status": "EXHAUSTED" if self.get_energy_level() < 0.2 else "HEALTHY"
        }

    # --- Restored Methods for Tests (Maintained) ---

    def maintain_homeostasis(self, model: nn.Module, learning_rate: float = 0.01):
        """
        [Test Compatibility]
        過剰な興奮を検知した場合、重みを減衰させる。
        """
        glutamate_val = max(self._glutamate_level.item(), self.modulators.get("glutamate", 0.0))
        
        if glutamate_val > 0.8:
            decay_factor = 1.0 - learning_rate
            decay_factor = max(0.0, decay_factor)
            
            for param in model.parameters():
                if param.requires_grad:
                    with torch.no_grad():
                        param.data.mul_(decay_factor)

    def handle_neuron_death(self, layer: nn.Module, death_rate: float = 0.01):
        """
        [Test Compatibility]
        ニューロン死をシミュレートし、エネルギーを消費する。
        """
        if hasattr(layer, 'weight'):
            # ここで cast して Tensor であることを mypy に教える
            weight = cast(torch.Tensor, layer.weight)
            
            with torch.no_grad():
                mask = torch.rand_like(weight) > death_rate
                # Tensorとしての演算であることを保証
                weight.mul_(mask.float())
                self.consume_energy("neuron_death_repair", 10.0)