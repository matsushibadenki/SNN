# ファイルパス: snn_research/cognitive_architecture/astrocyte_network.py
# Title: Astrocyte Network (Neuromorphic OS Scheduler) v2.0
# Description:
#   ROADMAP Phase 7 "The Brain OS" に基づく実装。
#   グリア細胞（アストロサイト）の機能を拡張し、脳内のエネルギー（グルコース/計算リソース）を
#   管理・配分する「ニューロモーフィック・スケジューラ」として機能させる。
#   各モジュールの活動を監視し、エネルギー不足時には優先度の低いモジュールを抑制（Inhibition）する。

import torch
import torch.nn as nn
from typing import List, Dict, Type, Optional, Any
import logging
import math

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

logger = logging.getLogger(__name__)

class AstrocyteNetwork:
    """
    SNN全体のエネルギー管理と恒常性維持（ホメオスタシス）を担うカーネルモジュール。
    OSの「タスクスケジューラ」および「パワーマネージャ」に相当する。
    """
    def __init__(
        self, 
        snn_model: Optional[nn.Module] = None, 
        monitoring_interval: int = 100, 
        evolution_threshold: float = 0.1,
        total_energy_capacity: float = 1000.0,
        basal_metabolic_rate: float = 0.5
    ):
        self.snn_model = snn_model
        self.monitoring_interval = monitoring_interval
        self.evolution_threshold = evolution_threshold
        
        # --- エネルギー管理パラメータ ---
        self.max_energy = total_energy_capacity
        self.current_energy = total_energy_capacity
        self.basal_metabolic_rate = basal_metabolic_rate # 基礎代謝（何もしなくても減る量）
        self.fatigue_toxin = 0.0 # 疲労毒素（睡眠で除去される）
        
        # モジュールごとの優先度設定 (デフォルト)
        self.module_priorities: Dict[str, float] = {
            "amygdala": 10.0,       # 生存に関わるため最優先
            "basal_ganglia": 9.0,   # 行動決定
            "perception": 8.0,      # 状況認識
            "visual_cortex": 7.0,   # 視覚（高コスト）
            "hippocampus": 6.0,     # 記憶
            "prefrontal_cortex": 5.0, # 高次推論（エネルギー不足時はサボる）
            "cortex": 5.0,
            "causal_inference": 4.0, # バックグラウンド処理
            "symbol_grounding": 4.0
        }

        self.step_counter = 0
        self.monitored_neurons: List[nn.Module] = []
        if self.snn_model:
            self.monitored_neurons = self._find_monitored_neurons()
            
        self.long_term_spike_rates: Dict[str, torch.Tensor] = {}
        
        logger.info(f"🧠 Astrocyte Network initialized. Energy: {self.current_energy:.1f}")

    def _find_monitored_neurons(self) -> List[nn.Module]:
        """モデル内の監視対象ニューロン(LIF or Izhikevich)を再帰的に探索する。"""
        neurons: List[nn.Module] = []
        if self.snn_model:
            for module in self.snn_model.modules():
                if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                    neurons.append(module)
        return neurons

    def request_resource(self, module_name: str, estimated_cost: float) -> bool:
        """
        [OS Scheduler] モジュールからの実行許可リクエストを処理する。
        
        Args:
            module_name (str): リクエスト元のモジュール名。
            estimated_cost (float): 実行に必要な推定エネルギーコスト。
            
        Returns:
            bool: 実行許可 (True) または 拒否 (False)。
        """
        # 1. 基礎代謝の消費
        self.current_energy = max(0.0, self.current_energy - self.basal_metabolic_rate * 0.1)

        # 2. クリティカル状態チェック
        if self.current_energy <= 0:
            logger.warning(f"⚠️ Energy Depleted! Denying request from {module_name}.")
            return False

        # 3. 優先度に基づく配分ロジック
        energy_ratio = self.current_energy / self.max_energy
        priority = self.module_priorities.get(module_name, 5.0)
        
        # エネルギーが低下すると、低優先度のタスクは拒否されやすくなる
        # 閾値: エネルギー50%以下で優先度4未満をカット、20%以下で優先度8未満をカット
        required_priority = 0.0
        if energy_ratio < 0.2:
            required_priority = 8.0
        elif energy_ratio < 0.5:
            required_priority = 5.0
            
        if priority < required_priority:
            # logger.debug(f"🛑 Inhibition: {module_name} denied (Low Energy Mode).")
            return False

        # 4. コスト消費と疲労蓄積
        self.current_energy -= estimated_cost
        self.fatigue_toxin += estimated_cost * 0.1 # 活動量に応じて疲労が蓄積
        
        return True

    def replenish_energy(self, amount: float):
        """エネルギーを補充する（食事、休憩など）"""
        self.current_energy = min(self.max_energy, self.current_energy + amount)
        # logger.info(f"🍎 Energy replenished: +{amount:.1f} -> {self.current_energy:.1f}")

    def clear_fatigue(self, amount: float):
        """疲労物質を除去する（睡眠中）"""
        self.fatigue_toxin = max(0.0, self.fatigue_toxin - amount)

    def step(self):
        """
        定期的な監視サイクル。
        """
        self.step_counter += 1
        if self.step_counter % self.monitoring_interval == 0:
            self.monitor_and_regulate()

    @torch.no_grad()
    def monitor_and_regulate(self):
        """
        ニューロン活動の監視と恒常性維持。
        """
        # logger.info(f"🔬 Astrocyte Monitor: Energy={self.current_energy:.1f}, Fatigue={self.fatigue_toxin:.2f}")
        
        # エネルギー効率が悪すぎる場合、全体的な抑制をかける（抑制性神経伝達物質の放出）
        if self.current_energy < self.max_energy * 0.3:
            # logger.info("   📉 Low Energy: Increasing inhibition globally.")
            self._adjust_global_inhibition(increase=True)
        else:
            self._adjust_global_inhibition(increase=False)

        # 個別ニューロンの監視 (既存機能)
        if not self.monitored_neurons:
            self.monitored_neurons = self._find_monitored_neurons()

        for i, layer in enumerate(self.monitored_neurons):
            layer_name = f"{type(layer).__name__}_{i}"
            if not hasattr(layer, 'spikes'): continue
            
            current_rate = layer.spikes.mean().item()
            
            if layer_name in self.long_term_spike_rates:
                self.long_term_spike_rates[layer_name] = (
                    0.99 * self.long_term_spike_rates[layer_name] + 0.01 * torch.tensor(current_rate)
                )
            else:
                self.long_term_spike_rates[layer_name] = torch.tensor(current_rate)

            # 動的進化 (LIF -> Izhikevich)
            if isinstance(layer, AdaptiveLIFNeuron):
                target_rate = layer.target_spike_rate
                long_term_rate = self.long_term_spike_rates[layer_name].item()
                
                if long_term_rate < (target_rate * self.evolution_threshold):
                    # logger.info(f"   🧬 Evolution Triggered for {layer_name}")
                    self._evolve_neuron_model(layer, IzhikevichNeuron)

    def _adjust_global_inhibition(self, increase: bool):
        """全ニューロンの閾値を微調整して活動レベルを制御する"""
        delta = 0.05 if increase else -0.01
        for neuron in self.monitored_neurons:
            if hasattr(neuron, 'base_threshold'):
                if isinstance(neuron.base_threshold, torch.Tensor):
                    neuron.base_threshold.add_(delta)
                else:
                    # floatの場合などは対応外だが通常はParameter
                    pass
            elif hasattr(neuron, 'v_threshold'):
                 # BioLIFNeuronなどの場合
                 pass

    def _evolve_neuron_model(self, layer_to_evolve: nn.Module, target_class: Type[nn.Module]):
        """指定されたニューロン層を進化させる"""
        if not self.snn_model: return
        
        for name, module in self.snn_model.named_modules():
            for child_name, child_module in module.named_children():
                if child_module is layer_to_evolve:
                    if hasattr(layer_to_evolve, 'features'):
                        features = layer_to_evolve.features # type: ignore
                        new_neuron = target_class(features=features)
                        # デバイス移動
                        try:
                            device = next(layer_to_evolve.parameters()).device
                            new_neuron.to(device)
                        except StopIteration: pass
                        
                        setattr(module, child_name, new_neuron)
                        # logger.info(f"      -> {child_name} evolved to {target_class.__name__}")
                        return
