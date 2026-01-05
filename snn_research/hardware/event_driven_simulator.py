# ファイルパス: snn_research/hardware/event_driven_simulator.py
# Title: イベント駆動型SNNシミュレータ (Trace Saturation Fix)
# Description:
#   ROADMAP Phase 6 "Hardware Native Transition" 実装。
#   修正: STDPトレースの飽和(Saturation)を導入し、過剰なLTDによる学習崩壊を防ぐ。

import torch
import torch.nn as nn
import heapq
import logging
import math
from typing import List, Dict, Any
from dataclasses import dataclass, field

# 既存のニューロン定義を利用
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

logger = logging.getLogger(__name__)


@dataclass(order=True)
class SpikeEvent:
    timestamp: float
    neuron_id: int = field(compare=False)
    layer_index: int = field(compare=False)
    source_index: int = field(compare=False)
    payload: float = field(compare=False, default=1.0)


class EventDrivenNeuronState:
    """
    ニューロン状態保持クラス。STDP用のトレースを追加。
    """

    def __init__(self, v_threshold: float, tau_mem: float, v_reset: float, tau_trace: float = 20.0):
        self.v = 0.0
        self.last_update_time = 0.0

        self.trace = 0.0
        self.last_spike_time = -1.0
        self.tau_trace = tau_trace

        self.v_threshold = v_threshold
        self.tau_mem = tau_mem
        self.v_reset = v_reset

    def decay_and_integrate(self, current_time: float, input_weight: float) -> None:
        """
        時間経過による減衰と入力の統合を行う（発火判定はしない）。
        """
        dt = current_time - self.last_update_time

        # 1. 膜電位の減衰
        decay_factor = math.exp(-dt / self.tau_mem)
        self.v = self.v * decay_factor

        # 2. トレースの減衰
        trace_decay = math.exp(-dt / self.tau_trace)
        self.trace *= trace_decay

        # 3. 入力の統合
        self.v += input_weight

        self.last_update_time = current_time

    def check_fire_and_update_trace(self, current_time: float) -> bool:
        """
        発火判定を行い、発火した場合はリセットとトレース更新を行う。
        """
        if self.v >= self.v_threshold:
            self.v = self.v_reset

            # --- 修正: トレースの更新ロジック ---
            # 加算し続けるとバースト時に値が爆発するため、上限を設けるかリセットする。
            # ここでは「発火直後はトレースが最大(1.0)になる」モデルを採用。
            self.trace = 1.0

            self.last_spike_time = current_time
            return True
        return False


class EventDrivenSimulator:
    """
    イベント駆動型SNNシミュレータ (with On-Chip Plasticity).
    """

    def __init__(
        self,
        model: nn.Module,
        enable_learning: bool = False,
        learning_rate: float = 0.001,
        stdp_window: float = 20.0
    ):
        self.model = model
        self.event_queue: List[SpikeEvent] = []
        self.current_time = 0.0
        self.total_ops = 0
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window

        self.layers: List[List[EventDrivenNeuronState]] = []
        self.weights: List[torch.Tensor] = []

        self._parse_model(model)

        logger.info(
            f"⚙️ Event-Driven Simulator initialized. Learning: {self.enable_learning}")

    def _parse_model(self, model: nn.Module):
        current_weights = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                current_weights = mod.weight.detach().cpu()
                self.weights.append(current_weights)
            elif isinstance(mod, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                if hasattr(mod, 'base_threshold'):
                    v_th = mod.base_threshold
                    if isinstance(v_th, torch.Tensor):
                        v_th = v_th.mean().item()
                else:
                    v_th = 1.0

                if hasattr(mod, 'tau_mem'):
                    tau = float(mod.tau_mem)
                elif hasattr(mod, 'log_tau_mem'):
                    tau = (torch.exp(mod.log_tau_mem) + 1.1).mean().item()
                else:
                    tau = 20.0

                v_reset = getattr(mod, 'v_reset', 0.0)

                if hasattr(mod, 'features'):
                    n_neurons = mod.features
                elif current_weights is not None:
                    n_neurons = current_weights.shape[0]
                else:
                    continue

                layer_states = [
                    EventDrivenNeuronState(
                        v_th, tau, v_reset, self.stdp_window)
                    for _ in range(n_neurons)
                ]
                self.layers.append(layer_states)

    def set_input_spikes(self, input_spikes: torch.Tensor):
        spike_indices = torch.nonzero(input_spikes)
        count = 0
        for t, n_idx in spike_indices:
            event = SpikeEvent(
                timestamp=float(t),
                neuron_id=int(n_idx),
                layer_index=-1,
                source_index=int(n_idx)
            )
            heapq.heappush(self.event_queue, event)
            count += 1
        logger.info(f"📥 Registered {count} input events.")

    def run(self, max_time: float = 100.0) -> Dict[str, Any]:
        logger.info(
            f"🚀 Running simulation (Learning={self.enable_learning})...")
        processed_events = 0
        output_spikes_count = 0
        weight_updates = 0

        last_spike_times: Dict[int, Dict[int, float]] = {-1: {}}

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.timestamp > max_time:
                break

            self.current_time = event.timestamp
            processed_events += 1

            src_layer_idx = event.layer_index
            src_neuron_idx = event.source_index

            # 発火時刻の記録
            if src_layer_idx not in last_spike_times:
                last_spike_times[src_layer_idx] = {}
            last_spike_times[src_layer_idx][src_neuron_idx] = self.current_time

            target_layer_idx = src_layer_idx + 1
            if target_layer_idx >= len(self.layers):
                output_spikes_count += 1
                continue
            if target_layer_idx >= len(self.weights):
                continue

            W = self.weights[target_layer_idx]

            relevant_weights = W[:, src_neuron_idx]
            active_indices = torch.nonzero(
                torch.abs(relevant_weights) > 0.001).flatten()

            for tgt_neuron_idx in active_indices:
                w = relevant_weights[tgt_neuron_idx].item()
                target_neuron = self.layers[target_layer_idx][tgt_neuron_idx]

                # 1. 減衰と統合
                self.total_ops += 1
                target_neuron.decay_and_integrate(self.current_time, w)

                # --- LTD (Long-Term Depression) ---
                if self.enable_learning:
                    if target_neuron.trace > 0.05:
                        # LTDの強さを少し控えめに設定
                        dw = -self.learning_rate * target_neuron.trace * 0.25
                        W[tgt_neuron_idx, src_neuron_idx] += dw
                        weight_updates += 1

                # 2. 発火判定
                fired = target_neuron.check_fire_and_update_trace(
                    self.current_time)

                if fired:
                    delay = 1.0
                    new_event = SpikeEvent(
                        timestamp=self.current_time + delay,
                        neuron_id=int(tgt_neuron_idx),
                        layer_index=target_layer_idx,
                        source_index=int(tgt_neuron_idx)
                    )
                    heapq.heappush(self.event_queue, new_event)

                    # --- LTP (Long-Term Potentiation) ---
                    if self.enable_learning:
                        pre_spike_times = last_spike_times.get(
                            src_layer_idx, {})

                        for pre_idx, t_pre in pre_spike_times.items():
                            dt = self.current_time - t_pre
                            if 0 <= dt < self.stdp_window:
                                stdp_factor = math.exp(-dt / self.stdp_window)
                                dw = self.learning_rate * stdp_factor
                                W[tgt_neuron_idx, pre_idx] += dw
                                weight_updates += 1

        logger.info(f"✅ Simulation complete. Updates: {weight_updates}")
        return {
            "processed_events": processed_events,
            "total_ops": self.total_ops,
            "output_spikes": output_spikes_count,
            "weight_updates": weight_updates
        }
