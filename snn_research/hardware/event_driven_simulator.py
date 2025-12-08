# ファイルパス: snn_research/hardware/event_driven_simulator.py
# Title: イベント駆動型SNNシミュレータ (On-Chip Plasticity 実装版)
# Description:
#   ROADMAP Phase 6 "Hardware Native Transition" 実装。
#   イベント駆動型の推論に加え、STDPに基づくオンチップ学習（On-Chip Plasticity）をサポート。
#   スパイクタイミングに基づいて、シミュレーション実行中に動的に重みを更新する。

import torch
import torch.nn as nn
import heapq
import logging
import math
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field

# 既存のニューロン定義を利用
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

logger = logging.getLogger(__name__)

@dataclass(order=True)
class SpikeEvent:
    """
    スパイクイベント。優先度付きキューで時間順に処理される。
    """
    timestamp: float        # 発火時刻
    neuron_id: int = field(compare=False) # レイヤー内ID
    layer_index: int = field(compare=False) # 所属レイヤー
    source_index: int = field(compare=False) # (同上)
    payload: float = field(compare=False, default=1.0) # スパイクの重み

class EventDrivenNeuronState:
    """
    ニューロン状態保持クラス。STDP用のトレースを追加。
    """
    def __init__(self, v_threshold: float, tau_mem: float, v_reset: float, tau_trace: float = 20.0):
        self.v = 0.0          # 現在の膜電位
        self.last_update_time = 0.0 # 最後に膜電位が更新された時刻
        
        # 学習用トレース (Pre/Post spike trace)
        self.trace = 0.0      
        self.last_spike_time = -1.0
        self.tau_trace = tau_trace
        
        self.v_threshold = v_threshold
        self.tau_mem = tau_mem
        self.v_reset = v_reset

    def update_trace(self, current_time: float):
        """トレースを現在時刻まで減衰させる"""
        dt = current_time - self.last_update_time
        if dt > 0:
            decay = math.exp(-dt / self.tau_trace)
            self.trace *= decay

    def update(self, current_time: float, input_weight: float) -> bool:
        """
        イベントを受け取り、状態を更新する。
        """
        dt = current_time - self.last_update_time
        
        # 1. 膜電位の減衰
        decay_factor = math.exp(-dt / self.tau_mem)
        self.v = self.v * decay_factor
        
        # 2. トレースの減衰 (学習用)
        trace_decay = math.exp(-dt / self.tau_trace)
        self.trace *= trace_decay
        
        # 3. 入力の統合
        self.v += input_weight
        
        self.last_update_time = current_time
        
        # 4. 発火判定
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.trace += 1.0 # 発火時にトレースを加算
            self.last_spike_time = current_time
            return True
            
        return False

class EventDrivenSimulator:
    """
    イベント駆動型SNNシミュレータ (with On-Chip Plasticity)。
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
        self.stdp_window = stdp_window # トレース時定数
        
        self.layers: List[List[EventDrivenNeuronState]] = []
        self.weights: List[torch.Tensor] = [] 
        
        self._parse_model(model)
        
        logger.info(f"⚙️ Event-Driven Simulator initialized. Learning: {self.enable_learning}")

    def _parse_model(self, model: nn.Module):
        """モデル解析と状態初期化"""
        current_weights = None
        
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                current_weights = mod.weight.detach().cpu() # (Out, In)
                self.weights.append(current_weights)
                
            elif isinstance(mod, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                if hasattr(mod, 'base_threshold'):
                    v_th = mod.base_threshold
                    if isinstance(v_th, torch.Tensor): v_th = v_th.mean().item()
                else:
                    v_th = 1.0
                    
                if hasattr(mod, 'log_tau_mem'):
                    tau = (torch.exp(mod.log_tau_mem) + 1.1).mean().item()
                elif hasattr(mod, 'tau_mem'):
                    tau = mod.tau_mem
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
                    EventDrivenNeuronState(v_th, tau, v_reset, self.stdp_window) 
                    for _ in range(n_neurons)
                ]
                self.layers.append(layer_states)

    def set_input_spikes(self, input_spikes: torch.Tensor):
        """入力スパイクを登録"""
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

    def _apply_stdp(self, pre_layer_idx: int, pre_neuron_idx: int, post_layer_idx: int, post_neuron_idx: int, event_type: str):
        """
        簡易的なSTDP学習則の適用 (On-Chip Plasticity)。
        event_type: 'pre_spike' (入力スパイク到着時) or 'post_spike' (出力ニューロン発火時)
        """
        # 重み行列の取得
        W = self.weights[post_layer_idx] # weights[0] is for Input->Layer0
        
        # ニューロン状態の取得
        # pre_neuron は Layer[pre_layer_idx] にある (Inputの場合は管理外なのでTrace近似が必要だが今回は簡易化)
        post_neuron = self.layers[post_layer_idx][post_neuron_idx]
        
        # 簡略化: 入力層(-1)からのスパイクの場合、PreTraceは「今スパイクした」として1.0とするか、
        # 本来は入力層も NeuronState を持つべきだが、ここではイベントドリブンLTPのみ実装する。
        
        current_w = W[post_neuron_idx, pre_neuron_idx].item()
        dw = 0.0

        # LTP (Long-Term Potentiation): Postが発火したとき、直近に発火していたPreとの結合を強める
        if event_type == 'post_spike':
            # Preニューロンのトレースを参照したいが、Inputの場合はトレースがない。
            # 簡易実装: Inputイベントのタイムスタンプを履歴として保持する必要があるが、
            # ここでは「Preが発火した直後にPostが発火した」とみなして、
            # 到着したスパイク(イベント)に対して即座にLTPを計算するのは難しい（因果が逆）。
            # 正しくは: Post発火時に、接続されている全Preニューロンのトレースを見て更新する。
            pass

    def run(self, max_time: float = 100.0) -> Dict[str, Any]:
        """シミュレーション実行"""
        logger.info(f"🚀 Running simulation (Learning={self.enable_learning})...")
        processed_events = 0
        output_spikes_count = 0
        weight_updates = 0
        
        # 学習用: 直近の入力イベント時間をキャッシュ (簡易的なPre-Trace代わり)
        # {layer_idx: {neuron_idx: last_spike_time}}
        last_spike_times: Dict[int, Dict[int, float]] = {-1: {}} 
        
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.timestamp > max_time: break
                
            self.current_time = event.timestamp
            processed_events += 1
            
            src_layer_idx = event.layer_index
            src_neuron_idx = event.source_index
            
            # 発火時刻の記録 (Pre-Trace用)
            if src_layer_idx not in last_spike_times: last_spike_times[src_layer_idx] = {}
            last_spike_times[src_layer_idx][src_neuron_idx] = self.current_time
            
            # LTD (Long-Term Depression): Pre発火時、Postのトレースが高ければ弱める (Post-before-Pre)
            # ここでは計算量削減のため省略、またはPost側のループで処理
            
            # --- 次層への伝播 ---
            tgt_layer_idx = src_layer_idx + 1
            if tgt_layer_idx >= len(self.layers):
                output_spikes_count += 1
                continue
                
            if tgt_layer_idx >= len(self.weights): continue
            
            W = self.weights[tgt_layer_idx]
            
            # スパース接続: 関連する重みのみ取得
            relevant_weights = W[:, src_neuron_idx]
            active_indices = torch.nonzero(torch.abs(relevant_weights) > 0.001).flatten()
            
            for tgt_neuron_idx in active_indices:
                w = relevant_weights[tgt_neuron_idx].item()
                target_neuron = self.layers[tgt_layer_idx][tgt_neuron_idx]
                
                # ターゲット更新
                self.total_ops += 1
                fired = target_neuron.update(self.current_time, w)
                
                # LTDの簡易実装: Preスパイク到着時に、Postのトレース(最近発火したか)を見て弱める
                if self.enable_learning and target_neuron.trace > 0.01:
                     # Post fired recently, now Pre fires -> Post before Pre -> Depression
                     dw = -self.learning_rate * target_neuron.trace
                     W[tgt_neuron_idx, src_neuron_idx] += dw
                     weight_updates += 1

                if fired:
                    # 発火イベント登録
                    delay = 1.0
                    new_event = SpikeEvent(
                        timestamp=self.current_time + delay,
                        neuron_id=int(tgt_neuron_idx),
                        layer_index=tgt_layer_idx,
                        source_index=int(tgt_neuron_idx)
                    )
                    heapq.heappush(self.event_queue, new_event)
                    
                    # LTP (On-Chip Plasticity): Post発火時、接続元(Pre)のトレースを見て強める
                    if self.enable_learning:
                        # 接続されているPreレイヤーのニューロン活動を確認
                        # (全結合前提: Wの列数分)
                        num_pre_neurons = W.shape[1]
                        pre_spike_times = last_spike_times.get(src_layer_idx, {})
                        
                        for pre_idx, t_pre in pre_spike_times.items():
                            # Preが最近発火していれば強化 (Pre before Post)
                            dt = self.current_time - t_pre
                            if 0 < dt < self.stdp_window * 3: # 時間窓内
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
