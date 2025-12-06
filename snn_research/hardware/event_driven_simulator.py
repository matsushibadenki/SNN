# ファイルパス: snn_research/hardware/event_driven_simulator.py
# Title: イベント駆動型SNNシミュレータ (Event-Driven Simulator)
# Description:
#   ROADMAP Phase 6 "Hardware Native Transition" の中核。
#   従来の同期型（タイムステップ刻み）シミュレーションではなく、
#   スパイクイベントが発生したタイミングでのみニューロン状態を更新する
#   非同期・イベント駆動型の計算モデルを実装する。
#   これにより、SNNの真のメリットである「スパース性による計算量削減」を実証する。

import torch
import torch.nn as nn
import heapq
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field

# 既存のニューロン定義を利用（パラメータ参照用）
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

logger = logging.getLogger(__name__)

@dataclass(order=True)
class SpikeEvent:
    """
    スパイクイベントを表すデータクラス。
    優先度付きキューで時間順に処理される。
    """
    timestamp: float        # 発火時刻
    neuron_id: int = field(compare=False) # 発火したニューロンのグローバルID
    layer_index: int = field(compare=False) # 所属レイヤー
    source_index: int = field(compare=False) # レイヤー内のインデックス
    payload: float = field(compare=False, default=1.0) # スパイクの重み（通常1.0）

class EventDrivenNeuronState:
    """
    イベント駆動型更新のためのニューロン状態保持クラス。
    """
    def __init__(self, v_threshold: float, tau_mem: float, v_reset: float):
        self.v = 0.0          # 現在の膜電位
        self.last_update_time = 0.0 # 最後に更新された時刻
        self.v_threshold = v_threshold
        self.tau_mem = tau_mem
        self.v_reset = v_reset

    def update(self, current_time: float, input_weight: float) -> bool:
        """
        イベントを受け取り、状態を更新する。発火した場合はTrueを返す。
        V(t) = V(t_last) * exp(-(t - t_last) / tau) + Input
        """
        dt = current_time - self.last_update_time
        
        # 1. 漏れ（Leak）の計算: 前回の更新から現在までの自然減衰
        decay_factor = torch.exp(torch.tensor(-dt / self.tau_mem)).item()
        self.v = self.v * decay_factor
        
        # 2. 入力の統合（Integrate）
        self.v += input_weight
        
        self.last_update_time = current_time
        
        # 3. 発火判定（Fire）
        if self.v >= self.v_threshold:
            # リセット
            self.v = self.v_reset
            return True
        return False

class EventDrivenSimulator:
    """
    優先度付きキューを用いたイベント駆動型SNNシミュレータ。
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.event_queue: List[SpikeEvent] = []
        self.current_time = 0.0
        self.total_ops = 0 # 演算回数（ベンチマーク用）
        
        # ネットワーク構造の解析と状態初期化
        self.layers: List[List[EventDrivenNeuronState]] = []
        self.weights: List[torch.Tensor] = [] # [Layer0 -> Layer1, Layer1 -> Layer2, ...]
        
        self._parse_model(model)
        logger.info(f"⚙️ Event-Driven Simulator initialized. Layers: {len(self.layers)}")

    def _parse_model(self, model: nn.Module):
        """
        PyTorchモデル（nn.SequentialやSNNCore）を解析し、
        イベント駆動シミュレーション用の内部構造に変換する。
        """
        modules = list(model.modules())
        # ニューロン層と線形層（重み）を抽出
        current_weights = None
        
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                current_weights = mod.weight.detach().cpu() # (Out, In)
                # バイアスは簡易化のため今回は省略（または入力として扱う）
                self.weights.append(current_weights)
                
            elif isinstance(mod, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                # ニューロン層の初期化
                # パラメータ取得
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
                
                # ニューロン数 (直前の重みの出力次元、またはfeatures属性)
                if hasattr(mod, 'features'):
                    n_neurons = mod.features # type: ignore
                elif current_weights is not None:
                    n_neurons = current_weights.shape[0]
                else:
                    continue # 特定できない場合はスキップ

                layer_states = [
                    EventDrivenNeuronState(v_th, tau, v_reset) 
                    for _ in range(n_neurons)
                ]
                self.layers.append(layer_states)

    def set_input_spikes(self, input_spikes: torch.Tensor):
        """
        入力スパイク列をイベントキューに登録する。
        input_spikes: (Time, Neurons) の Tensor (0 or 1)
        """
        time_steps, num_inputs = input_spikes.shape
        count = 0
        
        # 入力層への接続重みがない場合、最初の層を仮想的にレイヤー0として扱うためのダミー処理が必要だが、
        # ここでは「入力 -> Layer 0」の重みが self.weights[0] にあると仮定する。
        # 入力スパイクは「Layer -1」からのイベントとして扱う。
        
        # スパイクがある場所を探す
        spike_indices = torch.nonzero(input_spikes)
        for t, n_idx in spike_indices:
            # 入力イベント: Layer -1, Neuron n_idx から発火
            # これを Layer 0 のニューロンに伝播させる必要がある
            # シミュレータの構造上、キューには「受信イベント」を入れるのが効率的だが、
            # ここでは「発火イベント」を入れ、process_eventで次層への伝播を処理する。
            
            event = SpikeEvent(
                timestamp=float(t),
                neuron_id=int(n_idx),
                layer_index=-1, # Input Layer
                source_index=int(n_idx)
            )
            heapq.heappush(self.event_queue, event)
            count += 1
            
        logger.info(f"📥 Input spikes registered: {count} events.")

    def run(self, max_time: float = 100.0) -> Dict[str, Any]:
        """
        イベントキューが空になるか、最大時間に達するまでシミュレーションを実行する。
        """
        logger.info(f"🚀 Running event-driven simulation (Max Time: {max_time})...")
        processed_events = 0
        output_spikes_count = 0
        
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            if event.timestamp > max_time:
                break
                
            self.current_time = event.timestamp
            processed_events += 1
            
            # --- イベント処理: スパイクの伝播 ---
            
            # 現在のイベントは「layer_indexのneuron_source_indexが発火した」ことを示す
            source_layer_idx = event.layer_index
            source_neuron_idx = event.source_index
            
            # 次の層（ターゲット層）が存在するか確認
            target_layer_idx = source_layer_idx + 1
            if target_layer_idx >= len(self.layers):
                # ネットワークの出力層からのスパイク
                output_spikes_count += 1
                continue
                
            # 次の層への結合重みを取得
            # weights[target_layer_idx] ではなく、層間の重みなのでインデックスに注意
            # self.weights[0] は input -> layer0
            # self.weights[1] は layer0 -> layer1
            weight_idx = target_layer_idx
            if weight_idx >= len(self.weights):
                continue
                
            W = self.weights[weight_idx] # (Out, In)
            
            # スパース性を活用: 接続されている（重みが0でない）ニューロンのみ更新
            # W[:, source_neuron_idx] がこのスパイクの影響を受ける重み列
            relevant_weights = W[:, source_neuron_idx]
            
            # 重みが閾値以上のターゲットのみ処理 (近似計算/スパース計算)
            # 全結合でも0に近い重みは無視することで計算量を削減できる（Hardware Nativeの挙動）
            active_indices = torch.nonzero(torch.abs(relevant_weights) > 0.001).flatten()
            
            for target_neuron_idx in active_indices:
                w = relevant_weights[target_neuron_idx].item()
                target_neuron = self.layers[target_layer_idx][target_neuron_idx]
                
                # ターゲットニューロンの状態更新
                self.total_ops += 1 # 演算カウント
                fired = target_neuron.update(self.current_time, w)
                
                if fired:
                    # 発火したら、新しいイベントをキューに追加
                    # 軸索遅延 (Axonal Delay) をシミュレートして少し未来に追加
                    delay = 1.0 # 1タイムステップ相当
                    new_event = SpikeEvent(
                        timestamp=self.current_time + delay,
                        neuron_id=int(target_neuron_idx), # これはレイヤー内ID
                        layer_index=target_layer_idx,
                        source_index=int(target_neuron_idx)
                    )
                    heapq.heappush(self.event_queue, new_event)

        logger.info(f"✅ Simulation complete.")
        logger.info(f"   - Processed Events: {processed_events}")
        logger.info(f"   - Total Operations (Neuron Updates): {self.total_ops}")
        logger.info(f"   - Output Spikes: {output_spikes_count}")
        
        return {
            "processed_events": processed_events,
            "total_ops": self.total_ops,
            "output_spikes": output_spikes_count
        }