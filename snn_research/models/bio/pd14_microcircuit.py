# ファイルパス: snn_research/models/bio/pd14_microcircuit.py
# Title: Potjans-Diesmann Cortical Microcircuit Model (PD14)
# Description:
#   doc/assignment.md 第3章に基づき、大脳皮質1mm^2の微小回路モデルを実装。
#   4つの層(L2/3, L4, L5, L6) × 2つの細胞種(Excitatory, Inhibitory) の
#   計8集団で構成され、生物学的な接続確率に基づいて配線される。
#   各ニューロンには TwoCompartmentLIF (能動的樹状突起) を採用可能。

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

from snn_research.core.base import BaseModel
from snn_research.core.neurons.multi_compartment import TwoCompartmentLIF
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore

logger = logging.getLogger(__name__)

class PD14Microcircuit(BaseModel):
    """
    Potjans-Diesmann (2014) モデルに基づく皮質マイクロサーキット。
    """
    def __init__(
        self,
        scale_factor: float = 0.1, # ニューロン数のスケーリング (1.0 = フルスケール ~77k)
        time_steps: int = 16,
        neuron_type: str = "two_compartment", # "two_compartment" or "lif"
        input_dim: int = 100, # 外部入力(Thalamic Input)の次元
        output_dim: int = 10  # L5からの読み出し次元
    ):
        super().__init__()
        self.time_steps = time_steps
        self.scale_factor = scale_factor
        
        # --- 1. ポピュレーション定義 (PD14に基づく比率) ---
        # フルスケールでのニューロン数
        base_counts = {
            "L23e": 20683, "L23i": 5834,
            "L4e": 21915,  "L4i": 5479,
            "L5e": 4850,   "L5i": 1065,
            "L6e": 14395,  "L6i": 2948
        }
        
        # スケーリングされたニューロン数 (最低10個は確保)
        self.pop_counts = {
            k: max(10, int(v * scale_factor)) for k, v in base_counts.items()
        }
        
        self.populations = nn.ModuleDict()
        
        # ニューロン層の構築
        for name, count in self.pop_counts.items():
            if neuron_type == "two_compartment":
                # 多区画モデル: 樹状突起計算を有効化
                # 興奮性(e)はNMDAゲイン高く、抑制性(i)は低く設定するなどの調整も可能
                nmda_gain = 0.5 if 'e' in name else 0.1
                self.populations[name] = TwoCompartmentLIF(
                    features=count, 
                    nmda_gain=nmda_gain
                )
            else:
                # 標準LIF
                self.populations[name] = AdaptiveLIFNeuron(features=count)
                
        # --- 2. 接続性の定義 (Connectivity) ---
        # PD14の接続確率行列 (簡略版: From -> To)
        # 実際にはより詳細な数値があるが、ここでは主要な経路を定義
        # 値は接続確率 (0.0 - 1.0)
        
        # ソースとターゲットのリスト
        pop_names = ["L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
        
        # 接続確率テーブル (行: From, 列: To) - assignment.mdの表2参照
        # 簡易実装: 主要なパスのみ確率を高めに設定
        self.connections = nn.ModuleDict()
        
        for src in pop_names:
            for tgt in pop_names:
                prob = self._get_connection_prob(src, tgt)
                if prob > 0:
                    src_dim = self.pop_counts[src]
                    tgt_dim = self.pop_counts[tgt]
                    
                    # 線形層を作成 (スパースマスクを適用するためのベース)
                    layer = nn.Linear(src_dim, tgt_dim, bias=False)
                    
                    # 確率に基づいて重みをスパース化 (マスク適用)
                    with torch.no_grad():
                        mask = (torch.rand(tgt_dim, src_dim) < prob).float()
                        # 抑制性ニューロンからの出力は負の重みに初期化
                        if 'i' in src:
                            nn.init.uniform_(layer.weight, -0.5, -0.01)
                        else:
                            nn.init.uniform_(layer.weight, 0.01, 0.5)
                        layer.weight *= mask
                        
                    conn_name = f"{src}_to_{tgt}"
                    self.connections[conn_name] = layer

        # --- 3. 外部入出力 ---
        # 視床(Thalamus)入力 -> L4e, L6e (主要経路)
        self.thalamic_input_L4 = nn.Linear(input_dim, self.pop_counts["L4e"])
        self.thalamic_input_L6 = nn.Linear(input_dim, self.pop_counts["L6e"])
        
        # トップダウン(Feedback)入力 -> L2/3e, L5e
        self.feedback_input_L23 = nn.Linear(input_dim, self.pop_counts["L23e"])
        
        # 出力 (Readout) -> L5e (主要出力層) から
        self.readout = nn.Linear(self.pop_counts["L5e"], output_dim)
        
        logger.info(f"🧠 PD14 Microcircuit initialized. Total Neurons: {sum(self.pop_counts.values())}")

    def _get_connection_prob(self, src: str, tgt: str) -> float:
        """
        PD14モデルに基づき、2つの集団間の接続確率を返す。
        (assignment.mdの表2に基づく簡略化ロジック)
        """
        # 自己結合・同層内
        if src == tgt: return 0.1
        if src[:3] == tgt[:3]: return 0.15 # 同じ層
        
        # 正準回路の流れ: L4 -> L2/3 -> L5 -> L6
        if "L4" in src and "L23" in tgt: return 0.15
        if "L23" in src and "L5" in tgt: return 0.15
        if "L5" in src and "L6" in tgt: return 0.1
        if "L6" in src and "L4" in tgt: return 0.05 # Feedback
        
        # 抑制性結合は広範囲
        if 'i' in src: return 0.2
        
        # その他
        return 0.01

    def forward(
        self, 
        thalamic_input: torch.Tensor, # (Batch, InputDim)
        topdown_input: Optional[torch.Tensor] = None # (Batch, InputDim)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        B = thalamic_input.shape[0]
        device = thalamic_input.device
        
        SJ_F.reset_net(self)
        
        # ニューロン状態のリセットとStateful設定
        for mod in self.populations.values():
            if hasattr(mod, 'set_stateful'): mod.set_stateful(True)
            
        # スパイク活動記録用
        spike_counts = {name: 0.0 for name in self.populations.keys()}
        
        # 出力スパイクの蓄積 (L5e)
        readout_accum = torch.zeros(B, self.readout.out_features, device=device)
        
        # 前ステップのスパイク状態 (初期値0)
        prev_spikes = {
            name: torch.zeros(B, count, device=device) 
            for name, count in self.pop_counts.items()
        }
        
        # --- 時間ステップループ ---
        for t in range(self.time_steps):
            current_spikes = {}
            
            # 各ポピュレーションの更新
            for name, neuron_layer in self.populations.items():
                # 1. 内部入力の集計 (他のポピュレーションからの入力)
                internal_current = torch.zeros(B, self.pop_counts[name], device=device)
                
                for src_name, src_spikes in prev_spikes.items():
                    conn_name = f"{src_name}_to_{name}"
                    if conn_name in self.connections:
                        # Linear層を通して電流を加算
                        internal_current += self.connections[conn_name](src_spikes)
                
                # 2. 外部入力の加算
                external_current_soma = torch.zeros_like(internal_current)
                external_current_dend = torch.zeros_like(internal_current)
                
                # L4, L6 -> Thalamic (Bottom-up) -> Somaへ
                if name == "L4e":
                    external_current_soma += self.thalamic_input_L4(thalamic_input)
                if name == "L6e":
                    external_current_soma += self.thalamic_input_L6(thalamic_input)
                    
                # L23, L5 -> Top-down -> Dendriteへ (多区画モデルの利点)
                if topdown_input is not None:
                    if name == "L23e":
                        external_current_dend += self.feedback_input_L23(topdown_input)
                    # L5へのトップダウンも想定可能
                
                # 3. ニューロン発火
                if isinstance(neuron_layer, TwoCompartmentLIF):
                    # 多区画モデル: Soma入力とDendrite入力を分けて渡す
                    # 内部結合はSoma、トップダウンはDendriteといった配分
                    s_input = internal_current + external_current_soma
                    d_input = external_current_dend # 内部結合の一部をDendriteに回すのもアリ
                    
                    spikes, _ = neuron_layer(input_soma=s_input, input_dend=d_input)
                else:
                    # 標準モデル: 全て加算
                    total_input = internal_current + external_current_soma + external_current_dend
                    spikes, _ = neuron_layer(total_input)
                
                current_spikes[name] = spikes
                
                # 統計
                spike_counts[name] += spikes.sum().item() / B

            # L5eの活動を出力として読み出し (積分)
            readout_accum += self.readout(current_spikes["L5e"])
            
            # 状態更新
            prev_spikes = current_spikes

        # --- 終了処理 ---
        for mod in self.populations.values():
            if hasattr(mod, 'set_stateful'): mod.set_stateful(False)

        # 平均発火率の計算
        avg_firing_rates = {
            k: v / self.time_steps for k, v in spike_counts.items()
        }
        
        return readout_accum, avg_firing_rates