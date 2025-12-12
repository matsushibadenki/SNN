# ファイルパス: snn_research/models/bio/pd14_microcircuit.py
# Title: Potjans-Diesmann Cortical Microcircuit Model (PD14) - Re-Tuned for Stability
# Description:
#   doc/assignment.md 第3章に基づき、大脳皮質1mm^2の微小回路モデルを実装。
#   修正 (v3):
#   - Scenario Bでの過剰発火(てんかん状態)を防ぐため、NMDAゲインを抑制(0.3 -> 0.15)。
#   - 抑制性結合(Inhibitory)の重みをさらに強化してE/Iバランスを安定化。
#   - TwoCompartmentLIFの閾値を調整。

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, cast
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
                # チューニング v3: NMDAゲインをさらに下げて過剰なバーストを防ぐ
                nmda_gain = 0.15 if 'e' in name else 0.05
                self.populations[name] = TwoCompartmentLIF(
                    features=count, 
                    nmda_gain=nmda_gain,
                    tau_soma=20.0,
                    v_threshold=1.0,
                    # NMDA発火の閾値を少し上げる (容易にバーストしないように)
                    nmda_threshold=1.2 
                )
            else:
                # 標準LIF
                self.populations[name] = AdaptiveLIFNeuron(features=count)
                
        # --- 2. 接続性の定義 (Connectivity) ---
        # PD14の接続確率行列 (簡略版: From -> To)
        pop_names = ["L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
        
        self.connections = nn.ModuleDict()
        
        for src in pop_names:
            for tgt in pop_names:
                prob = self._get_connection_prob(src, tgt)
                if prob > 0:
                    src_dim = self.pop_counts[src]
                    tgt_dim = self.pop_counts[tgt]
                    
                    # 線形層を作成 (バイアスなし)
                    layer = nn.Linear(src_dim, tgt_dim, bias=False)
                    
                    # チューニング: 重みの初期化をスケーリング (1/sqrt(N))
                    limit = 1.0 / np.sqrt(src_dim)
                    
                    with torch.no_grad():
                        # マスク作成 (確率的接続)
                        mask = (torch.rand(tgt_dim, src_dim) < prob).float()
                        
                        # 重み初期化
                        if 'i' in src:
                            # チューニング v3: 抑制性結合をさらに強化 (x8.0) して暴走を止める
                            nn.init.uniform_(layer.weight, -limit * 8.0, -limit * 2.0)
                        else:
                            # 興奮性: 正の重み (少し控えめに)
                            nn.init.uniform_(layer.weight, limit * 0.05, limit * 0.8)
                            
                        # 接続がない部分は0にする
                        layer.weight *= mask
                        
                    conn_name = f"{src}_to_{tgt}"
                    self.connections[conn_name] = layer

        # --- 3. 外部入出力 ---
        # 重み初期化も同様にスケーリング
        in_limit = 1.0 / np.sqrt(input_dim)
        
        self.thalamic_input_L4 = nn.Linear(input_dim, self.pop_counts["L4e"])
        nn.init.uniform_(self.thalamic_input_L4.weight, in_limit * 0.5, in_limit * 2.0)
        
        self.thalamic_input_L6 = nn.Linear(input_dim, self.pop_counts["L6e"])
        nn.init.uniform_(self.thalamic_input_L6.weight, in_limit * 0.5, in_limit * 2.0)
        
        self.feedback_input_L23 = nn.Linear(input_dim, self.pop_counts["L23e"])
        # トップダウン入力の重みも少し抑える
        nn.init.uniform_(self.feedback_input_L23.weight, in_limit * 0.2, in_limit * 1.5)
        
        # 出力 (Readout) -> L5e (主要出力層) から
        self.readout = nn.Linear(self.pop_counts["L5e"], output_dim)
        
        logger.info(f"🧠 PD14 Microcircuit initialized. Total Neurons: {sum(self.pop_counts.values())}")

    def _get_connection_prob(self, src: str, tgt: str) -> float:
        """
        PD14モデルに基づき、2つの集団間の接続確率を返す。
        """
        # 自己結合・同層内
        if src == tgt: return 0.05
        if src[:3] == tgt[:3]: return 0.1 
        
        # 正準回路の流れ
        if "L4" in src and "L23" in tgt: return 0.15
        if "L23" in src and "L5" in tgt: return 0.15
        if "L5" in src and "L6" in tgt: return 0.1
        if "L6" in src and "L4" in tgt: return 0.05
        
        # 抑制性結合は広範囲
        if 'i' in src: return 0.2
        
        return 0.01

    def forward(
        self, 
        thalamic_input: torch.Tensor, # (Batch, InputDim)
        topdown_input: Optional[torch.Tensor] = None # (Batch, InputDim)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        B = thalamic_input.shape[0]
        device = thalamic_input.device
        
        SJ_F.reset_net(self)
        
        # mypy対応 (castを使用)
        for mod in self.populations.values():
            if hasattr(mod, 'set_stateful'):
                cast(Any, mod).set_stateful(True)
            
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
                # 1. 内部入力の集計
                internal_current = torch.zeros(B, self.pop_counts[name], device=device)
                
                for src_name, src_spikes in prev_spikes.items():
                    conn_name = f"{src_name}_to_{name}"
                    if conn_name in self.connections:
                        internal_current += self.connections[conn_name](src_spikes)
                
                # 2. 外部入力の加算
                external_current_soma = torch.zeros_like(internal_current)
                external_current_dend = torch.zeros_like(internal_current)
                
                if name == "L4e":
                    external_current_soma += self.thalamic_input_L4(thalamic_input)
                if name == "L6e":
                    external_current_soma += self.thalamic_input_L6(thalamic_input)
                    
                if topdown_input is not None:
                    if name == "L23e":
                        external_current_dend += self.feedback_input_L23(topdown_input)
                
                # 3. ニューロン発火
                if isinstance(neuron_layer, TwoCompartmentLIF):
                    s_input = internal_current + external_current_soma
                    d_input = external_current_dend
                    spikes, _ = neuron_layer(input_soma=s_input, input_dend=d_input)
                else:
                    total_input = internal_current + external_current_soma + external_current_dend
                    spikes, _ = neuron_layer(total_input)
                
                current_spikes[name] = spikes
                spike_counts[name] += spikes.sum().item() / B

            readout_accum += self.readout(current_spikes["L5e"])
            prev_spikes = current_spikes

        # mypy対応 (castを使用)
        for mod in self.populations.values():
            if hasattr(mod, 'set_stateful'):
                cast(Any, mod).set_stateful(False)

        # 平均発火率の計算
        avg_firing_rates = {
            k: v / self.time_steps for k, v in spike_counts.items()
        }
        
        return readout_accum, avg_firing_rates
