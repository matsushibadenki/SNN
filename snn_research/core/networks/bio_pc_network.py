# snn_research/core/networks/bio_pc_network.py
# 生物学的予測符号化（Bio-PC）ネットワーク v2.0
#
# 変更点:
# - 双方向反復推論 (Iterative Bidirectional Inference) の実装。
# - トップダウン予測とボトムアップ誤差の統合管理。

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union, Callable, cast, Type

from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
from ..neurons import AdaptiveLIFNeuron

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク。
    Deep Predictive Coding のアーキテクチャを採用し、
    各層が状態(Representation)と誤差(Error)を保持して相互作用する。
    """
    def __init__(self, 
                 layer_sizes: List[int], 
                 sparsity: float = 0.05, 
                 input_gain: float = 1.0,
                 inference_steps: int = 8,  # 推論時の反復回数
                 neuron_class: Optional[Type[nn.Module]] = None,
                 neuron_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.input_gain = input_gain
        self.inference_steps = inference_steps
        
        self.neuron_class = neuron_class or AdaptiveLIFNeuron
        self.neuron_params = neuron_params or {"tau_mem": 20.0, "base_threshold": 1.0}

        self.pc_layers = nn.ModuleList()
        # PredictiveCodingLayerは「上位層の状態」から「下位層の入力」を予測するモジュール
        # Layer[i] connects Size[i] (bottom) and Size[i+1] (top)
        for i in range(len(layer_sizes) - 1):
            layer = PredictiveCodingLayer(
                layer_sizes[i],     # Bottom size (Prediction Target)
                layer_sizes[i+1],   # Top size (State Source)
                self.neuron_class,
                self.neuron_params,
                sparsity=sparsity,
                inference_steps=1   # Network側でループ制御するため層内ループは最小限に
            )
            self.pc_layers.append(layer)
            
        # 各層の状態保持用バッファ (推論ループ間で使用)
        # 実際にはforward内で動的に確保するが、構造を示すために記載
        # states[i] corresponds to layer_sizes[i]

    def reset_state(self) -> None:
        for m in self.modules():
            if m is self: continue
            reset_func = getattr(m, 'reset_state', None)
            if callable(reset_func):
                try:
                    reset_func()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Iterative Inference Process:
        1. 入力を最下層に固定。
        2. 全層の状態を更新 (Bottom-up Error & Top-down Prediction)。
        3. 指定ステップ数だけ繰り返し、平衡状態に近づける。
        """
        x = x * self.input_gain
        batch_size = x.size(0)
        device = x.device
        
        # 状態の初期化
        # states[i] は layer_sizes[i] の次元を持つ
        states = [torch.zeros(batch_size, size, device=device) for size in self.layer_sizes]
        
        # 最下層 (Sensory Layer) に入力をセット
        # Layer 0 は外部入力により固定（あるいは強いナッジを受ける）
        states[0] = x
        
        # 推論ループ (Relaxation)
        for t in range(self.inference_steps):
            # Error Propagation & State Update Logic
            # 予測誤差は下から上へ、予測は上から下へ流れる
            
            # 1. Calculate Errors and Bottom-Up Inputs
            # 各PC Layer i は state[i+1] を元に state[i] を予測する
            # forward返り値: (updated_state_top, error_bottom, mem)
            
            # 注意: PredictiveCodingLayer.forward(bottom_up_input, top_down_state)
            # bottom_up_input = states[i]
            # top_down_state = states[i+1]
            
            new_states = [s.clone() for s in states]
            
            # 最下層は入力で固定 (Clamp)
            new_states[0] = x 

            for i, layer in enumerate(self.pc_layers):
                # layer i connects state[i] (bottom) and state[i+1] (top)
                
                bottom_val = states[i]
                top_val = states[i+1]
                
                # 上位層からのさらに上の予測誤差があれば受け取るべきだが、
                # 現在のPredictiveCodingLayerは「自身の予測誤差」を出力し、
                # 「自身の状態」を更新する設計。
                # ここでは layer.forward を呼ぶことで state[i+1] を更新する
                
                updated_top, error_bottom, _ = layer(bottom_val, top_val)
                
                # Update top state (accumulate updates if multiple connections existed)
                # ここでは単純なチェーンなのでそのまま更新
                new_states[i+1] = updated_top
                
                # 誤差信号は本来、さらに下の層の状態更新に使われるべきだが、
                # PredictiveCodingLayer内で bottom_up_input (state[i]) と prediction の差分計算は済んでいる。
                # Deep PCでは、Layer i の更新には「Layer i-1からの誤差(Bottom-up)」と「Layer i+1からの予測(Top-down)」が必要。
                # 現在のlayer実装は "Inference Path" (Error -> State Update) を含んでいるため、
                # layer(state[i], state[i+1]) を呼ぶことで state[i+1] がボトムアップ誤差に基づいて更新される。
                
                # つまり、このループで下層から順に layer を呼ぶことで、
                # 誤差が上に伝播し、上位層の状態が更新されていく。
            
            states = new_states

        # 最終的な出力は最上位層の状態 (または分類タスクなら最上位の出力)
        return states[-1]

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            loss_attr = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_attr):
                total_loss += loss_attr()
            else:
                total_loss += cast(torch.Tensor, torch.as_tensor(loss_attr))
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
