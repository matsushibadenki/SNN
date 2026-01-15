# ファイルパス: snn_research/cognitive_architecture/som_feature_map.py
# Title: Self-Organizing Feature Map (Robust)
# Description:
# - STDP学習結果の受け取り処理に安全対策を追加。
# - デバイス不整合の防止を追加。

import torch
import torch.nn as nn
from typing import Tuple

from snn_research.learning_rules.stdp import STDP

class SomFeatureMap(nn.Module):
    """
    STDPを用いて特徴を自己組織化する、単層のSNN。
    """
    def __init__(self, input_dim: int, map_size: Tuple[int, int], stdp_params: dict):
        super().__init__()
        self.input_dim = input_dim
        self.map_size = map_size
        self.num_neurons = map_size[0] * map_size[1]
        
        self.weights = nn.Parameter(torch.rand(self.input_dim, self.num_neurons))
        
        self.stdp = STDP(**stdp_params)
        
        self.neuron_pos = torch.stack(torch.meshgrid(
            torch.arange(map_size[0]),
            torch.arange(map_size[1]),
            indexing='xy'
        )).float().reshape(2, -1).T
        
        print(f"🗺️ 自己組織化マップが初期化されました ({map_size[0]}x{map_size[1]})。")

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        # デバイス同期
        if input_spikes.device != self.weights.device:
            input_spikes = input_spikes.to(self.weights.device)

        activation = input_spikes @ self.weights
        winner_index = torch.argmax(activation)
        
        output_spikes = torch.zeros(self.num_neurons, device=input_spikes.device)
        output_spikes[winner_index] = 1.0
        
        return output_spikes

    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        STDPと近傍学習則に基づき、重みを更新する。
        """
        # デバイス同期
        if pre_spikes.device != self.weights.device:
            pre_spikes = pre_spikes.to(self.weights.device)
        if post_spikes.device != self.weights.device:
            post_spikes = post_spikes.to(self.weights.device)

        winner_index = torch.argmax(post_spikes)
        
        # 1. 近傍関数
        # neuron_pos も同じデバイスにある必要がある
        if self.neuron_pos.device != self.weights.device:
            self.neuron_pos = self.neuron_pos.to(self.weights.device)
            
        distances = torch.linalg.norm(self.neuron_pos - self.neuron_pos[winner_index], dim=1)
        neighborhood_factor = torch.exp(-distances**2 / (2 * (self.map_size[0]/4)**2))
        
        # 2. STDPベースの重み更新 (安全対策追加)
        result = self.stdp.update(pre_spikes, post_spikes, self.weights.T)
        
        if result is None:
            # 学習則が何も返さなかった場合は更新スキップ
            return

        dw_transposed, _ = result
        dw = dw_transposed.T
        
        # 3. 近傍関数で学習率を変調
        # shape合わせ: neighborhood_factor (N_out) -> (1, N_out) or similar?
        # dw is (N_in, N_out), neighborhood_factor is (N_out)
        # Broadcasting: (N_in, N_out) * (N_out) -> Works
        
        modulated_dw = dw * neighborhood_factor
        
        self.weights.data += modulated_dw
        self.weights.data = torch.clamp(self.weights.data, 0, 1)