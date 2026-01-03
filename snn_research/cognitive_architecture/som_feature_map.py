# ファイルパス: snn_research/cognitive_architecture/som_feature_map.py
# (新規作成)
#
# Title: Self-Organizing Feature Map (自己組織化特徴マップ)
#
# Description:
# - ロードマップ「自己組織化する特徴マップ」を実装するコンポーネント。
# - STDP学習則に基づき、入力データから教師なしで特徴を学習し、
#   トポロジー的なマップを形成する。
# - 脳の視覚野などが持つ、特定の刺激に選択的に反応するニューロン群の
#   自己組織化プロセスを模倣する。

import torch
import torch.nn as nn
from typing import Tuple

from snn_research.learning_rules.stdp import STDP

class SomFeatureMap(nn.Module):
    """
    STDPを用いて特徴を自己組織化する、単層のSNN。
    """
    def __init__(self, input_dim: int, map_size: Tuple[int, int], stdp_params: dict):
        """
        Args:
            input_dim (int): 入力ベクトルの次元数。
            map_size (Tuple[int, int]): 特徴マップのサイズ (例: (10, 10))。
            stdp_params (dict): STDP学習則のパラメータ。
        """
        super().__init__()
        self.input_dim = input_dim
        self.map_size = map_size
        self.num_neurons = map_size[0] * map_size[1]
        
        # 全結合の重み
        self.weights = nn.Parameter(torch.rand(self.input_dim, self.num_neurons))
        
        # 学習則
        self.stdp = STDP(**stdp_params)
        
        # ニューロンの位置をグリッド上に保存
        self.neuron_pos = torch.stack(torch.meshgrid(
            torch.arange(map_size[0]),
            torch.arange(map_size[1]),
            indexing='xy'
        )).float().reshape(2, -1).T
        
        print(f"🗺️ 自己組織化マップが初期化されました ({map_size[0]}x{map_size[1]})。")

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        入力スパイクを受け取り、マップ上のニューロンの発火を計算する。

        Args:
            input_spikes (torch.Tensor): 単一タイムステップの入力スパイク (input_dim,)

        Returns:
            torch.Tensor: マップ上のニューロンの出力スパイク (num_neurons,)
        """
        # 1. 最も強く反応するニューロン（勝者）を見つける
        activation = input_spikes @ self.weights
        winner_index = torch.argmax(activation)
        
        # 2. Winner-Take-All (WTA): 勝者のみが発火
        output_spikes = torch.zeros(self.num_neurons, device=input_spikes.device)
        output_spikes[winner_index] = 1.0
        
        return output_spikes

    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        STDPと近傍学習則に基づき、重みを更新する。

        Args:
            pre_spikes (torch.Tensor): 入力層のスパイク (input_dim,)
            post_spikes (torch.Tensor): 出力層（マップ）のスパイク (num_neurons,)
        """
        winner_index = torch.argmax(post_spikes)
        
        # 1. 近傍関数: 勝者の周りのニューロンも学習に参加させる
        distances = torch.linalg.norm(self.neuron_pos - self.neuron_pos[winner_index], dim=1)
        neighborhood_factor = torch.exp(-distances**2 / (2 * (self.map_size[0]/4)**2))
        
        # 2. STDPベースの重み更新
        # STDPのdwは [post, pre] の形状を期待するため、転置して渡す
        dw_transposed, _ = self.stdp.update(pre_spikes, post_spikes, self.weights.T)
        dw = dw_transposed.T
        
        # 3. 近傍関数で学習率を変調
        modulated_dw = dw * neighborhood_factor
        
        self.weights.data += modulated_dw
        self.weights.data = torch.clamp(self.weights.data, 0, 1) # 重みを正規化