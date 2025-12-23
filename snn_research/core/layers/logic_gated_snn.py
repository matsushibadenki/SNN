# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー
# 目的: 行列演算をビット論理演算に置き換え、生体のような構造的可塑性（配線の結合・解離）をシミュレートする。

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class LogicGatedSNN(nn.Module):
    """
    Logic-Gated Spiking Neural Network Layer.
    
    特徴:
    1. Dendritic Logic: 入力スパイクに対して、Tsetlin Machine的な論理ゲートを適用。
    2. 1.58-bit Weights: シナプス結合は {-1, 0, 1} の3値。
    3. Structural Plasticity: 重みの更新は「状態」のインクリメント/デクリメントで行う (BP不要)。
    """
    def __init__(self, in_features: int, out_features: int, max_states: int = 100):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 各シナプスの内部「状態」 (1〜max_states)
        # 初期状態は閾値付近 (結合するかしないかの境界)
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        # 膜電位
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.spike_threshold = 1.0

    def get_ternary_weights(self) -> torch.Tensor:
        """
        内部状態を 1.58ビット (-1, 0, 1) の重みに変換。
        状態 > 閾値 なら 1 (Include)、それ以外は 0 (Exclude)。
        ※本実装では簡略化のため 0, 1 の2値(配線の有無)として扱う。
        """
        return (self.synapse_states > self.threshold).float()

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        論理演算ベースの順伝播:
        1. 重みの3値化
        2. 入力スパイクとの AND/累積 (行列積の排除)
        3. 発火判定
        """
        # 重みをバイナリ/3値として取得
        w = self.get_ternary_weights()
        
        # 行列積 (GEMM) ではなく、スパイクが存在するインデックスのみを加算 (累積)
        # デジタル回路では単純な Accumulation に相当
        if spike_input.dim() == 1:
            spike_input = spike_input.unsqueeze(0)
        
        # 出力電流の計算
        current = torch.matmul(spike_input, w.t())
        
        # 膜電位の更新と発火
        self.membrane_potential += current.squeeze(0)
        spikes = (self.membrane_potential >= self.spike_threshold).float()
        
        # 発火したニューロンの電位をリセット
        self.membrane_potential[spikes > 0] = 0.0
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        構造的可塑性の更新則 (Local Structural Learning):
        - ポストが発火し、プレも発火していた場合 -> 結合状態を強化 (状態+)
        - ポストが発火したが、プレが発火していなかった場合 -> 無駄な結合とみなし弱体化 (状態-)
        - ホメオスタシス: 過剰な発火は全体の閾値を上げる
        """
        with torch.no_grad():
            # プレとポストの相関 (Hebbian Count)
            # (out, 1) * (1, in) -> (out, in)
            correlation = torch.matmul(post_spikes.unsqueeze(1), pre_spikes.unsqueeze(0))
            
            # 強化: 共発火したシナプスの状態を上げる
            self.synapse_states += correlation
            
            # 抑圧: ポストだけが発火した原因でないシナプスの状態を下げる
            # (ポスト発火 1 かつ プレ不発火 0 の箇所)
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            self.synapse_states[depression_mask] -= 0.5
            
            # 状態のクリッピング (オートマトンの境界)
            self.synapse_states.clamp_(1, self.max_states)

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, threshold={self.threshold}'