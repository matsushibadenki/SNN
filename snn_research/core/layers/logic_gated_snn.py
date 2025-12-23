# ファイルパス: snn_research/core/layers/logic_gated_snn.py
# 日本語タイトル: 1.58ビット・ロジックゲート樹状突起レイヤー (完全型安全版)
# 目的: 構造的可塑性をシミュレートし、mypyの厳格な型チェックを完全にパスする実装を提供する。

import torch
import torch.nn as nn
from typing import cast

class LogicGatedSNN(nn.Module):
    """
    Logic-Gated Spiking Neural Network Layer.
    
    特徴:
    1. Dendritic Logic: 入力スパイクに対して論理ゲートを適用。
    2. 1.58-bit Weights: シナプス結合を3値化 (Include/Exclude)。
    3. Structural Plasticity: カウントベースの状態遷移による学習。
    """
    def __init__(self, in_features: int, out_features: int, max_states: int = 100) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_states = max_states
        self.threshold = max_states // 2
        
        # 内部状態バッファ: [out_features, in_features]
        self.register_buffer('synapse_states', torch.randint(
            self.threshold - 5, self.threshold + 5, (out_features, in_features)
        ).float())
        
        # 膜電位バッファ: [out_features]
        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.spike_threshold = 1.0

    @property
    def states(self) -> torch.Tensor:
        """バッファをTensor型としてキャストして取得"""
        return cast(torch.Tensor, self.synapse_states)

    @property
    def v_mem(self) -> torch.Tensor:
        """膜電位をTensor型としてキャストして取得"""
        return cast(torch.Tensor, self.membrane_potential)

    def get_ternary_weights(self) -> torch.Tensor:
        """状態に基づき 1.58ビット相当の重み(0/1)を生成"""
        # (states > threshold) は BoolTensor を返すため、明示的に float に変換
        mask = self.states > self.threshold
        return mask.to(torch.float32)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """順伝播実行"""
        # 重みの取得 (3値/バイナリ)
        w = self.get_ternary_weights()
        
        # 入力が [Batch, Features] であることを保証
        x = spike_input if spike_input.dim() > 1 else spike_input.unsqueeze(0)
        
        # 電流計算 (行列演算を論理的な累積として実行)
        current = torch.matmul(x, w.t())
        
        # 膜電位更新
        # squeeze(0)によりBatch次元を考慮したベクトル演算を行う
        new_v = self.v_mem + current.view(-1)
        
        # 発火判定 (1.0 または 0.0 のスパイクログ)
        spikes = (new_v >= self.spike_threshold).to(torch.float32)
        
        # リセット処理: 
        # mypyエラーを回避するため、演算結果を一度別のTensorに格納してからcopy_を使用
        # 発火した箇所(spikes=1)は 0 に、そうでない箇所は new_v を保持
        reset_mask = 1.0 - spikes
        updated_v = new_v * reset_mask
        self.v_mem.copy_(updated_v)
        
        return spikes

    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """局所的な構造的可塑性 (Structural Plasticity) の更新"""
        with torch.no_grad():
            # プレとポストの相関計算
            # ポスト(out,) と プレ(in,) の外積で [out, in] の更新行列を作成
            correlation = torch.outer(post_spikes, pre_spikes)
            
            # LTP (長期増強): 共発火したシナプスの状態をインクリメント
            self.states.add_(correlation)
            
            # LTD (長期抑圧): ポストだけが発火した原因でないシナプス(無駄な配線)を弱体化
            # (post == 1 かつ pre == 0) のマスクを作成
            depression_mask = (post_spikes.unsqueeze(1) > 0) & (pre_spikes.unsqueeze(0) == 0)
            
            # 弱体化の強度は 0.5 固定 (カウントベース)
            self.states[depression_mask] -= 0.5
            
            # 状態を [1, max_states] の範囲に制限
            self.states.clamp_(1, self.max_states)

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, threshold={self.threshold}'
