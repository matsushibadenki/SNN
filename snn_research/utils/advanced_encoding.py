# ファイルパス: snn_research/utils/advanced_encoding.py
# タイトル: SDRエンコーダ (Sparse Distributed Representation)
# 内容: k-WTAを用いたスパース表現生成により、強力なノイズ除去を実現

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparsePatternSeparator(nn.Module):
    """
    SDR生成モジュール
    高次元空間へ射影後、k-Winner-Take-All (k-WTA) を適用して
    ノイズに強いスパース分散表現を作成する。
    """
    
    def __init__(
        self,
        input_dim: int,
        expansion_ratio: int = 4,  # 3 -> 4 (表現力確保のため少し戻す)
        sparsity: float = 0.15,    # 重み行列の疎性
        activity_sparsity: float = 0.1 # k-WTAの活性化率 (上位10%のみ残す)
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim * expansion_ratio
        self.activity_sparsity = activity_sparsity
        
        # 固定ランダム重み
        weights = torch.randn(self.output_dim, input_dim)
        mask = (torch.rand_like(weights) < sparsity).float()
        
        # スケーリング
        weights = weights * mask * (1.5 / math.sqrt(input_dim * sparsity))
        
        self.register_buffer('projection_weights', weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # バイポーラ入力
        if x.min() >= 0:
            x_bipolar = (x - 0.5) * 2.0
        else:
            x_bipolar = x
            
        # 線形射影
        projected = torch.matmul(x_bipolar, self.projection_weights.t())
        
        # === k-WTA (k-Winner-Take-All) ===
        # 上位k個の強い反応だけを残し、それ以外（ノイズ成分）をゼロにする
        k = int(self.output_dim * self.activity_sparsity)
        if k < 1: k = 1
        
        # top-kの値を取得
        top_values, _ = torch.topk(projected, k, dim=1)
        # k番目の値を閾値とする
        threshold = top_values[:, -1].unsqueeze(1)
        
        # 閾値以下の活動をゼロにするマスク
        # ReLUも含めて、正の強い相関のみを通す
        mask = (projected >= threshold).float()
        encoded = F.relu(projected) * mask
        
        # 正規化
        encoded = F.normalize(encoded, p=2, dim=1)
        
        return encoded

class HybridEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        use_multiscale: bool = True,
        use_error_correction: bool = False,
        use_adaptive_contrast: bool = False
    ):
        super().__init__()
        # SDRエンコーダを使用
        self.separator = SparsePatternSeparator(
            input_dim, 
            expansion_ratio=4, 
            sparsity=0.15,
            activity_sparsity=0.1 # 上位10%のみ活性化
        )
        self.output_dim = self.separator.output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.separator(x)