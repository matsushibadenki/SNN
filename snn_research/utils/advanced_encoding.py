# ファイルパス: snn_research/utils/advanced_encoding.py
# タイトル: 空間認識ハイブリッドエンコーダ (Spatial-Aware Hybrid Encoder)
# 内容: 空間フィルタリングでノイズを除去してから、コントラスト強調とスパース射影を行う3段構成
# 修正: Mypy型エラー修正 (Tensor cast, .T property)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import cast


class SpatialDenoising(nn.Module):
    """
    空間的なつながりを利用してノイズを除去するモジュール。
    ランダムなノイズは孤立していることが多いが、数字は線でつながっていることを利用する。
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.side_len = int(math.sqrt(input_dim))

        # 1D入力を2D画像とみなして処理するための準備
        if self.side_len * self.side_len != input_dim:
            # 正方形でない場合は処理をスキップするためのフラグ
            self.skip = True
        else:
            self.skip = False

            # 3x3の平滑化フィルタ（ガウシアンライク）
            kernel = torch.tensor([
                [0.5, 1.0, 0.5],
                [1.0, 2.0, 1.0],
                [0.5, 1.0, 0.5]
            ]).unsqueeze(0).unsqueeze(0)
            kernel = kernel / kernel.sum()  # 正規化
            self.register_buffer('kernel', kernel)
            self.kernel: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return x

        batch_size = x.size(0)

        # [Batch, 784] -> [Batch, 1, 28, 28]
        x_img = x.view(batch_size, 1, self.side_len, self.side_len)

        # 畳み込みによる平滑化（ノイズ低減）
        # Explicit cast for mypy
        kernel_tensor = cast(torch.Tensor, self.kernel)
        x_smooth = F.conv2d(x_img, kernel_tensor, padding=1)

        # 元の入力との積を取ることで、「周囲も活性化しているピクセル」だけを残す
        # これにより孤立したノイズ（周囲が0）は抑制される
        x_filtered = x_img * x_smooth

        # [Batch, 784] に戻す
        return x_filtered.view(batch_size, -1)


class SparsePatternSeparator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_ratio: int = 3,  # 速度重視で3倍
        sparsity: float = 0.2,
        activity_sparsity: float = 0.15
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * expansion_ratio
        self.activity_sparsity = activity_sparsity

        weights = torch.randn(self.output_dim, input_dim)
        mask = (torch.rand_like(weights) < sparsity).float()
        weights = weights * mask * (1.5 / math.sqrt(input_dim * sparsity))
        self.register_buffer('projection_weights', weights)
        self.projection_weights: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # バイポーラ変換なし（フィルタリング済みの強度をそのまま使う）
        # .T uses the property which returns Tensor, avoiding "Tensor not callable" on .t() method calls in some envs
        proj_w = cast(torch.Tensor, self.projection_weights)
        projected = torch.matmul(x, proj_w.T)

        # k-WTA
        k = int(self.output_dim * self.activity_sparsity)
        if k < 1:
            k = 1

        top_values, _ = torch.topk(projected, k, dim=1)
        threshold = top_values[:, -1].unsqueeze(1)

        mask = (projected >= threshold).float()
        encoded = F.relu(projected) * mask
        encoded = F.normalize(encoded, p=2, dim=1)
        return encoded


class AdaptiveContrastEncoder(nn.Module):
    def __init__(self, input_dim, power=2.0):
        super().__init__()
        self.power = power

    def forward(self, x):
        # 信号を強調
        return x.pow(self.power)


class HybridEncoder(nn.Module):
    """
    3段階の処理を行う高性能エンコーダ
    1. 空間デノイズ: 孤立ノイズを除去
    2. コントラスト強調: 信号成分を増幅
    3. スパース射影: パターン分離
    """

    def __init__(
        self,
        input_dim: int,
        use_multiscale: bool = True,
        use_error_correction: bool = False,
        use_adaptive_contrast: bool = True
    ):
        super().__init__()

        # 1. 空間フィルタリング
        self.denoiser = SpatialDenoising(input_dim)

        # 2. コントラスト強調
        self.contrast = AdaptiveContrastEncoder(input_dim, power=2.0)

        # 3. スパース射影 (次元拡張)
        self.separator = SparsePatternSeparator(
            input_dim,
            expansion_ratio=4,  # ここで次元を増やす
            sparsity=0.2
        )

        self.output_dim = self.separator.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Denoise
        x = self.denoiser(x)

        # Step 2: Contrast
        x = self.contrast(x)

        # Step 3: Separate
        x = self.separator(x)

        return x
