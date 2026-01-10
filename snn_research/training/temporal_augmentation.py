# ファイルパス: snn_research/training/temporal_augmentation.py
# 日本語タイトル: SNN特化型 時間方向データ拡張 v1.0
# 目的・内容:
#   ROADMAP Phase 2.4 「CIFAR-10精度96%への到達」対応。
#   SNNの時間方向の特性を活かしたデータ拡張を実装。
#   - TemporalJitter: スパイク時刻の揺らぎ
#   - TemporalDrop: ランダムな時間ステップ削除
#   - TemporalShift: 時間方向シフト
#   - TemporalMixup: 時間方向でのMixup

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import random


class TemporalJitter(nn.Module):
    """
    スパイク時刻にランダムな揺らぎを加える。
    生物学的なスパイクタイミングの変動を模倣。
    """

    def __init__(self, jitter_sigma: float = 0.5, max_shift: int = 2):
        """
        Args:
            jitter_sigma: ジッターの標準偏差（時間ステップ単位）
            max_shift: 最大シフト量
        """
        super().__init__()
        self.jitter_sigma = jitter_sigma
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: スパイクテンソル [Batch, Time, ...] または [Time, ...]

        Returns:
            ジッターを適用したテンソル
        """
        if not self.training:
            return x

        # 時間次元を特定
        has_batch = x.dim() >= 3

        if has_batch:
            B, T = x.shape[:2]
        else:
            T = x.shape[0]
            B = 1
            x = x.unsqueeze(0)

        # 各時間ステップに対してランダムシフト
        shifts = torch.round(
            torch.randn(B, T, device=x.device) * self.jitter_sigma
        ).long()
        shifts = torch.clamp(shifts, -self.max_shift, self.max_shift)

        # シフト適用
        result = torch.zeros_like(x)
        for b in range(B):
            for t in range(T):
                new_t = t + shifts[b, t].item()
                if 0 <= new_t < T:
                    result[b, int(new_t)] = x[b, t]

        if not has_batch:
            result = result.squeeze(0)

        return result


class TemporalDrop(nn.Module):
    """
    ランダムな時間ステップをドロップ（ゼロ化）する。
    スパイク欠落やセンサーノイズを模倣。
    """

    def __init__(self, drop_prob: float = 0.1, contiguous: bool = False):
        """
        Args:
            drop_prob: ドロップ確率
            contiguous: Trueなら連続した時間ステップをドロップ
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: スパイクテンソル [Batch, Time, ...] または [Time, ...]
        """
        if not self.training:
            return x

        has_batch = x.dim() >= 3
        if not has_batch:
            x = x.unsqueeze(0)

        B, T = x.shape[:2]

        if self.contiguous:
            # 連続ドロップ: ランダムな開始点から一定範囲をドロップ
            drop_len = max(1, int(T * self.drop_prob))
            mask = torch.ones(B, T, device=x.device)

            for b in range(B):
                if random.random() < self.drop_prob * 3:  # 適用確率
                    start = random.randint(0, T - drop_len)
                    mask[b, start:start + drop_len] = 0

            # マスクを空間次元に拡張
            for _ in range(x.dim() - 2):
                mask = mask.unsqueeze(-1)

            result = x * mask
        else:
            # 独立ドロップ
            mask = (torch.rand(B, T, device=x.device) > self.drop_prob).float()
            for _ in range(x.dim() - 2):
                mask = mask.unsqueeze(-1)
            result = x * mask

        if not has_batch:
            result = result.squeeze(0)

        return result


class TemporalShift(nn.Module):
    """
    時間方向に全体をシフトする。
    刺激タイミングの変動を模倣。
    """

    def __init__(self, max_shift: int = 3, pad_mode: str = "zero"):
        """
        Args:
            max_shift: 最大シフト量
            pad_mode: パディングモード ("zero", "replicate", "wrap")
        """
        super().__init__()
        self.max_shift = max_shift
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, ...] または [Time, ...]
        """
        if not self.training:
            return x

        has_batch = x.dim() >= 3
        if not has_batch:
            x = x.unsqueeze(0)

        shift = random.randint(-self.max_shift, self.max_shift)

        if shift == 0:
            result = x
        else:
            result = torch.zeros_like(x)

            if shift > 0:
                # 右シフト（未来に移動）
                result[:, shift:] = x[:, :-shift]
                if self.pad_mode == "replicate":
                    result[:, :shift] = x[:,
                                          0:1].expand(-1, shift, *x.shape[2:])
                elif self.pad_mode == "wrap":
                    result[:, :shift] = x[:, -shift:]
            else:
                # 左シフト（過去に移動）
                shift = -shift
                result[:, :-shift] = x[:, shift:]
                if self.pad_mode == "replicate":
                    result[:, -shift:] = x[:, -
                                           1:].expand(-1, shift, *x.shape[2:])
                elif self.pad_mode == "wrap":
                    result[:, -shift:] = x[:, :shift]

        if not has_batch:
            result = result.squeeze(0)

        return result


class TemporalMixup(nn.Module):
    """
    時間方向でのMixup拡張。
    異なるサンプルの時間ステップを混合する。
    """

    def __init__(self, alpha: float = 1.0, time_aware: bool = True):
        """
        Args:
            alpha: Beta分布のパラメータ
            time_aware: Trueなら時間ステップごとに異なるλを使用
        """
        super().__init__()
        self.alpha = alpha
        self.time_aware = time_aware

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        y2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: 入力1 [Batch, Time, ...]
            y: ラベル1 [Batch] or [Batch, Classes]
            x2: 入力2（Noneならバッチ内シャッフル）
            y2: ラベル2

        Returns:
            mixed_x, y1, y2, lam
        """
        if not self.training:
            return x, y, y, 1.0

        B, T = x.shape[:2]

        # λの生成
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 時間対応Mixup
        if self.time_aware:
            # 時間ステップごとにλを変える
            lam_t = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(1, T))
            ).float().to(x.device)
            for _ in range(x.dim() - 2):
                lam_t = lam_t.unsqueeze(-1)
        else:
            # 修正: Mypyエラー対応のためTensor型にキャスト
            lam_t = torch.tensor(lam, device=x.device, dtype=x.dtype)

        # x2がない場合はバッチ内シャッフル
        if x2 is None:
            indices = torch.randperm(B, device=x.device)
            x2 = x[indices]
            y2 = y[indices]

        # y2がNoneの場合の対応
        if y2 is None:
            y2 = y

        # Mixup
        mixed_x = lam_t * x + (1 - lam_t) * x2

        return mixed_x, y, y2, float(lam_t.mean() if isinstance(lam_t, torch.Tensor) else lam_t)


class TemporalAugmentationPipeline(nn.Module):
    """
    時間方向拡張のパイプライン。
    複数の拡張を順番に適用する。
    """

    def __init__(
        self,
        jitter_sigma: float = 0.5,
        drop_prob: float = 0.1,
        max_shift: int = 2,
        enable_jitter: bool = True,
        enable_drop: bool = True,
        enable_shift: bool = True
    ):
        super().__init__()

        self.augmentations = nn.ModuleList()

        if enable_jitter:
            self.augmentations.append(
                TemporalJitter(jitter_sigma=jitter_sigma))

        if enable_drop:
            self.augmentations.append(TemporalDrop(drop_prob=drop_prob))

        if enable_shift:
            self.augmentations.append(TemporalShift(max_shift=max_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        全ての拡張を順番に適用する。
        """
        for aug in self.augmentations:
            x = aug(x)
        return x


# 画像→スパイクテンソル変換用のユーティリティ
def image_to_spike_tensor(
    images: torch.Tensor,
    time_steps: int = 8,
    coding: str = "rate"
) -> torch.Tensor:
    """
    画像テンソルをスパイクテンソルに変換する。

    Args:
        images: [Batch, Channels, H, W] の画像テンソル (0-1正規化済み)
        time_steps: 時間ステップ数
        coding: "rate" (レートコーディング) または "latency" (レイテンシコーディング)

    Returns:
        spike_tensor: [Batch, Time, Channels, H, W]
    """
    B, C, H, W = images.shape

    if coding == "rate":
        # レートコーディング: 値が高いほど発火確率が高い
        probs = images.unsqueeze(1).expand(-1, time_steps, -1, -1, -1)
        spikes = (torch.rand_like(probs) < probs).float()

    elif coding == "latency":
        # レイテンシコーディング: 値が高いほど早く発火
        # 各ピクセルの発火タイミングを計算
        latency = ((1 - images) * (time_steps - 1)).long()
        spikes = torch.zeros(B, time_steps, C, H, W, device=images.device)

        for t in range(time_steps):
            spikes[:, t] = (latency <= t).float()
            # 一度発火したら次は発火しない（バースト抑制）
            if t > 0:
                spikes[:, t] = spikes[:, t] - spikes[:, t - 1]
                spikes[:, t] = torch.clamp(spikes[:, t], 0, 1)

    else:
        raise ValueError(f"Unknown coding: {coding}")

    return spikes


def spike_tensor_to_image(
    spikes: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    スパイクテンソルを画像テンソルに変換する（可視化用）。

    Args:
        spikes: [Batch, Time, Channels, H, W]
        method: "mean", "sum", "first"

    Returns:
        images: [Batch, Channels, H, W]
    """
    if method == "mean":
        return spikes.mean(dim=1)
    elif method == "sum":
        return spikes.sum(dim=1).clamp(0, 1)
    elif method == "first":
        # 最初のスパイク時刻を強度に変換
        first_spike = spikes.argmax(dim=1).float()
        return 1.0 - (first_spike / spikes.shape[1])
    else:
        raise ValueError(f"Unknown method: {method}")
