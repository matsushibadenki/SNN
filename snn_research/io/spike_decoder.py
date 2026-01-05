# ファイルパス: snn_research/io/spike_decoder.py
# (新規作成)
#
# Title: スパイクデコーダ
# Description:
# - SNNの出力スパイク列から、分類ラベルや回帰値（アナログ値）を復元する。
# - Rate Decoding, Membrane Potential Decoding など。
#
# mypy --strict 準拠。

import torch
import torch.nn as nn


class SpikeDecoder(nn.Module):
    """スパイクデコーダの基底クラス"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RateDecoder(SpikeDecoder):
    """
    レートデコーディング:
    指定された時間窓内の平均発火率を出力とする。
    """

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (B, T, ...)
        # out: (B, ...)
        return spikes.mean(dim=1)


class SumDecoder(SpikeDecoder):
    """
    合計デコーディング:
    総スパイク数を出力とする。
    """

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        return spikes.sum(dim=1)


class MembranePotentialDecoder(SpikeDecoder):
    """
    膜電位デコーディング:
    最終層のニューロンの膜電位を直接出力として使用する。
    (これはSNNモデル自体が膜電位を返す構造になっている必要があるため、
     デコーダとしては「膜電位を受け取って整形する」役割になるか、
     あるいは非発火の積分ニューロンを内部に持つ形になる)

    ここでは、「出力層のスパイクを入力として受け取り、積分してアナログ値にする」
    Readoutニューロンとして実装する。
    """

    def __init__(self, num_outputs: int, tau: float = 10.0) -> None:
        super().__init__()
        self.tau = tau
        # 学習可能な重みを持つ場合は nn.Linear を使うが、
        # ここでは単純な leaky integrator とする

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (B, T, N)
        B, T, N = spikes.shape

        mem = torch.zeros(B, N, device=spikes.device)
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=spikes.device))

        # 最後のステップの膜電位、あるいは最大膜電位、あるいは全ステップの平均膜電位などを返す
        # ここでは「最終ステップの膜電位」を返す

        for t in range(T):
            spike_t = spikes[:, t, :]
            mem = mem * decay + spike_t

        return mem


class FirstToSpikeDecoder(SpikeDecoder):
    """
    First-To-Spike (FTS) デコーディング:
    最も早く発火したニューロンのインデックス、またはその時刻に基づく値を出力する。
    分類タスクで使用され、発火時刻が早いクラスを予測とする。
    """

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (B, T, N)
        # 戻り値: (B, N) のスコア (早いほど大きい値)

        B, T, N = spikes.shape

        # 各ニューロンの最初の発火時刻を見つける
        # 発火していない場合は T (最大遅延) とする

        # 時間軸で累積和をとり、0より大きくなった最初のインデックスを探す
        cumsum = spikes.cumsum(dim=1)

        # has_fired が True になった最初のインデックスを取得したい
        # argmax は最初の True のインデックスを返す (すべて False なら 0 だが...)
        # 確実に処理するため、has_fired を float にして重み付けするなどの工夫が必要

        # 方法: T - t_first をスコアとする
        # 未発火なら 0

        # 時間ごとの重み (T, T-1, ..., 1)
        time_weights = torch.arange(
            T, 0, -1, device=spikes.device).view(1, T, 1)

        # 各時刻のスパイクに重みを掛ける。
        # 最初のスパイクだけを考慮したい -> has_fired の変化点を使う

        first_spike_mask = torch.zeros_like(spikes)
        first_spike_mask[:, 0, :] = spikes[:, 0, :]
        # t > 0: spike[t] == 1 AND sum(spike[0:t]) == 0
        # 簡易的に: cumsum == 1 AND spike == 1
        mask = (cumsum == 1) & (spikes > 0.5)

        # 重み付き和をとる (1回しか発火しない前提ならこれでOK)
        scores = (mask.float() * time_weights).sum(dim=1)

        return scores
