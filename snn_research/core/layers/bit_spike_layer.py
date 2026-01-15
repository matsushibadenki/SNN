# snn_research/core/layers/bit_spike_layer.py
# Title: BitNet Quantization Layers (CPU Backend - Tuned)
# 修正内容: _pair関数のインポート追加と呼び出し修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, cast
from torch.nn.modules.utils import _pair  # [Fix] 内部ユーティリティをインポート

# [Refactor] 共通ロジックを ht8b_cpu からインポート
from snn_research.core.ops.ht8b_cpu import bit_quantize_weight


class BitSpikeLinear(nn.Linear):
    """
    BitNet Linear Layer.
    重みを1.58bitに量子化して演算を行う線形層。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, quantize_inference: bool = True):
        # 型安全性のためのキャスト
        in_features = int(in_features)
        out_features = int(out_features)
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_inference = quantize_inference
        self.eps = 1e-5
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 推論用キャッシュバッファ
        self.register_buffer('cached_w_quant', None)
        self.is_cached = False
        self.cached_w_quant: Optional[torch.Tensor]

    def train(self, mode: bool = True):
        super().train(mode)
        # 学習モードに戻ったらキャッシュを無効化
        if mode:
            self.is_cached = False
            self.cached_w_quant = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimization: Prioritize Inference Path
        if not self.training:
            if self.is_cached and self.cached_w_quant is not None:
                # キャッシュヒット: 量子化済み重みを使用
                return F.linear(x, self.cached_w_quant, self.bias)
            else:
                # キャッシュミス: 生成して保存
                with torch.no_grad():
                    self.cached_w_quant = bit_quantize_weight(
                        self.weight, self.eps)
                self.is_cached = True
                return F.linear(x, cast(torch.Tensor, self.cached_w_quant), self.bias)

        # Training Path (QAT: Quantization Aware Training)
        w_quant = bit_quantize_weight(self.weight, self.eps)
        return F.linear(x, w_quant, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, 1.58bit=True'


class BitSpikeConv2d(nn.Conv2d):
    """
    BitNet Conv2d Layer.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):

        # 型安全性のためのキャスト
        super().__init__(int(in_channels), int(out_channels), kernel_size,  # type: ignore
                         stride, padding, dilation, groups, bias, padding_mode)  # type: ignore

        self.eps = 1e-5
        nn.init.kaiming_normal_(
            self.weight, mode='fan_out', nonlinearity='relu')

        self.register_buffer('cached_w_quant', None)
        self.is_cached = False
        self.cached_w_quant: Optional[torch.Tensor]

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.is_cached = False
            self.cached_w_quant = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if self.is_cached and self.cached_w_quant is not None:
                w_quant = self.cached_w_quant
            else:
                with torch.no_grad():
                    self.cached_w_quant = bit_quantize_weight(
                        self.weight, self.eps)
                self.is_cached = True
                w_quant = cast(torch.Tensor, self.cached_w_quant)
        else:
            w_quant = bit_quantize_weight(self.weight, self.eps)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            w_quant, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)  # [Fix] self._pair -> _pair
        return F.conv2d(input, w_quant, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s + ', 1.58bit=True'
