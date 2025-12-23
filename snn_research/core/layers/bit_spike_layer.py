# ファイルパス: snn_research/core/layers/bit_spike_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BitSpikeWeightQuantizer(torch.autograd.Function):
    """
    BitNet b1.58に準拠した重み量子化関数 (AbsMean Quantization)。
    Forward: 重みを {-1, 0, 1} * gamma に量子化する。
    Backward: Straight-Through Estimator (STE) を用いて勾配を通過させる。
    """
    @staticmethod
    def forward(ctx, weight, eps=1e-5):
        # ドキュメントに基づき、重みの絶対値の平均をスケーリング係数 gamma とする
        # 数値安定性のために epsilon を加算
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        
        # スケーリング
        scaled_weight = weight / gamma
        
        # {-1, 0, 1} に丸める
        # round() は 0.5 を最も近い偶数に丸める挙動があるが、ここでは単純な四捨五入的な挙動を期待
        # clampして範囲を制限してからroundする
        quantized_weight = torch.round(torch.clamp(scaled_weight, -1.0, 1.0))
        
        # スケールを戻して返す
        # これにより、出力のスケールは元の重みのスケールと整合する
        return quantized_weight * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        # 量子化関数は微分不可能だが、勾配をそのまま流すことで学習を可能にする
        return grad_output, None

def bit_quantize_weight(weight, eps=1e-5):
    return BitSpikeWeightQuantizer.apply(weight, eps)

class BitSpikeLinear(nn.Linear):
    """
    BitNet b1.58 アーキテクチャを採用した線形層。
    推論時は乗算フリー（加算・減算のみ）の演算が可能になるように重みを制約する。
    SNNにおいては、入力スパイク(0/1)との積和演算が、重みの条件付き加算に還元されるため、
    極めて高いエネルギー効率を実現する。
    """
    def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_inference = quantize_inference
        self.eps = 1e-5
        # 初期化: BitNetは重みのスケールに敏感なため、Kaiming Uniformで分散を保つ
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 学習時および推論時ともに量子化重みを使用する (Quantization-Aware Training)
        # これにより、学習された重みは {-gamma, 0, +gamma} の分布に収束していく
        w_quant = bit_quantize_weight(self.weight, self.eps)
        
        # 線形変換
        # x がスパイク入力 (0 or 1) の場合、実質的に w_quant の加算のみとなる
        return F.linear(x, w_quant, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantize={self.quantize_inference}'
