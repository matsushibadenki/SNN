# ファイルパス: snn_research/core/layers/frequency_encoding.py
# Title: Frequency Encoding Layer (FEEL)
# Description:
#   FEEL-SNN (NeurIPS 2024) に基づく、周波数領域エンコーディング層。
#   入力画像に対してFFT（高速フーリエ変換）を適用し、
#   時間ステップごとに異なる周波数帯域をスキャンして特徴を抽出する。

import torch
import torch.nn as nn
import torch.fft
import math

class FrequencyEncodingLayer(nn.Module):
    """
    画像を周波数領域に変換し、時間ステップに応じて特定の周波数帯域を通過させるエンコーダ。
    """
    def __init__(self, time_steps: int, encoding_method: str = "fft"):
        super().__init__()
        self.time_steps = time_steps
        self.encoding_method = encoding_method
        self.register_buffer('mask', None)

    def _create_bandpass_mask(self, height: int, width: int, time_step: int, total_steps: int) -> torch.Tensor:
        """
        特定の周波数帯域を通すマスクを生成する。
        低周波（中心）から高周波（外側）へとスキャンする。
        """
        mask = torch.zeros((height, width))
        
        # 中心（低周波）からの距離に基づくバンドパスフィルタ
        # FFTの出力は、通常 DC成分が隅にあるが、ここではシフトなしを想定して左上を低周波として扱う簡易実装
        # (厳密には fftshift が必要だが、計算コスト削減のため象限を考慮して生成)
        
        y = torch.arange(height).unsqueeze(1).float()
        x = torch.arange(width).unsqueeze(0).float()
        
        # 正規化された周波数距離 (0.0 ~ 1.414)
        dist = torch.sqrt((y / height)**2 + (x / width)**2)
        
        # 帯域幅
        max_dist = math.sqrt(2) # 対角線
        band_width = max_dist / total_steps
        
        # 現在のステップが担当する距離範囲
        # 低周波から開始
        start_dist = time_step * band_width * 0.8 # 少しオーバーラップさせる
        end_dist = (time_step + 1) * band_width * 1.2
        
        mask[(dist >= start_dist) & (dist < end_dist)] = 1.0
        
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力画像 (Batch, Channels, Height, Width)
        
        Returns:
            torch.Tensor: 周波数エンコードされた時系列特徴 (Batch, Time, Channels, Height, Width)
        """
        B, C, H, W = x.shape
        device = x.device
        
        outputs = []
        
        # 1. 周波数変換 (FFT)
        # rfft2 は実数入力用で、幅が W/2+1 になる。情報を維持するため fft2 を使用。
        freq_repr = torch.fft.fft2(x, norm="ortho")
        
        # 振幅スペクトルを使用 (位相情報は今回は破棄または単純化)
        freq_mag = torch.abs(freq_repr)
        freq_phase = torch.angle(freq_repr)
        
        # 2. 時間方向への展開とフィルタリング
        for t in range(self.time_steps):
            # マスク生成
            mask = self._create_bandpass_mask(H, W, t, self.time_steps).to(device)
            mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            # フィルタリング (振幅のみ変調)
            filtered_mag = freq_mag * mask
            
            # 逆変換 (空間領域に戻す)
            # フィルタリングされた振幅と元の位相を組み合わせて復元
            reconstructed_complex = torch.polar(filtered_mag, freq_phase)
            spatial_out = torch.fft.ifft2(reconstructed_complex, norm="ortho").real
            
            outputs.append(spatial_out)

        # (Batch, Time, C, H, W)
        return torch.stack(outputs, dim=1)