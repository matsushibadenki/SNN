# ファイルパス: snn_research/io/universal_encoder.py
# 日本語タイトル: Universal Spike Encoder (万能スパイクエンコーダ)
# 目的・内容:
#   Phase 8-6: 多様なモダリティ（画像、音声、テキスト、DVS）を
#   統一されたスパイク列形式 (Batch, Time, Features) に変換するインターフェース。
#   - Image: Rate Coding / Latency Coding
#   - Audio: Delta Coding (変化分検出)
#   - Text: Embedding -> Rate Coding
#   - DVS: 空間次元の平坦化 (Flatten)

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

class UniversalSpikeEncoder(nn.Module):
    """
    あらゆる入力データをSNN用のスパイク列に変換する統合エンコーダ。
    """
    def __init__(self, time_steps: int = 16, device: str = 'cpu'):
        super().__init__()
        self.time_steps = time_steps
        self.device = device
        
    def encode(self, data: torch.Tensor, modality: Literal['image', 'audio', 'text', 'dvs'], method: str = 'rate') -> torch.Tensor:
        """
        メインのエンコードメソッド。モダリティに応じて適切な処理にディスパッチする。
        
        Args:
            data: 入力データ
            modality: 'image', 'audio', 'text', 'dvs'
            method: エンコーディング手法 ('rate', 'latency', 'delta' など)
            
        Returns:
            spikes: (Batch, Time, Features) のバイナリスパイク列
        """
        if modality == 'image':
            return self._encode_image(data, method)
        elif modality == 'audio':
            return self._encode_audio(data, method)
        elif modality == 'text':
            return self._encode_text(data, method)
        elif modality == 'dvs':
            return self._encode_dvs(data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _encode_image(self, image: torch.Tensor, method: str) -> torch.Tensor:
        """
        画像 (Batch, Channels, H, W) または (Batch, Features) をスパイク化。
        """
        # 入力が画像形式 (B, C, H, W) の場合、フラット化して (B, Features) にするオプション
        # ここでは Liquid Brain 向けに (B, T, Features) への変換を主眼とするためフラット化
        if image.ndim == 4:
            B, C, H, W = image.shape
            flat_features = image.view(B, -1) # (B, C*H*W)
        else:
            flat_features = image # (B, Features)

        # 値を 0-1 に正規化 (想定)
        if flat_features.max() > 1.0:
            flat_features = flat_features / (flat_features.max() + 1e-8)

        if method == 'rate':
            # Rate Coding: 値を確率としてベルヌーイ試行
            # (B, F) -> (B, T, F)
            input_expanded = flat_features.unsqueeze(1).repeat(1, self.time_steps, 1)
            spikes = (torch.rand_like(input_expanded) < input_expanded).float()
            return spikes.to(self.device)
            
        elif method == 'latency':
            # Latency Coding: 値が大きいほど早い時間に発火
            # 値 1.0 -> t=0, 値 0.0 -> t=T
            # 簡易実装: 線形マッピング
            B, F = flat_features.shape
            spikes = torch.zeros(B, self.time_steps, F, device=self.device)
            
            # 発火タイミング計算: (1 - val) * T
            fire_times = ((1.0 - flat_features) * (self.time_steps - 1)).long()
            fire_times = torch.clamp(fire_times, 0, self.time_steps - 1)
            
            # Scatterで1を立てる
            # batch_indices: (B, F) -> (B, 1, F) -> broadcast
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, F)
            feature_indices = torch.arange(F, device=self.device).unsqueeze(0).expand(B, F)
            
            spikes[batch_indices, fire_times, feature_indices] = 1.0
            return spikes
            
        else:
            raise ValueError(f"Unknown image encoding method: {method}")

    def _encode_audio(self, waveform: torch.Tensor, method: str) -> torch.Tensor:
        """
        音声波形/スペクトログラム (Batch, Features) または (Batch, Time, Features) をスパイク化。
        method='delta': 時間変化分(diff)が閾値を超えたら発火
        """
        # 入力が (Batch, Features) の場合、それは静的特徴量とみなしてRate Coding
        if waveform.ndim == 2:
            return self._encode_image(waveform, 'rate') # 再利用
            
        # 入力が (Batch, Time, Features) の場合
        if waveform.ndim == 3:
            if method == 'delta':
                # Delta Coding: |x[t] - x[t-1]| > threshold
                threshold = 0.1 # パラメータ化すべきだが簡易実装
                
                # 時間微分の絶対値
                diff = torch.abs(waveform[:, 1:, :] - waveform[:, :-1, :])
                # 先頭に0を追加してサイズを合わせる
                zeros = torch.zeros_like(waveform[:, 0:1, :])
                diff = torch.cat([zeros, diff], dim=1)
                
                spikes = (diff > threshold).float()
                
                # 指定された time_steps に合わせる（リサンプリング or 切り出し）
                current_steps = spikes.shape[1]
                if current_steps != self.time_steps:
                    # 簡易的にスライスまたはゼロパディング
                    if current_steps > self.time_steps:
                        spikes = spikes[:, :self.time_steps, :]
                    else:
                        pad = torch.zeros(spikes.shape[0], self.time_steps - current_steps, spikes.shape[2], device=self.device)
                        spikes = torch.cat([spikes, pad], dim=1)
                        
                return spikes.to(self.device)
            else:
                # デフォルトは入力を確率とみなす
                return (torch.rand_like(waveform) < waveform).float().to(self.device)

        raise ValueError("Audio input shape must be (B, F) or (B, T, F)")

    def _encode_text(self, embedding: torch.Tensor, method: str) -> torch.Tensor:
        """
        テキスト埋め込みベクトル (Batch, EmbedDim) をスパイク化。
        通常はRate Codingを使用。
        """
        # 値の範囲が -1~1 や unbounded なので、Sigmoid等で 0-1 に収める
        normalized_emb = torch.sigmoid(embedding)
        return self._encode_image(normalized_emb, 'rate')

    def _encode_dvs(self, dvs_spikes: torch.Tensor) -> torch.Tensor:
        """
        DVS入力 (Batch, Time, C, H, W) を (Batch, Time, Features) に平坦化。
        DVSは既にスパイクなので、Encode処理は形状変換のみ。
        """
        if dvs_spikes.ndim != 5:
            raise ValueError("DVS input must be (B, T, C, H, W)")
            
        B, T, C, H, W = dvs_spikes.shape
        
        # 時間長調整
        if T != self.time_steps:
            # 簡易リサンプリング (実際は積分や補間が必要だが、ここではスライス/パッド)
            if T > self.time_steps:
                dvs_spikes = dvs_spikes[:, :self.time_steps, ...]
            else:
                pad = torch.zeros(B, self.time_steps - T, C, H, W, device=self.device)
                dvs_spikes = torch.cat([dvs_spikes, pad], dim=1)
        
        # (B, T, C*H*W) に変形
        flattened = dvs_spikes.reshape(B, self.time_steps, -1)
        return flattened.to(self.device)