# ファイルパス: snn_research/io/universal_encoder.py
# Title: Universal Spike Encoder (Video Support Fixed & Alias Added)
# 修正内容:
# 1. 'vision' を 'image' のエイリアスとして追加し、ValueErrorを回避。
# 2. _encode_image に 5次元入力(Batch, Time, C, H, W)のサポートを維持。

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalSpikeEncoder(nn.Module):
    def __init__(self, time_steps: int = 16, d_model: int = 64, device: str = 'cpu'):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.device = device

        # テキスト用: Byte Embedding
        self.byte_embedding = nn.Embedding(256, d_model)
        nn.init.orthogonal_(self.byte_embedding.weight)
        self.byte_embedding.weight.requires_grad = False

    def encode(self, data: torch.Tensor, modality: str, method: str = 'rate') -> torch.Tensor:
        """
        統合エンコードインターフェース
        """
        if data.device != torch.device(self.device):
            data = data.to(self.device)

        # モダリティのエイリアス処理
        if modality == 'vision':
            modality = 'image'
        elif modality == 'auditory':
            modality = 'audio'

        if modality == 'image':
            return self._encode_image(data, method)
        elif modality == 'audio':
            return self._encode_audio(data, method)
        elif modality == 'text':
            return self._encode_text_embedding(data, method)
        elif modality == 'dvs':
            return self._encode_dvs(data)
        elif modality == 'tactile': # 触覚用 (簡易実装)
            return self._encode_generic_sequence(data)
        elif modality == 'olfactory': # 嗅覚用 (簡易実装)
            return self._encode_generic_sequence(data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _encode_image(self, x: torch.Tensor, method: str) -> torch.Tensor:
        # Case 1: Video Input (Batch, Time, C, H, W)
        if x.dim() == 5:
            B, T_in, C, H, W = x.shape
            # 時間次元(1)を残し、空間次元(2,3,4)をフラット化 -> (B, T, Features)
            x_flat = x.flatten(2) 
            
            # 時間ステップの調整
            if T_in != self.time_steps:
                x_flat = x_flat.permute(0, 2, 1) # (B, F, T)
                x_flat = F.interpolate(x_flat, size=self.time_steps, mode='linear', align_corners=False)
                x_flat = x_flat.permute(0, 2, 1) # (B, T, F)
            
            # Rate Coding (各フレームの画素値を確率としてスパイク生成)
            spikes = (torch.rand_like(x_flat, device=self.device) < x_flat).float()
            return spikes

        # Case 2: Static Image (Batch, C, H, W) or (Batch, H, W)
        if x.dim() > 2:
            x = x.flatten(1)  # (B, F)

        batch, features = x.shape

        if method == 'latency':
            spike_times = ((1.0 - x) * (self.time_steps - 1)).long()
            spike_times = torch.clamp(spike_times, 0, self.time_steps - 1)

            spikes = torch.zeros(batch, self.time_steps,
                                 features, device=self.device)
            for b in range(batch):
                spikes[b, spike_times[b], torch.arange(features)] = 1.0
            return spikes

        else:  # rate
            x_expanded = x.unsqueeze(1).expand(-1, self.time_steps, -1)
            spikes = torch.rand_like(
                x_expanded, device=self.device) < x_expanded
            return spikes.float()

    def _encode_audio(self, x: torch.Tensor, method: str) -> torch.Tensor:
        # x: (B, Time_in, Feat)
        if x.dim() != 3:
            if x.dim() == 4:
                x = x.flatten(2)
            else:
                raise ValueError(
                    f"Audio input must be 3D tensor (B, T, F), got {x.shape}")

        B, T_in, num_features = x.shape

        # 時間方向のリサイズ
        if T_in != self.time_steps:
            x = x.permute(0, 2, 1)  # (B, F, T_in)
            x = F.interpolate(x, size=self.time_steps,
                              mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # (B, T_out, F)

        if method == 'delta':
            threshold = 0.05
            diff = torch.abs(x[:, 1:] - x[:, :-1])
            diff = torch.cat(
                [torch.zeros(B, 1, num_features, device=self.device), diff], dim=1)
            spikes = (diff > threshold).float()
            return spikes
        else:
            return (torch.rand_like(x) < x).float()
            
    def _encode_generic_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """触覚・嗅覚などの一般的な時系列データ (B, T, F) 用"""
        if x.dim() == 2: # (B, F) -> (B, 1, F)
             x = x.unsqueeze(1)
        
        B, T_in, F_dim = x.shape
        if T_in != self.time_steps:
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=self.time_steps, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)
            
        spikes = (torch.rand_like(x, device=self.device) < x).float()
        return spikes

    def _encode_text_embedding(self, x: torch.Tensor, method: str) -> torch.Tensor:
        probs = torch.sigmoid(x)
        probs_expanded = probs.unsqueeze(1).expand(-1, self.time_steps, -1)
        spikes = (torch.rand_like(probs_expanded, device=self.device)
                  < probs_expanded).float()
        return spikes

    def _encode_dvs(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, C, H, W = x.shape
        x_flat = x.flatten(2)

        if T_in != self.time_steps:
            x_flat = x_flat.permute(0, 2, 1)
            x_flat = F.interpolate(
                x_flat, size=self.time_steps, mode='nearest')
            x_flat = x_flat.permute(0, 2, 1)

        return x_flat.float()

    def encode_text_str(self, text: str) -> torch.Tensor:
        bytes_data = list(text.encode('utf-8'))
        if not bytes_data:
            return torch.zeros(1, self.time_steps, self.d_model, device=self.device)
        input_ids = torch.tensor(
            bytes_data, device=self.device, dtype=torch.long)
        embeddings = self.byte_embedding(input_ids)
        context = embeddings.mean(dim=0, keepdim=True)
        return self._encode_text_embedding(context.expand(1, -1), 'rate')

    def to(self, device):
        self.device = device
        return super().to(device)


# --- Backward Compatibility Wrapper ---
class UniversalEncoder(UniversalSpikeEncoder):
    """
    AutonomousAgentなどの既存コードとの互換性のためのラッパー。
    古い引数名 (time_window) を新しい引数名 (time_steps) にマッピングする。
    """

    def __init__(self, d_model: int = 64, time_window: int = 16, device: str = 'cpu', **kwargs):
        # time_window 引数を time_steps に変換して親クラスに渡す
        super().__init__(time_steps=time_window, d_model=d_model, device=device)

    def encode_text(self, text: str) -> torch.Tensor:
        # 古いメソッド名の互換性
        return self.encode_text_str(text)