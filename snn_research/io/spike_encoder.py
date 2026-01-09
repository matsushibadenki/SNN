# ファイルパス: snn_research/io/spike_encoder.py
# Title: スパイクエンコーダ v2.0 (Device Aware)
# Description:
# - Device引数に対応し、GPU/CPU上でのテンソル生成を制御可能に変更。
# - サブクラス(RateEncoder等)もdevice引数を継承するように修正。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import numpy as np

# 意味的埋め込み用
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpikeEncoder(nn.Module):
    """スパイクエンコーダの基底クラス"""
    _embedding_model = None  # モデルをクラスレベルでキャッシュ

    def __init__(self, num_neurons: Optional[int] = None, device: str = 'cpu') -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device

    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        raise NotImplementedError

    def _get_embedding_model(self):
        """Embeddingモデルの遅延読み込み (シングルトン)"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        if SpikeEncoder._embedding_model is None:
            logger.info("Loading SentenceTransformer for semantic encoding...")
            # 軽量なモデルを使用
            SpikeEncoder._embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2')
        return SpikeEncoder._embedding_model

    def _char_ngram_projection(self, text: str, dimension: int, n: int = 3) -> torch.Tensor:
        """
        Transformerがない場合のフォールバック。
        """
        vector = np.zeros(dimension, dtype=np.float32)
        text_len = len(text)

        if text_len < n:
            h = hash(text)
            np.random.seed(h % (2**32))
            return torch.from_numpy(np.random.rand(dimension)).float().to(self.device)

        for i in range(text_len - n + 1):
            ngram = text[i:i+n]
            h = abs(hash(ngram))
            np.random.seed(h % (2**32))
            sign_vector = np.random.choice([-1.0, 1.0], size=dimension)
            vector += sign_vector

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return torch.from_numpy(vector).float().to(self.device)

    def encode_text(self, text: str, duration: int = 10) -> torch.Tensor:
        """
        テキスト文字列をスパイク列にエンコードする。
        """
        return self.encode({"content": text}, duration)

    def encode(self, sensory_info: Dict[str, Any], duration: int) -> torch.Tensor:
        """
        感覚情報（辞書）を受け取り、スパイクパターン（Tensor）に変換する。
        """
        content = sensory_info.get("content")

        # 1. 数値の場合
        if isinstance(content, (int, float)):
            x = torch.tensor([[float(content)]], device=self.device)
            return self.forward(x, duration)

        # 2. リストの場合 (数値リストを想定)
        elif isinstance(content, list):
            try:
                x = torch.tensor(content, device=self.device).float()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.forward(x, duration)
            except Exception:
                pass

        # 3. テキストの場合
        content_str = str(content)
        N = self.num_neurons if self.num_neurons is not None else 256

        model = self._get_embedding_model()
        if model is not None:
            with torch.no_grad():
                embedding = model.encode(content_str, convert_to_tensor=True)
                if isinstance(embedding, list):
                    embedding = torch.tensor(embedding, device=self.device)
            
            # Ensure embedding is on the correct device
            embedding = embedding.to(self.device)
            
            current_dim = embedding.shape[0]
            if current_dim != N:
                # 次元合わせ
                embedding = torch.nn.functional.interpolate(
                    embedding.view(1, 1, -1), size=N, mode='linear', align_corners=False
                ).view(-1)
            probs = torch.sigmoid(embedding)
        else:
            logger.warning(
                "SentenceTransformer not available. Using N-gram projection fallback.")
            projected_vector = self._char_ngram_projection(content_str, N)
            probs = torch.sigmoid(projected_vector * 2.0)

        probs_expanded = probs.unsqueeze(0).expand(duration, -1)
        spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()
        return spikes


class RateEncoder(SpikeEncoder):
    """レートコーディング"""
    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_seq = x.unsqueeze(1).repeat(1, duration, *([1] * (x.ndim - 1)))
        spikes = torch.rand_like(x_seq) < x_seq
        return spikes.float()


class LatencyEncoder(SpikeEncoder):
    """レイテンシコーディング"""
    def __init__(self, tau: float = 1.0, threshold: float = 0.01, num_neurons: Optional[int] = None, device: str = 'cpu') -> None:
        super().__init__(num_neurons, device)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_expanded = x.unsqueeze(1)
        time_axis = torch.arange(duration, device=self.device).view(
            1, duration, 1).float()
        latency = (1.0 - x_expanded.clamp(0, 1)) * (duration - 1)
        fire_mask = (time_axis - latency).abs() < 0.5
        return fire_mask.float()


class DeltaEncoder(SpikeEncoder):
    """デルタコーディング"""
    def __init__(self, threshold: float = 0.1, num_neurons: Optional[int] = None, device: str = 'cpu') -> None:
        super().__init__(num_neurons, device)
        self.threshold = threshold

    def forward(self, x_seq: torch.Tensor, duration: int = 0) -> torch.Tensor:
        x_seq = x_seq.to(self.device)
        if x_seq.dim() < 3:
            raise ValueError(
                "DeltaEncoder requires input with time dimension (Batch, Duration, Features).")
        diff = torch.zeros_like(x_seq)
        diff[:, 1:, ...] = x_seq[:, 1:, ...] - x_seq[:, :-1, ...]
        diff[:, 0, ...] = x_seq[:, 0, ...]
        spikes = (diff.abs() > self.threshold).float()
        return spikes


class DifferentiableTTFSEncoder(SpikeEncoder):
    """学習可能なTTFS (Time-to-First-Spike) エンコーダ"""

    def __init__(self, num_neurons: int, duration: int, initial_sensitivity: float = 1.0, device: str = 'cpu') -> None:
        super().__init__(num_neurons, device)
        self.duration = duration
        self.sensitivity = nn.Parameter(
            torch.ones(num_neurons, device=device) * initial_sensitivity)
        self.v_th = 1.0
        self.tau = 2.0
        self.to(device) # Parameter移動

    def forward(self, x: torch.Tensor, duration: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        T = duration if duration is not None else self.duration
        current = x * self.sensitivity.unsqueeze(0)
        spikes_list = []
        mem = torch.zeros_like(current)
        has_fired = torch.zeros_like(current, dtype=torch.bool)
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=self.device))
        
        for t in range(T):
            mem = mem * decay + current * (1 - decay)
            spike = (mem >= self.v_th).float()
            effective_spike = spike * (~has_fired).float()
            has_fired = has_fired | (spike > 0.5)
            mem = mem * (1.0 - spike)
            spikes_list.append(effective_spike)
        return torch.stack(spikes_list, dim=1)


class HybridTemporal8BitEncoder(SpikeEncoder):
    """Hybrid Temporal-8-Bit Encoder."""
    def __init__(self, duration: int = 8, num_neurons: Optional[int] = None, device: str = 'cpu') -> None:
        super().__init__(num_neurons, device)
        self.duration = min(duration, 8)

    def forward(self, x: torch.Tensor, duration: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        T = duration if duration is not None else self.duration
        T = min(T, 8)

        if x.max() <= 1.0 and x.dtype.is_floating_point:
            x_int = (x * 255).int()
        else:
            x_int = x.int()

        x_int = torch.clamp(x_int, 0, 255)

        spikes_list = []
        for t in range(T):
            shift = 7 - t
            bit_plane = (x_int >> shift) & 1
            spikes_list.append(bit_plane.float())

        spikes = torch.stack(spikes_list, dim=1)
        return spikes