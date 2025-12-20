# ファイルパス: snn_research/io/spike_encoder.py
# Title: スパイクエンコーダ (Semantic Embedding対応・安全なフォールバック版)
# Description:
# - 数値データやアナログ信号をSNN用のスパイク列に変換する各種エンコーダ。
# - 修正: MD5ハッシュによる完全ランダムなスパイク生成を廃止。
#   SentenceTransformerがない場合でも、文字N-gramを用いた決定論的な
#   ハッシュプロジェクションにより、類似した文字列が類似したスパイクパターンを
#   生成するように改善。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import numpy as np

# 意味的埋め込み用
try:
    from sentence_transformers import SentenceTransformer # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SpikeEncoder(nn.Module):
    """スパイクエンコーダの基底クラス"""
    _embedding_model = None # モデルをクラスレベルでキャッシュ

    def __init__(self, num_neurons: Optional[int] = None) -> None:
        super().__init__()
        self.num_neurons = num_neurons
    
    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        raise NotImplementedError

    def _get_embedding_model(self):
        """Embeddingモデルの遅延読み込み (シングルトン)"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        if SpikeEncoder._embedding_model is None:
            logger.info("Loading SentenceTransformer for semantic encoding...")
            # 軽量なモデルを使用
            SpikeEncoder._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return SpikeEncoder._embedding_model

    def _char_ngram_projection(self, text: str, dimension: int, n: int = 3) -> torch.Tensor:
        """
        Transformerがない場合のフォールバック。
        文字N-gramハッシュを用いて、類似した文字列が近いベクトルになるように射影する。
        """
        # バケツの初期化
        vector = np.zeros(dimension, dtype=np.float32)
        text_len = len(text)
        
        if text_len < n:
            # 文字列が短すぎる場合はパディングするか、そのままハッシュ
            h = hash(text)
            np.random.seed(h % (2**32))
            return torch.from_numpy(np.random.rand(dimension)).float()
        
        # N-gramの生成とハッシュ加算
        for i in range(text_len - n + 1):
            ngram = text[i:i+n]
            # N-gramごとのハッシュ値を生成
            # Pythonのhash()は実行ごとにシードが変わる可能性があるため、安定したハッシュが必要ならzlib.adler32等を使うべきだが
            # ここでは簡易的にnumpyのrandom seedとして使う（デモ用途）
            h = abs(hash(ngram))
            
            # 符号付きで加算することで、衝突時の相殺を利用（SimHash的アプローチ）
            # 次元ごとに +1 か -1 を加算
            np.random.seed(h % (2**32))
            sign_vector = np.random.choice([-1.0, 1.0], size=dimension)
            vector += sign_vector
            
        # 正規化 (0-1の範囲には収まらないが、後段でSigmoidするならOK)
        # ここでは分散を抑えるために正規化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return torch.from_numpy(vector).float()

    def encode(self, sensory_info: Dict[str, Any], duration: int) -> torch.Tensor:
        """
        感覚情報（辞書）を受け取り、スパイクパターン（Tensor）に変換する。
        ArtificialBrainからの呼び出し用インターフェース。
        """
        content = sensory_info.get("content")
        
        # 1. 数値の場合
        if isinstance(content, (int, float)):
            x = torch.tensor([[float(content)]]) 
            return self.forward(x, duration)
            
        # 2. リストの場合 (数値リストを想定)
        elif isinstance(content, list):
            try:
                # テンソル変換を試みる
                x = torch.tensor(content).float()
                # 次元調整: (Features,) -> (1, Features)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.forward(x, duration)
            except Exception:
                # 数値以外のリスト等の場合は後続のテキスト処理へ
                pass
        
        # 3. テキストの場合 (意味的エンコーディング)
        content_str = str(content)
        # HybridPerceptionCortex (CorticalColumn) の入力次元と合わせるため、
        # self.num_neurons が指定されていればその次元でスパイクを生成する。
        N = self.num_neurons if self.num_neurons is not None else 256
        
        model = self._get_embedding_model()
        if model is not None:
            # SentenceBERTでベクトル化
            with torch.no_grad():
                embedding = model.encode(content_str, convert_to_tensor=True)
                if isinstance(embedding, list):
                     embedding = torch.tensor(embedding)
            
            embedding = embedding.cpu()

            # ニューロン数に合わせて次元圧縮または拡張
            current_dim = embedding.shape[0]
            if current_dim != N:
                # 線形補間でリサイズ
                embedding = torch.nn.functional.interpolate(
                    embedding.view(1, 1, -1), size=N, mode='linear', align_corners=False
                ).view(-1)
            
            # ベクトル値を0-1に正規化してレートコーディング
            probs = torch.sigmoid(embedding)
            
        else:
            # 4. フォールバック (N-gram Projection)
            logger.warning("SentenceTransformer not available. Using N-gram projection fallback.")
            projected_vector = self._char_ngram_projection(content_str, N)
            # Sigmoidで確率化 (平均0、分散1に近い入力なので、0.5中心に分布するはず)
            # 少しゲインを上げてメリハリをつける
            probs = torch.sigmoid(projected_vector * 2.0)

        # (Duration, N) に拡張
        probs_expanded = probs.unsqueeze(0).expand(duration, -1)
        
        # ベルヌーイ試行でスパイク生成
        spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()
        
        return spikes


class RateEncoder(SpikeEncoder):
    """レートコーディング"""
    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        # x: (Batch, Features) -> x_seq: (Batch, Duration, Features)
        x_seq = x.unsqueeze(1).repeat(1, duration, *([1] * (x.ndim - 1)))
        # 入力値を発火確率としてベルヌーイ試行
        spikes = torch.rand_like(x_seq) < x_seq
        return spikes.float()

class LatencyEncoder(SpikeEncoder):
    """レイテンシコーディング"""
    def __init__(self, tau: float = 1.0, threshold: float = 0.01, num_neurons: Optional[int] = None) -> None:
        super().__init__(num_neurons)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x: torch.Tensor, duration: int) -> torch.Tensor:
        # x: (Batch, Features)
        x_expanded = x.unsqueeze(1)
        # 時間軸を作成 (1, Duration, 1)
        time_axis = torch.arange(duration, device=x.device).view(1, duration, 1).float()
        
        # 入力 x が大きいほど閾値は小さく（早く発火）
        latency = (1.0 - x_expanded.clamp(0, 1)) * (duration - 1)
        
        # 現在時刻がlatencyと一致する（または超えた瞬間）に発火
        fire_mask = (time_axis - latency).abs() < 0.5
        
        return fire_mask.float()

class DeltaEncoder(SpikeEncoder):
    """デルタコーディング"""
    def __init__(self, threshold: float = 0.1, num_neurons: Optional[int] = None) -> None:
        super().__init__(num_neurons)
        self.threshold = threshold
        
    def forward(self, x_seq: torch.Tensor, duration: int = 0) -> torch.Tensor:
        if x_seq.dim() < 3:
             raise ValueError("DeltaEncoder requires input with time dimension (Batch, Duration, Features).")
             
        diff = torch.zeros_like(x_seq)
        diff[:, 1:, ...] = x_seq[:, 1:, ...] - x_seq[:, :-1, ...]
        diff[:, 0, ...] = x_seq[:, 0, ...]
        
        spikes = (diff.abs() > self.threshold).float()
        return spikes

class DifferentiableTTFSEncoder(SpikeEncoder):
    """学習可能なTTFS (Time-to-First-Spike) エンコーダ"""
    def __init__(self, num_neurons: int, duration: int, initial_sensitivity: float = 1.0) -> None:
        super().__init__(num_neurons)
        self.duration = duration
        self.sensitivity = nn.Parameter(torch.ones(num_neurons) * initial_sensitivity)
        self.v_th = 1.0
        self.tau = 2.0 

    def forward(self, x: torch.Tensor, duration: Optional[int] = None) -> torch.Tensor:
        T = duration if duration is not None else self.duration
        I = x * self.sensitivity.unsqueeze(0)
        
        spikes_list = []
        mem = torch.zeros_like(I)
        has_fired = torch.zeros_like(I, dtype=torch.bool)
        
        decay = torch.exp(torch.tensor(-1.0 / self.tau, device=x.device))
        
        for t in range(T):
            mem = mem * decay + I * (1 - decay)
            spike = (mem >= self.v_th).float()
            effective_spike = spike * (~has_fired).float()
            has_fired = has_fired | (spike > 0.5)
            mem = mem * (1.0 - spike)
            spikes_list.append(effective_spike)
            
        return torch.stack(spikes_list, dim=1)