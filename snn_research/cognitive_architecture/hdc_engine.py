# ファイルパス: snn_research/cognitive_architecture/hdc_engine.py
# 日本語タイトル: HDC Engine v2.1 (Full Suite)
# 機能説明: 
#   HDC演算エンジン、Neuro-Symbolic Bridge、および HDCReasoningAgent を統合。
#   シンボル接地と推論機能を提供する。

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

class HDCEngine:
    """
    Hyperdimensional Computing (HDC) エンジン。
    MAP (Multiply, Add, Permute) アーキテクチャ。
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        self.dim = dim
        self.device = torch.device(device if device else "cpu")
        self.item_memory: Dict[str, torch.Tensor] = {}
        logger.info(f"🌌 HDC Engine initialized (Dim: {dim}, Device: {self.device})")

    def create_hypervector(self, name: Optional[str] = None) -> torch.Tensor:
        """ランダムなバイナリハイパーベクトル (-1, +1) を生成"""
        hv = torch.randint(0, 2, (self.dim,), device=self.device, dtype=torch.float32)
        hv = torch.where(hv == 0, -1.0, 1.0)
        if name:
            self.item_memory[name] = hv
        return hv

    def get_hypervector(self, name: str) -> torch.Tensor:
        """メモリからベクトルを取得、なければ新規作成"""
        if name not in self.item_memory:
            return self.create_hypervector(name)
        return self.item_memory[name]

    # --- Operations ---

    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Binding (XOR equivalent in bipolar)"""
        return hv1 * hv2

    def bundle(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """Bundling (Superposition with normalization)"""
        if not hvs:
            return torch.zeros(self.dim, device=self.device)
        stacked = torch.stack(hvs)
        summed = torch.sum(stacked, dim=0)
        bundled = torch.sign(summed)
        bundled[bundled == 0] = 1.0 
        return bundled

    def permute(self, hv: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """Permutation (Cyclic shift)"""
        return torch.roll(hv, shifts=shifts, dims=0)

    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """Cosine Similarity"""
        return torch.nn.functional.cosine_similarity(hv1.unsqueeze(0), hv2.unsqueeze(0)).item()

    def query_memory(self, query_hv: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """連想メモリ検索"""
        results = []
        for name, mem_hv in self.item_memory.items():
            sim = self.similarity(query_hv, mem_hv)
            results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class NeuroSymbolicBridge(nn.Module):
    """
    SNN(サブシンボリック)とHDC(シンボリック)の架け橋。
    ランダム射影を用いて、低次元のスパイク活動を高次元のハイパーベクトルへ、
    あるいはその逆へと変換する。
    """
    def __init__(self, snn_features: int, hdc_dim: int = 10000, device: Optional[str] = None):
        super().__init__()
        self.snn_features = snn_features
        self.hdc_dim = hdc_dim
        self.device = torch.device(device if device else "cpu")
        
        # 固定ランダム射影行列 (学習不要、生物学的妥当性が高い)
        # SNN -> HDC (Encoder)
        self.projection_matrix = torch.randn(hdc_dim, snn_features, device=self.device)
        # HDC -> SNN (Decoder) - 擬似逆行列に近い役割だが、ここでは転置を使用(双方向性)
        
        # 2値化のための閾値
        self.threshold = 0.0

    def spikes_to_hypervector(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        SNNのスパイク活動 (Batch, Features) or (Features,) を ハイパーベクトルに変換。
        Symbol Grounding: 「知覚」を「概念」に変換。
        """
        if spikes.dim() > 1:
            # 時間平均またはバッチ平均をとる（簡易化）
            spikes = spikes.mean(dim=0)
            
        # 射影: HV = sign(W @ spikes)
        projected = torch.matmul(self.projection_matrix, spikes)
        hv = torch.sign(projected)
        hv[hv == 0] = 1.0
        return hv

    def hypervector_to_spikes(self, hv: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        ハイパーベクトルをSNNの入力電流/スパイクに変換。
        Top-down Attention / Imagination: 「概念」から「知覚イメージ」を生成。
        """
        # 逆射影: Input = W.T @ HV
        currents = torch.matmul(self.projection_matrix.t(), hv)
        
        # レートコーディングへの変換 (簡易的なポアソン生成)
        # 電流値を発火確率とみなす（正規化必要）
        probs = torch.sigmoid(currents) # 0.0 ~ 1.0
        
        # 時間ステップ分生成
        spike_train = (torch.rand(steps, self.snn_features, device=self.device) < probs).float()
        return spike_train

class HDCReasoningAgent:
    """
    HDCを用いた簡易推論エージェント。
    """
    def __init__(self, engine: HDCEngine):
        self.hdc = engine
        
    def learn_concept(self, subject: str, relation: str, obj: str):
        """
        知識を結合して記憶する。
        Memory = Memory + (Subject * Relation * Object)
        """
        # 例: (Japan * Capital * Tokyo)
        h_sub = self.hdc.get_hypervector(subject)
        h_rel = self.hdc.get_hypervector(relation)
        h_obj = self.hdc.get_hypervector(obj)
        
        fact = self.hdc.bind(self.hdc.bind(h_sub, h_rel), h_obj)
        
        # "Global Knowledge" という概念に束ねていく（簡易実装）
        global_mem = self.hdc.get_hypervector("GLOBAL_KNOWLEDGE")
        new_mem = self.hdc.bundle([global_mem, fact])
        self.hdc.item_memory["GLOBAL_KNOWLEDGE"] = new_mem
        
    def query(self, subject: str, relation: str) -> List[Tuple[str, float]]:
        """
        推論を行う。
        Query: Japan * Capital * ? = Tokyo
        """
        global_mem = self.hdc.get_hypervector("GLOBAL_KNOWLEDGE")
        h_sub = self.hdc.get_hypervector(subject)
        h_rel = self.hdc.get_hypervector(relation)
        
        # Query = Memory * Subject * Relation
        query_res = self.hdc.bind(self.hdc.bind(global_mem, h_sub), h_rel)
        
        # ノイズの中から最も近い概念を探す
        return self.hdc.query_memory(query_res, top_k=3)