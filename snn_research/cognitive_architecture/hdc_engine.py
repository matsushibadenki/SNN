# ファイルパス: snn_research/cognitive_architecture/hdc_engine.py
# 日本語タイトル: Hyperdimensional Computing (HDC) エンジン
# 機能説明: 
#   10,000次元以上の超高次元ベクトル（Hypervector）を用いた計算エンジン。
#   GPUを使用せず、ビット演算（XOR等）と整数演算のみで高度なシンボル処理を実現する。
#   従来のベクトル検索よりもノイズに強く、ワンショット学習が可能。
#   Green AI / 省エネコンピューティングの中核モジュール。

import torch
import logging
from typing import Dict, Optional, List, Union, Tuple # 修正: Tupleを追加

logger = logging.getLogger(__name__)

class HDCEngine:
    """
    Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) エンジン。
    MAP (Multiply, Add, Permute) アーキテクチャに基づく。
    
    Attributes:
        dim (int): ハイパーベクトルの次元数 (通常 10,000 以上)。
        device (torch.device): 計算に使用するデバイス (CPU推奨だがCUDAも可)。
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        self.dim = dim
        # HDCはCPUのビット演算が高速であるため、デフォルトはCPUでも良いが、
        # 大規模並列化のためにCUDAもサポートする。
        self.device = torch.device(device if device else "cpu")
        
        # アイテムメモリ（既知の基本ベクトル）
        self.item_memory: Dict[str, torch.Tensor] = {}
        
        logger.info(f"🌌 HDC Engine initialized (Dim: {dim}, Device: {self.device})")

    def create_hypervector(self, name: Optional[str] = None) -> torch.Tensor:
        """
        ランダムなバイナリハイパーベクトル (-1, +1) を生成する。
        """
        # {0, 1} を生成
        hv = torch.randint(0, 2, (self.dim,), device=self.device, dtype=torch.float32)
        # {0, 1} -> {-1, 1} に変換 (Bipolar表現)
        hv = torch.where(hv == 0, -1.0, 1.0)
        
        if name:
            self.item_memory[name] = hv
            
        return hv

    def get_hypervector(self, name: str) -> torch.Tensor:
        """メモリからベクトルを取得、なければ新規作成"""
        if name not in self.item_memory:
            return self.create_hypervector(name)
        return self.item_memory[name]

    # --- Operations (MAP) ---

    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        Binding (結合) 操作。
        Bipolar表現では要素ごとの乗算（XORに相当）。
        結合されたベクトルは元のベクトルとは直交する（似ていない）。
        例: Color * Red
        """
        return hv1 * hv2

    def bundle(self, hvs: List[torch.Tensor]) -> torch.Tensor:
        """
        Bundling (重ね合わせ) 操作。
        要素ごとの加算を行い、符号関数(sign)で正規化する。
        重ね合わせたベクトルは、元の全てのベクトルと類似する。
        例: Red + Car + Fast
        """
        if not hvs:
            return torch.zeros(self.dim, device=self.device)
            
        stacked = torch.stack(hvs)
        summed = torch.sum(stacked, dim=0)
        
        # マジョリティ投票による正規化 {-1, 1}
        # 0の場合はランダムに割り振ることで情報を保存するのが一般的だが、ここでは1とする
        bundled = torch.sign(summed)
        bundled[bundled == 0] = 1.0 
        return bundled

    def permute(self, hv: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """
        Permutation (置換) 操作。
        ベクトルの要素を巡回シフトする。順序情報をエンコードする際に使用。
        例: A then B -> A + Permute(B)
        """
        return torch.roll(hv, shifts=shifts, dims=0)

    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """
        コサイン類似度計算。
        Bipolar HDCでは、内積を次元数で割ったものとほぼ等価。
        +1: 同一, 0: 直交(無関係), -1: 正反対
        """
        return torch.nn.functional.cosine_similarity(hv1.unsqueeze(0), hv2.unsqueeze(0)).item()

    # --- High-Level Cognitive Functions ---

    def encode_sequence(self, sequence: List[str]) -> torch.Tensor:
        """
        系列情報をエンコードする。
        H = v1 + P(v2) + P(P(v3)) ...
        """
        if not sequence:
            return self.create_hypervector() # random noise
            
        accumulated = torch.zeros(self.dim, device=self.device)
        
        for i, token in enumerate(sequence):
            hv = self.get_hypervector(token)
            permuted_hv = self.permute(hv, shifts=i)
            accumulated += permuted_hv
            
        # 正規化
        result = torch.sign(accumulated)
        result[result == 0] = 1.0
        return result

    def query_memory(self, query_hv: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        クエリベクトルに最も近い概念をメモリから検索する。
        """
        results = []
        for name, mem_hv in self.item_memory.items():
            sim = self.similarity(query_hv, mem_hv)
            results.append((name, sim))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class HDCReasoningAgent:
    """
    HDCを用いた簡易推論エージェントの例。
    「日本の首都は？」のようなクエリを、行列演算なしで解く。
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
        Unbind: Memory * Subject * Relation
        """
        global_mem = self.hdc.get_hypervector("GLOBAL_KNOWLEDGE")
        h_sub = self.hdc.get_hypervector(subject)
        h_rel = self.hdc.get_hypervector(relation)
        
        # BipolarにおけるUnbind(逆演算)はBindと同じ
        # Query = Memory * Subject * Relation
        # 期待値: Object + Noise
        query_res = self.hdc.bind(self.hdc.bind(global_mem, h_sub), h_rel)
        
        # ノイズの中から最も近い概念を探す
        return self.hdc.query_memory(query_res, top_k=3)