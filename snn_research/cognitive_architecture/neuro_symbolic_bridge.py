# ファイルパス: snn_research/cognitive_architecture/neuro_symbolic_bridge.py
# 日本語タイトル: Neuro-Symbolic Bridge (NSB)
# 目的: SNNのスパイク活動（Neural）と記号論理（Symbolic）の相互変換を行い、
#       「直感」と「論理」を融合した推論を可能にする。

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Symbol:
    name: str
    confidence: float = 1.0
    vector: Optional[torch.Tensor] = None

class NeuroSymbolicBridge(nn.Module):
    """
    ニューラル活動(Spikes)とシンボル(Logic)の変換アダプタ。
    
    Architecture:
    1. Concept Extraction: スパイク発火率 -> 概念シンボル (Grounding)
    2. Logic Processing: 外部ルールエンジンによる推論 (Neural Module外で実行)
    3. Concept Injection: 推論結果シンボル -> スパイク刺激 (Embedding)
    """
    def __init__(self, input_dim: int, embed_dim: int, concepts: List[str]):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.concepts = concepts
        
        # 1. Extraction Layer (Neural -> Symbol)
        # スパイク活動から、各概念の活性度を推定する線形層
        self.extractor = nn.Linear(input_dim, len(concepts))
        
        # 2. Injection Layer (Symbol -> Neural)
        # 各概念に対応する「脳内表現（Embedding）」
        self.concept_embeddings = nn.Embedding(len(concepts), embed_dim)
        
        # 概念名とIDのマッピング
        self.concept_to_id = {c: i for i, c in enumerate(concepts)}
        
        # 初期化
        nn.init.xavier_normal_(self.extractor.weight)
        nn.init.normal_(self.concept_embeddings.weight, std=0.02)

    def extract_symbols(self, spike_activity: torch.Tensor, threshold: float = 0.5) -> List[Symbol]:
        """
        脳の活動状態から、現在活性化している概念シンボルを抽出する。
        
        Args:
            spike_activity: (Batch, InputDim) - ニューロンの発火率や膜電位
        Returns:
            active_symbols: 検出されたシンボルのリスト
        """
        # (B, NumConcepts)
        logits = self.extractor(spike_activity)
        probs = torch.sigmoid(logits)
        
        # 閾値を超えた概念を抽出 (Batchサイズ1を想定した簡易実装)
        # 実際はバッチごとにリストを返す形になる
        active_indices = torch.nonzero(probs[0] > threshold, as_tuple=False).squeeze(-1)
        
        symbols = []
        for idx in active_indices:
            idx_int = idx.item()
            sym_name = self.concepts[idx_int]
            conf = probs[0, idx_int].item()
            vec = self.concept_embeddings(idx)
            symbols.append(Symbol(name=sym_name, confidence=conf, vector=vec))
            
        return symbols

    def inject_symbols(self, symbols: List[str]) -> torch.Tensor:
        """
        記号論理で導き出された結論（概念リスト）を、脳への入力信号に変換する。
        
        Args:
            symbols: 活性化させたい概念名のリスト
        Returns:
            neural_input: (1, EmbedDim) - 脳へフィードバックする刺激
        """
        if not symbols:
            return torch.zeros(1, self.embed_dim, device=self.concept_embeddings.weight.device)
            
        indices = []
        for s in symbols:
            if s in self.concept_to_id:
                indices.append(self.concept_to_id[s])
        
        if not indices:
            return torch.zeros(1, self.embed_dim, device=self.concept_embeddings.weight.device)
            
        idx_tensor = torch.tensor(indices, device=self.concept_embeddings.weight.device)
        
        # 複数の概念ベクトルの和（または平均）を生成
        embeddings = self.concept_embeddings(idx_tensor)
        neural_input = torch.sum(embeddings, dim=0, keepdim=True)
        
        return neural_input

class SimpleLogicEngine:
    """
    簡易的なルールベース推論エンジン（Python実装）。
    本来はPrologやASPなどのソルバーを連携させるが、ここではPython辞書でルールを定義。
    """
    def __init__(self, rules: List[Tuple[str, str, str]]):
        # Rules format: (Condition1, Condition2, Result) -> If C1 and C2 then Result
        # Simple AND rule
        self.rules = rules

    def infer(self, active_facts: List[str]) -> List[str]:
        """
        既知の事実(facts)から、ルールに基づいて新しい事実を導出する。
        """
        inferred_facts = set(active_facts)
        newly_inferred = True
        
        while newly_inferred:
            newly_inferred = False
            for c1, c2, result in self.rules:
                # C2がNoneの場合は単項条件 (If C1 then Result)
                if c2 is None:
                    if c1 in inferred_facts and result not in inferred_facts:
                        inferred_facts.add(result)
                        newly_inferred = True
                else:
                    if c1 in inferred_facts and c2 in inferred_facts and result not in inferred_facts:
                        inferred_facts.add(result)
                        newly_inferred = True
                        
        return list(inferred_facts)