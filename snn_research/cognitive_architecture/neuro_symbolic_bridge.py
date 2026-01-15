# snn_research/cognitive_architecture/neuro_symbolic_bridge.py
# Title: Neuro-Symbolic Bridge v1.1
# Description: SNNの分散表現（アトラクタ）と知識グラフのシンボル表現を相互変換するブリッジ。
#              v1.1: ConceptAugmentedTrainer向けにテンソル変換機能（symbol_to_spike）を追加。

from typing import Dict, Optional, List, Tuple, Any, Union
import numpy as np
import logging
import torch
import torch.nn as nn

# 型ヒント用のダミーインターフェース
class SNNInterface:
    def create_attractor(self, concept: str) -> np.ndarray:
        # デモ用: ランダムなアトラクタパターンを生成
        return np.random.randn(256).astype(np.float32)

    def strengthen_connection(self, pattern1: np.ndarray, pattern2: np.ndarray):
        pass


class KGInterface:
    def extract_entities(self, text: str) -> List[str]:
        return [w for w in text.split() if len(w) > 3]

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        return []

    def add_facts(self, entities: List[str], relations: List[Any]):
        pass

    def get_triplets(self) -> List[Tuple[str, str, str]]:
        return []


class NeuroSymbolicBridge(nn.Module):
    """
    神経系（SNN）と記号系（GraphRAG）の橋渡しを行うモジュール。
    抽象的な「シンボル（言葉）」を具体的な「スパイクパターン（数値）」に変換し、その逆も行います。
    """

    def __init__(self, snn_network: Optional[Any] = None, knowledge_graph: Optional[Any] = None,
                 input_dim: int = 32, embed_dim: int = 256, concepts: List[str] = []):
        super().__init__()
        self.snn = snn_network
        self.kg = knowledge_graph
        self.embed_dim = embed_dim

        # コンセプト抽出用の簡易線形層 (デモ用)
        # SNNの内部状態からどの概念が活性化しているかを読み取る
        self.extractor = nn.Linear(input_dim, len(concepts) if concepts else 32)

        # アトラクタ状態 → シンボル のマッピング
        self.attractor_to_symbol: Dict[str, str] = {}

        # シンボル → SNNパターン のマッピング (Grounding dictionary)
        self.symbol_to_pattern: Dict[str, np.ndarray] = {}

        # コンセプトリスト
        self.concepts = concepts

        logging.info("NeuroSymbolicBridge v1.1 initialized.")

    def ground_symbol(self, concept: str) -> np.ndarray:
        """
        シンボルグラウンディング: 概念文字列 → スパイクパターン(Numpy)
        
        既知の概念であれば保存されたパターンを返し、
        未知の概念であればSNNインターフェースを通じて新しいアトラクタを生成します。
        """
        if concept in self.symbol_to_pattern:
            return self.symbol_to_pattern[concept]

        # 新しい概念の場合、SNN側でアトラクタを作成
        attractor_pattern = np.random.randn(self.embed_dim).astype(np.float32)
        if self.snn and hasattr(self.snn, 'create_attractor'):
            attractor_pattern = self.snn.create_attractor(concept)

        self.symbol_to_pattern[concept] = attractor_pattern
        return attractor_pattern

    def symbol_to_spike(self, concepts: Union[str, List[str]], batch_size: int = 1) -> torch.Tensor:
        """
        [New] トレーニング用: 概念(リスト)をPyTorchのテンソル形式のスパイクパターンに変換します。
        
        Args:
            concepts: 概念文字列、またはそのリスト。
                      単一文字列の場合はバッチサイズ分複製されます。
                      リストの場合はバッチサイズと長さが一致する必要があります。
            batch_size: 単一文字列が渡された場合のバッチサイズ。

        Returns:
            torch.Tensor: (Batch, Embedding_Dim) の形状を持つテンソル。
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'

        if isinstance(concepts, str):
            # 単一概念の場合、パターンを取得してバッチサイズ分複製
            pattern = self.ground_symbol(concepts)
            tensor_pattern = torch.from_numpy(pattern).float().to(device)
            return tensor_pattern.unsqueeze(0).expand(batch_size, -1)
        
        elif isinstance(concepts, list):
            # リストの場合、各概念を変換してスタック
            patterns = []
            for c in concepts:
                pat = self.ground_symbol(c)
                patterns.append(pat)
            
            # (Batch, Dim)
            batch_patterns = np.stack(patterns)
            return torch.from_numpy(batch_patterns).float().to(device)
        
        else:
            raise ValueError("Concepts must be a string or a list of strings.")

    def abstract_pattern(self, spike_pattern: np.ndarray) -> Optional[str]:
        """パターン抽象化: スパイクパターン→概念"""

        # 最も近いアトラクタを検索
        best_match = None
        max_similarity = -1.0

        norm_input = np.linalg.norm(spike_pattern) + 1e-9

        for symbol, pattern in self.symbol_to_pattern.items():
            # 簡易コサイン類似度
            similarity = np.dot(spike_pattern, pattern) / \
                (norm_input * np.linalg.norm(pattern) + 1e-9)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = symbol

        if max_similarity > 0.7:  # 類似度閾値
            return best_match

        return None

    def extract_symbols(self, neural_features: torch.Tensor, threshold: float = 0.5) -> List[Any]:
        """ニューラル特徴量からシンボルを抽出 (PyTorch Tensor対応)"""
        # 単純な線形変換と閾値処理
        with torch.no_grad():
            logits = self.extractor(neural_features)
            probs = torch.sigmoid(logits)

        detected = []
        if self.concepts:
            for i, concept in enumerate(self.concepts):
                if probs[0, i] > threshold:
                    # ダミーのシンボルオブジェクトを返す
                    detected.append(type('Symbol', (), {'name': concept}))
        return detected

    def inject_symbols(self, facts: List[str]) -> torch.Tensor:
        """推論された事実をニューラル信号に変換 (レガシー互換用)"""
        # 今後は symbol_to_spike の使用を推奨
        # ここでは単純に最初の事実を変換して返す
        if not facts:
            return torch.randn(self.embed_dim)
        return self.symbol_to_spike(facts[0], batch_size=1).squeeze(0)

    def learn_from_dialogue(self, user_input: str, snn_response_pattern: np.ndarray):
        """対話から学習 - GraphとSNNの両方を更新"""
        if self.kg:
            entities = self.kg.extract_entities(user_input)
            relations = self.kg.extract_relations(user_input)
            self.kg.add_facts(entities, relations)

            for entity in entities:
                if entity not in self.symbol_to_pattern:
                    self.symbol_to_pattern[entity] = snn_response_pattern.copy()

    def sleep_integration(self):
        """睡眠中に神経パターンと知識グラフを統合"""
        if self.kg and self.snn:
            for (entity1, relation, entity2) in self.kg.get_triplets():
                pattern1 = self.ground_symbol(entity1)
                pattern2 = self.ground_symbol(entity2)
                self.snn.strengthen_connection(pattern1, pattern2)

class SimpleLogicEngine:
    """簡易的なルールベース論理エンジン"""

    def __init__(self, rules: List[Tuple[str, Optional[str], str]]):
        self.rules = rules

    def infer(self, facts: List[str]) -> List[str]:
        """与えられた事実から新しい事実を推論する"""
        current_facts = set(facts)
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                cond1, cond2, result = rule

                # 条件1のチェック
                if cond1 not in current_facts:
                    continue

                # 条件2のチェック（ある場合）
                if cond2 is not None and cond2 not in current_facts:
                    continue

                # 結果がまだ事実セットになければ追加
                if result not in current_facts:
                    current_facts.add(result)
                    changed = True

        return list(current_facts)