# ファイルパス: snn_research/social/emergent_language.py
# Title: Emergent Language Game (Naming Game) v1.2
# Description:
#   2つのエージェント（Speaker, Listener）がオブジェクトに対して名前を付け合い、
#   共通の語彙を形成するシミュレーション。
#   [Fix] SymbolGroundingの初期化に必要なRAGSystem引数を追加。

import torch
import random
import logging
from typing import Dict, Optional, Any
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
# [Fix] RAGSystemのインポートを追加
from snn_research.cognitive_architecture.rag_snn import RAGSystem

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, id: str, grounding_system: SymbolGrounding):
        self.id = id
        self.grounding = grounding_system
        self.vocabulary: Dict[str, str] = {}  # ConceptID -> Word
        self.reverse_vocabulary: Dict[str, str] = {}  # Word -> ConceptID

    def observe_and_name(self, object_features: torch.Tensor, context: str) -> str:
        """
        [Speaker] オブジェクトを見て、それに対応する単語を発する。
        """
        # 1. 接地 (Pattern -> Concept)
        concept_id = self.grounding.ground_neural_pattern(
            object_features, context)

        # 2. 命名 (Concept -> Word)
        if concept_id in self.vocabulary:
            word = self.vocabulary[concept_id]
        else:
            # 新しい単語を発明
            word = f"word_{random.randint(1000, 9999)}"
            self.vocabulary[concept_id] = word
            self.reverse_vocabulary[word] = concept_id
            logger.info(
                f"🗣️ Agent {self.id} invented word '{word}' for concept '{concept_id}'.")

        return word

    def listen_and_guess(self, word: str, object_features: torch.Tensor, context: str) -> bool:
        """
        [Listener] 単語を聞き、自分が思っている概念と一致するか確認する。
        """
        # 1. 接地 (Pattern -> Concept)
        my_concept_id = self.grounding.ground_neural_pattern(
            object_features, context)

        # 2. 解釈 (Word -> Concept)
        guessed_concept = self.reverse_vocabulary.get(word)

        success = False
        if guessed_concept:
            if guessed_concept == my_concept_id:
                success = True
            else:
                # 単語は知っているが、違う概念だと思った (Synonym/Homonym conflict)
                pass
        else:
            # 単語を知らない -> 学習する
            self.vocabulary[my_concept_id] = word
            self.reverse_vocabulary[word] = my_concept_id
            success = True  # 新しく覚えたので成功とみなす（Alignment）
            logger.info(
                f"👂 Agent {self.id} learned word '{word}' maps to '{my_concept_id}'.")

        return success


class NamingGameSimulation:
    def __init__(self, agent_a: Agent, agent_b: Agent):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.success_count = 0
        self.total_rounds = 0

    def play_round(self):
        """
        1ラウンドのゲームを実行。役割（Speaker/Listener）はランダム。
        """
        self.total_rounds += 1

        # 1. 共通のオブジェクトを提示 (Random Feature)
        object_features = torch.randn(10)  # 10次元の特徴ベクトル
        context = f"round_{self.total_rounds}"

        # role assignment
        if random.random() < 0.5:
            speaker, listener = self.agent_a, self.agent_b
        else:
            speaker, listener = self.agent_b, self.agent_a

        # 2. 会話
        word = speaker.observe_and_name(object_features, context)
        success = listener.listen_and_guess(word, object_features, context)

        if success:
            self.success_count += 1
            # 報酬: 両者のvigilanceを高める、または結合を強化するなど
            logger.debug("✅ Communication Validated!")
        else:
            logger.debug("❌ Communication Failed.")

        return success

class EmergentLanguageProtocol:
    """
    Agentクラスをラップして、社会的学習実験スクリプトから利用しやすくするためのプロトコルクラス。
    """
    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        
        # [Fix] RAGSystemのインスタンスを作成して渡す
        # 簡易的なRAGSystemを作成（embedding_dimは適当に設定）
        self.rag = RAGSystem(embedding_dim=64)
        self.grounding = SymbolGrounding(rag_system=self.rag)
        
        self.agent = Agent("protocol_agent", self.grounding)

    def process_message(self, message: str) -> None:
        """メッセージを処理する（ダミー）"""
        pass