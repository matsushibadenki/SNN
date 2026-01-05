# /snn_research/cognitive_architecture/causal_inference_engine.py
# 日本語タイトル: 因果推論エンジン (DEMOCRITUS Pipeline実装版)
# 参照: "Large Causal Models from Large Language Models" (arXiv:2512.07796)
# 概要: 
#   DEMOCRITUS (Decentralized Extraction of Manifold Ontologies of Causal Relations 
#   Integrating Topos Universal Slices) パイプラインを統合し、
#   テキストストリームからLCM (Large Causal Model) を構築するためのエンジン。

from typing import Dict, Any, Optional, List, Callable
import logging
import re
from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

class DemocritusPipeline:
    """
    DEMOCRITUSシステムのパイプライン実装。
    LLMを利用してテキストから因果関係のトリプルを抽出し、LCMを構築するための前処理を行う。
    
    Modules:
    1. Topic Extraction: 関連するトピックの特定
    2. Causal Question Generation: 因果関係を問う質問の生成
    3. Causal Statement Extraction: 回答からの因果記述の抽出
    4. Relational Triple Construction: (Subject, Relation, Object) 形式への変換
    """
    def __init__(self, generator_callback: Callable[[str], str]):
        self.generator = generator_callback

    def run_pipeline(self, text: str) -> List[Dict[str, Any]]:
        """
        テキストから因果トリプルを抽出する完全なパイプラインを実行。
        """
        # Module 1: Topic Graph (簡易版: 主要トピックの抽出)
        topics = self._extract_topics(text)
        if not topics:
            return []
        
        extracted_triples = []
        for topic in topics:
            # Module 2: Causal Questions
            questions = self._generate_causal_questions(text, topic)
            
            for question in questions:
                # Module 3 & 4: Statement to Triple
                answer = self._get_model_response(f"Context: {text}\nQuestion: {question}\nAnswer concisely:")
                triples = self._extract_triples_from_answer(topic, question, answer)
                extracted_triples.extend(triples)
                
        return self._deduplicate_triples(extracted_triples)

    def _extract_topics(self, text: str) -> List[str]:
        prompt = (
            f"Analyze the following text and list up to 3 main scientific or logical topics mentioned.\n"
            f"Text: {text[:500]}...\n"
            f"Output format: Topic1, Topic2, Topic3"
        )
        response = self._get_model_response(prompt)
        # 簡易的なパース
        topics = [t.strip() for t in response.split(',')]
        return [t for t in topics if t]

    def _generate_causal_questions(self, text: str, topic: str) -> List[str]:
        prompt = (
            f"Based on the text about '{topic}', generate 2 questions that ask about cause-and-effect relationships.\n"
            f"Start questions with 'What causes' or 'What is the effect of'.\n"
            f"Text: {text[:500]}..."
        )
        response = self._get_model_response(prompt)
        questions = [q.strip() for q in response.split('\n') if '?' in q]
        return questions[:2]

    def _extract_triples_from_answer(self, topic: str, question: str, answer: str) -> List[Dict[str, Any]]:
        prompt = (
            f"Extract causal triples from the statement below. \n"
            f"Format: [Subject] -> [Relation] -> [Object] (Strength: 0.0-1.0)\n"
            f"Statement: Since {answer}, it implies a causal link related to {topic}."
        )
        response = self._get_model_response(prompt)
        return self._parse_triple_response(response)

    def _parse_triple_response(self, response: str) -> List[Dict[str, Any]]:
        triples = []
        # Regex for "[Subject] -> [Relation] -> [Object] (Strength: X.X)"
        pattern = r"\[(.*?)\] -> \[(.*?)\] -> \[(.*?)\] \(Strength: (0\.\d+|1\.0)\)"
        matches = re.findall(pattern, response)
        
        for subj, rel, obj, strength in matches:
            triples.append({
                "subject": subj.strip(),
                "predicate": rel.strip(),
                "object": obj.strip(),
                "strength": float(strength),
                "type": "causal_triple"
            })
        return triples

    def _get_model_response(self, prompt: str) -> str:
        # LLM生成コールのラッパー
        return self.generator(prompt).strip()

    def _deduplicate_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique = {}
        for t in triples:
            key = f"{t['subject']}_{t['predicate']}_{t['object']}"
            if key not in unique:
                unique[key] = t
            else:
                # Keep the one with higher strength
                if t['strength'] > unique[key]['strength']:
                    unique[key] = t
        return list(unique.values())


class CausalInferenceEngine:
    def __init__(
        self, 
        rag_system: RAGSystem, 
        workspace: GlobalWorkspace, 
        inference_threshold: float = 0.6,
        llm_backend: Optional[Callable[[str], str]] = None
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        self.workspace.subscribe(self.handle_conscious_broadcast)
        
        # DEMOCRITUS Pipelineの初期化
        # llm_backendが提供されない場合は、ダミー（または後で設定）
        self.pipeline = DemocritusPipeline(llm_backend if llm_backend else self._dummy_generator)
        
        logger.info("🔥 CausalInferenceEngine (w/ DEMOCRITUS pipeline) initialized.")

    def set_llm_backend(self, generator: Callable[[str], str]):
        """
        ReasoningEngineなどの生成機能を持つバックエンドを登録する。
        """
        self.pipeline.generator = generator

    def _dummy_generator(self, prompt: str) -> str:
        logger.warning("LLM backend not set for CausalInferenceEngine. Returning empty string.")
        return ""

    def process_text_for_causality(self, text: str, source: str = "perception"):
        """
        テキストデータに対してDEMOCRITUSパイプラインを実行し、
        抽出されたLCM（大規模因果モデル）トリプルをRAGに統合する。
        """
        logger.info(f"🧪 Running DEMOCRITUS pipeline on text from {source}...")
        triples = self.pipeline.run_pipeline(text)
        
        for triple in triples:
            self._crystallize_causal_triple(triple, source)
            
        return len(triples)

    def _crystallize_causal_triple(self, triple: Dict[str, Any], context_source: str):
        """
        抽出された因果トリプルを知識ベースとワークスペースに登録。
        """
        subj = triple['subject']
        pred = triple['predicate']
        obj = triple['object']
        strength = triple.get('strength', 1.0)
        
        logger.info(f"💎 Causal Triple Crystallized: [{subj}] --{pred}--> [{obj}] (s={strength})")
        
        # RAGへの登録: トリプル形式と因果形式の両方で登録を試みる
        # 1. トリプルとして登録
        self.rag_system.add_triple(
            subj=subj, 
            pred=pred, 
            obj=obj, 
            metadata={
                "source": context_source,
                "strength": strength,
                "paradigm": "DEMOCRITUS_LCM"
            }
        )
        
        # 2. 単純な因果関係としても登録 (互換性維持)
        if "cause" in pred.lower() or "lead" in pred.lower() or "result" in pred.lower():
            self.rag_system.add_causal_relationship(
                cause=subj,
                effect=obj,
                strength=strength,
                condition=f"via {pred}"
            )

        # ワークスペースへのフィードバック (顕著性が高い場合)
        if strength > self.inference_threshold:
            self.workspace.upload_to_workspace(
                source="causal_inference_engine",
                data={
                    "type": "new_causal_discovery", 
                    "triple": triple
                },
                salience=strength
            )

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float):
        """
        レガシーメソッド: 単純な因果関係を発見した場合の登録。
        """
        logger.info(f"🔥 Simple Causal Discovery: {cause} -> {effect} (strength={strength:.2f})")
        
        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            strength=strength
        )
        
        self.workspace.upload_to_workspace(
            source="causal_inference_engine",
            data={
                "type": "causal_credit", 
                "cause": cause, 
                "effect": effect, 
                "strength": strength
            },
            salience=0.8
        )

    def handle_conscious_broadcast(self, source: str, conscious_data: Any):
        """
        意識に上った情報（Conscious Broadcast）を監視し、
        テキスト情報が含まれていれば因果抽出パイプラインを起動する。
        """
        # テキストデータが含まれているかチェック
        text_content = None
        
        if isinstance(conscious_data, str):
            text_content = conscious_data
        elif isinstance(conscious_data, dict):
            if "text" in conscious_data:
                text_content = conscious_data["text"]
            elif "observation" in conscious_data:
                text_content = conscious_data["observation"]
        
        # 十分な長さがあればパイプラインを実行
        if text_content and len(text_content) > 50:
            # バックグラウンド処理として実行するのが理想だが、ここでは同期的に実行
            self.process_text_for_causality(text_content, source=source)