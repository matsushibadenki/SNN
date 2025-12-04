# ファイルパス: snn_research/cognitive_architecture/rag_snn.py
# (修正: NetworkXのFutureWarning解消)
#
# Title: RAG System (Vector Store + Knowledge Graph)
#
# Description:
# - ベクトルストア（意味記憶）とナレッジグラフ（構造化記憶）を統合したRAGシステム。
# - NetworkXの `node_link_graph` / `node_link_data` 使用時に `edges="links"` を
#   明示的に指定することで、FutureWarning を解消し、将来のバージョンとの互換性を確保。

import os
import json
from typing import List, Optional, Dict, Any, Tuple
import networkx as nx # type: ignore[import-untyped]

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class RAGSystem:
    """
    ベクトルストア（意味記憶）とナレッジグラフ（構造化記憶）を統合したRAGシステム。
    知識の修正、構造化、ハイブリッド検索をサポートする。
    """
    def __init__(self, vector_store_path: str = "runs/vector_store"):
        self.vector_store_path = vector_store_path
        self.graph_path = os.path.join(os.path.dirname(vector_store_path), "knowledge_graph.json")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 1. ベクトルストア (FAISS) のロード
        self.vector_store: Optional[FAISS] = self._load_vector_store()
        
        # 2. ナレッジグラフ (NetworkX) のロード
        self.knowledge_graph: nx.DiGraph = self._load_knowledge_graph()

    def _load_vector_store(self) -> Optional[FAISS]:
        if os.path.exists(self.vector_store_path):
            print(f"📚 既存のベクトルストアをロード中: {self.vector_store_path}")
            try:
                return FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"⚠️ ベクトルストアのロード失敗: {e}")
                return None
        return None

    def _load_knowledge_graph(self) -> nx.DiGraph:
        if os.path.exists(self.graph_path):
            print(f"🕸️ 既存のナレッジグラフをロード中: {self.graph_path}")
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # --- ▼ 修正: edges="links" を明示的に指定 ▼ ---
                return nx.node_link_graph(data, edges="links")
                # --- ▲ 修正 ▲ ---
            except Exception as e:
                print(f"⚠️ ナレッジグラフのロード失敗: {e}。新規作成します。")
        return nx.DiGraph()

    def _save_knowledge_graph(self):
        """ナレッジグラフをJSONとして保存"""
        # --- ▼ 修正: edges="links" を明示的に指定 ▼ ---
        data = nx.node_link_data(self.knowledge_graph, edges="links")
        # --- ▲ 修正 ▲ ---
        
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def setup_vector_store(self, knowledge_dir: str = "doc", memory_file: str = "runs/agent_memory.jsonl"):
        """ベクトルストアの初期構築（既存機能）"""
        print("🛠️ ベクトルストアの構築を開始します...")
        docs = []
        if os.path.exists(knowledge_dir):
            doc_loader = DirectoryLoader(knowledge_dir, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
            docs += doc_loader.load()
        
        if not docs:
             docs = [Document(page_content="System Memory Initialized.", metadata={"source": "system"})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
        self.vector_store.save_local(self.vector_store_path)
        print(f"✅ ベクトルストア構築完了。")

    # --- グラフ操作 API (New) ---

    def add_triple(self, subj: str, pred: str, obj: str, metadata: Optional[Dict[str, Any]] = None):
        """
        知識グラフにトリプル (主語, 述語, 目的語) を追加する。
        """
        if metadata is None: metadata = {}
        
        # ノードとエッジを追加
        self.knowledge_graph.add_node(subj)
        self.knowledge_graph.add_node(obj)
        self.knowledge_graph.add_edge(subj, obj, relation=pred, **metadata)
        
        # ベクトルストアにも自然言語として記録 (検索用)
        text_repr = f"{subj} {pred} {obj}."
        self.add_document(text_repr, metadata={"type": "triple", "subj": subj})
        
        self._save_knowledge_graph()
        print(f"🔗 グラフ更新: ({subj}) --[{pred}]--> ({obj})")

    def update_knowledge(self, subj: str, pred: str, new_obj: str, reason: str = "correction"):
        """
        既存の知識を修正する。古いエッジを削除し、新しいエッジを張る。
        """
        if self.knowledge_graph.has_node(subj):
            # 同じ述語を持つ既存のエッジを探して削除 (簡易的な修正ロジック)
            edges_to_remove = []
            for neighbor in self.knowledge_graph.successors(subj):
                edge_data = self.knowledge_graph.get_edge_data(subj, neighbor)
                if edge_data.get('relation') == pred:
                    edges_to_remove.append(neighbor)
            
            for neighbor in edges_to_remove:
                self.knowledge_graph.remove_edge(subj, neighbor)
                print(f"🗑️ 旧知識削除: ({subj}) --[{pred}]--> ({neighbor})")

        # 新しい知識を追加
        self.add_triple(subj, pred, new_obj, metadata={"modified_by": reason, "timestamp": "now"})
        print(f"✨ 知識修正完了: ({subj}) --[{pred}]--> ({new_obj})")

    def get_subgraph_info(self, concept: str, depth: int = 1) -> List[str]:
        """
        特定の概念周辺のグラフ構造をテキスト化して返す。
        """
        if not self.knowledge_graph.has_node(concept):
            return []
        
        info = []
        # 自ノードから出るエッジ
        for neighbor in self.knowledge_graph.successors(concept):
            edge = self.knowledge_graph.get_edge_data(concept, neighbor)
            rel = edge.get('relation', 'related_to')
            info.append(f"{concept} {rel} {neighbor}")
            
        # 自ノードに入るエッジ
        for predecessor in self.knowledge_graph.predecessors(concept):
            edge = self.knowledge_graph.get_edge_data(predecessor, concept)
            rel = edge.get('relation', 'related_to')
            info.append(f"{predecessor} {rel} {concept}")
            
        return info

    # --- 検索 API ---

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        ハイブリッド検索: ベクトル検索結果 + 関連するグラフ知識
        """
        results = []
        
        # 1. ベクトル検索
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=k)
            results.extend([d.page_content for d in docs])
            
            # 2. ベクトル検索でヒットした主要な単語(主語)についてグラフを探索
            for doc in docs:
                # メタデータに主語があればそれを使う
                subj = doc.metadata.get("subj")
                if subj:
                    graph_info = self.get_subgraph_info(subj)
                    results.extend(graph_info)
        
        return list(set(results)) # 重複排除

    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """テキストをベクトルストアに追加"""
        if self.vector_store is None:
             self.vector_store = FAISS.from_texts([text], self.embedding_model, metadatas=[metadata or {}])
        else:
             self.vector_store.add_texts([text], metadatas=[metadata or {}])
        self.vector_store.save_local(self.vector_store_path)

    # 互換性のためのラッパー
    def add_relationship(self, source: str, relation: str, target: str):
        self.add_triple(source, relation, target)
    
    def add_causal_relationship(self, cause: str, effect: str, condition: Optional[str] = None):
        rel = "causes" if not condition else f"causes_under_{condition}"
        self.add_triple(cause, rel, effect)