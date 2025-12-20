# matsushibadenki/snn2/scripts/build_knowledge_base.py
# Phase 3: RAG-SNNのための知識ベース（ベクトルストア）を構築するスクリプト

import sys
from pathlib import Path
import argparse

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from snn_research.cognitive_architecture.rag_snn import RAGSystem

def main():
    """
    ドキュメントとエージェントの記憶から、RAGシステムのための
    ベクトルストア（知識ベース）を構築する。
    """
    parser = argparse.ArgumentParser(
        description="RAG-SNN知識ベース構築ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default="doc",
        help="知識源となるドキュメントが格納されているディレクトリ。\n例: 'doc'"
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="runs/agent_memory.jsonl",
        help="エージェントの長期記憶ログファイル。\n例: 'runs/agent_memory.jsonl'"
    )
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="runs/vector_store",
        help="構築したベクトルストアの保存先パス。"
    )
    args = parser.parse_args()

    print("RAG知識ベースの構築を開始します...")
    
    rag_system = RAGSystem(vector_store_path=args.vector_store_path)
    rag_system.setup_vector_store(
        knowledge_dir=args.knowledge_dir,
        memory_file=args.memory_file
    )
    
    print("\n知識ベースの構築が正常に完了しました。")
    print(f"ベクトルストアは '{args.vector_store_path}' に保存されました。")

if __name__ == "__main__":
    main()