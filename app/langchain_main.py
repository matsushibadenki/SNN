# matsushibadenki/snn4/app/langchain_main.py
# LangChainと連携したSNNチャットアプリケーション
#
# 機能:
# - ロードマップ フェーズ2「2.4. プロトタイプ開発」に対応。
# - DIコンテナからSNNLangChainAdapterを取得。
# - LangChain Expression Language (LCEL) を使用して、モダンなチェインを構築。
# - 共通UIビルダー関数を呼び出してUIを構築・起動する。
# - --model_config 引数を追加し、ベース設定とモデル設定を分けて読み込めるようにした。
# - 修正: Gradio 4.20.0 との互換性のために .queue() を削除し、app/utils.py 経由で queue=False を設定
#
# 修正 (v5):
# - mypyエラー [import-untyped] を解消。

import gradio as gr  # type: ignore[import-untyped]
import argparse
import sys
import time
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- ▼ 修正: List[List[Optional[str]]] をインポート ▼ ---
from typing import Iterator, Tuple, List, Dict, Optional 
# --- ▲ 修正 ▲ ---

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import AppContainer
from app.utils import build_gradio_ui

def main():
    parser = argparse.ArgumentParser(description="SNN + LangChain 連携AIプロトタイプ")
    parser.add_argument("--config", type=str, default="configs/templates/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイルのパス")
    parser.add_argument("--model_path", type=str, help="モデルのパス (設定ファイルを上書き)")
    args = parser.parse_args()

    # DIコンテナを初期化
    container = AppContainer()
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)

    # コマンドラインからモデルパスが指定された場合は、設定を上書き
    if args.model_path:
        container.config.model.path.from_value(args.model_path)

    # コンテナからLangChainアダプタを取得
    snn_llm = container.langchain_adapter()
    print(f"Loading SNN model from: {container.config.model.path()}")
    print("✅ SNN model loaded and wrapped for LangChain successfully.")

    # LangChain Expression Language (LCEL) を使用してチェインを構築
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは、簡潔で役立つアシスタントです。ユーザーからの質問に日本語で答えてください。"),
        ("user", "{question}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | snn_llm | output_parser

    # --- ▼ 修正: history の型ヒントを List[List[Optional[str]]] に変更 ▼ ---
    def stream_response(message: str, history: List[List[Optional[str]]]) -> Iterator[Tuple[List[List[Optional[str]]], str]]:
    # --- ▲ 修正 ▲ ---
        """GradioのBlocks UIのために、チャット履歴と統計情報をストリーミング生成する。"""
        history.append([message, ""])
        
        print("-" * 30)
        print(f"Input question to LCEL Chain: {message}")
        
        start_time = time.time()
        full_response = ""
        token_count = 0
        
        # LCELチェインのstreamメソッドを使用
        for chunk in chain.stream({"question": message}):
            full_response += chunk
            token_count += 1
            history[-1][1] = full_response
            
            duration = time.time() - start_time
            stats = snn_llm.snn_engine.last_inference_stats
            total_spikes = stats.get("total_spikes", 0)
            spikes_per_second = total_spikes / duration if duration > 0 else 0
            tokens_per_second = token_count / duration if duration > 0 else 0

            stats_md = f"""
            **Inference Time:** `{duration:.2f} s`
            **Tokens/Second:** `{tokens_per_second:.2f}`
            ---
            **Total Spikes:** `{total_spikes:,.0f}`
            **Spikes/Second:** `{spikes_per_second:,.0f}`
            """
            
            yield history, stats_md

        duration = time.time() - start_time
        stats = snn_llm.snn_engine.last_inference_stats
        total_spikes = stats.get("total_spikes", 0)
        print(f"\nGenerated response: {full_response.strip()}")
        print(f"Inference time: {duration:.4f} seconds")
        print(f"Total spikes: {total_spikes:,.0f}")
        print("-" * 30)

    # 共通UIビルダーを使用してUIを構築
    demo = build_gradio_ui(
        stream_fn=stream_response,
        title="🤖 SNN + LangChain Prototype (LCEL)",
        description="""
        SNNモデルをLangChain Expression Language (LCEL)経由で利用するプロトタイプ。
        右側のパネルには、推論時間や総スパイク数などの統計情報がリアルタイムで表示されます。
        """,
        chatbot_label="SNN+LangChain Chat",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime")
    )
    
    # Webアプリケーションの起動
    server_port = container.config.app.server_port() + 1 # ポートが衝突しないように+1する
    print("\nStarting Gradio web server for LangChain app...")
    print(f"Please open http://{container.config.app.server_name()}:{server_port} in your browser.")
    # --- ▼ 修正: .queue() を削除 ▼ ---
    demo.launch(
        server_name=container.config.app.server_name(),
        server_port=server_port,
    )
    # --- ▲ 修正 ▲ ---

if __name__ == "__main__":
    main()