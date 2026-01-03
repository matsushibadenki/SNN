# ファイルパス: scripts/runners/run_active_learning_loop.py
# 日本語タイトル: Active Learning Loop (能動学習実行スクリプト)
# 目的: 不確実性検知 -> Web検索 -> 蒸留学習 のサイクルを実証する。

import asyncio
import torch
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from app.services.web_crawler import WebCrawler

async def run_active_learning_demo():
    """
    不確実な事象に対し、能動的に情報を取得し、脳をアップデートするデモ。
    """
    # 1. 初期化 (Mock-safe な構成)
    brain = ArtificialBrain() 
    crawler = WebCrawler()
    
    print("🚀 Starting Active Learning Cycle...")

    # 未知の入力 (不確実性が高いと想定されるデータ)
    unknown_input = torch.randn(1, 3, 224, 224) 

    # 2. 知覚と判断
    cycle_result = brain.run_cognitive_cycle(unknown_input)
    uncertainty = cycle_result.get("uncertainty", 0.0)
    
    print(f"Initial Perception: Uncertainty = {uncertainty:.4f}")

    # 3. 能動的介入 (System 2 起動と外部検索)
    if uncertainty > 0.6: # 閾値を超えた場合に「能動学習」を開始
        print("🤔 High uncertainty detected. Activating WebCrawler for information gathering...")
        
        # 外部知識の取得
        external_info = await crawler.fetch_query("Next-generation SNN architectures")
        
        # 思考トレースの作成 (System 2 の推論結果として保存)
        trace = {
            "input": unknown_input,
            "thought_trace": f"Verified via web: {external_info[:50]}...",
            "final_answer": "Updated conceptual model of SNN"
        }
        
        # 4. 睡眠による記憶の更新
        print("🛌 Initiating Sleep Cycle for Distillation...")
        if brain.sleep_manager:
            brain.sleep_manager.add_experience(trace)
            brain.sleep_cycle()

    print("✅ Active learning cycle completed successfully.")

if __name__ == "__main__":
    asyncio.run(run_active_learning_demo())