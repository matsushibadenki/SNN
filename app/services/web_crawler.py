# ファイルパス: app/services/web_crawler.py
# Title: Web Crawler Service
# 修正: crawlメソッドの追加

from typing import List
import random

class WebCrawler:
    def __init__(self):
        # 模擬インターネット空間
        self.knowledge_base = {
            "pattern": [
                "Patterns are regularities in the world.",
                "Fractals serve as efficient compression of visual data.",
                "Temporal patterns indicate causality."
            ],
            "snn": [
                "Spiking Neural Networks replicate brain dynamics.",
                "Energy efficiency comes from event-driven processing.",
                "STDP is a local learning rule."
            ],
            "ai": [
                "AI is transforming industries.",
                "Deep learning requires massive data.",
                "Neuromorphic computing is the next wave."
            ],
            "default": [
                "The world is full of unknown data.",
                "Learning is the process of reducing surprise.",
                "Curiosity drives exploration."
            ]
        }

    def search(self, query: str) -> List[str]:
        """クエリに基づいて情報を検索する。"""
        print(f"    🌐 [Web] Searching for: '{query}'...")
        results = []
        query_lower = query.lower()
        
        hit = False
        for key, facts in self.knowledge_base.items():
            if key in query_lower:
                results.extend(facts)
                hit = True
        
        if not hit:
            results.extend(self.knowledge_base["default"])
            
        random.shuffle(results)
        return results[:2]

    def crawl(self, start_url: str, max_pages: int = 5) -> str:
        """
        指定されたURLからクローリングを行い、結果をテキストファイルに保存する（モック）。
        RunWebLearningスクリプト用。
        """
        print(f"    🕷️ [Web] Crawling from: '{start_url}' (Max: {max_pages})")
        
        # モックコンテンツ生成
        content = f"Source: {start_url}\n"
        content += "--- Extracted Content ---\n"
        
        # 検索機能を利用してコンテンツを生成
        topics = ["ai", "snn", "pattern"]
        for topic in topics:
            res = self.search(topic)
            for r in res:
                content += f"- {r}\n"
                
        output_path = "crawled_data_mock.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"    📄 [Web] Saved crawled data to {output_path}")
        return output_path