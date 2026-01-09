# ファイルパス: app/services/web_crawler.py
# タイトル: 強化版 Web Crawler Service
# 目的: 実際のWebページからのコンテンツ抽出、クリーニング、および接続不可時の高度なフォールバック機能を提供する。
# 内容:
#   - requestsとBeautifulSoupを用いた実際のスクレイピング機能
#   - HTMLタグ除去、ノイズ除去などのテキスト正規化
#   - 知識ベースを用いた模擬検索機能（オフライン/エラー時用）
#   - 複数のトピックに関連する情報を収集・統合する機能

from typing import List, Optional, Set
import random
import re
import os
import time

# 外部ライブラリのインポート（環境にない場合のフォールバック付き）
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_ACCESS = True
except ImportError:
    HAS_WEB_ACCESS = False
    print(
        "⚠️ [WebCrawler] 'requests' or 'bs4' not found. Running in Offline Mock Mode.")


class WebCrawler:
    def __init__(self, user_agent: str = "Mozilla/5.0 (compatible; SNN-Bot/1.0)"):
        self.user_agent = user_agent
        self.visited_urls: Set[str] = set()

        # 模擬インターネット空間（オフライン/テスト用）
        self.knowledge_base = {
            "pattern": [
                "Patterns are regularities in the world defined by mathematical structures.",
                "Fractals serve as efficient compression of visual data in nature.",
                "Temporal patterns indicate causality and sequence in neural processing.",
                "Symmetry breaking leads to diverse pattern formation in biology."
            ],
            "snn": [
                "Spiking Neural Networks replicate brain dynamics using discrete events.",
                "Energy efficiency comes from event-driven processing and sparse coding.",
                "STDP (Spike-Timing-Dependent Plasticity) is a local learning rule for synapses.",
                "Neuromorphic hardware minimizes the Von Neumann bottleneck."
            ],
            "ai": [
                "Artificial Intelligence is transforming industries through automation.",
                "Deep learning requires massive data and computational resources.",
                "Neuromorphic computing is the next wave of AI efficiency.",
                "Neuro-symbolic AI bridges the gap between logic and neural networks."
            ],
            "brain": [
                "The brain operates on approximately 20 watts of power.",
                "Synaptic plasticity is the biological basis of learning and memory.",
                "Sleep plays a crucial role in memory consolidation.",
                "Predictive coding suggests the brain constantly generates models of the world."
            ],
            "default": [
                "The world is full of unknown data waiting to be structured.",
                "Learning is the process of reducing surprise (free energy minimization).",
                "Curiosity drives exploration towards novel information.",
                "Feedback loops are essential for self-regulating systems."
            ]
        }

    def _clean_text(self, text: str) -> str:
        """テキストから不要な空白や特殊文字を除去する。"""
        # 改行をスペースに置換
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        # 連続するスペースを1つに
        text = re.sub(r'\s+', ' ', text)
        # 前後の空白削除
        return text.strip()

    def _fetch_page(self, url: str) -> Optional[str]:
        """指定されたURLのHTMLを取得する（実通信）。"""
        if not HAS_WEB_ACCESS:
            return None

        try:
            # 安全のため、特定のドメインや過度なアクセスを制限するロジックをここに挟むのが理想
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"    ⚠️ [Web] Connection failed for {url}: {e}")
            return None

    def _extract_content_from_html(self, html: str) -> List[str]:
        """HTMLから有益なテキストコンテンツを抽出する。"""
        if not HAS_WEB_ACCESS or not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # スクリプトとスタイルを除去
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        # 本文らしいテキストを抽出（pタグやliタグを中心に）
        lines = []
        for tag in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']):
            text = self._clean_text(tag.get_text())
            # 短すぎる行や意味のない行を除外
            if len(text) > 30:
                lines.append(text)

        return lines

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """HTMLから次の探索候補となるリンクを抽出する。"""
        if not HAS_WEB_ACCESS or not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # 簡易的なURL正規化（相対パス対応などが必要だがここでは省略）
            if href.startswith('http'):
                links.append(href)

        # ランダムにシャッフルして返す
        random.shuffle(links)
        return links

    def search(self, query: str) -> List[str]:
        """
        クエリに基づいて情報を検索する。
        オンラインなら検索エンジンの結果（を模倣）、オフラインなら知識ベースを使用。
        """
        print(f"    🔍 [Web] Searching knowledge for: '{query}'...")
        results = []
        query_lower = query.lower()

        # 1. 知識ベースからの検索（高速・確実）
        hit = False
        for key, facts in self.knowledge_base.items():
            if key in query_lower:
                results.extend(facts)
                hit = True

        # 2. オフラインでヒットしなかった場合のフォールバック
        if not hit:
            # 関連しそうなキーをランダムに選ぶ（連想）
            random_key = random.choice(list(self.knowledge_base.keys()))
            results.extend(self.knowledge_base[random_key])
            print(
                f"    💡 [Web] No direct hit. Associating with '{random_key}'...")

        random.shuffle(results)
        return results[:5]

    def crawl(self, start_url: str, max_pages: int = 5, topic_filter: Optional[str] = None) -> str:
        """
        指定されたURLからクローリングを行い、結果をテキストファイルに保存する。
        オンライン環境であれば実際にWebアクセスを試み、失敗すればモックデータを使用する。

        Args:
            start_url: 開始URL
            max_pages: クロールする最大ページ数
            topic_filter: 特定のトピック（文字列）に関連する情報のみを優先する（オプション）

        Returns:
            保存されたファイルのパス
        """
        print(
            f"    🕷️ [Web] Crawling started. Root: '{start_url}' (Max: {max_pages})")

        collected_data = []
        queue = [start_url]
        pages_crawled = 0

        # モック判定: URLがダミーっぽい、またはライブラリがない場合
        is_mock_url = "http" not in start_url or "example.com" in start_url
        use_mock_mode = not HAS_WEB_ACCESS or is_mock_url

        if use_mock_mode:
            print("    🤖 [Web] Running in SIMULATION mode.")
            # モック: トピックに関連する情報を生成
            base_topics = ["ai", "snn", "brain", "pattern"]
            if topic_filter:
                base_topics.insert(0, topic_filter)

            for _ in range(max_pages):
                topic = random.choice(base_topics)
                facts = self.search(topic)
                collected_data.append(
                    f"\n--- Simulated Page about {topic.upper()} ---\n")
                collected_data.extend([f"- {fact}" for fact in facts])
                pages_crawled += 1

        else:
            # 実クローリングループ
            while queue and pages_crawled < max_pages:
                url = queue.pop(0)
                if url in self.visited_urls:
                    continue

                print(f"      Reading: {url} ...")
                html = self._fetch_page(url)
                self.visited_urls.add(url)

                if html:
                    content_lines = self._extract_content_from_html(html)
                    if content_lines:
                        # トピックフィルタがある場合、関連ワードが含まれるか簡易チェック
                        if topic_filter and topic_filter.lower() not in html.lower():
                            pass  # 関連性が低そうならスキップ、または優先度を下げる
                        else:
                            collected_data.append(f"\n--- Source: {url} ---\n")
                            collected_data.extend(content_lines)
                            pages_crawled += 1

                    # 次のリンクを取得してキューに追加
                    new_links = self._extract_links(html, url)
                    queue.extend(new_links[:3])  # 1ページあたり最大3リンクを追加

                time.sleep(1)  # マナーのための待機

        # 結果の保存
        if not collected_data:
            print(
                "    ⚠️ [Web] No data collected. Generating default knowledge.")
            collected_data = self.search("default")

        output_dir = "data/crawled"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"web_knowledge_{timestamp}.txt"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Crawl Root: {start_url}\n")
            f.write(f"Topic Filter: {topic_filter}\n")
            f.write(f"Date: {time.ctime()}\n")
            f.write("========================================\n\n")
            f.write("\n".join(collected_data))

        print(
            f"    📄 [Web] Saved {len(collected_data)} lines to {output_path}")
        return output_path
