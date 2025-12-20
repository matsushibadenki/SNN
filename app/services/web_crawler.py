# snn_research/tools/web_crawler.py
# Title: Web Crawler Tool
# Description: 指定されたURLからWebページのコンテンツを取得し、HTMLからテキストデータを抽出するツール。
# 改善点: URLがMarkdownリンク形式で渡された場合や、末尾に不要な文字が含まれている場合でも対応できるようにサニタイズ処理を強化。

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse
from typing import Set, List, Optional
import time
import os
import json
import re

class WebCrawler:
    """
    Webを巡回し、テキストコンテンツを収集するシンプルなクローラー。
    """
    def __init__(self, output_dir: str = "workspace/web_data"):
        self.visited_urls: Set[str] = set()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_url(self, url: str) -> str:
        """Markdownリンク形式や不要な文字が含まれるURLをサニタイズする。"""
        # [https://example.com](...) or (https://example.com) のような形式からURLを抽出
        match = re.search(r'https?://[^\]\)]+', url)
        if match:
            # 抽出したURLの末尾にある可能性のある不要な文字（バックスラッシュなど）を削除
            return match.group(0).rstrip('\\/')
        return url.rstrip('\\/')

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """巡回対象として適切なURLか判断する。"""
        parsed_url = urlparse(url)
        return (
            parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == base_domain and
            url not in self.visited_urls
        )

    def _extract_text_from_html(self, html_content: str) -> str:
        """BeautifulSoupを使ってHTMLから主要なテキストを抽出する。"""
        soup = BeautifulSoup(html_content, 'html.parser')
        # ヘッダー、フッター、ナビゲーションなどの不要な部分を削除
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        return text

    def crawl(self, start_url: str, max_pages: int = 5) -> str:
        """
        指定されたURLからクロールを開始し、収集したテキストデータをファイルに保存する。

        Returns:
            str: 保存されたjsonlファイルのパス。
        """
        sanitized_start_url = self._sanitize_url(start_url)
        urls_to_visit: List[str] = [sanitized_start_url]
        base_domain = urlparse(sanitized_start_url).netloc
        
        output_filename = f"crawled_data_{int(time.time())}.jsonl"
        output_filepath = os.path.join(self.output_dir, output_filename)

        page_count = 0
        with open(output_filepath, 'w', encoding='utf-8') as f:
            while urls_to_visit and page_count < max_pages:
                current_url = urls_to_visit.pop(0)
                if not self._is_valid_url(current_url, base_domain):
                    continue

                try:
                    print(f"📄 クロール中: {current_url}")
                    # ユーザーエージェントを設定して、ブロックされるリスクを低減
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
                    response = requests.get(current_url, timeout=10, headers=headers)
                    response.raise_for_status()
                    self.visited_urls.add(current_url)
                    page_count += 1

                    text_content = self._extract_text_from_html(response.text)
                    
                    if text_content:
                        record = {"text": text_content, "source_url": current_url}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        if isinstance(link, Tag):
                            href = link.get('href')
                            if isinstance(href, str):
                                absolute_link = urljoin(current_url, href)
                                if self._is_valid_url(absolute_link, base_domain):
                                    urls_to_visit.append(absolute_link)
                    
                    time.sleep(1)

                except requests.RequestException as e:
                    print(f"❌ クロールエラー: {current_url} ({e})")

        print(f"✅ クロール完了。{page_count}ページのデータを '{output_filepath}' に保存しました。")
        return output_filepath