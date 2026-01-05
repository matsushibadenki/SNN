# ファイルパス: app/services/chat_service.py
# (動的ロードUI対応 v4 - max_len 修正)
# チャット機能のビジネスロジックを担うサービス
#
# 機能:
# - DIコンテナから推論エンジンを受け取る。
# - Gradioからの入力を処理し、整形して推論エンジンに渡す。
# - 推論結果をGradioが扱える形式で返す。
# - ストリーミング応答をサポート。
# - 推論完了後に総スパイク数をコンソールに出力。
# - UI表示用に、リアルタイムの統計情報も生成する。
# - __init__ から max_len を削除し、推論エンジンから取得するよう修正。

import time
from app.deployment import SNNInferenceEngine
from typing import Iterator, Tuple, List, Optional
from omegaconf import OmegaConf

class ChatService:
    def __init__(self, snn_engine: SNNInferenceEngine):
        """
        ChatServiceを初期化します。

        Args:
            snn_engine: テキスト生成に使用するSNN推論エンジン。
        """
        self.snn_engine = snn_engine

    def stream_response(self, message: str, history: List[List[Optional[str]]]) -> Iterator[Tuple[List[List[Optional[str]]], str]]:
        """
        GradioのBlocks UIのために、チャット履歴と統計情報をストリーミング生成する。
        """
        # --- ▼ 修正: SNNInferenceEngine の config から max_len を取得 ▼ ---
        max_len_config = OmegaConf.select(self.snn_engine.config, "app.max_len", default=100)
        max_len = int(max_len_config) if max_len_config is not None else 100
        # --- ▲ 修正 ▲ ---

        # 履歴を List[List[str]] からプロンプト文字列に変換
        prompt = ""
        for pair in history:
            user_msg = pair[0]
            bot_msg = pair[1]
            if user_msg is not None:
                prompt += f"User: {user_msg}\n"
            if bot_msg is not None:
                prompt += f"Assistant: {bot_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        # 新しい応答を履歴に追加 (Gradioの List[List[str]] 形式)
        history.append([message, ""])

        print("-" * 30)
        print(f"Input prompt to SNN (len: {len(prompt)}):\n{prompt}")

        start_time = time.time()
        
        full_response = ""
        token_count = 0
        
        # SNNInferenceEngineのgenerateメソッドを呼び出し
        for chunk, stats in self.snn_engine.generate(prompt, max_len=max_len, stop_sequences=["User:"]):
            full_response += chunk
            token_count += 1
            history[-1][1] = full_response # 最後のペアのアシスタント応答を更新
            
            duration = time.time() - start_time
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
            
            yield history, stats_md # 更新された履歴全体と統計を返す

        # Final log to console
        duration = time.time() - start_time
        # ループ終了後の最終的な統計情報を取得
        final_stats = self.snn_engine.last_inference_stats
        total_spikes = final_stats.get("total_spikes", 0)
        print(f"\nGenerated response: {full_response.strip()}")
        print(f"Inference time: {duration:.4f} seconds")
        print(f"Total spikes: {total_spikes:,.0f}")
        print("-" * 30)

