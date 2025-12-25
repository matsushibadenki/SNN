# ファイルパス: snn_research/distillation/async_manager.py
# Title: Async Distillation Manager (Brain Integration)
# Description:
#   Brain Kernelの非同期イベントループ内で動作する蒸留マネージャー。
#   思考完了イベント(THOUGHT_COMPLETE)をフックし、
#   System 2の思考過程をSystem 1(BitSpikeMamba)にバックグラウンドで学習させる。

import asyncio
import logging
from typing import Any, Dict, Optional
import torch

from snn_research.distillation.thought_distiller import ThoughtDistillationManager

logger = logging.getLogger(__name__)

class AsyncDistillationManager:
    """
    非同期・知識蒸留マネージャー。
    System 2の思考結果をキューに溜め、バックグラウンドでSystem 1を再学習させる。
    """
    def __init__(self, system1_model: Any, teacher_engine: Any = None):
        self.manager = ThoughtDistillationManager(system1_model, teacher_engine)
        # [Fix] Added type annotation
        self.learning_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        logger.info("⚗️ Async Distillation Manager initialized.")

    async def start_worker(self):
        """学習ワーカーを起動"""
        self.is_running = True
        asyncio.create_task(self._learning_loop())

    async def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: Any, force_retrain: bool = False):
        """
        Web学習などからのリクエストを処理するインターフェース
        （Brain Kernelのシグネチャに合わせる）
        """
        # ここでは簡易的に、タスク記述をそのまま学習データとしてキューに入れる
        logger.info(f"⚗️ Schedule Distillation for: {task_description}")
        
        # ダミーの思考データを作成（本来はWeb検索結果から抽出）
        training_sample = {
            "input": task_description,
            "thought_chain": "Researching... Found verifiable facts... Analyzing...",
            "answer": f"Learned knowledge about {task_description}"
        }
        await self.learning_queue.put(training_sample)
        return True

    async def schedule_learning(self, thought_event_payload: Dict[str, Any]):
        """
        System 2の思考結果を学習キューに追加
        payload: {'input': str, 'thought': str, 'result': str}
        """
        if not isinstance(thought_event_payload, dict):
            return

        # 必要なフィールドがあるか確認
        if "input" in thought_event_payload and "thought" in thought_event_payload:
            logger.info("📥 Queuing new thought for distillation...")
            # 形式を変換
            sample = {
                "input": thought_event_payload["input"],
                "thought_chain": thought_event_payload["thought"],
                "answer": thought_event_payload.get("result", "")
            }
            await self.learning_queue.put(sample)

    async def _learning_loop(self):
        logger.info("⚗️ Distillation Worker Started (Background).")
        while self.is_running:
            try:
                # データが来るまで待機
                sample = await self.learning_queue.get()
                
                logger.info(f"🧠 Improving System 1 on: '{sample['input']}'...")
                
                # 同期的な学習処理をExecutorで実行（ブロック回避）
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._execute_training_step, sample)
                
                logger.info("✅ System 1 updated.")
                self.learning_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Distillation failed: {e}")

    def _execute_training_step(self, sample: Dict[str, Any]):
        """1サンプルに対する学習実行"""
        # ThoughtDistillationManagerのdistillメソッドを再利用
        # リスト形式で渡す
        self.manager.distill([sample], epochs=1)
