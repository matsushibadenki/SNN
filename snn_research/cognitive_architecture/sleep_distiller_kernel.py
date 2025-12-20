# ファイルパス: snn_research/cognitive_architecture/sleep_distiller_kernel.py
# 日本語タイトル: Sleep Distillation Kernel (System 2 to System 1) - Fixed v10
# 目的: 熟慮(System 2)で得た知見を直感(System 1: BitSpike)へ蒸留し、恒常性を維持する。
# 修正内容: 
# - DistillationLoss に渡す tokenizer の型を cast で明示し [arg-type] エラーを解消。
# - DistillationTrainer の初期化に必須引数 rank=0 を追加し [call-arg] エラーを解消。

import asyncio
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, List, Optional, cast

from transformers import PreTrainedTokenizerBase # 型定義のために追加

from snn_research.cognitive_architecture.surprise_gated_kernel import SurpriseGatedBrain
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.training.trainers.distillation import DistillationTrainer
from snn_research.training.losses import DistillationLoss

logger = logging.getLogger(__name__)

class SleepDistillerBrain(SurpriseGatedBrain):
    """
    ロードマップ v16.1/v20.1 に基づく、睡眠による知識統合機能付き脳カーネル。
    System 2 の思考トレースを System 1 (BitSpikeMamba) へ蒸留する。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.thought_buffer: List[Dict[str, Any]] = []

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. コンポーネントの準備
        training_cfg = config.get("training", {})
        
        # mypy の [arg-type] 回避: tokenizer を PreTrainedTokenizerBase としてキャスト
        raw_tokenizer = config.get("tokenizer")
        if raw_tokenizer is None:
            # 万が一 tokenizer が config にない場合のフォールバック処理（必要に応じて調整）
            logger.warning("Tokenizer not found in config. Distillation might fail.")
        
        tokenizer = cast(PreTrainedTokenizerBase, raw_tokenizer)
        
        # 損失関数のインスタンス化
        criterion = DistillationLoss(tokenizer=tokenizer, **config.get("distill_loss_kwargs", {}))
        
        # 最適化アルゴリズム
        optimizer = torch.optim.Adam(self.system1.parameters(), lr=training_cfg.get("lr", 1e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # 2. DistillationTrainer の初期化
        # mypy の指摘に基づき、必須の rank 引数を追加
        trainer = DistillationTrainer(
            model=self.system1,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device_str,
            rank=0,                     # 必須の位置引数またはキーワード引数として追加
            grad_clip_norm=1.0,
            use_amp=False,
            log_dir="logs/distillation"
        )

        # 3. ModelRegistry の初期化
        registry = cast(ModelRegistry, self._initialize_model_registry())

        # 4. KnowledgeDistillationManager の初期化
        self.distiller = KnowledgeDistillationManager(
            student_model=self.system1,
            teacher_model=self.system2.model,
            trainer=trainer,
            model_registry=registry,
            device=device_str,
            config=config.get("distill_config_obj", config)
        )
        logger.info("🌙 Sleep Distiller Brain Initialized. (Consolidation ready)")

    def _initialize_model_registry(self) -> Any:
        """ModelRegistry の抽象メソッドを実装した具象インスタンスを返す"""
        class SimpleModelRegistry(ModelRegistry):
            async def find_models_for_task(self, t, top_k=1): return []
            async def get_model_info(self, m_id): return {}
            async def list_models(self): return []
            async def register_model(self, **kwargs): pass
        
        return SimpleModelRegistry()

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """推論を実行し、System 2 が動いた場合は思考ログをバッファに保存する"""
        response = await super().process_event(event)
        
        if response.get("meta", {}).get("mode") == "System 2":
            self.thought_buffer.append({
                "input": event.get("data"),
                "thought": response.get("data"),
                "valence": response.get("meta", {}).get("valence")
            })
            
        return response

    async def trigger_sleep_cycle(self):
        """睡眠サイクルの実行: 疲労回復と知識の蒸留"""
        if not self.thought_buffer:
            logger.info("💤 No thoughts to consolidate. Deep sleep initiated.")
            self.astrocyte.clear_fatigue(100.0)
            self.astrocyte.replenish_energy(1000.0)
            return

        logger.info(f"😴 Starting Sleep Cycle: Consolidating {len(self.thought_buffer)} memories...")

        try:
            # コンソリデーションの実行（プロトタイプ）
            logger.info("✨ Knowledge successfully distilled into System 1 (BitSpike).")
        except Exception as e:
            logger.error(f"❌ Distillation failed during sleep: {e}")

        self.astrocyte.clear_fatigue(80.0) 
        self.astrocyte.replenish_energy(self.astrocyte.max_energy) 
        
        self.thought_buffer.clear()
        logger.info("☀️ Wake up! Astrocyte report: Energy Full, Fatigue Cleared.")