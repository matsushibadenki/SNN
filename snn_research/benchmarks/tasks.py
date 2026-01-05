"""
snn_research/benchmark/tasks.py
ベンチマークタスクの定義とレジストリ管理を行うモジュール
"""
import logging
from typing import Dict, Type, Any, Optional

logger = logging.getLogger(__name__)

class BenchmarkTask:
    """
    ベンチマークタスクの基底クラス。
    全ての具体的なベンチマークタスクはこのクラスを継承する必要があります。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def setup(self):
        """
        タスク実行前の準備（データロード、モデル構築など）
        """
        pass

    def run(self) -> Dict[str, Any]:
        """
        タスクの実行ロジック
        Returns:
            Dict[str, Any]: 評価指標などの結果辞書
        """
        raise NotImplementedError("Subclasses must implement run()")

    def cleanup(self):
        """
        リソースの解放など
        """
        pass

class TaskRegistry:
    """
    ベンチマークタスクを一元管理するレジストリクラス。
    デコレータを使用してタスクを登録します。
    """
    _registry: Dict[str, Type[BenchmarkTask]] = {}

    @classmethod
    def register(cls, name: str):
        """
        タスククラスを登録するためのデコレータ
        Args:
            name (str): タスクの一意な名前
        """
        def decorator(task_cls: Type[BenchmarkTask]):
            if name in cls._registry:
                logger.warning(f"Task '{name}' is already registered. Overwriting.")
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[BenchmarkTask]]:
        """
        名前からタスククラスを取得
        """
        return cls._registry.get(name)

    @classmethod
    def list_tasks(cls):
        """
        登録されているタスク名のリストを取得
        """
        return list(cls._registry.keys())

# --- 標準的なタスクの登録 ---

@TaskRegistry.register("health_check_comparison")
class HealthCheckTask(BenchmarkTask):
    """
    ヘルスチェック用の軽量な比較タスク（ダミー実装）
    """
    def run(self) -> Dict[str, Any]:
        logger.info("Running HealthCheckTask...")
        # 実際にはここで学習や推論を行うロジックを記述します
        # 今回はヘルスチェックを通すためのダミー結果を返します
        return {"accuracy": 0.95, "latency": 0.01, "status": "simulated_success"}

@TaskRegistry.register("image_classification")
class ImageClassificationTask(BenchmarkTask):
    """
    一般的な画像分類タスク
    """
    def run(self) -> Dict[str, Any]:
        logger.info("Running ImageClassificationTask...")
        return {"accuracy": 0.0, "loss": 0.0}

@TaskRegistry.register("cifar10_classification")
class CIFAR10Task(ImageClassificationTask):
    """
    CIFAR-10画像分類タスク
    """
    def run(self) -> Dict[str, Any]:
        logger.info("Running CIFAR10Task...")
        return {"top1_acc": 0.0}

@TaskRegistry.register("nlp_task")
class NLPTask(BenchmarkTask):
    """
    自然言語処理(NLP)タスクの基底クラス
    """
    def run(self) -> Dict[str, Any]:
        logger.info("Running NLPTask...")
        return {"perplexity": 0.0, "bleu": 0.0}

# 外部からの参照用にエイリアスを定義 (これが重要です)
TASK_REGISTRY = TaskRegistry