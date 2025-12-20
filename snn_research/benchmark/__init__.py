# ファイルパス: snn_research/benchmark/__init__.py
# Title: ベンチマークモジュール初期化
# Description:
#   ベンチマークタスクと関連クラスをエクスポートする。
#
# 修正:
#   - tasks モジュールから正しいクラスをインポートするように修正。
#   - 存在しないタスククラス (SST2Task 等) の参照を削除または修正。

from .tasks import (
    BenchmarkTask,
    ImageClassificationTask,
    NLPTask,
    TASK_REGISTRY
)

# tasks.py に定義されていないクラスをインポートしようとするとエラーになるため、
# 存在するクラスのみを公開する形に修正。
# 必要であればエイリアスを作成する。

CIFAR10Task = ImageClassificationTask
SST2Task = NLPTask
# 他のタスクも汎用クラスにマッピング (仮)
MRPCTask = NLPTask
CIFAR10DVSTask = ImageClassificationTask
SHDTask = ImageClassificationTask # 仮

__all__ = [
    "BenchmarkTask",
    "ImageClassificationTask",
    "NLPTask",
    "TASK_REGISTRY",
    "CIFAR10Task",
    "SST2Task",
    "MRPCTask",
    "CIFAR10DVSTask",
    "SHDTask"
]
