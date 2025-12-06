# ファイルパス: snn_research/training/trainers/__init__.py
# Title: トレーナーパッケージ初期化 (修正版)
# Description:
#   各モジュールに分割されたトレーナークラスをまとめてエクスポートします。
#   これにより、外部からは `from snn_research.training.trainers import BreakthroughTrainer`
#   のようにアクセス可能です。

from .breakthrough import BreakthroughTrainer
from .distillation import DistillationTrainer
from .self_supervised import SelfSupervisedTrainer
from .physics_informed import PhysicsInformedTrainer
from .probabilistic import ProbabilisticEnsembleTrainer
from .planner import PlannerTrainer
from .particle_filter import ParticleFilterTrainer
from .bptt import BPTTTrainer

__all__ = [
    "BreakthroughTrainer",
    "DistillationTrainer",
    "SelfSupervisedTrainer",
    "PhysicsInformedTrainer",
    "ProbabilisticEnsembleTrainer",
    "PlannerTrainer",
    "ParticleFilterTrainer",
    "BPTTTrainer"
]
