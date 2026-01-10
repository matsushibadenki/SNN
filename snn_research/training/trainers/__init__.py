# ファイルパス: snn_research/training/trainers/__init__.py
# タイトル: トレーナーモジュール初期化 (No-BP Fix)
# 修正: BPTTTrainer (誤差逆伝播) を削除

from .stdp import STDPTrainer
from .distillation import DistillationTrainer
from .probabilistic import ProbabilisticTrainer
from .physics_informed import PhysicsInformedTrainer
from .self_supervised import SelfSupervisedTrainer
from .particle_filter import ParticleFilterTrainer
from .planner import PlannerTrainer
from .breakthrough import BreakthroughTrainer
from .forward_forward import ForwardForwardTrainer

__all__ = [
    "STDPTrainer",
    "DistillationTrainer",
    "ProbabilisticTrainer",
    "PhysicsInformedTrainer",
    "SelfSupervisedTrainer",
    "ParticleFilterTrainer",
    "PlannerTrainer",
    "BreakthroughTrainer",
    "ForwardForwardTrainer",
]
