# ファイルパス: snn_research/training/trainers/__init__.py
# タイトル: トレーナーモジュール初期化
# 機能説明: 
#   各種学習パラダイムに対応したトレーナーをエクスポートする。
#   v2.6追加: Forward-Forward Trainerのエクスポート。

from .bptt import BPTTTrainer
from .stdp import STDPTrainer
from .distillation import DistillationTrainer
from .probabilistic import ProbabilisticTrainer
from .physics_informed import PhysicsInformedTrainer
from .self_supervised import SelfSupervisedTrainer
from .particle_filter import ParticleFilterTrainer
from .planner import PlannerTrainer
from .breakthrough import BreakthroughTrainer

# --- Non-BP / Forward-Only Trainers ---
from .forward_forward import ForwardForwardTrainer

__all__ = [
    "BPTTTrainer",
    "STDPTrainer",
    "DistillationTrainer",
    "ProbabilisticTrainer",
    "PhysicsInformedTrainer",
    "SelfSupervisedTrainer",
    "ParticleFilterTrainer",
    "PlannerTrainer",
    "BreakthroughTrainer",
    # New
    "ForwardForwardTrainer",
]
