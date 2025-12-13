# ファイルパス: snn_research/training/trainers/__init__.py
# タイトル: トレーナーモジュール初期化 (修正版)
# 機能説明: 
#   各種学習パラダイムに対応したトレーナーをエクスポートする。
#   mypyエラー修正: STDPTrainerの追加、ProbabilisticTrainerの名称確認。

from .bptt import BPTTTrainer
from .stdp import STDPTrainer
from .distillation import DistillationTrainer
# probabilistic.py 内の実装を確認し、クラス名が ProbabilisticEnsembleTrainer ならエイリアスを貼る
from .probabilistic import ProbabilisticTrainer
from .physics_informed import PhysicsInformedTrainer
from .self_supervised import SelfSupervisedTrainer
from .particle_filter import ParticleFilterTrainer
from .planner import PlannerTrainer
from .breakthrough import BreakthroughTrainer
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
    "ForwardForwardTrainer",
]
