# ファイルパス: snn_research/training/trainers/physics_informed.py
# Title: Physics Informed Trainer
# Description: 物理的制約を取り入れた学習を行うトレーナー。

from .breakthrough import BreakthroughTrainer


class PhysicsInformedTrainer(BreakthroughTrainer):
    # BreakthroughTrainer のロジックをそのまま継承し、
    # 必要に応じて PhysicsInformedLoss と連携する。
    pass
