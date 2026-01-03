# ファイルパス: snn_research/agent/__init__.py
# Title: Agent Package Init
# 機能: エージェントモジュールの公開

from .base_agent import BaseAgent
from .reinforcement_learner_agent import ReinforcementLearnerAgent
# 既存のエージェント（エラーが出ていたがWebCrawler修正により復旧）
try:
    from .self_evolving_agent import SelfEvolvingAgentMaster
except ImportError:
    pass

# 新しい自律エージェント
from .autonomous_agent import AutonomousAgent

__all__ = [
    "BaseAgent",
    "ReinforcementLearnerAgent",
    "SelfEvolvingAgentMaster",
    "AutonomousAgent",
]