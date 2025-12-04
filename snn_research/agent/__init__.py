# matsushibadenki/snn4/snn_research/agent/__init__.py
# 修正: SelfEvolvingAgentMaster に対応

from .autonomous_agent import AutonomousAgent
from .memory import Memory
# --- ▼ 修正 ▼ ---
from .self_evolving_agent import SelfEvolvingAgentMaster # クラス名を Master に変更
# --- ▲ 修正 ▲ ---
from .digital_life_form import DigitalLifeForm
from .reinforcement_learner_agent import ReinforcementLearnerAgent

__all__ = ["AutonomousAgent", "Memory",
           # --- ▼ 修正 ▼ ---
           "SelfEvolvingAgentMaster", # クラス名を Master に変更
           # --- ▲ 修正 ▲ ---
           "DigitalLifeForm", "ReinforcementLearnerAgent"]