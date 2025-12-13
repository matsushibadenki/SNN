# ファイルパス: snn_research/cognitive_architecture/__init__.py
# タイトル: 認知アーキテクチャ パッケージ初期化
# 機能説明: 
#   主要な認知モジュールをエクスポートし、外部からのアクセスを容易にする。
#   v2.6追加: HDCエンジン、Tsetlin Machineのエクスポート。

from .artificial_brain import ArtificialBrain, BrainConfig
from .global_workspace import GlobalWorkspace
from .perception_cortex import PerceptionCortex
from .motor_cortex import MotorCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus, MemoryTrace
from .basal_ganglia import BasalGanglia, ActionSelection
from .cerebellum import Cerebellum
from .amygdala import Amygdala, EmotionalState
from .astrocyte_network import AstrocyteNetwork, MetabolicState
from .neuromorphic_scheduler import NeuromorphicScheduler, BrainState
from .rag_snn import RAGSystem
from .symbol_grounding import SymbolGrounding
from .planner_snn import PlannerSNN
from .reasoning_engine import ReasoningEngine
from .meta_cognitive_snn import MetaCognitiveSNN
from .sleep_consolidation import SleepConsolidation
from .intrinsic_motivation import IntrinsicMotivationSystem
from .causal_inference_engine import CausalInferenceEngine
from .emergent_system import EmergentSystem
from .physics_evaluator import PhysicsEvaluator
from .som_feature_map import SOMFeatureMap
from .hybrid_perception_cortex import HybridPerceptionCortex

# --- Green AI / Alternative Computing Modules ---
from .hdc_engine import HDCEngine, HDCReasoningAgent
from .tsetlin_machine import TsetlinMachine

__all__ = [
    "ArtificialBrain",
    "BrainConfig",
    "GlobalWorkspace",
    "PerceptionCortex",
    "MotorCortex",
    "PrefrontalCortex",
    "Hippocampus", "MemoryTrace",
    "BasalGanglia", "ActionSelection",
    "Cerebellum",
    "Amygdala", "EmotionalState",
    "AstrocyteNetwork", "MetabolicState",
    "NeuromorphicScheduler", "BrainState",
    "RAGSystem",
    "SymbolGrounding",
    "PlannerSNN",
    "ReasoningEngine",
    "MetaCognitiveSNN",
    "SleepConsolidation",
    "IntrinsicMotivationSystem",
    "CausalInferenceEngine",
    "EmergentSystem",
    "PhysicsEvaluator",
    "SOMFeatureMap",
    "HybridPerceptionCortex",
    # New
    "HDCEngine", "HDCReasoningAgent",
    "TsetlinMachine",
]
