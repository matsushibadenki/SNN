# snn_research/cognitive_architecture/__init__.py
# Title: Cognitive Architecture Init
# Description: 認知アーキテクチャコンポーネントのエクスポート定義。

from .artificial_brain import ArtificialBrain
from .global_workspace import GlobalWorkspace
from .perception_cortex import PerceptionCortex
from .motor_cortex import MotorCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .amygdala import Amygdala
from .astrocyte_network import AstrocyteNetwork
from .neuromorphic_scheduler import NeuromorphicScheduler
from .rag_snn import RAGSystem
from .symbol_grounding import SymbolGrounding
from .planner_snn import PlannerSNN
from .reasoning_engine import ReasoningEngine
from .meta_cognitive_snn import MetaCognitiveSNN

# New Components for Phase 1-4
from .memory_consolidation import HierarchicalMemorySystem
from .sleep_consolidation import SleepConsolidator
from .adaptive_moe import AdaptiveFrankenMoE
from .delta_learning import DeltaLearningSystem
from .neuro_symbolic_bridge import NeuroSymbolicBridge

from .intrinsic_motivation import IntrinsicMotivationSystem
from .causal_inference_engine import CausalInferenceEngine
from .emergent_system import EmergentCognitiveSystem
from .physics_evaluator import PhysicsEvaluator
from .som_feature_map import SomFeatureMap
from .hybrid_perception_cortex import HybridPerceptionCortex

from .hdc_engine import HDCEngine, HDCReasoningAgent
from .tsetlin_machine import TsetlinMachine

# Phase 2.1: Knowledge Graph Integration
from .curiosity_knowledge_integrator import CuriosityKnowledgeIntegrator, create_curiosity_integrator
from .theory_of_mind import TheoryOfMind
from .explainability import ExplainabilityEngine

__all__ = [
    "ArtificialBrain",
    "GlobalWorkspace",
    "PerceptionCortex",
    "MotorCortex",
    "PrefrontalCortex",
    "Hippocampus",
    "BasalGanglia",
    "Cerebellum",
    "Amygdala",
    "AstrocyteNetwork",
    "NeuromorphicScheduler",
    "RAGSystem",
    "SymbolGrounding",
    "PlannerSNN",
    "ReasoningEngine",
    "MetaCognitiveSNN",
    "SleepConsolidator",
    "HierarchicalMemorySystem",  # Added
    "AdaptiveFrankenMoE",       # Added
    "DeltaLearningSystem",      # Added
    "NeuroSymbolicBridge",      # Added
    "IntrinsicMotivationSystem",
    "CausalInferenceEngine",
    "EmergentCognitiveSystem",
    "PhysicsEvaluator",
    "SomFeatureMap",
    "HybridPerceptionCortex",
    "HDCEngine",
    "HDCReasoningAgent",
    "TsetlinMachine",
    # Phase 2.1
    "CuriosityKnowledgeIntegrator",
    "create_curiosity_integrator",
    # Phase 3.1
    "TheoryOfMind",
    "ExplainabilityEngine",
]
