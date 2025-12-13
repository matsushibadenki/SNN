# ファイルパス: snn_research/cognitive_architecture/__init__.py
# タイトル: 認知アーキテクチャ パッケージ初期化
# 機能説明: 
#   主要な認知モジュールをエクスポートし、外部からのアクセスを容易にする。
#   修正: 存在しない属性(Config等)のインポート削除、クラス名ミスの修正。

# 修正: 各ファイルに存在しないクラス(BrainConfigなど)のインポートを削除
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
# 修正: クラス名を正しいものに変更 (SleepConsolidation -> SleepConsolidator)
from .sleep_consolidation import SleepConsolidator
from .intrinsic_motivation import IntrinsicMotivationSystem
from .causal_inference_engine import CausalInferenceEngine
# 修正: クラス名を正しいものに変更 (EmergentSystem -> EmergentCognitiveSystem)
from .emergent_system import EmergentCognitiveSystem
from .physics_evaluator import PhysicsEvaluator
# 修正: クラス名を正しいものに変更 (SOMFeatureMap -> SomFeatureMap)
from .som_feature_map import SomFeatureMap
from .hybrid_perception_cortex import HybridPerceptionCortex

# --- Green AI / Alternative Computing Modules ---
from .hdc_engine import HDCEngine, HDCReasoningAgent
from .tsetlin_machine import TsetlinMachine

__all__ = [
    "ArtificialBrain",
    # "BrainConfig", # 削除
    "GlobalWorkspace",
    "PerceptionCortex",
    "MotorCortex",
    "PrefrontalCortex",
    "Hippocampus", # "MemoryTrace", # 削除
    "BasalGanglia", # "ActionSelection", # 削除
    "Cerebellum",
    "Amygdala", # "EmotionalState", # 削除
    "AstrocyteNetwork", # "MetabolicState", # 削除
    "NeuromorphicScheduler", # "BrainState", # 削除
    "RAGSystem",
    "SymbolGrounding",
    "PlannerSNN",
    "ReasoningEngine",
    "MetaCognitiveSNN",
    "SleepConsolidator", # 修正
    "IntrinsicMotivationSystem",
    "CausalInferenceEngine",
    "EmergentCognitiveSystem", # 修正
    "PhysicsEvaluator",
    "SomFeatureMap", # 修正
    "HybridPerceptionCortex",
    # New
    "HDCEngine", "HDCReasoningAgent",
    "TsetlinMachine",
]
