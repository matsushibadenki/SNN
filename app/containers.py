# ファイルパス: app/containers.py
# Title: DIコンテナ (依存関係注入) - リファクタリング版
# Description:
# - アプリケーション全体の依存関係を管理するコンテナ群。
# - 修正: 複雑な lambda プロバイダーをヘルパー関数に切り出し、可読性とデバッグ性を向上。
# - 循環参照リスクを最小限にするため、インポートを整理。

import torch
import os
import logging
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import TYPE_CHECKING, Dict, Any, Optional, List

from snn_research.core.snn_core import SNNCore
from app.deployment import SNNInferenceEngine
from snn_research.training.losses import (
    CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss
)
from snn_research.training.trainers import (
    BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer, PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer
)
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .services.image_classification_service import ImageClassificationService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry
from app.services.web_crawler import WebCrawler
from snn_research.learning_rules import ProbabilisticHebbian, get_bio_learning_rule
from snn_research.models.bio.simple_network import BioSNN
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.active_inference_agent import ActiveInferenceAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.visual_perception import VisualCortex
from snn_research.core.cortical_column import CorticalColumn
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.benchmark import TASK_REGISTRY
from .utils import get_auto_device
from snn_research.agent.digital_life_form import DigitalLifeForm
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# --- ヘルパー関数 (Providers の lambda を置き換え) ---

def get_tokenizer(config_dict: Dict[str, Any]) -> PreTrainedTokenizerBase:
    cfg = OmegaConf.create(config_dict)
    tokenizer_name = OmegaConf.select(cfg, "data.tokenizer_name", default="gpt2")
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")

def create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=max(1, warmup_epochs))
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max(1, epochs - warmup_epochs))
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def load_planner_snn(planner_model: PlannerSNN, model_path: Optional[str], device: str) -> PlannerSNN:
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            planner_model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded PlannerSNN from '{model_path}'.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load PlannerSNN: {e}")
    return planner_model.to(device)

def get_registry_path(config: Dict[str, Any]) -> str:
    cfg = OmegaConf.create(config)
    return OmegaConf.select(cfg, "model_registry.file.path", default="runs/model_registry.json")

def get_vector_store_path(log_dir: Optional[str]) -> str:
    return os.path.join(log_dir, "vector_store") if log_dir else "runs/vector_store"

def get_memory_path(log_dir: Optional[str]) -> str:
    return os.path.join(log_dir, "agent_memory.jsonl") if log_dir else "runs/agent_memory.jsonl"

# --- コンテナ定義 ---

class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    device = providers.Factory(get_auto_device)
    tokenizer = providers.Factory(get_tokenizer, config_dict=config)
    
    # SNNモデル (ファクトリ利用)
    snn_model = providers.Factory(
        SNNCore, 
        config=config.model, 
        vocab_size=providers.Callable(lambda t: len(t), t=tokenizer)
    )
    
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn = providers.Factory(MetaCognitiveSNN)
    
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(
        create_scheduler, 
        optimizer=optimizer, 
        epochs=config.training.epochs, 
        warmup_epochs=config.training.gradient_based.warmup_epochs
    )
    
    # トレーナー群
    standard_trainer = providers.Factory(
        BreakthroughTrainer, 
        criterion=providers.Factory(CombinedLoss, tokenizer=tokenizer, ce_weight=config.training.gradient_based.loss.ce_weight), 
        grad_clip_norm=config.training.gradient_based.grad_clip_norm, 
        use_amp=config.training.gradient_based.use_amp, 
        log_dir=config.training.log_dir, 
        meta_cognitive_snn=meta_cognitive_snn
    )
    
    distillation_trainer = providers.Factory(
        DistillationTrainer, 
        criterion=providers.Factory(DistillationLoss, tokenizer=tokenizer, temperature=config.training.gradient_based.distillation.loss.temperature),
        grad_clip_norm=config.training.gradient_based.grad_clip_norm, 
        use_amp=config.training.gradient_based.use_amp, 
        log_dir=config.training.log_dir, 
        meta_cognitive_snn=meta_cognitive_snn
    )
    
    # Bio-RL関連
    synaptic_rule = providers.Factory(get_bio_learning_rule, name=config.training.biologically_plausible.learning_rule, params=config.training.biologically_plausible)
    homeostatic_rule = providers.Factory(get_bio_learning_rule, name="BCM", params=config.training.biologically_plausible)
    
    bio_rl_agent = providers.Factory(
        ReinforcementLearnerAgent, 
        input_size=4, output_size=4, device=device, 
        synaptic_rule=synaptic_rule, homeostatic_rule=homeostatic_rule
    )
    grid_world_env = providers.Factory(GridWorldEnv, device=device)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=bio_rl_agent, env=grid_world_env)
    
    # Plannerモデル (学習用)
    planner_snn = providers.Factory(
        PlannerSNN, 
        vocab_size=providers.Callable(lambda t: len(t), t=tokenizer),
        d_model=128, d_state=64, num_layers=2, time_steps=16, n_head=4, num_skills=10,
        neuron_config={'type': 'lif'}
    )
    
    # Active Inference Agent
    active_inference_agent = providers.Factory(
        ActiveInferenceAgent,
        generative_model=snn_model,
        num_actions=4, action_dim=128, observation_dim=128, hidden_dim=64, lr=0.001
    )
    
    # 確率的学習トレーナー
    probabilistic_trainer = providers.Factory(
        BioRLTrainer,
        agent=providers.Factory(
            ReinforcementLearnerAgent,
            input_size=4, output_size=4, device=device,
            synaptic_rule=providers.Factory(ProbabilisticHebbian, learning_rate=0.01),
            homeostatic_rule=None
        ),
        env=grid_world_env
    )
    
    # 他のトレーナー (省略せずに記述)
    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer,
        criterion=providers.Factory(PhysicsInformedLoss, tokenizer=tokenizer),
        grad_clip_norm=1.0, use_amp=True, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn
    )
    self_supervised_trainer = providers.Factory(
        SelfSupervisedTrainer,
        criterion=providers.Factory(SelfSupervisedLoss, tokenizer=tokenizer, prediction_weight=1.0, spike_reg_weight=0.0, mem_reg_weight=0.0),
        grad_clip_norm=1.0, use_amp=True, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn
    )
    probabilistic_ensemble_trainer = providers.Factory(
        ProbabilisticEnsembleTrainer,
        criterion=providers.Factory(ProbabilisticEnsembleLoss, tokenizer=tokenizer, ce_weight=1.0, variance_reg_weight=0.1),
        grad_clip_norm=1.0, use_amp=True, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn
    )
    particle_filter_trainer = providers.Factory(
        ParticleFilterTrainer,
        base_model=providers.Factory(BioSNN, layer_sizes=[10,5,2], neuron_params={'tau_mem':10.0, 'v_threshold':1.0, 'v_reset':0.0, 'v_rest':0.0}, synaptic_rule=providers.Object(None)),
        config=config, device=device
    )


class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)
    
    device = providers.Factory(get_auto_device)
    
    # Registry
    model_registry = providers.Selector(
        providers.Callable(lambda c: c.get("model_registry", {}).get("provider", "file"), c=config),
        file=providers.Singleton(SimpleModelRegistry, registry_path=providers.Factory(get_registry_path, config=config)),
        distributed=providers.Singleton(DistributedModelRegistry, registry_path=providers.Factory(get_registry_path, config=config))
    )
    
    web_crawler = providers.Singleton(WebCrawler)
    
    rag_system = providers.Factory(
        RAGSystem, 
        vector_store_path=providers.Factory(get_vector_store_path, log_dir=config.training.log_dir)
    )
    
    memory = providers.Factory(
        Memory, 
        rag_system=rag_system, 
        memory_path=providers.Factory(get_memory_path, log_dir=config.training.log_dir)
    )
    
    # 読み込み済みPlannerモデル
    loaded_planner_snn = providers.Factory(
        load_planner_snn,
        planner_model=training_container.planner_snn,
        model_path=config.training.planner.model_path,
        device=device
    )
    
    hierarchical_planner = providers.Factory(
        HierarchicalPlanner,
        model_registry=model_registry,
        rag_system=rag_system,
        memory=memory,
        planner_model=loaded_planner_snn,
        tokenizer_name=config.data.tokenizer_name,
        device=device
    )
    
    autonomous_agent = providers.Singleton(
        AutonomousAgent,
        name="AutonomousAgentBase",
        planner=hierarchical_planner,
        model_registry=model_registry,
        memory=memory,
        web_crawler=web_crawler
    )
    
    self_evolving_agent_master = providers.Singleton(
        SelfEvolvingAgentMaster,
        name="SelfEvolvingAgentMaster",
        planner=hierarchical_planner,
        model_registry=model_registry,
        memory=memory,
        web_crawler=web_crawler,
        meta_cognitive_snn=training_container.meta_cognitive_snn,
        motivation_system=providers.Singleton(IntrinsicMotivationSystem),
        model_config_path=config.model.path
    )

class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    model_registry = providers.Singleton(SimpleModelRegistry, registry_path="runs/model_registry.json")
    snn_inference_engine = providers.Factory(SNNInferenceEngine)
    
    chat_service = providers.Factory(ChatService, snn_engine=snn_inference_engine)
    image_classification_service = providers.Factory(ImageClassificationService, engine=snn_inference_engine)
    langchain_adapter = providers.Factory(SNNLangChainAdapter, snn_engine=snn_inference_engine)


class BrainContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # サブコンテナの統合
    agent_container = providers.Container(AgentContainer, config=config)
    app_container = providers.Container(AppContainer, config=config)
    
    device = providers.Factory(get_auto_device)

    # Core Components
    global_workspace = providers.Singleton(
        GlobalWorkspace, 
        model_registry=agent_container.model_registry
    )
    
    motivation_system = providers.Callable(lambda ag: ag.motivation_system, ag=agent_container.self_evolving_agent_master)
    
    # I/O
    sensory_receptor = providers.Singleton(SensoryReceptor)
    spike_encoder = providers.Singleton(SpikeEncoder, num_neurons=256)
    actuator = providers.Singleton(Actuator, actuator_name="voice_synthesizer")

    # Cognitive Modules
    cortical_column = providers.Factory(
        CorticalColumn, input_dim=256, output_dim=64, column_dim=128,
        neuron_config=config.training.biologically_plausible.neuron
    )
    
    perception_cortex = providers.Singleton(
        HybridPerceptionCortex, 
        workspace=global_workspace, num_neurons=256, feature_dim=64, som_map_size=(8, 8),
        stdp_params=config.training.biologically_plausible.stdp, cortical_column=cortical_column
    )
    
    visual_cortex = providers.Singleton(
        VisualCortex,
        workspace=global_workspace,
        vision_model_config=providers.Object({"architecture_type": "spiking_cnn", "num_classes": 128, "time_steps": 16}), 
        projector_config=providers.Object({"visual_dim": 128, "lang_dim": 256}),
        device=device 
    )

    prefrontal_cortex = providers.Singleton(PrefrontalCortex, workspace=global_workspace, motivation_system=motivation_system)
    hippocampus = providers.Singleton(Hippocampus, workspace=global_workspace, capacity=50)
    cortex = providers.Singleton(Cortex, rag_system=agent_container.rag_system)
    amygdala = providers.Singleton(Amygdala, workspace=global_workspace)
    basal_ganglia = providers.Singleton(BasalGanglia, workspace=global_workspace)
    cerebellum = providers.Singleton(Cerebellum)
    motor_cortex = providers.Singleton(MotorCortex, actuators=['voice_synthesizer'])
    
    causal_inference_engine = providers.Singleton(
        CausalInferenceEngine, 
        rag_system=agent_container.rag_system, workspace=global_workspace
    )
    
    symbol_grounding = providers.Singleton(SymbolGrounding, rag_system=agent_container.rag_system)

    # The Artificial Brain
    artificial_brain = providers.Singleton(
        ArtificialBrain, 
        global_workspace=global_workspace, 
        motivation_system=motivation_system, 
        sensory_receptor=sensory_receptor, 
        spike_encoder=spike_encoder, 
        actuator=actuator, 
        perception_cortex=perception_cortex, 
        visual_cortex=visual_cortex, 
        prefrontal_cortex=prefrontal_cortex, 
        hippocampus=hippocampus, 
        cortex=cortex, 
        amygdala=amygdala, 
        basal_ganglia=basal_ganglia, 
        cerebellum=cerebellum, 
        motor_cortex=motor_cortex, 
        causal_inference_engine=causal_inference_engine,
        symbol_grounding=symbol_grounding 
    )
    
    # Digital Life Form Integration
    digital_life_form = providers.Singleton(
        DigitalLifeForm,
        planner=agent_container.hierarchical_planner,
        autonomous_agent=agent_container.autonomous_agent,
        rl_agent=agent_container.training_container.bio_rl_agent,
        self_evolving_agent=agent_container.self_evolving_agent_master,
        motivation_system=motivation_system,
        meta_cognitive_snn=agent_container.training_container.meta_cognitive_snn,
        memory=agent_container.memory,
        physics_evaluator=providers.Singleton(PhysicsEvaluator),
        symbol_grounding=symbol_grounding,
        langchain_adapter=app_container.langchain_adapter,
        global_workspace=global_workspace,
        active_inference_agent=agent_container.training_container.active_inference_agent
    )
