# ファイルパス: app/containers.py
# (修正: mypyエラー [attr-defined] 解消のため、castを追加)

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import os
from typing import TYPE_CHECKING, Dict, Any, cast, Optional, List, Union
from omegaconf import DictConfig, OmegaConf

# --- プロジェクト内モジュールのインポート ---
from snn_research.core.snn_core import SNNCore
from app.deployment import SNNInferenceEngine
from snn_research.training.losses import ( CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss )
from snn_research.training.trainers import ( BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer, PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer, PlannerTrainer )
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .services.image_classification_service import ImageClassificationService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry, ModelRegistry
from app.services.web_crawler import WebCrawler
from snn_research.learning_rules import ProbabilisticHebbian, get_bio_learning_rule, BioLearningRule, CausalTraceCreditAssignmentEnhancedV2
from snn_research.core.neurons import ProbabilisticLIFNeuron
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

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .adapters.snn_langchain_adapter import SNNLangChainAdapter
    from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = _calculate_t_max(epochs=epochs, warmup_epochs=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def _load_planner_snn_factory(planner_snn_instance, model_path: str, device: str):
    model = planner_snn_instance
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded PlannerSNN from '{model_path}'.")
        except Exception as e: print(f"⚠️ Failed to load PlannerSNN: {e}.")
    else: print(f"⚠️ PlannerSNN model not found: {model_path}.")
    return model.to(device)

def get_tokenizer(config_dict: Dict[str, Any]) -> PreTrainedTokenizerBase:
    cfg = OmegaConf.create(config_dict)
    tokenizer_name = OmegaConf.select(cfg, "data.tokenizer_name", default="gpt2")
    if tokenizer_name is None:
        tokenizer_name = "gpt2"
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        return AutoTokenizer.from_pretrained("gpt2")

class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    task_registry = providers.Object(TASK_REGISTRY)
    device = providers.Factory(get_auto_device)
    tokenizer = providers.Factory(get_tokenizer, config_dict=config)
    
    snn_model = providers.Factory(
        SNNCore, 
        config=config.model, 
        vocab_size=tokenizer.provided.vocab_size
    )
    
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    
    meta_cognitive_snn: providers.Provider[MetaCognitiveSNN] = providers.Factory(MetaCognitiveSNN)
    
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    standard_trainer = providers.Factory(BreakthroughTrainer, criterion=providers.Factory(CombinedLoss, ce_weight=config.training.gradient_based.loss.ce_weight, spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight, tokenizer=tokenizer, ewc_weight=config.training.gradient_based.loss.ewc_weight), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    distillation_trainer = providers.Factory(DistillationTrainer, criterion=providers.Factory(DistillationLoss, tokenizer=tokenizer, ce_weight=config.training.gradient_based.distillation.loss.ce_weight, distill_weight=config.training.gradient_based.distillation.loss.distill_weight, spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight, temperature=config.training.gradient_based.distillation.loss.temperature), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    physics_informed_trainer = providers.Factory(PhysicsInformedTrainer, criterion=providers.Factory(PhysicsInformedLoss, ce_weight=config.training.physics_informed.loss.ce_weight, spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight, mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight, tokenizer=tokenizer), grad_clip_norm=config.training.physics_informed.grad_clip_norm, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    synaptic_learning_rule = providers.Factory(get_bio_learning_rule, name=config.training.biologically_plausible.learning_rule, params=config.training.biologically_plausible.provided)
    homeostatic_learning_rule = providers.Factory(get_bio_learning_rule, name="BCM", params=config.training.biologically_plausible.provided)
    bio_rl_agent = providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device, synaptic_rule=synaptic_learning_rule, homeostatic_rule=homeostatic_learning_rule)
    grid_world_env = providers.Factory(GridWorldEnv, device=device)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=bio_rl_agent, env=grid_world_env)
    particle_filter_trainer = providers.Factory(ParticleFilterTrainer, base_model=providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0}, synaptic_rule=providers.Object(None), homeostatic_rule=providers.Object(None)), config=config.provided, device=device)
    
    planner_snn = providers.Factory(
        PlannerSNN, 
        vocab_size=providers.Callable(len, tokenizer), 
        d_model=providers.Callable(lambda c: c.get('model', {}).get('d_model', 128) if c else 128, c=config.provided), 
        d_state=providers.Callable(lambda c: c.get('model', {}).get('d_state', 64) if c else 64, c=config.provided), 
        num_layers=providers.Callable(lambda c: c.get('model', {}).get('num_layers', 2) if c else 2, c=config.provided), 
        time_steps=providers.Callable(lambda c: c.get('model', {}).get('time_steps', 16) if c else 16, c=config.provided), 
        n_head=providers.Callable(lambda c: c.get('model', {}).get('n_head', 4) if c else 4, c=config.provided), 
        num_skills=10, 
        neuron_config=providers.Callable(lambda c: c.get('model', {}).get('neuron', {'type': 'lif'}) if c else {'type': 'lif'}, c=config.provided)
    )
    
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)
    model_registry = providers.Selector(providers.Callable(lambda cfg: cfg.get("model_registry", {}).get("provider", "file"), cfg=config.provided), file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path.or_none()), distributed=providers.Singleton(DistributedModelRegistry, registry_path=config.model_registry.file.path.or_none()))
    probabilistic_neuron_params = providers.Factory(lambda cfg: cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_neuron', {}), cfg=config.provided)
    probabilistic_learning_rule = providers.Factory(lambda cfg: ProbabilisticHebbian(learning_rate=cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian', {}).get('learning_rate', 0.01), weight_decay=cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian', {}).get('weight_decay', 0.0)) if cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian') else None, cfg=config.provided)
    probabilistic_model = providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params=probabilistic_neuron_params.provider, synaptic_rule=probabilistic_learning_rule.provider, homeostatic_rule=providers.Object(None), sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification.provided)
    probabilistic_agent = providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device, synaptic_rule=probabilistic_learning_rule.provider, homeostatic_rule=providers.Object(None))
    probabilistic_trainer = providers.Factory(BioRLTrainer, agent=probabilistic_agent, env=grid_world_env)
    bio_learning_rule = providers.Factory(get_bio_learning_rule, name=config.training.biologically_plausible.learning_rule, params=config.training.biologically_plausible.provided)
    
    active_inference_agent = providers.Factory(
        ActiveInferenceAgent,
        generative_model=snn_model,
        num_actions=4, 
        action_dim=providers.Callable(lambda c: c.get('model', {}).get('d_model', 128) if c else 128, c=config.provided),
        observation_dim=providers.Callable(lambda c: c.get('model', {}).get('d_model', 128) if c else 128, c=config.provided),
        hidden_dim=providers.Callable(lambda c: c.get('model', {}).get('d_state', 64) if c else 64, c=config.provided),
        lr=0.001
    )

class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)
    
    device = providers.Factory(get_auto_device)
    
    # Registry
    model_registry = providers.Selector(
        providers.Callable(lambda c: c.get("model_registry", {}).get("provider", "file"), c=config),
        file=providers.Singleton(SimpleModelRegistry, registry_path=providers.Factory(lambda cfg: cfg.get("model_registry", {}).get("file", {}).get("path", "runs/model_registry.json"), cfg=config.provided)),
        distributed=providers.Singleton(DistributedModelRegistry, registry_path=providers.Factory(lambda cfg: cfg.get("model_registry", {}).get("file", {}).get("path", "runs/model_registry.json"), cfg=config.provided))
    )
    
    web_crawler = providers.Singleton(WebCrawler)
    
    rag_system = providers.Factory(
        RAGSystem, 
        vector_store_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "vector_store") if log_dir else "runs/vector_store", log_dir=config.training.log_dir.or_none())
    )
    
    memory = providers.Factory(
        Memory, 
        rag_system=rag_system, 
        memory_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "agent_memory.jsonl") if log_dir else "runs/agent_memory.jsonl", log_dir=config.training.log_dir.or_none())
    )
    
    # 読み込み済みPlannerモデル
    loaded_planner_snn = providers.Singleton(
        _load_planner_snn_factory,
        planner_snn_instance=providers.Callable(lambda tc: tc.planner_snn(), tc=training_container),
        model_path=config.training.planner.model_path.or_none(),
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
        meta_cognitive_snn=providers.Callable(lambda tc: tc.meta_cognitive_snn(), tc=training_container.provider),
        motivation_system=providers.Singleton(IntrinsicMotivationSystem),
        model_config_path=config.model.path.or_none(),
        training_config_path=providers.Object("configs/templates/base_config.yaml")
    )
    
    active_inference_agent = providers.Callable(lambda tc: tc.active_inference_agent(), tc=training_container)

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
        vision_model_config=providers.Callable(lambda: {"architecture_type": "spiking_cnn", "num_classes": 128}), 
        projector_config=providers.Callable(lambda: {"visual_dim": 128, "lang_dim": 256, "use_bitnet": False}),
        device=device 
    )

    prefrontal_cortex = providers.Singleton(PrefrontalCortex, workspace=global_workspace, motivation_system=motivation_system)
    hippocampus = providers.Singleton(Hippocampus, workspace=global_workspace, capacity=50)
    cortex = providers.Singleton(Cortex, rag_system=providers.Callable(lambda ac: ac.rag_system(), ac=agent_container))
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
    
    # Explicitly typed providers to help mypy
    rl_agent = providers.Callable(
        lambda ac: cast(TrainingContainer, ac.training_container()).bio_rl_agent(),
        ac=agent_container
    )
    
    self_evolving_agent = providers.Callable(
        lambda ac: ac.self_evolving_agent_master(),
        ac=agent_container
    )
    
    active_inference_agent = providers.Callable(
        lambda ac: cast(AgentContainer, ac).active_inference_agent(),
        ac=agent_container
    )
    
    digital_life_form = providers.Singleton(
        DigitalLifeForm,
        planner=providers.Callable(lambda ac: ac.hierarchical_planner(), ac=agent_container), # type: ignore[name-defined]
        autonomous_agent=providers.Callable(lambda ac: ac.autonomous_agent(), ac=agent_container), # type: ignore[name-defined]
        rl_agent=rl_agent, # type: ignore[name-defined]
        self_evolving_agent=self_evolving_agent, # type: ignore[name-defined]
        motivation_system=motivation_system, # type: ignore[name-defined]
        meta_cognitive_snn=providers.Callable(lambda ac_instance: cast(TrainingContainer, ac_instance.training_container()).meta_cognitive_snn(), ac_instance=agent_container), # type: ignore[name-defined]
        memory=providers.Callable(lambda ac: ac.memory(), ac=agent_container), # type: ignore[name-defined]
        physics_evaluator=providers.Singleton(PhysicsEvaluator),
        symbol_grounding=symbol_grounding, # type: ignore[name-defined]
        langchain_adapter=app_container.langchain_adapter, # type: ignore[name-defined, attr-defined]
        global_workspace=global_workspace, # type: ignore[name-defined]
        active_inference_agent=active_inference_agent # type: ignore[name-defined]
    )
