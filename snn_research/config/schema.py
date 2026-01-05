# snn_research/config/schema.py

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from omegaconf import MISSING


@dataclass
class NeuronConfig:
    type: str = "lif"
    tau_m_init: float = 2.0
    tau_s_init: float = 2.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    detach_reset: bool = True
    # For SFN (Scale-and-Fire)
    num_levels: Optional[int] = None
    base_threshold: Optional[float] = None
    # For Dendritic / Predictive
    num_branches: Optional[int] = None
    branch_features: Optional[int] = None


@dataclass
class ModelConfig:
    name: str = MISSING
    # architecture_type is string
    architecture_type: str = MISSING
    path: Optional[str] = None

    # Transformer / Spikformer specific
    d_model: int = 256
    d_state: int = 64  # For Predictive Coding / SSM
    num_layers: int = 4
    n_head: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    time_steps: int = 4  # T
    patch_size: int = 16

    # CNN specific
    in_channels: int = 3
    num_classes: int = 1000

    neuron: NeuronConfig = field(default_factory=NeuronConfig)


@dataclass
class DataConfig:
    path: str = "data/cifar10"
    format: str = "image_folder"  # or "jsonl", "h5"
    split_ratio: float = 0.1
    max_vocab_size: int = 10000
    tokenizer_name: str = "gpt2"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    enabled: bool = True
    type: str = "CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class GradientBasedConfig:
    type: str = "standard"  # standard, distillation
    learning_rate: float = 1e-3
    use_scheduler: bool = True
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    warmup_epochs: int = 0
    loss: Dict[str, Any] = field(default_factory=lambda: {"ewc_weight": 0.0})
    distillation: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3  # Global LR override?
    seed: int = 42

    paradigm: str = "gradient_based"  # gradient_based, bio-*, etc.

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    gradient_based: GradientBasedConfig = field(
        default_factory=GradientBasedConfig)

    # Extended experimental configs (using Dict for flexibility)
    quantization: Dict[str, Any] = field(default_factory=dict)
    pruning: Dict[str, Any] = field(default_factory=dict)
    meta_cognition: Dict[str, Any] = field(default_factory=dict)
    planner: Dict[str, Any] = field(default_factory=dict)
    biologically_plausible: Dict[str, Any] = field(default_factory=dict)
    self_supervised: Dict[str, Any] = field(default_factory=dict)
    physics_informed: Dict[str, Any] = field(default_factory=dict)
    probabilistic_ensemble: Dict[str, Any] = field(default_factory=dict)

    # Logging & Checkpointing
    log_dir: str = "workspace/runs/logs"
    save_dir: str = "workspace/runs/checkpoints"
    log_interval: int = 10
    eval_interval: int = 1

    # Device
    device: str = "auto"
    distributed: bool = False


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: [
        {"model": "small"},
        {"training": "base"},
        "_self_"
    ])

    # Global settings
    seed: int = 42
    device: str = "auto"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # System Components (Dict for now)
    model_registry: Dict[str, Any] = field(default_factory=dict)
    app: Dict[str, Any] = field(default_factory=dict)
    deployment: Dict[str, Any] = field(default_factory=dict)
