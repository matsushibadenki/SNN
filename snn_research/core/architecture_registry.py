# ファイルパス: snn_research/core/architecture_registry.py
# snn_research/core/architecture_registry.py
# Title: モデルアーキテクチャレジストリ (Bugfix: Added 'hybrid' registration)
# Description:
#   SpikingWorldModel のビルダ関数に sensory_configs 引数の生成ロジックを追加し、
#   mypy エラー (Missing positional argument) を修正。
#   また、ベンチマークスイートで使用される 'hybrid' アーキテクチャを登録。

import torch.nn as nn
from typing import Dict, Any, Callable, List, Optional, cast
import logging

logger = logging.getLogger(__name__)


class ArchitectureRegistry:
    """
    アーキテクチャタイプ名とビルダー関数をマッピングするレジストリ。
    """
    _registry: Dict[str, Callable[[Dict[str, Any], int], nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(builder_func: Callable[[Dict[str, Any], int], nn.Module]):
            if name in cls._registry:
                logger.warning(
                    f"Architecture '{name}' is already registered. Overwriting.")
            cls._registry[name] = builder_func
            return builder_func
        return decorator

    @classmethod
    def build(cls, arch_type: str, config: Dict[str, Any], vocab_size: int) -> nn.Module:
        if arch_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown architecture type '{arch_type}'. Available architectures: {available}")

        logger.info(f"🏗️ Building architecture: {arch_type}")
        return cls._registry[arch_type](config, vocab_size)

# --- 以下、各モデルのビルダー関数定義 ---


@ArchitectureRegistry.register("spiking_cnn")
def build_spiking_cnn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
    return SpikingCNN(
        vocab_size=config.get('num_classes', vocab_size),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("predictive_coding")
def build_predictive_coding(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.predictive_coding_model import PredictiveCodingModel
    return PredictiveCodingModel(
        input_dim=config.get('input_dim', 784),
        hidden_dims=config.get('hidden_dims', [128]),
        output_dim=vocab_size,
        neuron_params=config.get('neuron', {}),
        vocab_size=vocab_size
    )


@ArchitectureRegistry.register("hybrid_cnn_snn")
def build_hybrid_cnn_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel
    return HybridCnnSnnModel(
        vocab_size=vocab_size,
        time_steps=config.get('time_steps', 16),
        ann_frontend=config.get(
            'ann_frontend', {'name': 'mobilenet_v2', 'output_features': 1280}),
        snn_backend=config.get(
            'snn_backend', {'d_model': 1280, 'n_head': 8, 'num_layers': 4}),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("spiking_mamba")
def build_spiking_mamba(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.core.mamba_core import SpikingMamba
    return SpikingMamba(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 128),
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand=config.get('expand', 2),
        num_layers=config.get('num_layers', 2),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("tskips_snn")
def build_tskips_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.tskips_snn import TSkipsSNN
    forward_delays = cast(List[Optional[List[int]]],
                          config.get('forward_delays_per_layer', []))
    backward_delays = cast(List[Optional[List[int]]], config.get(
        'backward_delays_per_layer', []))
    return TSkipsSNN(
        input_features=config.get('input_features', 700),
        num_classes=vocab_size,
        hidden_features=config.get('hidden_features', 256),
        num_layers=config.get('num_layers', 3),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {}),
        forward_delays_per_layer=forward_delays,
        backward_delays_per_layer=backward_delays
    )


@ArchitectureRegistry.register("franken_moe")
def build_franken_moe(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.moe_model import SpikingFrankenMoE, ExpertContainer
    
    # Placeholder for required args 'experts' and 'gate' to satisfy signature
    experts: List[ExpertContainer] = [] 
    gate = nn.Linear(config.get('d_model', 128), len(config.get('expert_configs', [])) or 1)

    return SpikingFrankenMoE(
        experts=experts,
        gate=gate,
        config=config,
        vocab_size=vocab_size, # Passed as kwargs
        d_model=config.get('d_model', 128),
        expert_configs=config.get('expert_configs', []),
        expert_checkpoints=config.get('expert_checkpoints', []),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("bit_spiking_rwkv")
def build_bit_spiking_rwkv(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.spiking_rwkv import BitSpikingRWKV
    return BitSpikingRWKV(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 128),
        num_layers=config.get('num_layers', 4),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {}),
        config=config
    )


@ArchitectureRegistry.register("spiking_transformer")
def build_spiking_transformer(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.spiking_transformer import SpikingTransformerV2
    d_model = config.get('d_model', 256)
    return SpikingTransformerV2(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=config.get('n_head', 8),
        num_encoder_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', d_model * 4),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("temporal_snn")
def build_temporal_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.bio.temporal_snn import SimpleRSNN
    return SimpleRSNN(
        input_dim=config.get('input_dim', 1),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 2),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {}),
        output_spikes=config.get('output_spikes', False)
    )


@ArchitectureRegistry.register("sew_resnet")
def build_sew_resnet(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.sew_resnet import SEWResNet
    return SEWResNet(
        num_classes=config.get('num_classes', vocab_size),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("feel_snn")
def build_feel_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.feel_snn import FEELSNN
    return FEELSNN(
        num_classes=config.get('num_classes', vocab_size),
        time_steps=config.get('time_steps', 16),
        in_channels=config.get('in_channels', 3),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("sformer")
def build_sformer(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.sformer import SFormer
    return SFormer(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        nhead=config.get('n_head', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        neuron_config=config.get('neuron_config', config.get('neuron', {}))
    )


@ArchitectureRegistry.register("semm")
def build_semm(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.semm_model import SEMMModel
    return SEMMModel(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        num_layers=config.get('num_layers', 4),
        num_experts=config.get('num_experts', 4),
        top_k=config.get('top_k', 2),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("visual_cortex")
def build_visual_cortex(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.bio.visual_cortex import VisualCortex
    in_channels = config.get('in_channels', 3)
    base_channels = config.get('base_channels', 32)
    neuron_params = config.get('neuron', {})

    return VisualCortex(
        in_channels=in_channels,
        base_channels=base_channels,
        neuron_params=neuron_params
    )


@ArchitectureRegistry.register("spiking_vlm")
def build_spiking_vlm(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.spiking_vlm import SpikingUnifiedModel

    vision_config = config.get('vision_config', {})
    sensory_configs = {}

    # 既存のvision_configがある場合、'vision'モダリティとして登録
    if vision_config:
        sensory_configs['vision'] = vision_config

    # Brain v4 style の sensory_inputs があれば統合
    if 'sensory_inputs' in config:
        for k, v in config['sensory_inputs'].items():
            sensory_configs[k] = v

    return SpikingUnifiedModel(
        vocab_size=vocab_size,
        language_config=config.get('language_config', {}),
        sensory_configs=sensory_configs,
        projector_config=config.get('projector_config', {})
    )


@ArchitectureRegistry.register("tiny_recursive_model")
def build_tiny_recursive_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.core.trm_core import TinyRecursiveModel
    return TinyRecursiveModel(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 64),
        d_state=config.get('d_state', 32),
        num_layers=config.get('num_layers', 1),
        layer_dims=config.get('layer_dims', []),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("spiking_ssm")
def build_spiking_ssm(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.spiking_ssm import SpikingSSM
    return SpikingSSM(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 512),
        d_state=config.get('d_state', 64),
        num_layers=config.get('num_layers', 6),
        time_steps=config.get('time_steps', 16),
        d_conv=config.get('d_conv', 4),
        neuron_config=config.get('neuron', {})
    )


@ArchitectureRegistry.register("dsa_transformer")
def build_dsa_transformer(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.dsa_transformer import SpikingDSATransformer
    d_model = config.get('d_model', 64)
    return SpikingDSATransformer(
        input_dim=config.get('input_dim', vocab_size),
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', d_model * 4),
        time_window=config.get('time_steps', 16),
        use_bitnet=config.get('use_bitnet', True),
        num_classes=vocab_size
    )


@ArchitectureRegistry.register("spiking_world_model")
def build_spiking_world_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.world_model_snn import SpikingWorldModel

    # sensory_configsの抽出 (config構造の差異を吸収)
    sensory_configs = {}

    # 1. config['sensory_configs'] が直接ある場合
    if 'sensory_configs' in config:
        sensory_configs = config['sensory_configs']

    # 2. config['sensory_inputs'] (Brain v4 config style) がある場合
    elif 'sensory_inputs' in config:
        # { 'vision': {'input_dim': 784}, ... } -> { 'vision': 784, ... }
        for key, val in config['sensory_inputs'].items():
            if isinstance(val, dict) and 'input_dim' in val:
                sensory_configs[key] = val['input_dim']
            elif isinstance(val, int):
                sensory_configs[key] = val

    # 3. フォールバック: input_dimがあれば vision として扱う
    if not sensory_configs and 'input_dim' in config:
        sensory_configs = {'vision': config['input_dim']}

    # 4. それでもなければデフォルト (784=MNIST size)
    if not sensory_configs:
        sensory_configs = {'vision': 784}

    return SpikingWorldModel(
        vocab_size=vocab_size,
        action_dim=config.get('action_dim', 4),
        d_model=config.get('d_model', 256),
        d_state=config.get('d_state', 128),
        num_layers=config.get('num_layers', 4),
        time_steps=config.get('time_steps', 16),
        sensory_configs=sensory_configs,
        neuron_config=config.get('neuron', {}),
        use_bitnet=config.get('use_bitnet', False)
    )


@ArchitectureRegistry.register("hybrid")
def build_hybrid_core(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """
    Benchmarks用: HybridNeuromorphicCore (PhaseCriticalHybridCore Wrapper) を構築。
    """
    from snn_research.core.hybrid_core import HybridNeuromorphicCore
    # ベンチマークスイート側の config キーに合わせてパラメータを抽出
    # config = {"in_features": 64, "hidden_features": 128, "out_features": 10}

    in_features = config.get('in_features', 64)
    hidden_features = config.get('hidden_features', 128)
    out_features = config.get('out_features', vocab_size)

    return HybridNeuromorphicCore(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features
    )