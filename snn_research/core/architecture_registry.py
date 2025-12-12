# ファイルパス: snn_research/core/architecture_registry.py
# Title: モデルアーキテクチャレジストリ (Fixed)
# Description:
#   dsa_transformer の登録を追加し、ベンチマークスイートからの呼び出しに対応。

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
        """
        ビルダー関数を登録するデコレータ。
        """
        def decorator(builder_func: Callable[[Dict[str, Any], int], nn.Module]):
            if name in cls._registry:
                logger.warning(f"Architecture '{name}' is already registered. Overwriting.")
            cls._registry[name] = builder_func
            return builder_func
        return decorator

    @classmethod
    def build(cls, arch_type: str, config: Dict[str, Any], vocab_size: int) -> nn.Module:
        """
        指定されたアーキテクチャタイプのモデルを構築する。
        """
        if arch_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown architecture type '{arch_type}'. Available architectures: {available}")
        
        logger.info(f"🏗️ Building architecture: {arch_type}")
        return cls._registry[arch_type](config, vocab_size)

# --- 以下、各モデルのビルダー関数定義 ---

@ArchitectureRegistry.register("spiking_cnn")
def build_spiking_cnn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
    
    num_classes = config.get('num_classes', vocab_size)
    time_steps = config.get('time_steps', 16)
    neuron_config = config.get('neuron', {})
    
    return SpikingCNN(
        vocab_size=num_classes,
        time_steps=time_steps,
        neuron_config=neuron_config
    )

@ArchitectureRegistry.register("predictive_coding")
def build_predictive_coding(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.predictive_coding_model import BreakthroughSNN
    
    return BreakthroughSNN(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 128),
        d_state=config.get('d_state', 64),
        num_layers=config.get('num_layers', 2),
        time_steps=config.get('time_steps', 16),
        n_head=config.get('n_head', 4),
        neuron_config=config.get('neuron', {})
    )

@ArchitectureRegistry.register("hybrid_cnn_snn")
def build_hybrid_cnn_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel
    
    return HybridCnnSnnModel(
        vocab_size=vocab_size,
        time_steps=config.get('time_steps', 16),
        ann_frontend=config.get('ann_frontend', {'name': 'mobilenet_v2', 'output_features': 1280}),
        snn_backend=config.get('snn_backend', {'d_model': 1280, 'n_head': 8, 'num_layers': 4}),
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
    
    forward_delays = cast(List[Optional[List[int]]], config.get('forward_delays_per_layer', []))
    backward_delays = cast(List[Optional[List[int]]], config.get('backward_delays_per_layer', []))
    
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
    from snn_research.models.experimental.moe_model import SpikingFrankenMoE
    
    return SpikingFrankenMoE(
        vocab_size=vocab_size,
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
        config=config # BitNet設定のためにconfig全体を渡す
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
    
    num_classes = config.get('num_classes', vocab_size)
    return SEWResNet(
        num_classes=num_classes,
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )

@ArchitectureRegistry.register("feel_snn")
def build_feel_snn(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.feel_snn import FEELSNN
    
    num_classes = config.get('num_classes', vocab_size)
    return FEELSNN(
        num_classes=num_classes,
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
    
    return VisualCortex(
        input_channels=config.get('in_channels', 2),
        height=config.get('height', 32),
        width=config.get('width', 32),
        d_model=config.get('d_model', 128),
        d_state=config.get('d_state', 64),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )

@ArchitectureRegistry.register("spiking_vlm")
def build_spiking_vlm(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.transformer.spiking_vlm import SpikingVLM
    
    return SpikingVLM(
        vocab_size=vocab_size,
        vision_config=config.get('vision_config', {}),
        language_config=config.get('language_config', {}),
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
    from snn_research.models.transformer.dsa_transformer import DSASpikingTransformer
    
    return DSASpikingTransformer(
        vocab_size=vocab_size,
        output_dim=vocab_size,
        d_model=config.get('d_model', 64),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        top_k=config.get('top_k', 4),
        max_len=config.get('max_len', 128),
        dropout=config.get('dropout', 0.1),
        neuron_params=config.get('neuron_config', {})
    )
    
@ArchitectureRegistry.register("spiking_world_model")
def build_spiking_world_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    from snn_research.models.experimental.world_model_snn import SpikingWorldModel
    
    return SpikingWorldModel(
        vocab_size=vocab_size,
        action_dim=config.get('action_dim', 4), # デフォルトアクション数
        d_model=config.get('d_model', 256),
        d_state=config.get('d_state', 128),
        num_layers=config.get('num_layers', 4),
        time_steps=config.get('time_steps', 16),
        neuron_config=config.get('neuron', {})
    )