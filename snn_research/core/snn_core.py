# ファイルパス: snn_research/core/snn_core.py
# (修正: 不明なアーキテクチャタイプの場合にエラーを発生させ、問題を明確にする)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple, cast, Type
import logging

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    # ... (init, forward, reset_state は変更なし) ...
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int = 1000,
        backend: str = "spikingjelly"
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.backend = backend
        
        self.model: nn.Module = self._build_model()
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        arch_type = self.config.get('architecture_type', 'unknown')
        if param_count == 0:
            logger.error(f"❌ Built model '{arch_type}' has 0 parameters! Check model initialization.")
        else:
            logger.info(f"✅ SNNCore built model '{arch_type}' with {param_count:,} parameters ({trainable_count:,} trainable).")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if x is None:
            for key in ['input_ids', 'input_images', 'input_sequence', 'x']:
                if key in kwargs:
                    x = kwargs.pop(key)
                    break
        
        if x is None:
            return self.model(**kwargs)
        
        return self.model(x, **kwargs)
    
    def reset_state(self) -> None:
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state() # type: ignore
        elif hasattr(self.model, 'reset_spike_stats'):
             self.model.reset_spike_stats() # type: ignore
        elif hasattr(self.model, 'reset'):
             self.model.reset() # type: ignore

    def _build_model(self) -> nn.Module:
        """
        設定に基づきモデルを構築する。
        """
        # デフォルト値を削除し、設定不備を検出
        arch_type = self.config.get('architecture_type')
        neuron_config = self.config.get('neuron', {})
        time_steps = self.config.get('time_steps', 16)
        
        if not arch_type:
            raise ValueError("SNNCore: 'architecture_type' is missing in the configuration. Cannot build model.")

        if self.backend != "spikingjelly":
             raise ValueError(f"Unsupported backend: {self.backend}. Only 'spikingjelly' is supported.")

        # --- アーキテクチャに応じたモデルのインポートと構築 ---
        
        if arch_type == "spiking_cnn":
            from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
            num_classes = self.config.get('num_classes', self.vocab_size)
            return SpikingCNN(
                vocab_size=num_classes,
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "predictive_coding":
            from snn_research.models.experimental.predictive_coding_model import BreakthroughSNN
            return BreakthroughSNN(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 128),
                d_state=self.config.get('d_state', 64),
                num_layers=self.config.get('num_layers', 2),
                time_steps=time_steps,
                n_head=self.config.get('n_head', 4),
                neuron_config=neuron_config
            )
        
        # ... (他のモデルタイプは省略せずそのまま維持してください。変更なし) ...
        # (以下のelifブロックは全て以前の回答と同じ内容を維持)
        elif arch_type == "hybrid_cnn_snn":
            from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel
            return HybridCnnSnnModel(
                vocab_size=self.vocab_size,
                time_steps=time_steps,
                ann_frontend=self.config.get('ann_frontend', {}),
                snn_backend=self.config.get('snn_backend', {}),
                neuron_config=neuron_config
            )
        
        elif arch_type == "spiking_mamba":
            from snn_research.core.mamba_core import SpikingMamba
            return SpikingMamba(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 128),
                d_state=self.config.get('d_state', 16),
                d_conv=self.config.get('d_conv', 4),
                expand=self.config.get('expand', 2),
                num_layers=self.config.get('num_layers', 2),
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "tskips_snn":
            from snn_research.models.cnn.tskips_snn import TSkipsSNN
            return TSkipsSNN(
                input_features=self.config.get('input_features', 700),
                num_classes=self.vocab_size,
                hidden_features=self.config.get('hidden_features', 256),
                num_layers=self.config.get('num_layers', 3),
                time_steps=time_steps,
                neuron_config=neuron_config,
                forward_delays_per_layer=self.config.get('forward_delays_per_layer', []),
                backward_delays_per_layer=self.config.get('backward_delays_per_layer', [])
            )

        elif arch_type == "tiny_recursive_model":
            from snn_research.core.trm_core import TinyRecursiveModel
            return TinyRecursiveModel(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 64),
                d_state=self.config.get('d_state', 32),
                num_layers=self.config.get('num_layers', 1),
                layer_dims=self.config.get('layer_dims', []),
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "franken_moe":
            from snn_research.models.experimental.moe_model import SpikingFrankenMoE
            return SpikingFrankenMoE(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 128),
                expert_configs=self.config.get('expert_configs', []),
                expert_checkpoints=self.config.get('expert_checkpoints', []),
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "bit_spiking_rwkv":
            from snn_research.models.transformer.spiking_rwkv import BitSpikingRWKV
            return BitSpikingRWKV(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 128),
                num_layers=self.config.get('num_layers', 4),
                time_steps=time_steps,
                neuron_config=neuron_config,
                config=self.config
            )
        
        elif arch_type == "spiking_transformer":
            from snn_research.models.transformer.spiking_transformer import SpikingTransformerV2
            d_model = self.config.get('d_model', 256)
            return SpikingTransformerV2(
                vocab_size=self.vocab_size,
                d_model=d_model,
                nhead=self.config.get('n_head', 8),
                num_encoder_layers=self.config.get('num_layers', 6),
                dim_feedforward=self.config.get('dim_feedforward', d_model * 4),
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "temporal_snn":
            from snn_research.models.bio.temporal_snn import SimpleRSNN
            return SimpleRSNN(
                input_dim=self.config.get('input_dim', 1),
                hidden_dim=self.config.get('hidden_dim', 64),
                output_dim=self.config.get('output_dim', 2),
                time_steps=time_steps,
                neuron_config=neuron_config,
                output_spikes=self.config.get('output_spikes', False)
            )
        
        elif arch_type == "sew_resnet":
            from snn_research.models.cnn.sew_resnet import SEWResNet
            num_classes = self.config.get('num_classes', self.vocab_size)
            return SEWResNet(
                num_classes=num_classes,
                time_steps=time_steps,
                neuron_config=neuron_config
            )
        
        elif arch_type == "spiking_ssm":
            from snn_research.models.experimental.spiking_ssm import SpikingSSM
            return SpikingSSM(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 512),
                d_state=self.config.get('d_state', 64),
                num_layers=self.config.get('num_layers', 6),
                time_steps=time_steps,
                d_conv=self.config.get('d_conv', 4),
                neuron_config=neuron_config
            )

        elif arch_type == "feel_snn":
            from snn_research.models.experimental.feel_snn import FEELSNN
            num_classes = self.config.get('num_classes', self.vocab_size)
            return FEELSNN(
                num_classes=num_classes,
                time_steps=time_steps,
                in_channels=self.config.get('in_channels', 3),
                neuron_config=neuron_config
            )

        elif arch_type == "sformer":
            from snn_research.models.transformer.sformer import SFormer
            return SFormer(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 256),
                nhead=self.config.get('n_head', 8),
                num_layers=self.config.get('num_layers', 6),
                dim_feedforward=self.config.get('dim_feedforward', 1024),
                dropout=self.config.get('dropout', 0.1),
                neuron_config=neuron_config
            )

        elif arch_type == "semm":
            from snn_research.models.experimental.semm_model import SEMMModel
            return SEMMModel(
                vocab_size=self.vocab_size,
                d_model=self.config.get('d_model', 256),
                num_layers=self.config.get('num_layers', 4),
                num_experts=self.config.get('num_experts', 4),
                top_k=self.config.get('top_k', 2),
                neuron_config=neuron_config
            )
        
        elif arch_type == "visual_cortex":
            from snn_research.models.bio.visual_cortex import VisualCortex
            return VisualCortex(
                input_channels=self.config.get('in_channels', 2),
                height=self.config.get('height', 32),
                width=self.config.get('width', 32),
                d_model=self.config.get('d_model', 128),
                d_state=self.config.get('d_state', 64),
                time_steps=time_steps,
                neuron_config=neuron_config
            )
            
        elif arch_type == "spiking_vlm":
            from snn_research.models.transformer.spiking_vlm import SpikingVLM
            return SpikingVLM(
                vocab_size=self.vocab_size,
                vision_config=self.config.get('vision_config', {}),
                language_config=self.config.get('language_config', {}),
                projector_config=self.config.get('projector_config', {})
            )

        else:
            # --- 修正: 不明なタイプはエラーにする ---
            raise ValueError(f"Unknown architecture type '{arch_type}'. Please check your model config file.")
