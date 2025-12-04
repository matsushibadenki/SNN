# ファイルパス: snn_research/core/snn_core.py
# Title: SNN Core Model Factory (Phase 3 対応版)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple, cast, Type
import logging

logger = logging.getLogger(__name__)

class SNNCore(nn.Module):
    """
    SNNモデルの統合インターフェース（ファクトリクラス）。
    """
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
        
        # モデル構築
        self.model: nn.Module = self._build_model()
        
        # パラメータ数のチェックとログ出力
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        arch_type = self.config.get('architecture_type', 'unknown')
        if param_count == 0:
            logger.error(f"❌ Built model '{arch_type}' has 0 parameters! Check model initialization.")
        else:
            logger.info(f"✅ SNNCore built model '{arch_type}' with {param_count:,} parameters ({trainable_count:,} trainable).")

    def _get_neuron_class_and_params(self, neuron_config: Dict[str, Any], valid_keys: Optional[List[str]] = None) -> Tuple[Type[nn.Module], Dict[str, Any]]:
        """
        設定からニューロンクラスとパラメータを解決するヘルパーメソッド。
        """
        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)

        neuron_class: Type[nn.Module]
        
        if neuron_type == 'lif':
            from snn_research.core.neurons import AdaptiveLIFNeuron
            neuron_class = AdaptiveLIFNeuron
            if valid_keys is None:
                valid_keys = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']
                
        elif neuron_type == 'izhikevich':
            from snn_research.core.neurons import IzhikevichNeuron
            neuron_class = IzhikevichNeuron
            if valid_keys is None:
                valid_keys = ['features', 'a', 'b', 'c', 'd', 'dt']
                
        elif neuron_type == 'glif':
            from snn_research.core.neurons import GLIFNeuron
            neuron_class = GLIFNeuron
            if valid_keys is None:
                valid_keys = ['features', 'base_threshold', 'gate_input_features']

        elif neuron_type == 'tc_lif':
            from snn_research.core.neurons import TC_LIF
            neuron_class = TC_LIF
            if valid_keys is None:
                valid_keys = ['features', 'tau_s_init', 'tau_d_init', 'w_ds_init', 'w_sd_init', 'base_threshold', 'v_reset']

        elif neuron_type == 'dual_threshold':
            from snn_research.core.neurons import DualThresholdNeuron
            neuron_class = DualThresholdNeuron
            if valid_keys is None:
                valid_keys = ['features', 'tau_mem', 'threshold_high_init', 'threshold_low_init', 'v_reset']
                
        elif neuron_type == 'evolutionary_leak_lif':
            from snn_research.core.neurons.feel_neuron import EvolutionaryLeakLIF
            neuron_class = EvolutionaryLeakLIF
            if valid_keys is None:
                valid_keys = ['features', 'initial_tau', 'v_threshold', 'v_reset', 'detach_reset', 'learn_threshold']
                
        else:
            # デフォルト
            from snn_research.core.neurons import AdaptiveLIFNeuron
            neuron_class = AdaptiveLIFNeuron
            if valid_keys is None:
                valid_keys = ['features', 'tau_mem', 'base_threshold']

        filtered_params = {k: v for k, v in neuron_params.items() if k in valid_keys}
        return neuron_class, filtered_params


    def _build_model(self) -> nn.Module:
        """設定に基づきモデルを構築 (遅延インポート)"""
        arch_type = self.config.get('architecture_type', 'spiking_cnn')
        neuron_config = self.config.get('neuron', {})
        time_steps = self.config.get('time_steps', 16)
        
        if self.backend == "spikingjelly":
            # --- 1. Spiking CNN (画像分類) ---
            if arch_type == "spiking_cnn":
                from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
                num_classes = self.config.get('num_classes', self.vocab_size)
                return SpikingCNN(
                    vocab_size=num_classes,
                    time_steps=time_steps,
                    neuron_config=neuron_config
                )
            
            # --- 2. Predictive Coding (BreakthroughSNN) ---
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
            
            # --- 3. Hybrid CNN-SNN ---
            elif arch_type == "hybrid_cnn_snn":
                from snn_research.models.cnn.hybrid_cnn_snn_model import HybridCnnSnnModel
                return HybridCnnSnnModel(
                    vocab_size=self.vocab_size,
                    time_steps=time_steps,
                    ann_frontend=self.config.get('ann_frontend', {}),
                    snn_backend=self.config.get('snn_backend', {}),
                    neuron_config=neuron_config
                )
            
            # --- 4. Spiking Mamba ---
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
            
            # --- 5. TSkips SNN ---
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

            # --- 6. Tiny Recursive Model (TRM) ---
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
            
            # --- 7. FrankenMoE ---
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
            
            # --- 8. 1.58bit BitSpikingRWKV ---
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
            
            # --- 9. Spiking Transformer v2 (SDSA) ---
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
            
            # --- 10. Temporal SNN (SimpleRSNN) ---
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
            
            # --- 11. SEW ResNet ---
            elif arch_type == "sew_resnet":
                from snn_research.models.cnn.sew_resnet import SEWResNet
                num_classes = self.config.get('num_classes', self.vocab_size)
                return SEWResNet(
                    num_classes=num_classes,
                    time_steps=time_steps,
                    neuron_config=neuron_config
                )
            
            # --- 12. Spiking SSM (S4D) ---
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

            # --- 13. FEEL-SNN ---
            elif arch_type == "feel_snn":
                from snn_research.models.experimental.feel_snn import FEELSNN
                num_classes = self.config.get('num_classes', self.vocab_size)
                return FEELSNN(
                    num_classes=num_classes,
                    time_steps=time_steps,
                    in_channels=self.config.get('in_channels', 3),
                    neuron_config=neuron_config
                )

            # --- 14. SFormer (New Phase 3) ---
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

            # --- 15. SEMM (New Phase 3) ---
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

            # --- Fallback ---
            else:
                logger.warning(f"Unknown architecture type '{arch_type}'. Falling back to SpikingCNN.")
                from snn_research.models.cnn.spiking_cnn_model import SpikingCNN
                num_classes = self.config.get('num_classes', self.vocab_size)
                return SpikingCNN(
                    vocab_size=num_classes,
                    time_steps=time_steps,
                    neuron_config=neuron_config
                )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Only 'spikingjelly' is supported.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """順伝播"""
        model_any = cast(Any, self.model)
        return model_any(x, **kwargs)
    
    def reset_state(self) -> None:
        """状態のリセット"""
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state() # type: ignore[operator]
        elif hasattr(self.model, 'reset_spike_stats'):
             self.model.reset_spike_stats() # type: ignore[operator]
        elif hasattr(self.model, 'reset'):
             self.model.reset() # type: ignore[operator]