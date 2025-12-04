# ファイルパス: snn_research/architectures/__init__.py
# Title: SNN アーキテクチャ パッケージ (Phase 3 更新)
# Description: SFormer と SEMM を追加。

from .spiking_transformer_v2 import SpikingTransformerV2, SDSAEncoderLayer
from .hybrid_attention_transformer import HybridAttentionTransformer, AdaptiveTransformerLayer
from .hybrid_neuron_network import HybridSpikingCNN
from .hybrid_transformer import HybridSNNTransformer, HybridTransformerLayer
from .spiking_diffusion_model import SpikingDiffusionModel
from .sew_resnet import SEWResNet, SEWResidualBlock
from .tskips_snn import TSkipsSNN, TSkipsBlock
from .spiking_rwkv import SpikingRWKV, BitSpikingRWKV, SpikingRWKVBlock, BitSpikingRWKVBlock
from .spiking_ssm import SpikingSSM, S4DLIFBlock
from .feel_snn import FEELSNN
# --- ▼ 追加: Phase 3 Models ▼ ---
from .sformer import SFormer
from snn_research.models.experimental.semm_model import SEMMModel
# --- ▲ 追加 ▲ ---

from typing import List

__all__: List[str] = [
    "SpikingTransformerV2",
    "SDSAEncoderLayer",
    "HybridAttentionTransformer",
    "AdaptiveTransformerLayer",
    "HybridSpikingCNN",
    "HybridSNNTransformer",
    "HybridTransformerLayer",
    "SpikingDiffusionModel",
    "SEWResNet",
    "SEWResidualBlock",
    "TSkipsSNN",
    "TSkipsBlock",
    "SpikingRWKV",
    "BitSpikingRWKV",
    "SpikingRWKVBlock",
    "BitSpikingRWKVBlock",
    "SpikingSSM",
    "S4DLIFBlock",
    "FEELSNN",
    # --- ▼ 追加 ▼ ---
    "SFormer",
    "SEMMModel"
    # --- ▲ 追加 ▲ ---
]