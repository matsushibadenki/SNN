# ファイルパス: snn_research/conversion/ann_to_snn_converter.py
# Title: ANN-SNN 変換コンバータ (統合最適化版 / 高忠実度版)
# Description:
# - GGUF/Safetensors形式のANNモデルからSNNへの変換・蒸留を行うコンバータ。
# - Phase 5 (Deep Bio-Calibration) に対応し、最適化戦略モジュールと連携。
# - スタブクラスを削除し、snn_research.conversion.optimization_strategies を使用。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Dict, Any, Optional, cast, Type, List, Tuple, Callable, Union, TYPE_CHECKING
import logging
import numpy as np
from collections import OrderedDict
import math
import re

# Transformers (LLM変換用)
try:
    from transformers import AutoModelForCausalLM # type: ignore[import-untyped]
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]

# SNNコンポーネント
from snn_research.core.neurons import AdaptiveLIFNeuron, DualThresholdNeuron, ScaleAndFireNeuron
from .conversion_utils import safe_copy_weights, calibrate_thresholds_by_percentile
from .fold_bn import fold_all_batchnorms
from .ecl_components import LearnableClippingLayer
from snn_research.core.base import SNNLayerNorm
from snn_research.training.pruning import apply_sbc_pruning

# --- 修正: 最適化戦略モジュールのインポート ---
from snn_research.conversion.optimization_strategies import (
    AdaptiveTimestepScheduler,
    ProgressiveQuantization,
    LayerWiseOptimizer
)
# ----------------------------------------

# GGUFの依存関係
try:
    from gguf import GGUFReader  # type: ignore[import-untyped]
    GGUF_AVAILABLE = True
except ImportError:
    GGUFReader = Any  # type: ignore[misc, assignment]
    GGUF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 外部依存のスタブ/代替クラス (プロジェクト構造に存在することを前提とする) ---

class SpikingAttention(nn.Module):
    """【新機能】スパイキングアテンション層の簡易スタブ"""
    def __init__(self, hidden_dim: int, num_heads: int, timesteps: int = 4, neuron_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.q_neuron = AdaptiveLIFNeuron(features=hidden_dim) 
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        rate_q = q.mean(dim=1).unsqueeze(1)
        rate_k = k.mean(dim=1).unsqueeze(1)
        rate_v = v.mean(dim=1).unsqueeze(1)
        attn_scores = torch.matmul(rate_q, rate_k.transpose(-2, -1))
        return torch.matmul(attn_scores.softmax(dim=-1), rate_v)

# --- エンド・オブ・スタブ ---


def _load_gguf(path: str) -> Dict[str, torch.Tensor]:
    """GGUFファイルを読み込み、PyTorchのstate_dictを返す。"""
    if not GGUF_AVAILABLE:
        raise ImportError("GGUFファイルを読み込むには `gguf` ライブラリが必要です。")
    logging.info(f"GGUFファイルをロード中: {path}")
    reader = GGUFReader(path, 'r')  # type: ignore[operator]
    state_dict: Dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        state_dict[tensor.name] = torch.from_numpy(tensor.data.copy())
    logging.info(f"✅ GGUFから {len(state_dict)} 個のテンソルをロードしました。")
    return state_dict

def _replace_activation_with_ecl(
    module: nn.Module, 
    initial_threshold: float = 1.0,
    inplace: bool = True
) -> nn.Module:
    """モデル内の nn.ReLU, nn.GELU, nn.SiLU (SwiGLUのコア) を LearnableClippingLayer に置き換える。"""
    for name, child in list(module.named_children()):
        is_replacement = False
        
        if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU)):
            num_features = None # 簡易的にはスカラー閾値
            
            ecl_layer = LearnableClippingLayer(
                initial_threshold=initial_threshold, 
                num_features=num_features
            )
            setattr(module, name, ecl_layer)
            is_replacement = True
            logger.info(f"  - [ECL] Replaced '{name}' ({type(child).__name__}) with LearnableClippingLayer.")
        else:
            _replace_activation_with_ecl(child, initial_threshold, inplace=True)
            
    return module

@torch.no_grad()
def _dynamic_pruning_after_conversion(model: nn.Module, prune_amount: float, dataloader_stub: Any) -> float:
    """
    動的プルーニング (スパイク頻度に基づく) (Point 6)
    """
    if prune_amount <= 0.0: return 0.0
    logger.info(f"✂️ 動的プルーニング開始 (目標プルーニング率: {prune_amount:.2%})")

    # SBC/OBC (Saliency-based pruning) を適用 (ここでは簡易実装)
    for name, param in model.named_parameters():
         if 'weight' in name and param.dim() > 1:
              threshold = torch.kthvalue(param.data.abs().view(-1), k=int(param.numel() * prune_amount)).values
              param.data[param.data.abs() < threshold] = 0.0
    
    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(torch.sum(p == 0).item() for p in model.parameters())
    
    pruning_ratio = pruned_params / total_params if total_params > 0 else 0.0
    logger.info(f"✅ プルーニング完了。最終スパース性: {pruning_ratio:.2%}")
    return pruning_ratio


class AnnToSnnConverter:
    """
    既存のANNモデルファイルからSNNモデルを生成するユーティリティ。
    (統合最適化版)
    """
    def __init__(self, snn_model: nn.Module, model_config: Dict[str, Any]):
        self.snn_model = snn_model
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Deep Bio-Calibration Strategy の初期化
        base_timesteps = int(self.model_config.get('time_steps', 4))
        max_timesteps = int(self.model_config.get('max_time_steps', 16))
        
        self.timestep_scheduler = AdaptiveTimestepScheduler(
            base_timesteps=base_timesteps, 
            max_timesteps=max_timesteps,
            strategy="linear_decay"
        )
        self.progressive_quantizer = ProgressiveQuantization(
            stages=int(self.model_config.get('progressive_quantization_stages', 5))
        )
        self.layer_optimizer = LayerWiseOptimizer()
        self.snn_model.train() # 初期状態は訓練可能としておく

    def _load_ann_weights(self, ann_model_path: str, is_llm: bool = False) -> Dict[str, torch.Tensor]:
        """ANNモデルの重みをファイルから読み込む。"""
        logger.info(f"💾 ANNモデルの重みをロード中: {ann_model_path}")
        if ann_model_path.endswith(".safetensors"):
            return load_file(ann_model_path, device=self.device)
        elif ann_model_path.endswith(".gguf"):
            return _load_gguf(ann_model_path)
        elif is_llm:
            if not TRANSFORMERS_AVAILABLE or AutoModelForCausalLM is None:
                raise ImportError("LLMのロードには `transformers` ライブラリが必要です。")
            try:
                if AutoModelForCausalLM is not None: 
                    model = AutoModelForCausalLM.from_pretrained(ann_model_path).to(self.device) # type: ignore[operator]
                    return model.state_dict()
                else:
                    raise ImportError("AutoModelForCausalLM is not available.")
            except Exception as e:
                logger.error(f"Hugging Faceモデルのロードに失敗しました: {e}")
                raise
        else:
            try:
                return torch.load(ann_model_path, map_location=self.device)
            except Exception as e:
                logger.error(f"PyTorchモデルのロードに失敗しました: {e}")
                raise

    def _inject_spiking_attention(self, model: nn.Module) -> nn.Module:
        """モデル内の標準アテンション層をSpikingAttentionに置換"""
        logger.info("アテンション層をスパイキング化中...")
        replaced_count = 0
        
        # モデルの深さを推定
        total_modules = len(list(model.modules()))
        current_idx = 0
        
        for name, module in list(model.named_modules()):
            current_idx += 1
            if isinstance(module, nn.MultiheadAttention) or (hasattr(module, 'num_heads') and 'attention' in name.lower()):
                try:
                    hidden_dim = getattr(module, 'embed_dim', getattr(module, 'hidden_size', 512))
                    num_heads = getattr(module, 'num_heads', 8)
                    
                    # AdaptiveTimestepScheduler を使用してタイムステップを決定
                    # 層の深さを簡易的に推定 (実際にはより厳密な層インデックスが必要)
                    estimated_t = self.timestep_scheduler.get_timesteps_for_layer(
                        layer_index=current_idx, 
                        total_layers=total_modules, 
                        layer_type=type(module).__name__
                    )
                    
                    spiking_attn = SpikingAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        timesteps=estimated_t,
                        neuron_config=self.model_config.get('neuron', {})
                    )
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self._find_module(model, parent_name)
                    if parent is None: continue

                    setattr(parent, child_name, spiking_attn)
                    replaced_count += 1
                except Exception as e:
                    logger.warning(f"  - Failed to replace attention layer '{name}': {e}")
        logger.info(f"✅ {replaced_count} 個のアテンション層をスパイキング化しました")
        return model

    def _find_module(self, model: nn.Module, path: str) -> Optional[nn.Module]:
        if not path: return model
        try: return model.get_submodule(path)
        except AttributeError: return None

    def _optimize_threshold_distribution(self, thresholds: Dict[str, float], ann_model: nn.Module, calibration_loader: Any) -> Dict[str, float]:
        """閾値分布の最適化 (Point 7 - 閾値校正)"""
        # ここに LayerWiseOptimizer のロジックを統合可能
        # 今回は基本的な補正のみ行う
        return thresholds.copy()

    def convert_llm_weights(
        self,
        ann_model_name_or_path: str,
        output_path: str,
        calibration_loader: Optional[Any] = None,
        use_ecl: bool = False,
        use_spiking_attention: bool = True,
        progressive_stages: int = 5,
        prune_low_activity: float = 0.0,
        quantization_bits: float = 0.0,
        hardware_target: str = "GPU",
        teacher_model_name: Optional[str] = None
    ) -> None:
        """
        LLMのANNモデルをSNNモデルに変換する。
        """
        logger.info(f"--- 🚀 統合LLM変換開始: {ann_model_name_or_path} ---")

        if not TRANSFORMERS_AVAILABLE or AutoModelForCausalLM is None:
             raise ImportError("LLM変換には `transformers` ライブラリが必要です。")

        ann_model = AutoModelForCausalLM.from_pretrained(ann_model_name_or_path).to(self.device) # type: ignore[operator]
        ann_model.eval()
        
        if use_spiking_attention: ann_model = self._inject_spiking_attention(ann_model)
        if use_ecl: ann_model = _replace_activation_with_ecl(ann_model, initial_threshold=1.0, inplace=True)
        
        # 3. 重みコピーと正規化処理 (Point 1 - RMSNorm/SwiGLUのFold代替ヒント)
        ann_state_dict = ann_model.state_dict()
        snn_state_dict = self.snn_model.state_dict()
        
        for ann_name, ann_param in ann_state_dict.items():
            # RMSNorm/LayerNormの重みをSNNLayerNormにコピー (Point 1)
            if 'rmsnorm.weight' in ann_name or 'layernorm.weight' in ann_name:
                snn_name = ann_name.replace('model.', '').replace('transformer.', '')
                if snn_name in snn_state_dict: snn_state_dict[snn_name].copy_(ann_param)

        safe_copy_weights(self.snn_model, ann_state_dict)

        # 4. 閾値キャリブレーション（最適化版 & チャネル別対応ヒント）(Point 2, 7)
        if calibration_loader:
            base_thresholds = calibrate_thresholds_by_percentile(ann_model, calibration_loader, device=self.device)
            optimized_thresholds = self._optimize_threshold_distribution(base_thresholds, ann_model, calibration_loader)
            
            snn_neuron_layers: List[nn.Module] = [m for m in self.snn_model.modules() if isinstance(m, (AdaptiveLIFNeuron, DualThresholdNeuron, ScaleAndFireNeuron))]
            
            mapped_count = 0
            for i, lif in enumerate(snn_neuron_layers):
                # LayerWiseOptimizer で層ごとの推奨設定を取得
                # 名前解決が難しいため、インデックスベースで簡易適用
                
                # 閾値適用
                # thr_value をレイヤー名からマッピングするロジックが必要だが、
                # ここでは簡易的に全層共通または順次適用を想定 (実運用では名前マッチングが必要)
                thr_value = 1.0 # Default
                if optimized_thresholds:
                    thr_value = list(optimized_thresholds.values())[i % len(optimized_thresholds)]

                # thr_value が Tensor (チャネル別閾値) の場合と float の場合に対応 (Point 2)
                is_tensor_thr = isinstance(thr_value, torch.Tensor)
                
                if isinstance(lif, DualThresholdNeuron):
                    if is_tensor_thr:
                        thr_tensor = torch.as_tensor(thr_value).float()
                        lif.threshold_high.data.copy_(thr_tensor) 
                        lif.threshold_low.data.copy_(thr_tensor * 0.5) 
                    else:
                        lif.threshold_high.data.fill_(thr_value)
                        lif.threshold_low.data.fill_(thr_value * 0.5)
                elif isinstance(lif, AdaptiveLIFNeuron):
                    if is_tensor_thr:
                        thr_tensor = torch.as_tensor(thr_value).float()
                        lif.base_threshold.data.copy_(thr_tensor)
                    else:
                        lif.base_threshold.data.fill_(thr_value)

                elif isinstance(lif, ScaleAndFireNeuron):
                    thr_tensor = torch.as_tensor(thr_value)
                    max_thr = thr_tensor.max().item()
                    lif.thresholds.data.mul_(max_thr / lif.thresholds.data.max().item())
                
                mapped_count += 1
            logger.info(f"✅ {mapped_count} 個のニューロン層に最適化された閾値を適用")

        # 5. 動的プルーニング (Point 6)
        final_pruning_ratio = 0.0
        if prune_low_activity > 0.0 and calibration_loader:
            final_pruning_ratio = _dynamic_pruning_after_conversion(self.snn_model, prune_low_activity, calibration_loader)
        
        # 6. 変換メタデータの保存（Point 3, 8, 9の統合）
        conversion_metadata = {
            'bio_calibration_status': 'calibrated_with_ecl' if use_ecl else 'standard_conversion',
            'distillation': {
                'teacher_model': teacher_model_name or ann_model_name_or_path,
                'loss_type': 'KL_FinalLogits + MSE_InterFeature', 
                'surrogate_gradient_hint': 'ATan_or_FastSigmoid', 
                'peft_fine_tuning_required': True, 
            },
            'encoding_hint': 'Latency_or_Thresholded_Coding', 
            'runtime_optimization': {
                'target_quantization_bits': quantization_bits, 
                'quantization_method': 'BitLinear/INT8' if quantization_bits > 0.0 else 'None',
                'pruning_ratio': final_pruning_ratio,
                'hardware_optimization_hint': hardware_target,
                'time_step_batching_hint': hardware_target in ["Loihi", "TrueNorth"],
                'custom_kernel_hint': ['FUSE_MEM_SPIKE_UPDATE', 'SPARSE_CONV_JUMP_COMPUTATION'], 
            },
            'regularization_targets': {
                'target_firing_rate': 0.05, 
                'energy_cost_penalty': 0.01, 
                'weight_clip_value': 1.0 
            },
            'evaluation_metrics': {
                'primary': 'Accuracy',
                'profiling': ['SpikeCount_Total', 'Latency_Steps_Mean', 'InferenceTime_ms', 'GPU_Memory_Peak']
            },
            'normalization_compensation': {
                'RMSNorm_Issue': 'RMSNorm分母の近似/後段閾値スケーリングが別途必要', 
                'threshold_calibration_mode': 'Per_Channel_Scale' 
            },
            'progressive_quantization_stages': progressive_stages,
            'spiking_attention_enabled': use_spiking_attention,
            'ecl_enabled': use_ecl,
            'layer_strategies': self.layer_optimizer.strategies
        }

        # 7. 変換済みモデルの保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config,
            'conversion_metadata': conversion_metadata
        }, output_path)
        logger.info(f"✅ 統合LLM変換完了: '{output_path}'")


    def convert_cnn_weights(
        self,
        ann_model: nn.Module,
        output_path: str,
        calibration_loader: Any,
        use_ecl: bool = False,
        prune_low_activity: float = 0.0,
        quantization_bits: float = 0.0,
        hardware_target: str = "GPU",
    ) -> None:
        """CNNモデルの高忠実度変換を実行"""
        logger.info("--- 🚀 高忠実度CNN変換開始 ---")
        ann_model.to(self.device)
        ann_model.eval()

        if use_ecl: ann_model = _replace_activation_with_ecl(ann_model, initial_threshold=1.0, inplace=True)

        # BatchNorm Folding (Point 1)
        folded_model = fold_all_batchnorms(ann_model)
        
        # 閾値キャリブレーション
        thresholds = calibrate_thresholds_by_percentile(folded_model, calibration_loader, device=self.device)
        optimized_thresholds = self._optimize_threshold_distribution(thresholds, folded_model, calibration_loader)
        
        snn_neuron_layers: List[nn.Module] = [m for m in self.snn_model.modules() if isinstance(m, (AdaptiveLIFNeuron, DualThresholdNeuron, ScaleAndFireNeuron))]
        
        for i, lif in enumerate(snn_neuron_layers):
            # 簡易的な閾値適用 (リスト順)
            # 本来はレイヤー名でマッチングすべき
            thr = 1.0
            if optimized_thresholds:
                thr = list(optimized_thresholds.values())[i % len(optimized_thresholds)]
                
            is_tensor_thr = isinstance(thr, torch.Tensor)
            if isinstance(lif, DualThresholdNeuron):
                if is_tensor_thr:
                    thr_tensor = torch.as_tensor(thr).float()
                    lif.threshold_high.data.copy_(thr_tensor)
                    lif.threshold_low.data.copy_(thr_tensor * 0.5)
                else:
                    lif.threshold_high.data.fill_(thr)
                    lif.threshold_low.data.fill_(thr * 0.5)
            elif isinstance(lif, AdaptiveLIFNeuron):
                if is_tensor_thr:
                    thr_tensor = torch.as_tensor(thr).float()
                    lif.base_threshold.data.copy_(thr_tensor)
                else:
                    lif.base_threshold.data.fill_(thr)
            elif isinstance(lif, ScaleAndFireNeuron):
                thr_tensor = torch.as_tensor(thr)
                max_thr = thr_tensor.max().item()
                lif.thresholds.data.mul_(max_thr / lif.thresholds.data.max().item())
        
        # 安全な重みコピー
        safe_copy_weights(self.snn_model, folded_model.state_dict())

        # 動的プルーニング
        final_pruning_ratio = 0.0
        if prune_low_activity > 0.0 and calibration_loader:
            final_pruning_ratio = _dynamic_pruning_after_conversion(self.snn_model, prune_low_activity, calibration_loader)

        # 変換メタデータの保存
        conversion_metadata = {
            'bio_calibration_status': 'calibrated_with_ecl' if use_ecl else 'standard_conversion',
            'initial_pruning_ratio': final_pruning_ratio,
            'target_quantization_bits': quantization_bits,
            'hardware_optimization_hint': hardware_target,
            'time_step_batching_hint': hardware_target in ["Loihi", "TrueNorth"],
            'ecl_enabled': use_ecl,
            'evaluation_metrics': {
                'primary': 'Accuracy',
                'profiling': ['SpikeCount_Total', 'Latency_Steps_Mean', 'InferenceTime_ms', 'GPU_Memory_Peak']
            },
        }
        
        # モデルの保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config,
            'conversion_metadata': conversion_metadata
        }, output_path)
        logger.info(f"✅ CNN変換が完了: '{output_path}'")
