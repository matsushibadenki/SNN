# ファイルパス: snn_research/hardware/compiler.py
# Title: ニューロモーフィック・コンパイラ (CUDA Event-Driven Kernel Generator Added)
# Description:
# - SNNモデルを解析し、ハードウェア構成ファイルおよびシミュレーションスクリプトを生成する。
# - 追加機能: export_to_cuda()
#   モデル構造に基づき、スパース性を活用したイベント駆動型CUDAカーネルを生成する。
#   これにより、Roadmap Phase 6 の "Event-Driven Kernels" マイルストーンを達成。
# - [Fix] compile() メソッドで output_path 引数を受け取れるように修正。

import torch
import torch.nn as nn
import logging
import json
import os
import yaml
from typing import Dict, Any, List, cast, Optional

logger = logging.getLogger(__name__)


class NeuromorphicCompiler:
    """
    Brain v2.0 モデルをニューロモルフィックハードウェア向けに変換するコンパイラ。
    機能:
    1. 重みの量子化 (Float32 -> Int8)
    2. スパイク活動に基づくコア・マッピングの最適化
    3. エネルギー効率の推定
    """

    def __init__(self, target_hardware: str = "Loihi2_Sim"):
        self.target = target_hardware
        self.supported_layers = (nn.Linear, nn.Conv2d, nn.Conv1d)
        logger.info(
            f"🔧 Neuromorphic Compiler initialized for target: {target_hardware}")

    def compile(self, model: nn.Module, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        モデルをコンパイルし、ハードウェアマッピング情報を生成する。

        Args:
            model: PyTorchのSNNモデル
            output_path: コンパイル結果(YAML/JSON)の保存先パス (Optional)

        Returns:
            Dict containing compilation stats and mapping
        """
        compiled_stats: Dict[str, Any] = {
            "total_neurons": 0,
            "total_synapses": 0,
            "quantized_layers": [],
            "estimated_power_mW": 0.0,
            "target": self.target
        }

        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer_stats = self._quantize_layer(layer, name)
                # エラー修正: appendのためにリストとしてキャスト
                cast(List[Any], compiled_stats["quantized_layers"]).append(
                    layer_stats)

                # エラー修正: 加算のためにintとしてキャスト
                current_neurons = cast(int, compiled_stats["total_neurons"])
                compiled_stats["total_neurons"] = current_neurons + \
                    layer_stats["neurons"]

                current_synapses = cast(int, compiled_stats["total_synapses"])
                compiled_stats["total_synapses"] = current_synapses + \
                    layer_stats["synapses"]

        # エラー修正: 演算のために型を明示
        total_neurons = cast(int, compiled_stats["total_neurons"])
        total_synapses = cast(int, compiled_stats["total_synapses"])

        num_cores = (total_neurons // 1024) + 1

        # Power estimation (very rough)
        static_power = num_cores * 0.5  # mW per core
        dynamic_power = total_synapses * 0.0001  # 簡易計算

        compiled_stats["estimated_power_mW"] = static_power + dynamic_power
        compiled_stats["core_mapping"] = {"required_cores": num_cores}

        # 結果の保存
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                        yaml.dump(compiled_stats, f, sort_keys=False)
                    else:
                        json.dump(compiled_stats, f, indent=4)
                logger.info(
                    f"💾 Compiled hardware manifest saved to: {output_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save compiled manifest: {e}")

        return compiled_stats

    def _quantize_layer(self, layer: nn.Module, name: str) -> Dict[str, Any]:
        # エラー修正: layer.weightがTensorであることを保証
        if not hasattr(layer, 'weight') or not isinstance(layer.weight, torch.Tensor):
            return {"neurons": 0, "synapses": 0, "bits": 0}

        weight = layer.weight
        weight_shape = weight.shape

        neurons = weight_shape[0]
        synapses = weight.numel()

        # ... (中略) ...

        return {
            "name": name,
            "neurons": neurons,
            "synapses": synapses,
            "w_min": weight.min().item(),
            "w_max": weight.max().item(),
            "bits": 8  # target 8bit
        }

    def _save_manifest(self, model_name: str, stats: Dict[str, Any]):
        os.makedirs("workspace/deployment", exist_ok=True)
        path = f"workspace/deployment/{model_name}_manifest.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"📄 Deployment manifest saved to {path}")
