"""
ファイルパス: scripts/visualize_neuron_dynamics.py
タイトル: SNN ニューロンダイナミクス可視化ツール (ニューロン直接監視版)
機能説明:
  SNNモデルのニューロン層（LIF等）を直接監視し、膜電位とスパイクのダイナミクスを可視化します。
  (修正: 監視対象をPredictiveCodingLayerから実際のニューロンに変更)
"""

import sys
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Any, Callable, cast

# プロジェクトルートをsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron
    )
    from snn_research.core.layers.lif_layer import LIFLayer
    from snn_research.distillation.model_registry import ModelRegistry
    from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
except ImportError as e:
    logger.error(f"インポートエラー: {e}")
    sys.exit(1)


# --- 修正: 監視対象 ---
TARGET_LAYERS = (
    AdaptiveLIFNeuron, IzhikevichNeuron, LIFLayer
)


def create_dynamics_hook(recorder: NeuronDynamicsRecorder) -> Callable:
    def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor], output: Any) -> None:
        spk_tensor = None
        mem_tensor = None
        threshold_tensor: Optional[torch.Tensor] = None

        # ニューロン出力のタプル (spike, mem)
        if isinstance(output, tuple) and len(output) >= 2:
            spk_tensor = output[0]
            mem_tensor = output[1]

        elif isinstance(output, torch.Tensor):
            spk_tensor = output
            # 膜電位が返ってこない場合は記録スキップ
            return

        # 閾値
        if hasattr(module, 'adaptive_threshold'):
            at = getattr(module, 'adaptive_threshold')
            bt = getattr(module, 'base_threshold', 0.0)
            if at is not None:
                threshold_tensor = at + bt
        elif hasattr(module, 'v_threshold'):
            val = getattr(module, 'v_threshold')
            threshold_tensor = torch.tensor(val)

        # 記録
        if spk_tensor is not None and mem_tensor is not None:
            with torch.no_grad():
                spk_data = spk_tensor.detach().cpu()
                mem_data = mem_tensor.detach().cpu()
                thr_data = threshold_tensor.detach().cpu() if isinstance(
                    threshold_tensor, torch.Tensor) else None

                if spk_data.dim() >= 3:  # (T, B, F)
                    T = spk_data.shape[0]
                    for t in range(T):
                        recorder.record(mem_data[t], thr_data, spk_data[t])
                elif spk_data.dim() >= 2:  # (B, F)
                    recorder.record(mem_data, thr_data, spk_data)

    return hook


def load_config(config_path: str) -> DictConfig:
    try:
        return cast(DictConfig, OmegaConf.load(config_path))
    except Exception:
        sys.exit(1)


def create_dummy_input(model_config: DictConfig, batch_size: int, timesteps: int) -> torch.Tensor:
    input_shape = model_config.get("input_shape")
    if isinstance(input_shape, str):
        try:
            input_shape = eval(input_shape)
        except Exception:
            input_shape = None

    if isinstance(input_shape, (list, List)) and len(input_shape) == 3:
        dummy = torch.rand(timesteps, batch_size, *input_shape)
        return dummy
    else:
        vocab_size = model_config.get("vocab_size", 1000)
        dummy = torch.randint(0, vocab_size, (batch_size, timesteps))
        return dummy


def run_dynamics_visualization(config_path: str, timesteps: int, output_path_str: str) -> None:
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = load_config(config_path)
    model_config_dict = config_dict.get("model")

    try:
        model = ModelRegistry.get_model(
            cast(DictConfig, model_config_dict), load_weights=False)
        model.eval()
    except Exception as e:
        logger.error(f"モデルエラー: {e}")
        sys.exit(1)

    config_steps = model_config_dict.get("time_steps")
    estimated_steps = timesteps
    if isinstance(config_steps, int) and config_steps > 0:
        estimated_steps = max(timesteps, config_steps) * 2

    # フック登録
    recorder = NeuronDynamicsRecorder(max_timesteps=estimated_steps * 10)
    target_layer_found = False

    for name, layer in model.named_modules():
        if isinstance(layer, TARGET_LAYERS):
            logger.info(f"監視対象レイヤー発見: {name} ({type(layer).__name__})")
            layer.register_forward_hook(create_dynamics_hook(recorder))
            target_layer_found = True
            break  # 最初に見つかったニューロン層のみ監視

    if not target_layer_found:
        logger.error("監視可能なニューロン層が見つかりませんでした。")
        sys.exit(1)

    BATCH_SIZE = 4
    dummy_input = create_dummy_input(
        cast(DictConfig, model_config_dict), timesteps=timesteps, batch_size=BATCH_SIZE)

    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        sys.exit(1)

    # プロット (最適バッチ選択)
    if len(recorder.history['spikes']) == 0:
        logger.warning("データなし。")
        return

    try:
        spikes_stack = torch.stack(recorder.history['spikes'])
        if spikes_stack.dim() >= 2:
            # (Time, Batch, Features) -> sum -> (Batch,)
            spike_counts = spikes_stack.sum(dim=0)
            while spike_counts.dim() > 1:
                spike_counts = spike_counts.sum(dim=-1)
            # --- 修正: int() キャストを追加 ---
            best_batch_idx = int(torch.argmax(spike_counts).item())
            max_spikes = spike_counts[best_batch_idx].item()
        else:
            best_batch_idx = 0
            max_spikes = 0

        logger.info(f"Batch Spike Counts: {spike_counts.tolist()}")
        logger.info(
            f"✅ 最も活動的なサンプル (Batch Index {best_batch_idx}) を選択しました (Spikes: {int(max_spikes)})。")

        plot_neuron_dynamics(
            history=recorder.history,
            neuron_indices=None,
            save_path=output_path,
            batch_index=int(best_batch_idx)
        )
        logger.info(f"✅ ダイナミクスプロットを保存しました: {output_path}")
    except Exception as e:
        logger.error(f"プロットエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=16)
    parser.add_argument('--output_path', type=str,
                        default="workspace/runs/dynamics_viz/dynamics.png")
    args = parser.parse_args()

    if not os.path.exists(args.model_config):
        sys.exit(1)
    run_dynamics_visualization(
        args.model_config, args.timesteps, args.output_path)


if __name__ == "__main__":
    main()
