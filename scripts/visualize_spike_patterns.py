"""
ファイルパス: scripts/visualize_spike_patterns.py
タイトル: SNNスパイク活動可視化ツール (ニューロン直接監視・自動バッチ選択版)
機能説明:
  SNNモデルのニューロン層を直接監視し、最もスパイク活動が活発なサンプルを自動選択して
  ラスタプロットを生成します。
"""

import sys
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any, cast
from pathlib import Path

# --- 可視化ライブラリの確認 ---
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("エラー: matplotlib または numpy がインストールされていません。")
    sys.exit(1)

# プロジェクトルートをsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # --- 修正: 具体的なニューロンクラスをインポート ---
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, 
        TC_LIF, DualThresholdNeuron, ProbabilisticLIFNeuron
    )
    from snn_research.core.layers.lif_layer import LIFLayer
    from snn_research.distillation.model_registry import ModelRegistry
except ImportError as e:
    logger.error(f"ライブラリインポート失敗: {e}")
    sys.exit(1)


# --- 修正: 監視対象をニューロンクラスに変更 ---
TARGET_LAYERS = (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron, 
    TC_LIF, DualThresholdNeuron, ProbabilisticLIFNeuron, LIFLayer
)


class SpikeRecorder:
    def __init__(self) -> None:
        self.collected_spikes: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        self.collected_spikes.clear()
        self.remove_hooks()

        count = 0
        for name, layer in model.named_modules():
            if isinstance(layer, TARGET_LAYERS):
                if name not in self.collected_spikes:
                    self.collected_spikes[name] = []
                    h = layer.register_forward_hook(self._create_hook(name))
                    self.hooks.append(h)
                    count += 1
                    logger.debug(f"フック登録: {name}")
        
        logger.info(f"計 {count} 個のニューロン層にフックを登録しました。")

    def _create_hook(self, name: str) -> Any:
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            spk_tensor = None
            if isinstance(output, tuple):
                spk_tensor = output[0]
            elif isinstance(output, torch.Tensor):
                spk_tensor = output
            
            if spk_tensor is not None:
                self.collected_spikes[name].append(spk_tensor.detach().cpu())
        return hook

    def remove_hooks(self) -> None:
        for h in self.hooks: h.remove()
        self.hooks = []

    def get_stacked_spikes(self, name: str) -> Optional[torch.Tensor]:
        spike_list = self.collected_spikes.get(name)
        if not spike_list: return None
        try:
            # (B, F) のリスト -> (T, B, F)
            if spike_list[0].dim() == 2:
                return torch.stack(spike_list, dim=0)
            # (T, B, F) が1つ -> そのまま
            elif spike_list[0].dim() >= 3 and len(spike_list) == 1:
                return spike_list[0]
            else:
                return torch.stack(spike_list, dim=0)
        except Exception:
            return None


def plot_raster(spike_data: np.ndarray, title: str, output_path: Path, 
                total_steps: int, num_neurons: int, batch_idx: int) -> None:
    
    if spike_data.size == 0: return

    # スパイク (0/1) の位置を取得
    spike_times, neuron_ids = np.where(spike_data > 0.5)
    spike_count = spike_times.size
    possible_spikes = total_steps * num_neurons
    avg_rate = (spike_count / possible_spikes) * 100.0 if possible_spikes > 0 else 0.0

    logger.info(f"  Layer: {title} | Batch: {batch_idx} | Spikes: {spike_count} | Rate: {avg_rate:.2f}%")

    plt.figure(figsize=(12, 6))
    if spike_count > 0:
        plt.scatter(spike_times, neuron_ids, s=10, marker='|', alpha=0.8, c='black')

    plt.title(f"Raster Plot: {title} (Batch {batch_idx})\nTotal: {spike_count} spikes / Rate: {avg_rate:.2f}%")
    plt.xlabel(f"Time Step (Total: {total_steps})")
    plt.ylabel("Neuron ID")
    plt.xlim(0, max(total_steps, spike_data.shape[0]))
    plt.ylim(-0.5, num_neurons - 0.5)
    if num_neurons <= 20: plt.yticks(range(num_neurons))
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def process_and_visualize(recorder: SpikeRecorder, output_prefix: str, calc_T: int, seq_len: int) -> None:
    logger.info("--- スパイク活動の可視化 ---")
    
    layers = list(recorder.collected_spikes.keys())
    if not layers:
        logger.warning("データなし。")
        return

    total_vis_steps = calc_T * seq_len
    
    for name in layers:
        stacked_tensor = recorder.get_stacked_spikes(name)
        if stacked_tensor is None: continue

        # 最適なバッチ選択
        try:
             # (Time, Batch, Features) -> (Batch,)
             if stacked_tensor.dim() >= 2:
                 batch_spike_counts = stacked_tensor.sum(dim=0)
                 while batch_spike_counts.dim() > 1:
                     batch_spike_counts = batch_spike_counts.sum(dim=-1)
                 
                 # --- 修正: int() キャストを追加 ---
                 best_batch_idx = int(torch.argmax(batch_spike_counts).item())
                 max_spikes = batch_spike_counts[best_batch_idx].item()
             else:
                 best_batch_idx = 0
                 max_spikes = 0
        except:
             best_batch_idx = 0
             max_spikes = 0

        if max_spikes == 0:
             logger.warning(f"[{name}] スパイクなし。")

        # --- 修正: best_batch_idx は int であることが保証されている ---
        if stacked_tensor.shape[1] > best_batch_idx:
             spikes_best = stacked_tensor[:, best_batch_idx, ...].contiguous()
        else:
             spikes_best = stacked_tensor[:, 0, ...].contiguous()
        
        spikes_flat = spikes_best.view(spikes_best.shape[0], -1).numpy()
        time_steps, num_neurons = spikes_flat.shape
        
        safe_name = name.replace('.', '_')
        output_path = Path(f"{output_prefix}_{safe_name}.png")
        
        plot_raster(spikes_flat, name, output_path, max(time_steps, total_vis_steps), num_neurons, int(best_batch_idx))


def load_config(config_path: str) -> DictConfig:
    try:
        return cast(DictConfig, OmegaConf.load(config_path))
    except: sys.exit(1)


def create_dummy_input(model_config: DictConfig, batch_size: int, timesteps: int) -> torch.Tensor:
    input_shape = model_config.get("input_shape")
    if isinstance(input_shape, str):
        try: input_shape = eval(input_shape)
        except: input_shape = None

    if isinstance(input_shape, (list, List)) and len(input_shape) == 3:
        dummy = torch.rand(timesteps, batch_size, *input_shape)
        logger.info(f"画像ダミー入力: {dummy.shape}")
        return dummy
    else:
        vocab_size = model_config.get("vocab_size", 1000)
        dummy = torch.randint(0, vocab_size, (batch_size, timesteps))
        logger.info(f"シーケンスダミー入力: {dummy.shape}")
        return dummy


def run_spike_visualization(config_path: str, timesteps: int, batch_size: int, output_prefix: str) -> None:
    config_dict = load_config(config_path)
    model_config = config_dict.get("model")
    
    try:
        model = ModelRegistry.get_model(cast(DictConfig, model_config), load_weights=False)
        model.eval()
    except Exception as e:
        logger.error(f"モデル構築失敗: {e}")
        sys.exit(1)

    recorder = SpikeRecorder()
    recorder.register_hooks(model)

    cli_seq_len = timesteps
    model_internal_T = model_config.get("time_steps")
    calc_T = model_internal_T if isinstance(model_internal_T, int) and model_internal_T > 0 else cli_seq_len
    
    actual_batch_size = max(4, batch_size)
    dummy_input = create_dummy_input(cast(DictConfig, model_config), actual_batch_size, cli_seq_len)
    
    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        sys.exit(1)

    process_and_visualize(recorder, output_prefix, calc_T, cli_seq_len)
    recorder.remove_hooks()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_prefix', type=str, default="runs/spike_viz/plot")

    args = parser.parse_args()
    if not os.path.exists(args.model_config): sys.exit(1)
        
    run_spike_visualization(args.model_config, args.timesteps, args.batch_size, args.output_prefix)

if __name__ == "__main__":
    main()