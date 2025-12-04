# ファイルパス: scripts/report_sparsity_and_T.py

"""
SNNのスパース性(Sparsity)と処理時間ステップ(T)に関する効率性レポートを生成するスクリプト。
"""
import os
import sys
import torch
import logging
import argparse
import numpy as np
from omegaconf import OmegaConf

# プロジェクトルート設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# scripts/runners ディレクトリをパスに追加して、trainモジュールを解決可能にする
runners_dir = os.path.join(project_root, 'scripts', 'runners')
if runners_dir not in sys.path:
    sys.path.append(runners_dir)

from snn_research.core.snn_core import SNNCore

# trainモジュールのインポート（データローダーなどのユーティリティ再利用のため）
try:
    import train
except ImportError:
    logging.warning("Could not import 'train' module from scripts/runners. Data loading might fail if dependent on it.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_efficiency(model_config_path, data_path, num_samples=100):
    logger.info(f"Loading model config from {model_config_path}")
    cfg = OmegaConf.load(model_config_path)
    model_params = OmegaConf.to_container(cfg, resolve=True)
    
    # モデル構築
    model = SNNCore(model_params)
    model.eval()
    
    logger.info(f"Model built: {type(model)}")
    
    # ダミーデータ生成（データローダーが複雑なため、ここではランダム入力でスパース性を測定）
    # 実際のデータでの測定が理想だが、レポート用として簡易実装
    input_size = (1, 3, 32, 32) # CIFAR10 size default
    if 'input_shape' in model_params:
         input_size = (1, *model_params['input_shape'])
         
    timesteps = model_params.get('timesteps', 16)
    
    logger.info(f"Measuring with random inputs. Shape: {input_size}, T: {timesteps}")
    
    total_spikes = 0
    total_neurons = 0 # 概算
    
    # フックでスパイクをカウントする簡易実装
    spike_counts = []
    
    def hook_fn(module, input, output):
        # outputがスパイクテンソル(0 or 1)と仮定
        if isinstance(output, torch.Tensor):
            spikes = output.sum().item()
            elements = output.numel()
            spike_counts.append((spikes, elements))
            
    hooks = []
    # ネットワーク内の全LIFレイヤーにフックをかける（再帰的探索が必要だが簡易的に）
    # snn_researchの実装に依存するが、modules()で走査
    for name, module in model.named_modules():
        if "LIF" in module.__class__.__name__ or "Spike" in module.__class__.__name__:
            hooks.append(module.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        for _ in range(num_samples):
            # ランダム入力 (0-1正規化済みと仮定)
            dummy_input = torch.rand(input_size)
            # 時間次元が必要な場合は拡張
            # SNNCoreの仕様によるが、(B, T, C, H, W) か (T, B, C, H, W) か
            # ここでは (B, C, H, W) を渡して内部でT展開されるか、エンコーダ任せと想定
            
            try:
                _ = model(dummy_input)
            except Exception as e:
                logger.warning(f"Inference failed with simple dummy input: {e}. Trying with time dimension.")
                # 時間次元を追加して再トライ (B, T, ...)
                dummy_input_t = dummy_input.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
                try:
                    _ = model(dummy_input_t)
                except Exception as e2:
                    logger.error(f"Inference failed again: {e2}")
                    break

    # フック解除
    for h in hooks:
        h.remove()
        
    if not spike_counts:
        logger.warning("No spikes recorded. Check if hooks attached correctly to spiking layers.")
        return
        
    total_s = sum(s for s, t in spike_counts)
    total_elements = sum(t for s, t in spike_counts)
    
    sparsity = 1.0 - (total_s / total_elements) if total_elements > 0 else 0.0
    activity_rate = (total_s / total_elements) * 100 if total_elements > 0 else 0.0
    
    print("\n" + "="*40)
    print(f" Efficiency Report")
    print("="*40)
    print(f" Model Config: {model_config_path}")
    print(f" Timesteps (T): {timesteps}")
    print(f" Total Neurons/Steps Measured: {total_elements}")
    print(f" Total Spikes: {total_s}")
    print(f" Sparsity: {sparsity:.4f} (Avg Spikes: {activity_rate:.2f}%)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--data_path', type=str, help="Not used in dummy mode")
    args = parser.parse_args()
    
    measure_efficiency(args.model_config, args.data_path)