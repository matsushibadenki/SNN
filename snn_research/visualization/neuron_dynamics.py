# snn_research/visualization/neuron_dynamics.py
"""
Visualization tools for neuron dynamics.
(修正: バッチインデックス指定に対応)
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class NeuronDynamicsRecorder:
    """ニューロンの状態を記録するクラス"""
    
    def __init__(self, max_timesteps: int = 100):
        self.max_timesteps = max_timesteps
        self.history: Dict[str, List[torch.Tensor]] = {
            'membrane': [],
            'threshold': [],
            'spikes': []
        }
    
    def record(self, membrane: torch.Tensor, threshold: Optional[torch.Tensor], spikes: torch.Tensor):
        """1タイムステップの状態を記録"""
        if len(self.history['membrane']) < self.max_timesteps:
            self.history['membrane'].append(membrane.detach().cpu())
            if threshold is not None:
                self.history['threshold'].append(threshold.detach().cpu())
            self.history['spikes'].append(spikes.detach().cpu())
    
    def clear(self):
        """記録をクリア"""
        for key in self.history:
            self.history[key].clear()


def plot_neuron_dynamics(
    history: Dict[str, List[torch.Tensor]], 
    neuron_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
    batch_index: int = 0  # 追加: プロットするバッチのインデックス
) -> plt.Figure:
    """
    膜電位とスパイクの時系列をプロット。
    指定された batch_index のデータを描画します。
    """
    if not history['membrane']:
        raise ValueError("No data to plot")
    
    # データのスタック: (Time, Batch, Features) または (Time, Features)
    mem_values = torch.stack(history['membrane']).numpy()
    spike_values = torch.stack(history['spikes']).numpy()
    
    # 次元数の確認とデータの抽出
    ndim = mem_values.ndim
    
    if ndim == 3: # (Time, Batch, Features)
        # 指定されたバッチインデックスのデータを抽出
        if batch_index >= mem_values.shape[1]:
            print(f"Warning: batch_index {batch_index} is out of bounds. Using 0.")
            batch_index = 0
            
        mem_plot_data = mem_values[:, batch_index, :]
        spike_plot_data = spike_values[:, batch_index, :]
        
        if history['threshold']:
            threshold_stack = torch.stack(history['threshold']).numpy()
            if threshold_stack.ndim == 3:
                threshold_plot_data = threshold_stack[:, batch_index, :]
            else:
                threshold_plot_data = threshold_stack
        else:
            threshold_plot_data = None
            
    else: # (Time, Features) - バッチ次元なし
        mem_plot_data = mem_values
        spike_plot_data = spike_values
        if history['threshold']:
            threshold_plot_data = torch.stack(history['threshold']).numpy()
        else:
            threshold_plot_data = None

    time_steps = mem_plot_data.shape[0]
    num_neurons_total = mem_plot_data.shape[-1]

    num_neurons_to_plot = min(10, num_neurons_total)
    if neuron_indices is None:
        neuron_indices = list(range(num_neurons_to_plot))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # --- 1. 膜電位プロット ---
    for i in neuron_indices:
        val = mem_plot_data[:, i]
        axes[0].plot(val, label=f'Neuron {i}', alpha=0.7)
    
    # --- 2. 閾値プロット ---
    if threshold_plot_data is not None:
        # 最初のニューロンの閾値のみ表示（見やすさのため）
        thr_val = threshold_plot_data[:, neuron_indices[0]]
        axes[0].plot(thr_val, 'k--', alpha=0.8, label='Threshold')
    else:
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Base Threshold', alpha=0.5)
    
    axes[0].set_ylabel('Membrane Potential')
    axes[0].set_title(f'Neuron Membrane Dynamics (Batch Index: {batch_index})')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # --- 3. スパイク・ラスタプロット ---
    spike_times, spike_neurons = np.where(spike_plot_data > 0.5)
    
    axes[1].scatter(spike_times, spike_neurons, s=5, c='black', marker='|')
    axes[1].set_ylabel('Neuron Index')
    axes[1].set_title('Spike Raster Plot')
    
    # Y軸範囲の設定
    axes[1].set_ylim(-0.5, num_neurons_total - 0.5)
    axes[1].grid(True, alpha=0.3)
    
    # --- 4. 平均発火率プロット ---
    # 時間ステップごとの全ニューロン平均発火率
    spike_rate = spike_plot_data.mean(axis=1)

    axes[2].plot(spike_rate, color='blue', linewidth=2)
    axes[2].fill_between(range(time_steps), spike_rate, alpha=0.3)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Average Spike Rate')
    axes[2].set_title('Population Spike Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig