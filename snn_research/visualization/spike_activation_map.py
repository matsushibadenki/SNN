# ファイルパス: snn_research/visualization/spike_activation_map.py
# Title: Spike Activation Map (SAM) (mypy修正 & 記録ロジック改善)
# Description: Improvement-Plan.md に基づき、SNNの内部動作を可視化するための
#              Spike Activation Map (SAM) 生成機能を実装します。
#              スパイク間隔 (Inter-Spike Interval, ISI) を利用します。
#              フォールバック処理の注意喚起と可視化を改善。
#              mypyエラー [name-defined] を解消。
#              スパイク記録ロジックを改善し、内部/外部タイムステップループに対応試行。
#              mypyエラー [arg-type] を修正。 # <--- 修正コメント追加

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# --- ▼ 修正: cast をインポート ▼ ---
from typing import List, Tuple, Optional, Dict, cast, Union # Union をインポート
# --- ▲ 修正 ▲ ---
from torch.utils.hooks import RemovableHandle
import logging
from pathlib import Path # Path をインポート
from spikingjelly.activation_based import functional # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpikeActivationMap:
    """
    Spike Activation Map (SAM) を生成し、SNNの時空間的な活動を可視化するクラス。
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[RemovableHandle] = []
        self.spike_history: List[torch.Tensor] = [] # 各タイムステップのスパイクを記録
        self.target_layer_found = False # フックが正しく登録されたか
        logging.info("SpikeActivationMap initialized.")
        # モデルが配置されているデバイスを取得
        try:
             # --- ▼ 修正: self.model_device の型を明示 ▼ ---
             self.model_device: torch.device = next(model.parameters()).device
             # --- ▲ 修正 ▲ ---
        except StopIteration:
             # パラメータがないモデルの場合 (例: nn.Identity のみ)
             logging.warning("Model has no parameters, defaulting device to CPU for SAM.")
             self.model_device = torch.device('cpu')


    def _record_hook(self, module, input, output):
        """フォワードフック関数。スパイク出力を記録する。各タイムステップで呼ばれることを期待。"""
        spikes = None
        # 出力形式に応じてスパイクを抽出
        if isinstance(output, tuple):
            # SNN Coreの (logits, spikes, mem) やニューロンの (spikes, mem) を想定
            if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].numel() > 0: # 2番目がスパイクと仮定
                spikes = output[1]
            elif len(output) > 0 and isinstance(output[0], torch.Tensor) and output[0].numel() > 0: # 1番目がスパイクの場合も考慮
                 # 簡単な形状チェック (例: スパイクは {0, 1} に近い値を持つはず)
                 if output[0].float().mean().item() < 0.5: # ヒューリスティック
                      spikes = output[0]
        elif isinstance(output, torch.Tensor) and output.numel() > 0:
             # 単一テンソル出力の場合、それがスパイクと仮定 (形状チェックは難しい)
             spikes = output

        if spikes is not None:
            # 記録するのは最初のバッチ要素のみ、かつCPUへ移動
            # スパイク形状: (B, ...) or (B*?, ...)
            # 最初のバッチ要素を取得
            try:
                 # 多くのモデルは (B, T, F) or (B, F) or (B*T, F)
                 if spikes.dim() >= 2 and spikes.shape[0] > 0:
                      # 最初のバッチ要素のみ取得 (TやFがあっても最初のB次元でスライス)
                      first_batch_spike = spikes[0:1].detach().float().cpu()
                      self.spike_history.append(first_batch_spike)
                 elif spikes.dim() == 1 and spikes.shape[0] > 0: # 特徴量次元のみの場合 (例: 1サンプルの最後の層)
                      first_batch_spike = spikes.unsqueeze(0).detach().float().cpu() # バッチ次元を追加
                      self.spike_history.append(first_batch_spike)
                 # else: logging.warning(f"Skipping spike recording due to unexpected shape: {spikes.shape}")
            except Exception as e:
                 logging.warning(f"Error processing spikes in hook for module {module}: {e}, Shape: {spikes.shape}")
        # else:
            # logging.warning(f"Could not reliably extract spikes for module {module}. Output type: {type(output)}")


    def _register_hooks(self, target_layer_name: str):
        """指定されたレイヤー名にフォワードフックを登録する。"""
        self.clear_hooks()
        self.target_layer_found = False # フラグをリセット
        # 完全一致するモジュールを探す
        target_module = None
        for name, module in self.model.named_modules():
             if name == target_layer_name:
                  target_module = module
                  break

        if target_module is not None:
             try:
                 # フックを登録
                 self.hooks.append(target_module.register_forward_hook(self._record_hook))
                 logging.info(f"Registered hook for layer: {target_layer_name} ({type(target_module).__name__})")
                 self.target_layer_found = True
             except Exception as e:
                 logging.error(f"Failed to register hook for {target_layer_name}: {e}")
        else:
             logging.warning(f"Target layer '{target_layer_name}' not found.")


    def clear_hooks(self):
        """登録されているすべてのフックを解除し、履歴をクリアする。"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.spike_history = [] # 履歴もクリア

    def _run_model_forward(self, input_data: torch.Tensor, time_steps: int) -> bool:
        """
        モデルのフォワードパスを実行し、スパイクが記録されたかを確認する。
        内部タイムステップ処理と外部ループの両方に対応試行。
        Returns:
            bool: スパイクが記録されたかどうか。
        """
        self.model.eval()
        self.spike_history = [] # 実行前に履歴をクリア
        success = False

        # 入力データをモデルと同じデバイスに移動
        # --- ▼ 修正: mypyエラー [arg-type] 対策 (castを追加) ▼ ---
        input_data = input_data.to(cast(torch.device, self.model_device))
        # --- ▲ 修正 ▲ ---

        with torch.no_grad():
            try:
                # --- 試行1: モデルが time_steps を引数として受け取るか ---
                # (例: SpikingJellyベースのモデル)
                logging.info(f"Attempting forward pass with time_steps={time_steps} argument...")
                # SNNコアのインターフェースに合わせて return_spikes=True も渡す
                self.model(input_data, time_steps=time_steps, return_spikes=True)
                if self.spike_history:
                     logging.info(f"  -> Success: Model likely handles time steps internally. Recorded {len(self.spike_history)} spike tensors.")
                     success = True
                else:
                     logging.warning(f"  -> Model ran but no spikes recorded via hook. Trying external loop...")

            except TypeError as e:
                # --- 試行2: time_steps 引数を受け付けない場合、外部ループで処理 ---
                # (例: snnTorchベースのモデルや単純なRSNN)
                if 'time_steps' in str(e):
                    logging.info(f"Model does not accept 'time_steps'. Attempting external loop...")
                    functional.reset_net(self.model) # 外部ループの前にリセット
                    # snnTorchなどは内部状態を持つので stateful=True に設定する必要があるかも
                    # (ただし、全ニューロンに再帰的に適用するのは難しい)

                    # 最初の入力（t=0）
                    current_input = input_data # (B, F) or (B, T, F) ? -> (B, F)を仮定
                    if input_data.dim() == 3: # (B, T, F) の場合は最初のタイムステップを使う
                         current_input = input_data[:, 0, :]

                    for t in range(time_steps):
                        # 1ステップ実行
                        # このforward呼び出しでフックが呼ばれることを期待
                        output = self.model(current_input, return_spikes=True) # return_spikes=True を試す
                        # outputがタプルの場合、次の入力に何を使うか？ (通常はスパイク)
                        if isinstance(output, tuple):
                             # --- ▼ 修正: 次の入力がTensorであることを確認 ▼ ---
                             next_input_candidate = output[0]
                             if isinstance(next_input_candidate, torch.Tensor):
                                 current_input = next_input_candidate # 次の入力は現在の出力スパイクと仮定
                             else:
                                 logging.error(f"Output[0] is not a Tensor in external loop at step {t}. Aborting loop.")
                                 break
                             # --- ▲ 修正 ▲ ---
                        elif isinstance(output, torch.Tensor):
                             current_input = output # 次の入力は現在の出力と仮定
                        else:
                             logging.error(f"Unexpected output type {type(output)} in external loop at step {t}. Aborting loop.")
                             break


                    if self.spike_history:
                        logging.info(f"  -> Success: External loop completed. Recorded {len(self.spike_history)} spike tensors.")
                        success = True
                    else:
                        logging.warning(f"  -> External loop ran but no spikes recorded via hook.")
                else:
                    # time_steps 以外の TypeError
                    logging.error(f"Model forward pass failed with TypeError (not related to time_steps): {e}")
                    success = False

            except Exception as e_generic:
                 logging.error(f"An unexpected error occurred during model forward pass: {e_generic}")
                 success = False

        return success and bool(self.spike_history) # スパイクが実際に記録されたか


    def _record_spikes(self, input_data: torch.Tensor, target_layer_name: str, time_steps: int) -> List[torch.Tensor]:
        """
        指定されたレイヤーのスパイク時系列を記録します。
        内部タイムステップ処理と外部ループの両方に対応試行。
        """
        self._register_hooks(target_layer_name)

        if not self.target_layer_found:
             logging.error(f"Hook not registered for {target_layer_name}. Cannot record spikes.")
             self.clear_hooks()
             return []

        # フォワードパスを実行し、スパイクを記録
        # --- ▼ 修正: 入力データをモデルと同じデバイスに移動 ▼ ---
        spikes_recorded = self._run_model_forward(input_data.to(self.model_device), time_steps)
        # --- ▲ 修正 ▲ ---

        recorded_spikes_history = self.spike_history # フックによって記録されたスパイク
        self.clear_hooks() # 記録が終わったらフックを解除

        if not spikes_recorded:
             logging.warning(f"No spikes were recorded for layer {target_layer_name}. Check hook registration, model forward pass logic, or if the layer actually spiked.")
             return []

        # 記録されたスパイクの形状を確認し、整形
        if recorded_spikes_history:
            logging.info(f"Recorded {len(recorded_spikes_history)} spike tensors. Example shape: {recorded_spikes_history[0].shape}")

            # 形状を (TimeSteps, Batch=1, Features) に統一しようと試みる
            processed_history: List[torch.Tensor] = []
            expected_features = -1
            for t_idx, spikes_at_t in enumerate(recorded_spikes_history):
                 # (B=1, ...) の形状を期待
                 if spikes_at_t.dim() >= 1 and spikes_at_t.shape[0] == 1:
                      # 特徴量次元数を決定 (最初の有効なテンソルから)
                      if expected_features == -1:
                           if spikes_at_t.dim() == 2: # (B=1, F)
                                expected_features = spikes_at_t.shape[1]
                           elif spikes_at_t.dim() == 3: # (B=1, T=1?, F)
                                expected_features = spikes_at_t.shape[2]
                                spikes_at_t = spikes_at_t.squeeze(1) # T=1次元を削除 -> (B=1, F)
                           elif spikes_at_t.dim() > 3: # (B=1, C, H, W) など -> Flatten
                                expected_features = spikes_at_t[0].numel()
                                spikes_at_t = spikes_at_t.view(1, -1) # (B=1, F_flat)
                           else: # (B=1,) スカラー？
                                expected_features = 1
                                spikes_at_t = spikes_at_t.view(1, 1)

                      # 形状チェックと整形
                      if spikes_at_t.dim() == 2 and spikes_at_t.shape[1] == expected_features:
                           processed_history.append(spikes_at_t) # (B=1, F)
                      elif spikes_at_t.dim() > 2 and spikes_at_t.view(1, -1).shape[1] == expected_features:
                           processed_history.append(spikes_at_t.view(1, -1)) # Flattenして追加
                      else:
                           logging.warning(f"Skipping tensor at step {t_idx} due to inconsistent feature dimension. Expected {expected_features}, got shape {spikes_at_t.shape}")
                 else:
                      logging.warning(f"Skipping tensor at step {t_idx} due to unexpected batch dimension or shape: {spikes_at_t.shape}")

            if len(processed_history) != len(recorded_spikes_history):
                 logging.warning("Some recorded spike tensors were skipped due to shape inconsistencies.")

            # 内部ループの場合、記録されたテンソル数が time_steps より少ない可能性がある
            num_recorded = len(processed_history)
            if num_recorded > 0 and num_recorded < time_steps:
                 logging.warning(f"Recorded {num_recorded} steps, expected {time_steps}. Padding with zeros for ISI calculation (might be inaccurate).")
                 # 足りない分をゼロで埋める
                 last_spike_shape = processed_history[-1].shape
                 padding = [torch.zeros_like(processed_history[-1]) for _ in range(time_steps - num_recorded)]
                 processed_history.extend(padding)
            elif num_recorded > time_steps:
                 logging.warning(f"Recorded {num_recorded} steps, more than expected {time_steps}. Truncating.")
                 processed_history = processed_history[:time_steps]


            return processed_history

        return [] # スパイクが記録されなかった場合

    # _compute_isi は変更なし
    def _compute_isi(self, spikes_history: List[torch.Tensor], max_isi: float = 100.0) -> torch.Tensor:
        """
        スパイク時系列からスパイク間隔 (ISI) を計算します。
        入力形状: List[(Batch=1, Features)]
        """
        if not spikes_history or spikes_history[0].numel() == 0:
            logging.warning("No valid spike history to compute ISI from.")
            return torch.tensor([])

        # (TimeSteps, Batch=1, Features) -> (TimeSteps, Features)
        try:
             # Ensure all tensors have the same feature dimension
             feature_dim = spikes_history[0].shape[-1]
             # Remove batch dim, filter out inconsistent shapes
             valid_history = [t.squeeze(0) for t in spikes_history if t.dim() == 2 and t.shape[0] == 1 and t.shape[-1] == feature_dim]
             if not valid_history:
                  logging.error("Spike history tensors have inconsistent feature dimensions, unexpected batch size, or are empty after filtering.")
                  return torch.tensor([])
             # Stack along time dimension
             spikes_tensor = torch.stack(valid_history, dim=0) # (TimeSteps, Features)
        except Exception as e:
            logging.error(f"Error stacking spike history: {e}. Check tensor shapes. Example shape: {spikes_history[0].shape}")
            return torch.tensor([])

        time_steps, features = spikes_tensor.shape
        if time_steps <= 1:
             logging.warning("ISI calculation requires at least 2 time steps.")
             # スパイクがあればISI=0, なければmax_isiとする簡易処理
             avg_isi = torch.where(spikes_tensor.sum(dim=0) > 0, 0.0, max_isi)
             return avg_isi


        isi_sum = torch.zeros(features)
        spike_count_for_isi = torch.zeros(features) # ISI計算に使われたスパイク数
        last_spike_time = torch.full((features,), -1.0)

        for t in range(time_steps):
            # 現在のタイムステップでスパイクしたニューロン
            current_spikes_indices = torch.where(spikes_tensor[t] > 0.5)[0] # Use threshold for robustness

            for neuron_idx in current_spikes_indices:
                neuron_idx_item = neuron_idx.item() # Convert tensor index to int
                if last_spike_time[neuron_idx_item] >= 0: # 前回のスパイク記録がある場合
                    isi = t - last_spike_time[neuron_idx_item]
                    if isi > 0: # ISIは正でなければならない
                        isi_sum[neuron_idx_item] += isi
                        spike_count_for_isi[neuron_idx_item] += 1
                # 最後のスパイク時刻を更新
                last_spike_time[neuron_idx_item] = float(t)

        # 平均ISIを計算 (ISIが計算できたニューロンのみ)
        avg_isi = torch.full((features,), max_isi) # デフォルトは max_isi
        valid_mask = spike_count_for_isi > 0
        avg_isi[valid_mask] = isi_sum[valid_mask] / spike_count_for_isi[valid_mask]

        # スパイクしなかったニューロンも max_isi のまま
        logging.info(f"Computed average ISI for {features} neurons. Mean ISI: {avg_isi[valid_mask].mean().item() if valid_mask.any() else 'N/A'}")
        return avg_isi

    @torch.no_grad()
    def generate_temporal_heatmap(self, input_data: torch.Tensor, target_layer_name: str, time_steps: int, save_path: Optional[Union[str, Path]] = None):
        """
        指定されたレイヤーのSAM（ISIに基づくヒートマップ）を生成します。

        Args:
            input_data (torch.Tensor): モデルへの入力データ (Batch, ...)。 B>=1であること。
            target_layer_name (str): 可視化対象のレイヤー名。
            time_steps (int): SNNのシミュレーション時間ステップ数。
            save_path (Optional[Union[str, Path]]): ヒートマップを保存する場合のパス。
        """
        logging.info(f"Generating Spike Activation Map for layer '{target_layer_name}'...")

        if input_data.shape[0] == 0:
             logging.error("Input data batch size is 0.")
             return None

        # 1. スパイク時系列を記録 (最初のバッチ要素のみ使用し、モデルのデバイスに移動)
        # --- ▼ 修正: input_data[0:1] を渡す ▼ ---
        spikes_history = self._record_spikes(input_data[0:1], target_layer_name, time_steps)
        # --- ▲ 修正 ▲ ---

        if not spikes_history:
             logging.error("Failed to record spikes. Cannot generate heatmap.")
             return None

        # 2. ISIを計算
        avg_isi_map = self._compute_isi(spikes_history, max_isi=float(time_steps + 1)) # 最大ISIをタイムステップ数+1に設定

        if avg_isi_map.numel() == 0:
             logging.error("Failed to compute ISI or ISI map is empty. Cannot generate heatmap.")
             return None

        # 3. ヒートマップを生成・表示
        num_features = avg_isi_map.shape[0]
        # ニューロン数が多すぎる場合、一部をサンプリングして表示 (例: 1024個まで)
        max_neurons_to_plot = 1024
        if num_features > max_neurons_to_plot:
             indices = torch.linspace(0, num_features - 1, max_neurons_to_plot).long()
             plot_data = avg_isi_map[indices].unsqueeze(0).cpu().numpy()
             xlabel = f'Neuron Index (Sampled {max_neurons_to_plot}/{num_features})'
        else:
             plot_data = avg_isi_map.unsqueeze(0).cpu().numpy()
             xlabel = f'Neuron Index ({num_features})'


        fig, ax = plt.subplots(figsize=(max(12, num_features // 50), 3)) # 横幅を特徴量数に応じて調整
        # cmap='viridis_r' もしくは 'plasma_r' など、低い値が目立つカラーマップが良い
        im = ax.imshow(plot_data, cmap='hot_r', aspect='auto', interpolation='nearest', vmin=0, vmax=time_steps + 1)
        cbar = fig.colorbar(im, ax=ax, label='Average Inter-Spike Interval (ISI)')
        ax.set_title(f'Spike Activation Map (SAM) - Layer: {target_layer_name}\nLower ISI (darker color) indicates higher activity')
        ax.set_xlabel(xlabel)
        ax.set_yticks([]) # 1D表示なのでY軸は不要

        plt.tight_layout() # レイアウトを調整

        if save_path:
             save_path_obj = Path(save_path) # Pathオブジェクトに変換
             save_path_obj.parent.mkdir(parents=True, exist_ok=True) # 親ディレクトリを作成
             try:
                 plt.savefig(save_path_obj, dpi=150, bbox_inches='tight')
                 logging.info(f"SAM heatmap saved to {save_path_obj}")
             except Exception as e:
                 logging.error(f"Failed to save heatmap to {save_path_obj}: {e}")
        else:
             plt.show()

        plt.close(fig) # Figureオブジェクトを明示的に閉じる

        return avg_isi_map