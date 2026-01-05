"""
ファイルパス: scripts/debug_spike_activity.py
タイトル: SNNスパイク活動デバッガー (ニューロン直接監視版)
機能説明:
  SNNモデルの各ニューロン層（LIFなど）を直接フックし、正しいスパイク発火率を計測・表示します。
  (修正: 'cast' のインポート漏れを修正し、NameErrorを解消)
"""

import sys
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import logging
# 修正: 'cast' をインポートに追加
from typing import Dict, List, Tuple, Optional, Any, cast

# プロジェクトルートをsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 必要なライブラリのインポート
try:
    # 具体的なニューロンクラスをインポート
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
        TC_LIF, DualThresholdNeuron, ProbabilisticLIFNeuron
    )
    from snn_research.core.layers.lif_layer import LIFLayer
except ImportError as e:
    logger.error(f"ライブラリのインポートに失敗しました: {e}")
    logger.error("プロジェクトルートから実行しているか確認してください。")
    sys.exit(1)


# 監視対象のレイヤークラスをニューロンそのものに変更
TARGET_LAYERS = (
    AdaptiveLIFNeuron,
    IzhikevichNeuron,
    GLIFNeuron,
    TC_LIF,
    DualThresholdNeuron,
    ProbabilisticLIFNeuron,
    LIFLayer
)


# --- グローバル変数 ---
spike_counts: Dict[str, float] = {}
total_neurons: Dict[str, int] = {}
layer_names: List[str] = []


def create_hook(name: str):
    def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor, ...], output: Any):
        spk_tensor: Optional[torch.Tensor] = None

        # ニューロンの出力は通常 (spike, mem) のタプル
        if isinstance(output, tuple):
            spk_tensor = output[0]
        elif isinstance(output, torch.Tensor):
            spk_tensor = output

        if spk_tensor is not None:
            # スパイクの合計数をカウント
            with torch.no_grad():
                spike_counts[name] += float(torch.sum(spk_tensor).item())

                # 総ニューロン数の計算（初回のみ）
                if total_neurons[name] == 0:
                    # (Time, Batch, Features...) または (Batch, Features...)
                    if spk_tensor.dim() > 2:  # (T, B, F...)
                        num_neurons_per_step = spk_tensor.shape[2:].numel()
                    elif spk_tensor.dim() == 2:  # (B, F)
                        num_neurons_per_step = spk_tensor.shape[1:].numel()
                    else:
                        num_neurons_per_step = spk_tensor.numel()

                    total_neurons[name] = num_neurons_per_step
    return hook


def get_neuron_params_info(layer: nn.Module) -> str:
    """ニューロンパラメータの情報を文字列で返す"""
    info = []

    # 閾値
    if hasattr(layer, 'base_threshold'):
        # --- 修正: getattr と cast を使用して型を明確化 ---
        th = getattr(layer, 'base_threshold')
        # th が Tensor か float か判断してキャスト
        th_val: float
        if isinstance(th, torch.Tensor):
            th_val = th.mean().item()
        elif isinstance(th, (float, int)):
            th_val = float(th)
        else:
            th_val = 0.0  # フォールバック

        info.append(f"Th={th_val:.2f}")
    elif hasattr(layer, 'v_threshold'):
        info.append(f"Th={layer.v_threshold}")
    elif hasattr(layer, 'threshold'):  # ProbabilisticLIF
        info.append(f"Th={layer.threshold}")

    # 時定数
    if hasattr(layer, 'log_tau_mem'):  # AdaptiveLIF
        # log_tau から実際の tau を計算: tau = exp(log_tau) + 1.1
        # --- 修正: getattr と cast を使用 ---
        log_tau = cast(torch.Tensor, getattr(layer, 'log_tau_mem'))
        tau = (torch.exp(log_tau) + 1.1).mean().item()
        info.append(f"Tau≈{tau:.2f}")
    elif hasattr(layer, 'tau_mem'):
        info.append(f"Tau={layer.tau_mem}")

    return ", ".join(info) if info else ""


def register_hooks(model: nn.Module):
    global layer_names
    layer_names = []

    # 再帰的にモジュールを探索
    for name, layer in model.named_modules():
        if isinstance(layer, TARGET_LAYERS):
            if name in layer_names:
                continue

            layer_names.append(name)
            spike_counts[name] = 0.0
            total_neurons[name] = 0

            param_info = get_neuron_params_info(layer)
            logger.info(
                f"フック登録: {name} ({type(layer).__name__}) | {param_info}")

            layer.register_forward_hook(create_hook(name))

    logger.info(f"計 {len(layer_names)} 個のニューロン層にフックを登録しました。")


def reset_spike_counts():
    for name in spike_counts:
        spike_counts[name] = 0.0
        total_neurons[name] = 0


def load_config(config_path: str) -> DictConfig:
    try:
        conf = OmegaConf.load(config_path)
        # 修正: cast を使用して型ヒントを適合させる
        return cast(DictConfig, conf)
    except Exception as e:
        logger.error(f"設定ファイルのロードエラー: {e}")
        sys.exit(1)


def create_dummy_input(model_config: DictConfig, batch_size: int, timesteps: int) -> torch.Tensor:
    input_shape = model_config.get("input_shape")
    if isinstance(input_shape, str):
        try:
            import ast
            input_shape = ast.literal_eval(input_shape)
        except Exception:
            input_shape = None

    if isinstance(input_shape, (list, List, type(OmegaConf.create([])))) and len(input_shape) == 3:
        dummy = torch.rand(timesteps, batch_size, *input_shape)
        logger.info(f"画像型ダミー入力作成: {dummy.shape}")
        return dummy
    else:
        logger.info("シーケンス型ダミー入力作成: (B, T)")
        vocab_size = model_config.get("vocab_size", 1000)
        dummy = torch.randint(0, vocab_size, (batch_size, timesteps))
        return dummy


def run_spike_test(config_path: str, timesteps: int, batch_size: int):
    try:
        from snn_research.distillation.model_registry import ModelRegistry
    except ImportError:
        sys.exit(1)

    config_dict = load_config(config_path)
    model_config_dict = config_dict.get("model")

    if not model_config_dict:
        logger.error("'model' セクションが無効です。")
        sys.exit(1)

    model_name = model_config_dict.get("name")
    logger.info(f"対象モデル: {model_name}")

    try:
        model = ModelRegistry.get_model(model_config_dict, load_weights=False)
        model.eval()
    except Exception as e:
        logger.error(f"モデル構築失敗: {e}")
        sys.exit(1)

    register_hooks(model)
    if not layer_names:
        logger.warning("監視対象レイヤーが見つかりませんでした。")
        return

    cli_seq_len = timesteps
    model_internal_T = model_config_dict.get("time_steps")
    if isinstance(model_internal_T, int) and model_internal_T > 0:
        calc_T = model_internal_T
        logger.info(f"スパイク計算用タイムステップ: {calc_T} (Configより)")
    else:
        calc_T = cli_seq_len
        logger.warning(f"Configに 'time_steps' がないため、CLI引数を使用: {calc_T}")

    dummy_input = create_dummy_input(
        model_config_dict, batch_size, cli_seq_len)
    reset_spike_counts()

    try:
        with torch.no_grad():
            model(dummy_input)
        logger.info("フォワードパス完了。")
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"スパイク活動レポート: {model_name}")
    print("="*60)

    total_spikes_all = 0.0
    total_possible_all = 0.0

    for name in layer_names:
        spikes = spike_counts.get(name, 0.0)
        neurons = total_neurons.get(name, 0)

        if neurons == 0:
            continue

        # PredictiveCodingなど、内部でループするモデルの場合の総ステップ数推定
        # ここでは簡易的に calc_T * seq_len とする
        steps = calc_T * cli_seq_len

        possible_spikes = steps * batch_size * neurons
        spike_rate = (spikes / possible_spikes) * \
            100.0 if possible_spikes > 0 else 0.0

        total_spikes_all += spikes
        total_possible_all += possible_spikes

        print(f"Layer: {name}")
        print(f"  Type: {type(dict(model.named_modules())[name]).__name__}")
        print(f"  Neurons: {neurons:,}")
        print(f"  Spikes: {spikes:,.0f} / Max: {possible_spikes:,.0f}")
        print(f"  Rate: {spike_rate:.4f} %")
        print("-" * 30)

    print("="*60)
    overall_rate = (total_spikes_all / total_possible_all) * \
        100.0 if total_possible_all > 0 else 0.0
    print(f"Overall Mean Firing Rate: {overall_rate:.4f} %")
    print(f"Total Spikes: {total_spikes_all:,.0f}")
    print("="*60)

    if total_spikes_all == 0:
        logger.warning("⚠️  スパイクが全く検出されませんでした。閾値が高い可能性があります。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SNN Spike Activity Debugger")
    parser.add_argument('--model_config', type=str,
                        required=True, help="Path to model config")
    parser.add_argument('--timesteps', type=int, default=16, help="Time steps")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")

    args = parser.parse_args()

    if not os.path.exists(args.model_config):
        logger.error(f"File not found: {args.model_config}")
        sys.exit(1)

    run_spike_test(args.model_config, args.timesteps, args.batch_size)


if __name__ == "__main__":
    main()
