from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from snn_research.data.datasets import SimpleTextDataset
from snn_research.core.snn_core import SNNCore
import sys
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    print("🔍 SNN詳細信号診断 (Full Forward / Fixed) を開始します...")

    # 1. 設定とモデル
    config_path = "configs/models/bit_rwkv_micro.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return

    cfg = OmegaConf.load(config_path)

    print("  - Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = "cpu"

    print("  - Building model...")
    try:
        model_container = SNNCore(config=cfg.model, vocab_size=len(tokenizer))
        model = model_container.model
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ モデル構築エラー: {e}")
        return

    # 2. データ準備
    data_path = "data/smoke_test_data.jsonl"
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    dataset = SimpleTextDataset(data_path, tokenizer, max_seq_len=16)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    try:
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        elif isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
        else:
            print(f"❌ Unexpected batch type: {type(batch)}")
            return
    except StopIteration:
        print("❌ データセットが空です。")
        return

    # 3. 詳細診断実行
    print("\n📊 レイヤー別信号追跡:")

    # フック関数: 入出力の統計を表示 (修正: floatキャスト追加)
    def debug_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if isinstance(output, tuple):
                output = output[0]

            # float()にキャストしてから計算することでエラーを回避
            in_mean = input.float().abs().mean().item(
            ) if isinstance(input, torch.Tensor) else 0.0
            out_mean = output.float().abs().mean().item(
            ) if isinstance(output, torch.Tensor) else 0.0
            out_max = output.float().abs().max().item(
            ) if isinstance(output, torch.Tensor) else 0.0

            # スパイク数 (ニューロンの場合)
            spike_info = ""
            if "lif" in name.lower() or "neuron" in name.lower():
                if isinstance(output, torch.Tensor):
                    spike_count = output.sum().item()
                    spike_rate = output.float().mean().item() * 100
                    spike_info = f" | Spikes: {int(spike_count)} (Rate: {spike_rate:.2f}%)"

            print(f"  🔹 [{name}]")
            print(f"      Input Mean: {in_mean:.6f}")
            print(
                f"      Output Mean: {out_mean:.6f} | Max: {out_max:.6f}{spike_info}")

            if out_max == 0 and "lif" not in name.lower() and "neuron" not in name.lower():
                # Embedding入力(Long)は除く
                if "embedding" not in name.lower():
                    print(f"      🚨 信号消失警報: {name} の出力がゼロです！")
        return hook

    # 主要な層にフックを登録
    hooks = []

    # Embedding
    hooks.append(model.embedding.register_forward_hook(
        debug_hook("Embedding")))

    # 各レイヤーのコンポーネントを探索してフック
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            # BitLinear (time_mix_k)
            if hasattr(layer, 'time_mix_k'):
                hooks.append(layer.time_mix_k.register_forward_hook(
                    debug_hook(f"Layer{i}.TimeMix_K (BitLinear)")))

            # LIF (time_key_lif)
            if hasattr(layer, 'time_key_lif'):
                # 設定値の確認
                neuron = layer.time_key_lif
                thresh = neuron.base_threshold
                if isinstance(thresh, torch.Tensor):
                    thresh = thresh.mean().item()
                tau_val = 0.0
                if hasattr(neuron, 'log_tau_mem') and isinstance(neuron.log_tau_mem, torch.Tensor):
                    tau_val = (torch.exp(neuron.log_tau_mem) +
                               1.1).mean().item()
                elif hasattr(neuron, 'tau_mem'):
                    tau_val = float(neuron.tau_mem)

                print(
                    f"\n  [Layer {i} Config Check] Threshold: {thresh}, Tau: {tau_val:.2f}")
                hooks.append(layer.time_key_lif.register_forward_hook(
                    debug_hook(f"Layer{i}.LIF_K")))

    # 実行
    with torch.no_grad():
        try:
            print("\n  --- Forward Pass Start ---")
            model(input_ids, return_spikes=True)
            print("  --- Forward Pass End ---")
        except Exception as e:
            print(f"❌ 実行エラー: {e}")
            import traceback
            traceback.print_exc()

    # 後始末
    for h in hooks:
        h.remove()
    print("\n✅ 診断終了")


if __name__ == "__main__":
    main()
