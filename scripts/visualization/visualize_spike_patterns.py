# scripts/visualize_spike_patterns.py
# スパイク活動のパターンを可視化し、ニューロンの活動効率を解析するためのスクリプト
#
# ディレクトリ: scripts/visualize_spike_patterns.py
# ファイル名: スパイク活動可視化ツール
# 目的: 各レイヤーのスパイク発火率を計測し、ヒートマップやラスタプロットを生成する。
#
# 変更点:
# - [修正 v5] 全くスパイクがないレイヤーがある場合でも、他の活動的なレイヤーを優先してログ出力するよう改善。
# - [修正 v5] 可視化対象のサンプル選択時、全レイヤーの合計スパイク数が最大となるインデックスを検索するロジックに変更。
# - [修正 v5] 引数の batch_size がダミー入力生成に反映されるよう修正。

import argparse
import sys
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="SNNスパイク活動可視化ツール")
    parser.add_argument("--model_config", type=str, required=True, help="モデル設定ファイル")
    parser.add_argument("--timesteps", type=int, default=8, help="シミュレーション時間")
    parser.add_argument("--batch_size", type=int, default=2, help="解析用バッチサイズ")
    parser.add_argument("--output_prefix", type=str, default="workspace/runs/spike_viz/spike", help="出力ファイル名の接頭辞")
    args = parser.parse_args()

    # 出力ディレクトリの作成
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # モデルの準備
    try:
        container = TrainingContainer()
        container.config.from_yaml("configs/templates/base_config.yaml")
        container.config.from_yaml(args.model_config)
        model = container.snn_model()
        model.eval()
    except Exception as e:
        logger.error(f"モデルの初期化に失敗しました: {e}")
        sys.exit(1)

    # フックを使用してスパイクを記録
    spike_records = {}
    def get_hook(name):
        def hook(module, input, output):
            # outputはスパイクテンソル (T, B, ...)
            spike_records[name] = output.detach().cpu()
        return hook

    registered_count = 0
    for name, module in model.named_modules():
        # 各種ニューロン層を検知 (LIF, AdaptiveLIF, PredictiveCoding等のメンバ)
        if hasattr(module, 'spike_monitored') or "neuron" in name.lower():
            module.register_forward_hook(get_hook(name))
            registered_count += 1
    
    logger.info(f"{registered_count} 個のニューロン層にフックを登録しました。")

    # 入力生成 (修正: args.batch_size を使用)
    vocab_size = container.config.model.get("vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.timesteps))
    
    logger.info(f"ダミー入力生成: {input_ids.shape} (Batch Size: {args.batch_size})")

    # 推論実行
    with torch.no_grad():
        model(input_ids)

    if not spike_records:
        logger.error("スパイクデータが記録されませんでした。ニューロン層が正しく検知されていない可能性があります。")
        return

    # 最も活動的なサンプルを選択するロジック (全レイヤー合計)
    # spike_records[layer] shape: (T, B, N)
    total_spikes_per_sample = torch.zeros(args.batch_size)
    for spikes in spike_records.values():
        # (T, B, N) -> (B,)
        total_spikes_per_sample += spikes.sum(dim=(0, 2))

    best_sample_idx = torch.argmax(total_spikes_per_sample).item()
    max_spikes = total_spikes_per_sample[best_sample_idx].item()

    logger.info(f"--- スパイク活動解析 (Best Sample Index: {best_sample_idx}, Total Spikes: {max_spikes}) ---")

    for name, spikes in spike_records.items():
        # spikes: (T, B, N)
        sample_spikes = spikes[:, best_sample_idx, :] # (T, N)
        spike_count = sample_spikes.sum().item()
        spike_rate = (spike_count / sample_spikes.numel()) * 100
        
        if spike_count == 0:
            logger.warning(f"  Layer: {name} | Spikes: 0 (活動なし)")
            continue

        logger.info(f"  Layer: {name} | Spikes: {spike_count:.0f} | Rate: {spike_rate:.2f}%")

        # 可視化処理 (簡易ラスタプロット)
        plt.figure(figsize=(10, 4))
        # (T, N) を (N, T) にして描画
        plt.imshow(sample_spikes.t(), cmap='binary', aspect='auto', interpolation='nearest')
        plt.title(f"Spike Pattern: {name}\nRate: {spike_rate:.2f}%")
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Index")
        
        save_path = f"{args.output_prefix}_{name.replace('.', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    logger.info(f"✅ 可視化完了。出力先接頭辞: {args.output_prefix}")

if __name__ == "__main__":
    main()