# ファイルパス: scripts/runners/verify_and_train.py


from transformers import AutoTokenizer
from app.containers import TrainingContainer
from omegaconf import OmegaConf
import logging
import torch
import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# verify_and_train.py

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyTrain")


def main():
    # 1. 設定のロード (train.py と同じロジックを再現 + 強制上書き)
    config_path = "configs/models/bit_rwkv_micro.yaml"
    base_path = "configs/templates/base_config.yaml"

    base_conf = OmegaConf.load(base_path)
    model_conf = OmegaConf.load(config_path)

    # 設定のマージ
    conf = OmegaConf.merge(base_conf, model_conf)

    # --- 【重要】ここでコード内で直接値を強制上書きします ---
    # CLI引数に頼らず、ここで確定させます
    conf.model.neuron.base_threshold = 0.00001
    conf.model.neuron.tau_mem = 1000.0
    conf.model.neuron.noise_intensity = 0.2

    conf.training.paradigm = "gradient_based"
    conf.training.gradient_based.type = "standard"  # 蒸留ではなく標準学習
    conf.training.epochs = 5
    conf.data.path = "data/smoke_test_data.jsonl"

    print("\n" + "="*40)
    print("🧐 [VERIFICATION] 適用されるニューロン設定:")
    print(f"   Threshold: {conf.model.neuron.base_threshold}")
    print(f"   Tau Mem  : {conf.model.neuron.tau_mem}")
    print(f"   Noise    : {conf.model.neuron.noise_intensity}")
    print("="*40 + "\n")

    # 2. コンテナとモデルの準備
    container = TrainingContainer()
    # OmegaConf -> Dict 変換
    container.config.from_dict(OmegaConf.to_container(conf, resolve=True))

    device = "cpu"  # 安全のためCPU
    model = container.snn_model()
    model.to(device)

    # 3. 実モデルのパラメータ値を検査 (これが真実です)
    print("🧐 [VERIFICATION] モデル内部パラメータの実測値:")

    real_thresh = None
    real_tau = None

    # モデル内部を探索してLIFニューロンを探す
    for name, module in model.named_modules():
        if "lif" in name.lower() or "neuron" in name.lower():
            if hasattr(module, 'base_threshold'):
                # 値を取得
                th = module.base_threshold
                if isinstance(th, torch.Tensor):
                    th = th.mean().item()

                tau_param = getattr(module, 'log_tau_mem', None)
                if tau_param is not None:
                    tau_val = (torch.exp(tau_param) + 1.1).mean().item()
                elif hasattr(module, 'tau_mem'):
                    tau_val = module.tau_mem
                else:
                    tau_val = -1

                print(f"   - Layer: {name}")
                print(f"     -> Threshold: {th:.6f}")
                print(f"     -> Tau      : {tau_val:.2f}")

                if real_thresh is None:
                    real_thresh = th
                if real_tau is None:
                    real_tau = tau_val

                # 最初の1つだけチェックすれば十分
                break

    if real_thresh > 0.01:
        print("\n❌ 警告: 閾値が反映されていません！ 設定が上書きされています。")
    else:
        print("\n✅ 確認: 設定は正しく反映されています。")

    # 4. 学習の実行
    print("\n🚀 学習を開始します...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # --- ▼ 修正: type: ignore 追加 ▼ ---
    from train import collate_fn  # type: ignore[import-not-found]
    # --- ▲ 修正 ▲ ---
    from snn_research.data.datasets import SimpleTextDataset

    dataset = SimpleTextDataset(conf.data.path, tokenizer, max_seq_len=16)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, is_distillation=False)
    )

    optimizer = container.optimizer(params=model.parameters())
    scheduler = None

    trainer = container.standard_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1
    )

    for epoch in range(conf.training.epochs):
        trainer.train_epoch(loader, epoch)
        metrics = trainer.evaluate(loader, epoch)  # 検証も同じデータで簡易チェック

        spike_rate = metrics.get('spike_rate', 0.0)
        print(f"   -> Epoch {epoch} Spike Rate: {spike_rate:.6f}")

        if spike_rate > 0:
            print("🎉 成功！スパイクが発生しました！")
            return


if __name__ == "__main__":
    main()
