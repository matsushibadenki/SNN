#!/bin/bash
# matsushibadenki/snn/scripts/run_distributed_training.sh
# 大規模分散学習実行スクリプト (torchrun版)
#
# 機能:
# - 利用可能なGPU数を自動で検出し、分散学習のプロセス数を決定する。
# - torchrunユーティリティを使い、堅牢な分散学習ジョブを開始する。
# - 学習設定とモデル設定をこのスクリプト内で一元管理できる。

# エラーが発生した場合はスクリプトを終了する
set -e

# 使用するGPUの数を自動で取得
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "分散学習には2つ以上のGPUが必要です。利用可能なGPU: $NUM_GPUS"
    echo "単一デバイスで学習する場合は、直接 train.py を実行してください。"
    exit 1
fi

echo "✅ ${NUM_GPUS}個のGPUを検出しました。分散学習を開始します..."

# --- 設定 ---
# ここで学習設定とモデル設定を切り替える
BASE_CONFIG="configs/templates/base_config.yaml"
MODEL_CONFIG="configs/models/medium.yaml"

# 知識蒸留を行う場合は、training.typeをdistillationに設定し、
# データパスを事前計算済みロジットのディレクトリに設定する
DATA_PATH="precomputed_data/" 

# --- torchrunによる実行 ---
# torchrun は、各プロセスに適切な環境変数(RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)を自動で設定します。
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config $BASE_CONFIG \
    --model_config $MODEL_CONFIG \
    --data_path $DATA_PATH \
    --distributed

echo "✅ 分散学習が完了しました。"
