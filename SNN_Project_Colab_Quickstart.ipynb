{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SNN4プロジェクト Colabクイックスタート\n",
    "\n",
    "このノートブックは、SNN4プロジェクトの主要な機能をGoogle Colab上で簡単に実行するための最新ガイドです。\n",
    "プロジェクトの中心的なインターフェースである統合CLIツール `snn-cli.py` の使用方法を学びます。\n",
    "\n",
    "**主な内容:**\n",
    "1.  **環境設定**: プロジェクトのセットアップ\n",
    "2.  **モデルの学習**: `snn-cli gradient-train` を使ったSNNモデルの基本学習\n",
    "3.  **自律エージェントの実行**: `snn-cli agent solve` を用いたオンデマンドでの専門家モデル学習\n",
    "4.  **ANN vs SNN ベンチマーク**: `snn-cli benchmark run` による性能比較実験"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 環境設定\n",
    "\n",
    "まず、プロジェクトのセットアップスクリプトを実行して、必要なライブラリをインストールします。\n",
    "（注：実際の環境では、まず `git clone` でリポジトリをクローンする必要があります）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!bash setup_colab.sh"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. モデルの学習 (`snn-cli gradient-train`)\n",
    "\n",
    "`snn-cli.py` を使って、基本的なSNNモデルの学習を実行します。ここでは、動作確認用のマイクロモデル (`micro.yaml`) と小規模なテストデータ (`smoke_test_data.jsonl`) を使用します。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python snn-cli.py gradient-train \\\n",
    "    --model_config configs/models/micro.yaml \\\n",
    "    --data_path data/smoke_test_data.jsonl \\\n",
    "    --override_config \"training.epochs=3\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 自律エージェント (`snn-cli agent solve`) の実行\n",
    "\n",
    "自律エージェントにタスクを依頼します。エージェントは自己の能力（モデル登録簿）を調べ、最適な専門家モデルが存在しないと判断した場合、提供されたデータを使って新しい専門家をオンデマンドで学習します。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python snn-cli.py agent solve \\\n",
    "    --task \"文章要約\" \\\n",
    "    --unlabeled_data data/sample_data.jsonl \\\n",
    "    --force_retrain"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. ANN vs SNN ベンチマーク (`snn-cli benchmark run`)\n",
    "\n",
    "開発計画 `snn_4_ann_parity_plan.md` の中核的な目標である、ANNとSNNの性能比較を手軽に実行します。\n",
    "ここでは、CIFAR-10データセットを用いて、画像分類タスクにおけるANN (SimpleCNN) とSNN (SpikingCNN) の精度、速度、エネルギー効率を比較します。\n",
    "\n",
    "（注：この実験はデータセットのダウンロードと複数エポックの学習を含むため、完了までに時間がかかります。）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python snn-cli.py benchmark run \\\n",
    "    --experiment cifar10_comparison \\\n",
    "    --epochs 1 \\\n",
    "    --tag \"colab_quickstart_run\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}