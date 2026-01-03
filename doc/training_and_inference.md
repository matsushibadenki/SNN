# SNNプロジェクト: 学習・推論コマンドガイド

このドキュメントでは、モデルの学習（Training）と推論・デモ（Inference/Demo）を実行するための主要なコマンドについて解説します。
プロジェクトディレクトリのルートで実行してください。

## 1. 学習 (Training)

モデルを一から学習、あるいは継続学習させるためのコマンドです。

### 汎用トレーナー
CLIまたはスクリプト経由で、設定ファイル (`configs/`) を指定して学習を開始します。

```bash
# SNN CLIを使用する場合 (推奨)
snn-cli gradient-train --model_config configs/models/stable_small_snn.yaml --data_path data/smoke_test_data.jsonl

# 直接スクリプトを実行する場合
python scripts/training/train.py --config configs/experiments/brain_v14_config.yaml
```

### タスク特化型学習スクリプト
特定のタスクやデータセットに特化した学習スクリプトです。

*   **MNIST SNN学習**:
    ```bash
    python scripts/training/train_mnist_snn.py
    ```
*   **CIFAR-10 Bio-PC (Predictive Coding) 学習**:
    ```bash
    python scripts/training/train_bio_pc_cifar10.py
    ```
*   **Spiking VLM (Vision-Language Model) 学習**:
    ```bash
    python scripts/training/train_spiking_vlm.py
    ```
*   **Planner (推論エンジン) 学習**:
    ```bash
    python scripts/training/train_planner.py
    ```

---

## 2. 推論・デモ (Inference & Demos)

学習済みモデルや、初期化されたエージェントを使用して推論や動作デモを行います。
スクリプトは機能カテゴリごとに `scripts/demos/` 以下に整理されています。

### 🧠 Brain & Agent (脳モデル・エージェント)
*   **Brain v16 統合デモ**:
    ```bash
    python scripts/demos/brain/run_brain_v16_demo.py
    ```
*   **世界モデル (World Model)**:
    ```bash
    python scripts/demos/brain/run_world_model_demo.py
    ```
*   **好奇心エージェント**:
    ```bash
    python scripts/demos/brain/run_curiosity_demo.py
    ```

### 📚 Learning & Distillation (学習・蒸留)
*   **睡眠サイクル (記憶固定化)**:
    ```bash
    python scripts/demos/learning/run_sleep_cycle_demo.py
    ```
*   **継続学習デモ**:
    ```bash
    python scripts/demos/learning/run_continual_learning_demo.py
    ```

### ⚙️ Systems (システム制御・制御理論)
*   **能動的推論 (Active Inference)**:
    ```bash
    python scripts/demos/systems/run_active_inference_demo.py
    ```
*   **ニューロシンボリック推論**:
    ```bash
    python scripts/demos/systems/run_neuro_symbolic_demo.py
    ```

### 👁️ Visual & Sensors (視覚・センサー)
*   **産業用Eye (DVS処理)**:
    ```bash
    python scripts/demos/visual/run_industrial_eye_demo.py
    ```
*   **Forward-Forward アルゴリズム**:
    ```bash
    python scripts/demos/visual/run_forward_forward_demo.py
    ```

---

## 3. 実験 (Experiments)

研究開発フェーズの実験スクリプトは `scripts/experiments/` にあります。

*   **継続学習実験**: `python scripts/experiments/learning/run_continual_learning_experiment.py`
*   **蒸留実験**: `python scripts/experiments/learning/run_distillation_experiment.py`
*   **進化実験**: `python scripts/experiments/brain/run_brain_evolution.py`

詳細なテストコマンドについては `doc/test-command.md` を参照してください。
