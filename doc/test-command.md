# **Matsushiba Denki SNN - 統合テストコマンド完全版 (v20 Unified)**

このドキュメントは、SNNプロジェクトの全バージョン（Brain v20, v16, v14）および全機能（Phase 1-16）を網羅したテストコマンド集です。

---

## **📋 目次**

1. [環境準備・メンテナンス](#1-環境準備メンテナンス)
2. [Brain v20 (Current Stable)](#2-brain-v20-current-stable)
3. [新機能検証 (Bio-Foundation, Perception & Reasoning)](#3-新機能検証)
4. [SNN学習ワークフロー](#4-snn学習ワークフロー)
5. [脳型OS & 認知アーキテクチャ](#5-脳型os--認知アーキテクチャ)
6. [変換 & 最適化](#6-変換--最適化)
7. [エージェント & 自律行動](#7-エージェント--自律行動)
8. [性能証明・検証](#8-性能証明検証)
9. [Legacy Commands (Archive)](#9-legacy-commands-archive)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## **1. 環境準備・メンテナンス**

### **セットアップ**

```bash
pip install -r requirements.txt
# 静的型チェック
mypy .
```

### **プロジェクト健全性チェック (Health Check)**

全サブシステムの統合診断（学習、推論、エージェント、生物学的モデルを含む）。

```bash
# CLI経由（推奨）
python snn-cli.py health-check

# またはスクリプト直接実行
python scripts/run_project_health_check.py
```

### **クリーンアップ**

```bash
python snn-cli.py clean logs    # ログ削除
python snn-cli.py clean models  # モデル削除
```

---

## **2. Brain v20 (Current Stable)**

Brain v20は、非同期イベント駆動型カーネル (AsyncArtificialBrain) と、1.58bit量子化SNNモデル (BitSpikeMamba) を統合した最新アーキテクチャです。

### **A. Training (学習)**

モデルに言語パターンを学習させ、重みファイル (`models/checkpoints/trained_brain_v20.pth`) を生成します。

#### **デモ用・高速学習 (推奨)**

特定のフレーズを短時間で過学習(Overfit)させ、応答能力を確認するためのスクリプトです。数分でLossが0.1以下になり、対話が可能になります。

```bash
python scripts/trainers/train_overfit_demo.py
```

#### **一般学習 (General Training)**

テキストコーパスを用いて汎用的な学習を行います。データセットがない場合はダミーデータを使用します。

```bash
python scripts/trainers/train_bit_spike_mamba.py
```

### **B. Running the Prototype (自律動作デモ)**

学習した脳を非同期カーネル上で起動させます。視覚入力（シミュレーション）、思考、運動出力が並列に動作し、アストロサイトによるエネルギー管理が行われる様子を観察できます。

```bash
python scripts/runners/run_brain_v20_prototype.py
```

**期待される動作:**

- ログに `🧠 Async Brain Kernel Started` が表示される
- Energy replenished (エネルギー補給) が行われる
- 外部入力に対して `💡 Conscious Awareness` (意識) が発生する
- 裏で思考プロセスが走り、`🗣️ Brain Says: ...` と発話する

### **C. Interactive Dialogue (対話モード)**

学習済み脳とコマンドライン(CLI)で直接おしゃべりをするためのツールです。非同期処理の待ち時間を気にせず、言語生成能力をテストできます。

```bash
python scripts/runners/talk_to_brain.py
```

**終了方法:** `exit`, `quit`, `bye` と入力するか、`Ctrl+C` を押します。

### **D. Verification & Testing (品質保証)**

システムの堅牢性とモデルの動作を検証するためのテストスイートです。

#### **全テスト一括実行 (Master Runner) - 推奨**

モデル、カーネル、統合テストの全てを実行し、健全性を確認します。

```bash
python scripts/run_all_tests.py
```

#### **個別テスト実行**

特定のコンポーネントのみをデバッグする場合に使用します。

```bash
# BitSpikeMambaモデル (1.58bit量子化 & SNN動作)
python -m unittest tests/test_bit_spike_mamba.py

# 非同期カーネル (Async Event Bus)
python -m unittest tests/test_async_brain_kernel.py

# 統合テスト (Integration Cycle)
python -m unittest tests/test_brain_integration.py
```

---

## **3. 新機能検証**

**v16.1で強化された重要機能の単体・統合テスト。**

### **A. 生物学的マイクロサーキット (PD14 & Active Dendrites)**

Potjans-Diesmannモデルと能動的樹状突起による、生物学的妥当性の高い皮質演算デモ。

```bash
python scripts/run_bio_microcircuit_demo.py
```

**期待される結果:**

- **Scenario A:** ボトムアップ入力により L4 → L2/3 → L5 へと信号が伝播する
- **Scenario B:** トップダウン入力（予測）が樹状突起を活性化させ、弱い入力でも L5 が発火する（NMDAスパイク効果）

### **B. SNN-DSA (Dynamic Sparse Attention)**

動的スパース注意機構を持つTransformerの学習能力を検証。

```bash
python scripts/verify_dsa_learning.py
```

**期待される結果:** Accuracy > 80% で "PASSED" が表示される。

### **C. GRPO (Group Relative Policy Optimization)**

論理推論能力（思考の軌跡の自己改善）を検証。

```bash
python tests/test_grpo_logic.py
```

**期待される結果:** 重み更新が確認され、テストがPASSする。

### **D. DVS & Universal Encoder**

ニューロモルフィックデータセットと統一エンコーダの動作検証。

```bash
# DVSパイプライン (N-MNIST Mock)
python tests/test_dvs_pipeline.py

# Universal Spike Encoder (Image/Audio/Text/DVS)
python tests/test_universal_encoder.py
```

### **E. Liquid Association Cortex (LAC) & 五感統合**

リザーバ層によるモダリティ統合と、共感覚的想起デモ。

```bash
# LAC統合テスト (基本動作)
python tests/test_liquid_association.py

# Cross-Modal Demo ("Hearing Colors") - 音から色を想起
python scripts/run_cross_modal_demo.py
```

**期待される結果:** Association Improvement がプラスになり、音声のみから視覚概念が想起される。

### **F. Interactive Web Demo**

ブラウザ上で「Hearing Colors」やチャットを体験する。

```bash
python app/main.py --model-config configs/models/small.yaml
# または
python snn-cli.py ui start
```

**操作:** ブラウザで `http://127.0.0.1:7860` にアクセス。

---

## **4. SNN学習ワークフロー**

### **A. 標準・高速学習**

```bash
# クイックテスト (5エポック)
python scripts/runners/train.py \
    --config configs/templates/base_config.yaml \
    --model_config configs/models/micro.yaml \
    --data_path data/smoke_test_data.jsonl \
    --override_config "training.epochs=5"

# 1.58bit BitNet学習
python scripts/runners/train.py \
    --model_config configs/models/bit_rwkv_micro.yaml \
    --data_path data/smoke_test_data.jsonl \
    --paradigm gradient_based
```

### **B. 生物学的・因果学習**

```bash
# Bio-RL (強化学習)
python scripts/runners/run_rl_agent.py --episodes 100

# 因果駆動型学習 (Causal Trace V2)
python scripts/runners/train.py \
    --config configs/experiments/smoke_test_config.yaml \
    --model_config configs/models/small.yaml
```

---

## **5. 脳型OS & 認知アーキテクチャ**

### **A. 脳型OSシミュレーション**

複数の認知モジュールがリソース（エネルギー）を巡って競合する様子のデモ。

```bash
python scripts/runners/run_neuromorphic_os.py
```

### **B. イベント駆動 & オンチップ学習**

ハードウェアネイティブな学習のデモ。

```bash
# On-Chip Plasticity (STDPによる自己組織化)
python scripts/run_on_chip_learning.py

# イベント駆動型シミュレーション (推論のみ)
python scripts/run_hardware_simulation.py --model_config configs/models/micro.yaml
```

### **C. 人工脳統合シミュレーション (Full Cycle)**

対話、睡眠（記憶固定化）、進化のフルサイクル。

```bash
python scripts/runners/run_brain_v14.py
```

---

## **6. 変換 & 最適化**

### **A. Deep Bio-Calibration**

HSEOを用いてSNNパラメータ（閾値など）を自動チューニングする。

```bash
python scripts/run_deep_bio_calibration.py \
    --model_config configs/models/micro.yaml \
    --iterations 5 --particles 5
```

### **B. ANN-SNN 変換**

```bash
python scripts/convert_model.py \
    --method cnn-convert \
    --ann_model_path runs/dummy_ann.pth \
    --snn_model_config configs/models/micro.yaml \
    --output_snn_path runs/converted_snn.pth
```

---

## **7. エージェント & 自律行動**

### **自律タスク解決**

```bash
python scripts/runners/run_agent.py \
    --task_description "最新のAIトレンドについて教えて" \
    --force_retrain
```

### **デジタル生命体**

```bash
python scripts/runners/run_life_form.py --duration 60
```

---

## **8. 性能証明・検証**

### **性能検証レポートの発行**

目標値（`doc/Objective.md`）に対する達成度を自動判定し、証明書（Markdownレポート）を発行する。

```bash
# Mediumモデルの性能検証（シミュレーション）
python scripts/verify_performance.py --model_config configs/models/medium.yaml

# ターゲット設定ファイルを指定して実行（より厳しい基準など）
python scripts/verify_performance.py \
    --model_config configs/models/medium.yaml \
    --target_config configs/validation/targets_v1.yaml
```

---

## **9. Legacy Commands (Archive)**

以下のコマンドは旧バージョン (v14, v16) のものです。後方互換性テストのために残されています。

### **Brain v16 (Previous Gen)**

```bash
# Full Demo
python scripts/runners/run_brain_v16_demo.py

# Sleep Cycle Test
python scripts/runners/run_sleep_cycle_demo.py
```

### **Brain v14 (Stable SNN)**

```bash
# Standard Run
python scripts/runners/run_brain_v14.py
```

### **Functional Tests**

```bash
# Visual Cortex (Industrial Eye)
python scripts/runners/run_industrial_eye_demo.py

# Project Health Check
python scripts/run_project_health_check.py
```

---

## **10. トラブルシューティング**

### **ログが表示されない場合**

スクリプト内の `logging.basicConfig(..., force=True)` が設定されているか確認してください。他のライブラリがログ設定を上書きしている可能性があります。

### **モデルロードエラー**

- `models/checkpoints/trained_brain_v20.pth` が存在するか確認
- モデルのハイパーパラメータ（`d_model`, `num_layers` 等）が学習時と推論時で一致しているか確認

### **依存関係エラー**

```bash
pip install -r requirements.txt --upgrade
```

### **メモリ不足**

より小さいモデル設定（`micro.yaml`）を使用するか、バッチサイズを減らしてください。

---

## **📝 補足情報**

- **現在の安定版:** Brain v20 (Async Event-Driven Architecture + BitSpikeMamba)
- **推奨テストフロー:**
  1. 環境準備 → Health Check
  2. Brain v20 の高速学習 → 対話モード
  3. 新機能検証 (Bio-Foundation など)
  4. 性能検証レポート発行

- **開発時のベストプラクティス:**
  - 新機能追加時は必ず対応するテストを追加
  - `run_all_tests.py` で全体回帰テストを実施
  - Health Check を定期的に実行して品質を維持

---

**最終更新:** Brain v20 統合版 (2025-12-16)