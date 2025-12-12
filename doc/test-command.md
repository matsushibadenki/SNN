# **SNN Project 統合機能テストコマンド一覧 (Master Manual v16.1)**

本ドキュメントは、SNNプロジェクトの全機能を網羅的にチェック・実行するための統合コマンドリストです。  
Phase 1 ("Biological Foundation") から Phase 16 ("Stabilization") までの主要機能をカバーしています。

## **1\. 環境準備・メンテナンス (Setup & Maintenance)**

### **セットアップ**

pip install \-r requirements.txt  
\# 静的型チェック  
mypy .

### **プロジェクト健全性チェック (Health Check)**

全サブシステムの統合診断（学習、推論、エージェント、生物学的モデルを含む）。

\# CLI経由（推奨）  
python snn-cli.py health-check

\# またはスクリプト直接実行  
python scripts/run\_project\_health\_check.py

### **クリーンアップ**

python snn-cli.py clean logs    \# ログ削除  
python snn-cli.py clean models  \# モデル削除

## **2\. 新機能検証 (Bio-Foundation, Perception & Reasoning)**

**v16.1で強化された重要機能の単体・統合テスト。**

### **A. 生物学的マイクロサーキット (PD14 & Active Dendrites) \[New\]**

Potjans-Diesmannモデルと能動的樹状突起による、生物学的妥当性の高い皮質演算デモ。

python scripts/run\_bio\_microcircuit\_demo.py

* **期待される結果:**  
  * **Scenario A:** ボトムアップ入力により L4 → L2/3 → L5 へと信号が伝播する。  
  * **Scenario B:** トップダウン入力（予測）が樹状突起を活性化させ、弱い入力でも L5 が発火する（NMDAスパイク効果）。

### **B. SNN-DSA (Dynamic Sparse Attention)**

動的スパース注意機構を持つTransformerの学習能力を検証。

python scripts/verify\_dsa\_learning.py

* **期待される結果:** Accuracy \> 80% で "PASSED" が表示される。

### **C. GRPO (Group Relative Policy Optimization)**

論理推論能力（思考の軌跡の自己改善）を検証。

python tests/test\_grpo\_logic.py

* **期待される結果:** 重み更新が確認され、テストがPASSする。

### **D. DVS & Universal Encoder**

ニューロモルフィックデータセットと統一エンコーダの動作検証。

\# DVSパイプライン (N-MNIST Mock)  
python tests/test\_dvs\_pipeline.py

\# Universal Spike Encoder (Image/Audio/Text/DVS)  
python tests/test\_universal\_encoder.py

### **E. Liquid Association Cortex (LAC) & 五感統合**

リザーバ層によるモダリティ統合と、共感覚的想起デモ。

\# LAC統合テスト (基本動作)  
python tests/test\_liquid\_association.py

\# Cross-Modal Demo ("Hearing Colors") \- 音から色を想起  
python scripts/run\_cross\_modal\_demo.py

* **期待される結果:** Association Improvement がプラスになり、音声のみから視覚概念が想起される。

### **F. Interactive Web Demo**

ブラウザ上で「Hearing Colors」やチャットを体験する。

python app/main.py \--model-config configs/models/small.yaml  
\# または  
python snn-cli.py ui start

* **操作:** ブラウザで http://127.0.0.1:7860 にアクセス。

## **3\. SNN学習ワークフロー (Training \- Legacy & Standard)**

### **A. 標準・高速学習**

\# クイックテスト (5エポック)  
python scripts/runners/train.py \\  
    \--config configs/templates/base\_config.yaml \\  
    \--model\_config configs/models/micro.yaml \\  
    \--data\_path data/smoke\_test\_data.jsonl \\  
    \--override\_config "training.epochs=5"

\# 1.58bit BitNet学習  
python scripts/runners/train.py \\  
    \--model\_config configs/models/bit\_rwkv\_micro.yaml \\  
    \--data\_path data/smoke\_test\_data.jsonl \\  
    \--paradigm gradient\_based

### **B. 生物学的・因果学習**

\# Bio-RL (強化学習)  
python scripts/runners/run\_rl\_agent.py \--episodes 100

\# 因果駆動型学習 (Causal Trace V2)  
python scripts/runners/train.py \\  
    \--config configs/experiments/smoke\_test\_config.yaml \\  
    \--model\_config configs/models/small.yaml

## **4\. 脳型OS & 認知アーキテクチャ (Phase 7 & 14\)**

### **A. 脳型OSシミュレーション**

複数の認知モジュールがリソース（エネルギー）を巡って競合する様子のデモ。

python scripts/runners/run\_neuromorphic\_os.py

### **B. イベント駆動 & オンチップ学習**

ハードウェアネイティブな学習のデモ。

\# On-Chip Plasticity (STDPによる自己組織化)  
python scripts/run\_on\_chip\_learning.py

\# イベント駆動型シミュレーション (推論のみ)  
python scripts/run\_hardware\_simulation.py \--model\_config configs/models/micro.yaml

### **C. 人工脳統合シミュレーション (Full Cycle)**

対話、睡眠（記憶固定化）、進化のフルサイクル。

python scripts/runners/run\_brain\_v14.py

## **5\. 変換 & 最適化 (Deep Bio-Calibration)**

### **A. Deep Bio-Calibration**

HSEOを用いてSNNパラメータ（閾値など）を自動チューニングする。

python scripts/run\_deep\_bio\_calibration.py \\  
    \--model\_config configs/models/micro.yaml \\  
    \--iterations 5 \--particles 5

### **B. ANN-SNN 変換**

python scripts/convert\_model.py \\  
    \--method cnn-convert \\  
    \--ann\_model\_path runs/dummy\_ann.pth \\  
    \--snn\_model\_config configs/models/micro.yaml \\  
    \--output\_snn\_path runs/converted\_snn.pth

## **6\. エージェント & 自律行動**

### **自律タスク解決**

python scripts/runners/run\_agent.py \\  
    \--task\_description "最新のAIトレンドについて教えて" \\  
    \--force\_retrain

### **デジタル生命体**

python scripts/runners/run\_life\_form.py \--duration 60

## **7\. 性能証明・検証 (Validation & Proof)**

### **性能検証レポートの発行**

目標値（doc/Objective.md）に対する達成度を自動判定し、証明書（Markdownレポート）を発行する。

\# Mediumモデルの性能検証（シミュレーション）  
python scripts/verify\_performance.py \--model\_config configs/models/medium.yaml

\# ターゲット設定ファイルを指定して実行（より厳しい基準など）  
python scripts/verify\_performance.py \\  
    \--model\_config configs/models/medium.yaml \\  
    \--target\_config configs/validation/targets\_v1.yaml  
