# **SNN Project 統合機能テストコマンド一覧 (Master Manual)**

本ドキュメントは、プロジェクトの全機能を網羅的にチェック・実行するための統合コマンドリストです。

test-command.txt の具体的な実験パラメータと、SNN開発：プロジェクト機能テスト コマンド一覧.md の体系的な分類を統合しています。

## **1\. 環境準備・メンテナンス (Setup & Maintenance)**

開発環境の健全性を保つための基本コマンド群です。

### **セットアップ**

\# 依存パッケージのインストール

pip install \-r requirements.txt

\# 静的型チェック (プロジェクト全体)

mypy .

### **プロジェクト健全性チェック (Health Check)**

全サブシステム（学習、エージェント、変換、グラフなど）の統合診断を行います。

エラーが出た場合は runs/ 以下のログを確認してください。

\# CLI経由（推奨）

python snn-cli.py health-check

\# またはスクリプト直接実行（ダミーANN作成なども含まれます）

python scripts/run\_project\_health\_check.py

### **データ・ログのクリーンアップ**

実験の残骸を消去して環境をリセットします。

\# ログとキャッシュのみ削除 (モデル・データは保護)

python snn-cli.py clean logs

\# 学習済みモデルも含めて削除 (データは保護)

python snn-cli.py clean models

\# 全て削除 (注意: 生成データも消えます)

\# python snn-cli.py clean all

## **2\. データ管理 (Data Management)**

### **データセットの準備**

\# CIFAR-10 データセットのダウンロード

python \-c "from torchvision.datasets import CIFAR10; CIFAR10(root='data/cifar10', download=True)"

\# テスト用ダミーデータの生成 (存在しない場合)

\# (run\_project\_health\_check.py を実行すると自動生成されますが、手動で行う場合)

\# python scripts/data\_preparation.py ... (必要に応じて)

### **知識蒸留用データの事前生成**

教師モデルのロジットを事前計算し、学習を高速化します。

\# 1\. クリーンな状態で再作成 (推奨)

rm \-rf precomputed\_data/smoke\_distill

\# 2\. 蒸留データの生成 (GPT-2教師)

python scripts/prepare\_distillation\_data.py \\

\--input\_file data/smoke\_test\_data.jsonl \\

\--output\_dir precomputed\_data/smoke\_distill \\

\--teacher\_model gpt2

## **3\. SNN学習ワークフロー (Training Workflows)**

### **A. クイックテスト学習 (Smoke Test)**

動作確認用の極小モデルでの学習。パラメータが上書き設定されています。

\# テストモデル作成 (5エポック)

python scripts/runners/train.py \\

\--config configs/templates/base\_config.yaml \\

\--model\_config configs/models/micro.yaml \\

\--data\_path data/smoke\_test\_data.jsonl \\

\--override\_config "training.epochs=5" \\

\--override\_config "training.gradient\_based.type=standard"

\# スパイク自動チューニング（準備学習）

python scripts/runners/train.py \\

\--config configs/templates/base\_config.yaml \\

\--model\_config configs/models/small.yaml \\

\--data\_path data/smoke\_test\_data.jsonl \\

\--override\_config "training.epochs=10" \\

\--override\_config "training.batch\_size=4" \\

\--override\_config "training.gradient\_based.type=standard"

### **B. 標準SNN学習 (Standard Gradient-based)**

本格的な学習設定。

python scripts/runners/train.py \\

\--config configs/templates/base\_config.yaml \\

\--model\_config configs/models/medium.yaml \\

\--data\_path data/smoke\_test\_data.jsonl \\

\--paradigm gradient\_based \\

\--override\_config "training.epochs=50"

### **C. 知識蒸留学習 (Knowledge Distillation)**

事前計算データを用いた効率的な学習。

\# 事前計算データを使用する場合

python scripts/runners/train.py \\

\--model\_config configs/models/bit\_rwkv\_micro.yaml \\

\--data\_path precomputed\_data/smoke\_distill/distillation\_data.jsonl \\

\--paradigm gradient\_based \\

\--override\_config "training.gradient\_based.type=distillation" \\

\--override\_config "training.gradient\_based.distillation.teacher\_model=gpt2" \\

\--override\_config "training.epochs=5" \\

\--override\_config "device=cpu" \\

\--override\_config "training.gradient\_based.loss.spike\_reg\_weight=5.0"

### **D. 1.58bit BitNet (BitRWKV) 学習**

超低消費電力モデルの学習。

python scripts/runners/train.py \\

\--config configs/experiments/smoke\_test\_config.yaml \\

\--model\_config configs/models/bit\_rwkv\_micro.yaml \\

\--data\_path data/smoke\_test\_data.jsonl \\

\--paradigm gradient\_based

### **E. 生物学的・因果学習 (Biological & Causal Learning)**

非勾配法や特殊な学習則のテスト。

\# 因果駆動型学習（Causal Trace V2）テスト

python scripts/runners/train.py \\

\--config configs/experiments/smoke\_test\_config.yaml \\

\--model\_config configs/models/small.yaml

\# 生物学的強化学習 (Bio-PCNet / Causal Traceによる重み更新)

python scripts/train\_bio\_pc\_cifar10.py

### **F. 分散学習 (Multi-GPU)**

bash scripts/run\_distributed\_training.sh

### **G. 自動チューニング (Auto-Tuning)**

python scripts/auto\_tune\_efficiency.py \\

\--model-config configs/models/small.yaml \\

\--n-trials 20

## **4\. 評価・ベンチマーク・診断 (Evaluation & Diagnostics)**

### **ベンチマーク (ANN vs SNN)**

CIFAR-10等での性能比較実験。

\# CIFAR-10 比較実験 (5エポック)

python snn-cli.py benchmark run \--experiment cifar10\_comparison \--epochs 5

\# Phase 3 SFormer検証 (T=1)

python snn-cli.py benchmark run \\

\--experiment cifar10\_comparison \\

\--model-config configs/models/phase3\_sformer.yaml \\

\--epochs 1 \\

\--tag "SFormer\_T1\_Test"

### **精度評価 (Accuracy Evaluation)**

学習済みモデルの評価専用モード。

python snn-cli.py benchmark evaluate-accuracy \\

\--model-path runs/converted/spiking\_cnn\_from\_ann.pth \\

\--model-config configs/experiments/cifar10\_spikingcnn\_config.yaml \\

\--model-type SNN \\

\--experiment cifar10

### **効率性診断 (Efficiency Report)**

スパース性（Sparsity）と推論ステップ数（T）の診断レポートを発行。

\# CLI経由

python snn-cli.py diagnostics report-efficiency \\

\--model-path runs/snn\_experiment/best\_model.pth \\

\--model-config configs/models/medium.yaml \\

\--data-path data/smoke\_test\_data.jsonl

\# スクリプト直接実行

python scripts/report\_sparsity\_and\_T.py \\

\--model\_config configs/models/micro.yaml \\

\--data\_path data/smoke\_test\_data.jsonl \\

\--model\_path runs/snn\_experiment/best\_model.pth

## **5\. 認知アーキテクチャ & エージェント (Cognitive Systems)**

### **人工脳シミュレーション**

\# 単発実行

python scripts/runners/run\_brain\_simulation.py \\

\--model\_config configs/models/small.yaml \\

\--prompt "1+1の計算をして"

\# 思考プロセス観察 (対話モード)

python scripts/observe\_brain\_thought\_process.py \\

\--model\_config configs/models/micro.yaml

### **自律エージェント (Task Solver)**

タスクを与えて自律的に解決させる。

python scripts/runners/run\_agent.py \\

\--task\_description "最新のAIトレンドについて教えて" \\

\--force\_retrain

### **Web自律学習**

Webから情報を収集してモデルを学習。

python scripts/runners/run\_web\_learning.py \\

\--topic "量子コンピュータ" \\

\--start\_url "

$$https://ja.wikipedia.org/wiki/量子コンピュータ$$  
(https://ja.wikipedia.org/wiki/量子コンピュータ)" \\

\--max\_pages 3

### **デジタル生命体 (Digital Life Form)**

内発的動機に基づく自律行動シミュレーション。

python scripts/runners/run\_life\_form.py \--duration 60

### **ユニットテスト (Cognitive Core)**

認知アーキテクチャの中核機能テスト。

python tests/cognitive\_architecture/test\_artificial\_brain.py

## **6\. モデル管理 & FrankenMoE (Model Management)**

### **エキスパート登録**

\# デモ用エキスパート

python scripts/register\_demo\_experts.py

\# 1.58bit計算エキスパート

python scripts/register\_bitnet\_expert.py

### **FrankenMoE (モデル統合) の構築**

python scripts/manage\_models.py build-moe \\

\--keywords "science,history,calculation" \\

\--output my\_moe.yaml

※ 出力された configs/models/my\_moe.yaml は run\_brain\_simulation.py 等で使用可能。

## **7\. デモ・実験アプリケーション (Application Demos)**

各特化機能のデモスクリプト。

* **マルチモーダル処理 (Vision-Language)** python scripts/run\_multimodal\_demo.py  
* **ECG異常検知 (時系列SNN)** python scripts/run\_ecg\_analysis.py \\  
  \--model\_config configs/models/ecg\_temporal\_snn.yaml \\  
  \--time\_steps 500  
* **空間認識 (Spatial Demo)** python scripts/run\_spatial\_demo.py  
* **STDP学習 (Unsupervised)** python scripts/run\_stdp\_learning.py \--dataset mnist \--epochs 5  
* **Phase 3 統合検証 (SFormer/SEMM)** python scripts/verify\_phase3.py

## **8\. デバッグ & 可視化 (Debugging & Visualization)**

SNN内部挙動の詳細解析。

### **スパイク活動統計**

各層の発火率（Firing Rate）診断。

python scripts/debug\_spike\_activity.py \\

\--model\_config configs/models/micro.yaml \\

\--timesteps 16

### **ラスタプロット (Spike Patterns)**

スパイクタイミングの可視化。runs/spike\_viz/ に保存されます。

python snn-cli.py debug spike-visualize \\

\--model-config configs/models/micro.yaml \\

\--timesteps 8 \\

\--batch-size 2 \\

\--output-prefix "runs/spike\_viz/micro\_test"

\# スクリプト直接実行の場合

\# python scripts/visualize\_spike\_patterns.py ...

### **ニューロンダイナミクス (Membrane Potential)**

膜電位波形の可視化。runs/dynamics\_viz/ に保存されます。

\# CLI経由のコマンド例 (エイリアス設定されている場合)

\# python snn-cli.py debug dynamics ...

\# スクリプト直接実行 (確実)

python scripts/visualize\_neuron\_dynamics.py \\

\--model\_config configs/models/micro.yaml \\

\--output\_path runs/dynamics\_viz/micro\_dynamics.png \\

\--timesteps 8

## **9\. 知識グラフ操作 (GraphRAG)**

外部記憶（Vector Store \+ Knowledge Graph）の操作。

\# 知識の追加

python snn-cli.py knowledge add "SNN" "省エネな次世代AI" \--relation "is\_defined\_as"

\# 因果関係の記録

python snn-cli.py knowledge update-causal \--cause "学習" \--effect "賢くなる"

\# 検索

python snn-cli.py knowledge search "SNN"

## **10\. モデル変換 (ANN to SNN Conversion)**

学習済みANNをSNNに変換します。

\# CNN変換 (例: ダミーモデルを使用)

python scripts/convert\_model.py \\

\--method cnn-convert \\

\--ann\_model\_path runs/dummy\_ann.pth \\

\--snn\_model\_config configs/models/micro.yaml \\

\--output\_snn\_path runs/converted\_snn.pth

## **11\. ユーザーインターフェース (GUI)**

ブラウザベースのダッシュボードを起動します。

\# 標準UI (モデル動的ロード対応)

python snn-cli.py ui start

\# LangChain連携UI

python snn-cli.py ui start \--start-langchain
