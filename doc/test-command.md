# **Matsushiba Denki SNN \- 統合テストコマンド完全版 (v20 Unified)**

このドキュメントは、SNNプロジェクトの全バージョン（Brain v20, v16, v14）および全機能（Phase 1-16+）を網羅したテストコマンド集です。  
開発中の最新機能、ベンチマーク、ハードウェアシミュレーションを含むすべての実行手順を記載しています。

## **📋 目次**

1. [CLI & 環境準備](https://www.google.com/search?q=%231-cli--%E7%92%B0%E5%A2%83%E6%BA%96%E5%82%99)  
2. [Brain v20 (Current Stable)](https://www.google.com/search?q=%232-brain-v20-current-stable)  
3. [新機能検証 (Bio, Symbolic, Spatial)](https://www.google.com/search?q=%233-%E6%96%B0%E6%A9%9F%E8%83%BD%E6%A4%9C%E8%A8%BC-bio-symbolic-spatial)  
4. [SNN学習 & 蒸留ワークフロー](https://www.google.com/search?q=%234-snn%E5%AD%A6%E7%BF%92--%E8%92%B8%E7%95%99%E3%83%AF%E3%83%BC%E3%82%AF%E3%83%95%E3%83%AD%E3%83%BC)  
5. [脳型OS & 認知アーキテクチャ](https://www.google.com/search?q=%235-%E8%84%B3%E5%9E%8Bos--%E8%AA%8D%E7%9F%A5%E3%82%A2%E3%83%BC%E3%82%AD%E3%83%86%E3%82%AF%E3%83%81%E3%83%A3)  
6. [ハードウェア & 最適化](https://www.google.com/search?q=%236-%E3%83%8F%E3%83%BC%E3%83%89%E3%82%A6%E3%82%A7%E3%82%A2--%E6%9C%80%E9%81%A9%E5%8C%96)  
7. [エージェント & 自律行動](https://www.google.com/search?q=%237-%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88--%E8%87%AA%E5%BE%8B%E8%A1%8C%E5%8B%95)  
8. [性能証明・ベンチマーク](https://www.google.com/search?q=%238-%E6%80%A7%E8%83%BD%E8%A8%BC%E6%98%8E%E3%83%99%E3%83%B3%E3%83%81%E3%83%9E%E3%83%BC%E3%82%AF)  
9. [可視化 & デバッグ](https://www.google.com/search?q=%239-%E5%8F%AF%E8%A6%96%E5%8C%96--%E3%83%87%E3%83%90%E3%83%83%E3%82%B0)  
10. [Legacy Commands (Archive)](https://www.google.com/search?q=%2310-legacy-commands-archive)  
11. [トラブルシューティング](https://www.google.com/search?q=%2311-%E3%83%88%E3%83%A9%E3%83%96%E3%83%AB%E3%82%B7%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0)

## **1\. CLI & 環境準備**

プロジェクト全体の操作を簡略化するCLIツールと、基本的なセットアップコマンドです。

### **セットアップ**

\# 依存関係のインストール  
pip install \-r requirements.txt

\# 静的型チェック (開発者向け)  
mypy .

### **SNN CLI ツール (推奨)**

snn-cli.py を使用して主要なタスクを実行できます。

\# ヘルスチェック (全システムの健全性確認)  
python snn-cli.py health-check

\# 特定の実験設定でトレーニングを開始  
python snn-cli.py train \--config configs/models/medium.yaml

\# ベンチマークの実行  
標準ベンチマーク実行 (Benchmark Suite)
総合的なパフォーマンス測定を行います。
注意: --experiment オプションで実験IDの指定が必須です (例: all, cifar10)
python snn-cli.py benchmark run --experiment all

精度評価 (Evaluate Accuracy)
現在のモデルの認識精度のみを重点的に評価します。
python snn-cli.py benchmark evaluate-accuracy

継続学習実験 (Continual Learning)
逐次的なタスク学習における忘却率などを測定します。
python snn-cli.py benchmark continual


### **プロジェクト健全性チェック (スクリプト直接実行)**

CLIを使用しない場合の統合診断コマンドです。

\# プロジェクトヘルスチェック  
python scripts/run\_project\_health\_check.py

\# 全テストスイートの実行 (Unit & Integration)  
python scripts/run\_all\_tests.py

## **2\. Brain v20 (Current Stable)**

最新の統合人工脳モデル (Brain v20) の実行コマンドです。

### **対話型デモ (Chat Interface)**

ユーザーと自然言語で対話し、推論プロセスを確認できます。

\# 最新の対話インターフェース (推奨)  
python scripts/runners/talk\_to\_brain\_final.py

\# 旧バージョン (デバッグ用)  
python scripts/runners/talk\_to\_brain.py

### **Brain v20 プロトタイプ実行**

中核となる推論ループを実行します。

\# Brain v20 プロトタイプの起動  
python scripts/runners/run\_brain\_v20\_prototype.py

## **3\. 新機能検証 (Bio, Symbolic, Spatial)**

特定の研究テーマに特化した検証スクリプト群です。

### **Biomimetic & Microcircuits (生物学的妥当性)**

\# PD14/PFC マイクロサーキットのデモ  
python scripts/run\_bio\_microcircuit\_demo.py

\# Deep Bio Calibration (生物学的パラメータの校正)  
python scripts/run\_deep\_bio\_calibration.py

\# 心電図 (ECG) データの時系列解析  
python scripts/run\_ecg\_analysis.py

### **Neuro-Symbolic & Reasoning (推論・論理)**

\# ニューロシンボリック進化 (論理的推論能力の獲得)  
python scripts/run\_neuro\_symbolic\_evolution.py

\# ニューロシンボリック・デモ実行  
python scripts/runners/run\_neuro\_symbolic\_demo.py

### **Perception & Multimodal (知覚・マルチモーダル)**

\# マルチモーダル統合デモ (視覚・言語・聴覚など)  
python scripts/run\_multimodal\_demo.py  
\# または  
python scripts/runners/run\_multimodal\_brain.py

\# クロスモーダル学習のデモ  
python scripts/run\_cross\_modal\_demo.py

\# 空間認識 (Spatial Awareness) デモ  
python scripts/run\_spatial\_demo.py

\# 能動的推論 (Active Inference) デモ  
python scripts/run\_active\_inference\_demo.py

## **4\. SNN学習 & 蒸留ワークフロー**

モデルのトレーニング、知識蒸留、継続学習に関するコマンドです。

### **トレーニング実行**

\# 標準トレーニング (設定ファイル指定)  
python scripts/runners/train.py \--config configs/models/small.yaml

\# CIFAR-10データセットでのBio-PCネットワーク学習  
python scripts/train\_bio\_pc\_cifar10.py

\# STDP (Spike-Timing-Dependent Plasticity) 学習  
python scripts/run\_stdp\_learning.py

\# オンチップ学習シミュレーション  
python scripts/run\_on\_chip\_learning.py

### **知識蒸留 (Knowledge Distillation)**

大規模モデルからSNNへの知識転移を行います。

\# 蒸留実験の実行  
python scripts/run\_distillation\_experiment.py

\# 蒸留サイクルのテスト  
python scripts/runners/test\_distillation\_cycle.py

\# 蒸留用データの準備  
python scripts/prepare\_distillation\_data.py

### **継続学習 (Continual Learning)**

\# 継続学習実験の実行  
python scripts/run\_continual\_learning\_experiment.py

## **5\. 脳型OS & 認知アーキテクチャ**

ニューロモルフィックOSおよび高度な認知機能のシミュレーションです。

### **Neuromorphic OS**

\# Phase 7 OS シミュレーション (リソース管理・スケジューリング)  
python scripts/run\_phase7\_os\_simulation.py

\# OSランナー実行  
python scripts/runners/run\_neuromorphic\_os.py

### **睡眠・記憶固定化 (Sleep Consolidation)**

\# 睡眠学習デモ  
python scripts/runners/run\_sleep\_learning\_demo.py

\# 睡眠サイクルデモ  
python scripts/runners/run\_sleep\_cycle\_demo.py

\# 推論から睡眠への移行デモ  
python scripts/runners/run\_reasoning\_to\_sleep\_demo.py

## **6\. ハードウェア & 最適化**

ハードウェアへの実装可能性や効率化を検証するコマンドです。

### **ハードウェアシミュレーション**

\# 専用ハードウェア上での動作シミュレーション  
python scripts/run\_hardware\_simulation.py

\# コンパイラテスト  
python scripts/runners/run\_compiler\_test.py

### **効率化・最適化**

\# 効率性の自動チューニング  
python scripts/auto\_tune\_efficiency.py

\# スパース性と時間ステップ(T)のレポート出力  
python scripts/report\_sparsity\_and\_T.py

\# モデル変換スクリプト  
python scripts/convert\_model.py

## **7\. エージェント & 自律行動**

環境内で動作するエージェントの検証です。

\# 自律学習エージェントの実行  
python scripts/runners/run\_autonomous\_learning.py

\# 能動的学習ループ  
python scripts/runners/run\_active\_learning\_loop.py

\# 強化学習エージェント (RL Agent)  
python scripts/runners/run\_rl\_agent.py

\# デジタル生命体 (Digital Life Form) シミュレーション  
python scripts/runners/run\_life\_form.py

\# 世界モデル (World Model) デモ  
python scripts/runners/run\_world\_model\_demo.py

## **8\. 性能証明・ベンチマーク**

外部に対して性能を証明するためのベンチマークスイートです。

\# 総合ベンチマークスイートの実行  
python scripts/run\_benchmark\_suite.py

\# パフォーマンス検証 (詳細オプション指定)  
python scripts/verify\_performance.py \\  
    \--model\_config configs/models/medium.yaml \\  
    \--target\_config configs/validation/targets\_v1.yaml

\# Phase 3 (Transformation) の検証  
python scripts/verify\_phase3.py

\# DSA (Dynamic Sparse Attention) 学習の検証  
python scripts/verify\_dsa\_learning.py

\# 検証と学習の同時実行  
python scripts/runners/verify\_and\_train.py

## **9\. 可視化 & デバッグ**

ニューロンの挙動やスパイク活動を視覚的に確認します。

\# ニューロンダイナミクスの可視化  
python scripts/visualize\_neuron\_dynamics.py

\# スパイクパターンの可視化  
python scripts/visualize\_spike\_patterns.py

\# スパイク活動のデバッグ  
python scripts/debug\_spike\_activity.py

\# 信号診断  
python scripts/diagnose\_signal.py

## **10\. Legacy Commands (Archive)**

以下のコマンドは旧バージョン (v14, v16) のものです。後方互換性テストや比較のために残されています。

### **Brain v16 (Previous Gen)**

\# Full Demo v16.3  
python scripts/runners/run\_v16\_3\_demo.py

\# Full Demo v16.2  
python scripts/runners/run\_v16\_2\_final\_demo.py

\# Brain v16 Demo  
python scripts/runners/run\_brain\_v16\_demo.py

### **Brain v14 (Stable SNN)**

\# Standard Run  
python scripts/runners/run\_brain\_v14.py

\# Brain Simulation  
python scripts/runners/run\_brain\_simulation.py

### **Others**

\# Visual Cortex (Industrial Eye) \- 旧デモ  
python scripts/runners/run\_industrial\_eye\_demo.py

## **11\. トラブルシューティング**

### **ModuleNotFoundError が発生する場合**

プロジェクトルート（snn/）でコマンドを実行しているか確認してください。また、Pythonパスを通す必要がある場合があります。

export PYTHONPATH=$PYTHONPATH:.  
