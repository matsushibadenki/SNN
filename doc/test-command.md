# **Matsushiba Denki SNN \- 統合テストコマンド完全版 (v20 Unified)**

このドキュメントは、SNNプロジェクトの全バージョン（Brain v20, v16, v14）および全機能（Phase 1-16+）を網羅したテストコマンド集です。

開発中の最新機能、ベンチマーク、ハードウェアシミュレーションを含むすべての実行手順を記載しています。

## **📋 目次**

1. [CLI & 環境準備](https://www.google.com/search?q=%231-cli--%E7%92%B0%E5%A2%83%E6%BA%96%E5%82%99)  
2. [Brain v20 (Current Stable)](https://www.google.com/search?q=%232-brain-v20-current-stable)  
3. [Brain v20 Evolution (v20.1 \- v20.3)](https://www.google.com/search?q=%233-brain-v20-evolution-v201---v203) **(New\!)**  
4. [新機能検証 (Bio, Symbolic, Spatial)](https://www.google.com/search?q=%234-%E6%96%B0%E6%A9%9F%E8%83%BD%E6%A4%9C%E8%A8%BC-bio-symbolic-spatial)  
5. [SNN学習 & 蒸留ワークフロー](https://www.google.com/search?q=%235-snn%E5%AD%A6%E7%BF%92--%E8%92%B8%E7%95%99%E3%83%AF%E3%83%BC%E3%82%AF%E3%83%95%E3%83%AD%E3%83%BC)  
6. [旧バージョン (Legacy)](https://www.google.com/search?q=%236-%E6%97%A7%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3-legacy)

## **1\. CLI & 環境準備**

プロジェクトのセットアップと健全性チェックを行うための基本コマンドです。

\# 仮想環境のセットアップ (初回のみ)  
./setup\_colab.sh

\# プロジェクト全体の健全性チェック  
python scripts/run\_project\_health\_check.py

\# CLIツールのヘルプ確認  
python snn-cli.py \--help

## **2\. Brain v20 (Current Stable)**

現在安定稼働している人工脳のプロトタイプ実行コマンドです。

### **Brain v20 Prototype (Base)**

非同期カーネル、アストロサイト制御、BitSpikeMamba (System 1\) を統合した基本形です。

\# Brain v20 Prototype 実行  
python scripts/runners/run\_brain\_v20\_prototype.py

## **3\. Brain v20 Evolution (v20.1 \- v20.3)**

**★ 最新実装機能** 自律進化、マルチモーダル統合、世界モデルなど、高度な知能機能を検証するためのコマンドセットです。

### **Phase 1: 視覚と言語の統合 (SpikingVLM)**

1.58bit量子化を用いた視覚-言語モデルの学習と推論テスト。

\# 1\. 学習用ダミーデータの生成  
python scripts/generate\_vlm\_dummy\_data.py

\# 2\. SpikingVLMの学習 (BitNet有効化)  
python scripts/train\_spiking\_vlm.py \\  
    \--data\_path data/vlm\_dummy/train\_data.jsonl \\  
    \--use\_bitnet \\  
    \--batch\_size 4 \\  
    \--epochs 2

\# 3\. 推論時のオンチップ自己適応 (Test-Time Adaptation)  
\# 未知のデータに対してニューロンの閾値を動的に調整するデモ  
python scripts/run\_vlm\_adaptation.py

### **Phase 2: 睡眠と記憶の定着 (Sleep Consolidation)**

学習した記憶を、外部入力のない「睡眠状態」で整理・強化するプロセス。

\# 睡眠サイクル (Generative Replay) の実行  
\# 脳が自発的に「夢」を見る様子をシミュレーション  
python scripts/run\_vlm\_sleep.py

### **Phase 3: 世界モデルとメタ認知 (System 2\)**

自身の確信度を監視し、自信がない場合にのみ熟考モードへ移行する機能。

\# メタ認知と世界モデルによる行動計画デモ  
python scripts/runners/run\_world\_model\_demo.py

### **Phase 4: 思考の蒸留と自己進化 (Self-Evolution)**

System 2（論理・熟考）の解法を System 1（直感・BitSpike）に蒸留し、脳が自律的に進化するプロセス。

\# 1\. 思考蒸留の単体テスト (算数問題の解法学習)  
python scripts/runners/run\_distillation\_demo.py

\# 2\. Brain v20.2: 実視覚野 (SpikingCNN) との統合  
\# カメラ(想定)からの入力を非同期に処理  
python scripts/runners/run\_brain\_v20\_vision.py

\# 3\. Brain v20.3: 完全統合・自己進化デモ  
\# 未知の問題に遭遇 \-\> 熟考 \-\> 即時学習 \-\> 進化(即答化) のサイクル  
python scripts/runners/run\_brain\_evolution.py

## **4\. 新機能検証 (Bio, Symbolic, Spatial)**

### **生物学的モデル & マイクロ回路**

\# 生物学的マイクロ回路 (PFC Microcircuit)  
python scripts/run\_bio\_microcircuit\_demo.py

\# STDP学習 (Hebbian Learning)  
python scripts/run\_stdp\_learning.py

### **空間認識 & マルチモーダル**

\# 空間認識デモ (Hippocampus/Grid Cells)  
python scripts/run\_spatial\_demo.py

\# マルチモーダル統合 (旧デモ)  
python scripts/run\_multimodal\_demo.py

## **5\. SNN学習 & 蒸留ワークフロー**

### **CIFAR-10 学習ベンチマーク**

\# 生物学的PCネットワークによるCIFAR-10学習  
python scripts/train\_bio\_pc\_cifar10.py

### **論理ゲート駆動・自律学習 (Logic Gated)**

1.58ビット・ロジックゲート樹状突起を用いた、高精度な空間論理認識の自律学習テストです。

\# プロジェクトルートで実行  
export PYTHONPATH=$PYTHONPATH:.  
python scripts/run\_logic\_gated\_learning.py

* **期待される結果**:  
  * Acc: 学習が進むにつれ 99% 以上に到達。  
  * Robustness: ノイズレベル 0.40 でも高精度を維持。

## **6\. 旧バージョン (Legacy)**

過去のメジャーバージョン (v14, v16) のものです。後方互換性テストや比較のために残されています。

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
