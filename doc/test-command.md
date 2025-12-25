# **Matsushiba Denki SNN \- 統合テストコマンド完全版 (v20 Unified)**

このドキュメントは、SNNプロジェクトの全バージョン（Brain v20, v16, v14）および全機能（Phase 1-16+）を網羅したテストコマンド集です。

最新の **SCAL (Statistical Centroid Alignment Learning)** 技術、および **Brain v2.0 Spartan Training** の手順を含みます。

## **📋 目次**

1. [CLI & 環境準備](https://www.google.com/search?q=%231-cli--%E7%92%B0%E5%A2%83%E6%BA%96%E5%82%99)  
2. [Brain v20 / v16 Demo](https://www.google.com/search?q=%232-brain-v20--v16-demo) **(Updated\!)**  
3. [Training Workflow](https://www.google.com/search?q=%233-training-workflow) **(New\!)**  
4. [SCAL & Core Technology](https://www.google.com/search?q=%234-scal--core-technology) **(New\!)**  
5. [Legacy & Benchmarks](https://www.google.com/search?q=%235-legacy--benchmarks)

## **1\. CLI & 環境準備**

プロジェクトルートで実行してください。

\# 仮想環境の有効化 (推奨)  
source .venv/bin/activate

\# パスの設定 (必須)  
export PYTHONPATH=$PYTHONPATH:.

## **2\. Brain v20 / v16 Demo**

統合された人工脳の動作デモです。

### **Brain v16.3: SCAL統合・自律動作デモ**

視覚野、思考エンジン(Mamba)、反射モジュールが **SCAL (バイポーラ平均化)** 技術によって結合され、極限ノイズ環境でも動作します。

\# 統合デモの実行 (Greeting, Logic, Safety, Reflex, Fatigueシナリオ)  
python scripts/runners/run\_brain\_v16\_demo.py

### **Brain v20 Vision: 視覚野統合**

DVSセンサー（を模した入力）からの信号を処理し、行動を決定します。

python scripts/runners/run\_brain\_v20\_vision.py

## **3\. Training Workflow**

### **Brain v2.0 "Spartan" Training (Distinct Mode)**

言語モデルのモード崩壊（金太郎飴状態）を防ぎ、明確な回答能力を獲得させるための厳格な学習ループです。  
現在実行中のスクリプトです。  
\# ターゲットLoss: 0.05  
python scripts/trainers/train\_brain\_v5\_distinct.py

### **Web Learning (Self-Evolution)**

Web検索を通じて知識を獲得し、RAGデータベースを構築します。

python scripts/runners/run\_web\_learning.py

## **4\. SCAL & Core Technology**

今回確立された、ノイズ耐性限界(0.48)を突破するためのコア技術の検証コマンドです。

### **SCAL (Statistical Centroid Alignment Learning) 検証**

1.58ビット・ロジックゲート樹状突起とバイポーラ統計平均化を用いた、超ロバスト学習のシミュレーションです。

\# ノイズレベル0.48（信号成分4%）での学習実証  
python scripts/run\_logic\_gated\_learning.py

* **期待される結果**:  
  * Noise 0.45: Accuracy \> 85% (Excellent)  
  * Noise 0.48: Accuracy \> 35% (State-of-the-Art / Theoretical Limit)  
  * "Status: State-of-the-Art" が表示されれば成功です。

### **ニューロンダイナミクス可視化**

適応型LIFニューロンの膜電位挙動を確認します。

python scripts/visualize\_neuron\_dynamics.py

## **5\. Legacy & Benchmarks**

### **CIFAR-10 学習ベンチマーク**

生物学的PCネットワークによる画像認識学習。

python scripts/train\_bio\_pc\_cifar10.py

### **初期プロトタイプ (Brain v14)**

python scripts/run\_artificial\_brain\_v14.py  
