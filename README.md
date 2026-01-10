# **SNN Research: Next-Generation Neuromorphic Computing & Artificial Brain Architecture**. 
  

SNN Research is an open-source library designed to simulate Artificial Brains using Spiking Neural Networks (SNNs). Unlike traditional deep learning, this project focuses on biologically plausible mechanisms like STDP, Spiking Transformers, and Global Workspace Theory to achieve advanced cognitive functions including consciousness simulation, sleep consolidation, and embodied cognition.  
  

  

## **📖 Introduction & Philosophy**

This project is an ambitious research initiative aimed at bridging the gap between biological brains and artificial intelligence using **Spiking Neural Networks (SNNs)** and **Neuromorphic Computing** principles.

Unlike traditional Deep Learning, which relies on continuous activation values, this project focuses on **spike-based communication**, temporal dynamics, and biologically plausible learning rules (e.g., STDP, Hebbian Learning). The ultimate goal is to create an **Artificial Brain** capable of advanced cognitive functions such as:

* **Consciousness & Global Workspace Theory**  
* **Sleep, Dreaming & Memory Consolidation**  
* **Emotion & Intrinsic Motivation (Curiosity)**  
* **Social Cognition & Theory of Mind**  
* **Embodied Cognition (Sensory-Motor Integration)**

We integrate state-of-the-art architectures like **Spiking Transformers (Spikformer)**, **Spiking Mamba**, and **Logic-Gated SNNs** into a unified cognitive system.

## **🏗 Architecture**

The system is designed with a hierarchical and modular architecture, moving from low-level neuronal dynamics to high-level social interactions.

### **1\. Core Layer (snn\_research/core)**

* **Neuron Models:** Leaky Integrate-and-Fire (LIF), Adaptive Neurons, Multi-compartment models.  
* **Learning Rules:** STDP (Spike-Timing-Dependent Plasticity), Predictive Coding, Reward-modulated learning.  
* **Hardware Abstraction:** Support for simulating neuromorphic hardware constraints.

### **2\. Model Layer (snn\_research/models)**

* **Vision:** Spiking CNNs, DVS (Dynamic Vision Sensor) processing.  
* **Language & Sequence:** Spiking Transformers, Spiking Mamba/RWKV for efficient sequence modeling.  
* **Generative:** Spiking Diffusion Models.

### **3\. Cognitive Architecture (snn\_research/cognitive\_architecture)**

mimicking the mammalian brain structure:

* **Prefrontal Cortex:** Executive control, planning, and decision making.  
* **Hippocampus:** Episodic memory and spatial navigation.  
* **Thalamus:** Sensory relay and attention gating.  
* **Amygdala:** Emotional processing and survival instincts.  
* **Basal Ganglia:** Action selection and reinforcement learning.  
* **Global Workspace:** Consciousness simulation via information broadcasting.

### **4\. System & Agent Layer (scripts/agents, scripts/demos)**

* **Sleep Cycles:** Simulation of NREM/REM sleep for memory consolidation and structural plasticity.  
* **Social Agents:** Agents capable of communication (Naming Game) and understanding others' intent.

## **🚀 Installation**

### **Prerequisites**

* Python 3.9 or higher  
* PyTorch (CUDA support recommended for performance)

### **Setup**

1. **Clone the repository:**  
   git clone \<repository-url\>  
   cd SNN

2. Install dependencies:  
   You can use the provided setup script or install via pip.  
   \# Using the setup script (Recommended for Linux/Colab)  
   bash setup\_colab.sh

   \# Or manual installation  
   pip install \-r requirements.txt  \# If available  
   \# Or install the package in editable mode  
   pip install \-e .

## **💻 Basic Usage**

The project provides a CLI and numerous scripts to run demos and experiments.

### **1\. Using the CLI**

A central command-line interface is available for managing models and running tasks.

python snn-cli.py \--help  
python snn-cli.py demo list  
python snn-cli.py demo run brain\_v16

### **2\. Running Specific Demos**

Explore various cognitive capabilities through pre-configured demos in scripts/demos.

* **Consciousness Demo:**  
  python scripts/demos/brain/run\_conscious\_broadcast\_demo.py

* **Sleep & Dreaming Demo:**  
  python scripts/demos/systems/run\_sleep\_dream\_demo.py

* **Visual Perception:**  
  python scripts/demos/visual/run\_spiking\_ff\_demo.py

### **3\. Training & Experiments**

Run scientific experiments to validate hypotheses using scripts in scripts/experiments.

python scripts/experiments/learning/run\_continual\_learning\_experiment.py

## **📂 Project Structure**

.  
├── app/                        \# Web Interface / API (FastAPI, Dashboard)  
├── configs/                    \# YAML Configuration files for experiments and models  
├── doc/                        \# Documentation and research notes  
├── scripts/                    \# Executable scripts  
│   ├── agents/                 \# Autonomous agent runners  
│   ├── benchmarks/             \# Performance and latency benchmarks  
│   ├── demos/                  \# Demonstration of specific capabilities  
│   ├── experiments/            \# Scientific experiments and data collection  
│   ├── training/               \# Training loops and pipelines  
│   └── visualization/          \# Tools for visualizing spike trains and brain activity  
├── snn\_research/               \# Main Source Code Library  
│   ├── core/                   \# SNN primitives (Neurons, Synapses)  
│   ├── cognitive\_architecture/ \# High-level brain modules (Cortex, Hippocampus, etc.)  
│   ├── models/                 \# Neural Network Architectures (Spikformer, CNN, etc.)  
│   ├── learning\_rules/         \# STDP, BCM, etc.  
│   ├── systems/                \# Integrated agent systems  
│   └── utils/                  \# Helper functions  
└── tests/                      \# Unit and integration tests

# **SNN研究プロジェクト: 次世代ニューロモーフィック・コンピューティングと人工脳アーキテクチャ**

## **📖 プロジェクトの主旨と設計**

本プロジェクトは、**スパイキングニューラルネットワーク（SNN）とニューロモーフィック・コンピューティング**の原理を用いて、生物学的脳と人工知能の架け橋となることを目指す野心的な研究イニシアチブです。

従来のディープラーニングが連続的な活性化値に依存しているのに対し、本プロジェクトは**スパイクベースの通信**、時間的ダイナミクス、および生物学的に妥当な学習則（STDP、ヘッブ学習など）に焦点を当てています。究極の目標は、以下のような高度な認知機能を持つ\*\*人工脳（Artificial Brain）\*\*を構築することです。

* **意識とグローバルワークスペース理論**: 情報のブロードキャストによる意識の模倣  
* **睡眠・夢・記憶の定着**: 記憶の整理と構造的な可塑性の実現  
* **感情と内発的動機づけ**: 好奇心に基づく探索と自己保存  
* **社会性認知と心の理論**: 他者理解とコミュニケーション  
* **身体化された認知**: 感覚運動統合による環境との相互作用

また、**Spiking Transformer (Spikformer)** や **Spiking Mamba**、**Logic-Gated SNN** といった最新のアーキテクチャを、統一された認知システムへと統合しています。

## **🏗 アーキテクチャ**

システムは、低レベルの神経ダイナミクスから高レベルの社会的相互作用に至るまで、階層的かつモジュール化されたアーキテクチャで設計されています。

### **1\. コアレイヤー (snn\_research/core)**

* **ニューロンモデル**: LIF (Leaky Integrate-and-Fire)、適応型ニューロン、マルチコンパートメントモデルなど。  
* **学習則**: STDP (スパイクタイミング依存可塑性)、予測符号化、報酬変調型学習。  
* **ハードウェア抽象化**: ニューロモーフィックハードウェアの制約をシミュレーションするためのレイヤー。

### **2\. モデルレイヤー (snn\_research/models)**

* **視覚**: スパイキングCNN、DVS (Dynamic Vision Sensor) 処理。  
* **言語・系列**: Spiking Transformer、Spiking Mamba/RWKVによる効率的な系列モデリング。  
* **生成**: スパイキング拡散モデル (Diffusion Models)。

### **3\. 認知アーキテクチャ (snn\_research/cognitive\_architecture)**

哺乳類の脳構造を模倣したモジュール群:

* **前頭前野 (Prefrontal Cortex)**: 実行制御、計画、意思決定。  
* **海馬 (Hippocampus)**: エピソード記憶、空間ナビゲーション。  
* **視床 (Thalamus)**: 感覚情報の優先順位付けと注意のゲーティング。  
* **扁桃体 (Amygdala)**: 情動処理と生存本能。  
* **大脳基底核 (Basal Ganglia)**: 行動選択と強化学習。  
* **グローバルワークスペース**: 意識的な情報の共有と放送。

### **4\. システム＆エージェントレイヤー (scripts/agents, scripts/demos)**

* **睡眠サイクル**: ノンレム睡眠/レム睡眠をシミュレートし、記憶の定着とシナプスの最適化を行います。  
* **社会性エージェント**: 言語（ネーミングゲーム）を通じたコミュニケーションや、他者の意図理解を行うエージェント。

## **🚀 導入方法**

### **前提条件**

* Python 3.9 以上  
* PyTorch (パフォーマンスのためCUDAサポート推奨)

### **セットアップ**

1. **リポジトリのクローン:**  
   git clone \<repository-url\>  
   cd SNN

2. 依存関係のインストール:  
   提供されているセットアップスクリプトを使用するか、pip経由でインストールします。  
   \# セットアップスクリプトの使用（Linux/Colab環境推奨）  
   bash setup\_colab.sh

   \# または手動インストール  
   pip install \-e .

## **💻 基本的な使い方**

このプロジェクトには、デモや実験を実行するためのCLIと多数のスクリプトが用意されています。

### **1\. CLIの使用**

モデルの管理やタスクの実行を行うための中央コマンドラインインターフェースです。

python snn-cli.py \--help  
\# デモの一覧表示  
python snn-cli.py demo list  
\# 特定のデモの実行  
python snn-cli.py demo run brain\_v16

### **2\. デモの実行**

scripts/demos にある設定済みデモを通じて、様々な認知機能を確認できます。

* **意識のブロードキャストデモ:**  
  python scripts/demos/brain/run\_conscious\_broadcast\_demo.py

* **睡眠と夢のデモ:**  
  python scripts/demos/systems/run\_sleep\_dream\_demo.py

* **視覚知覚 (Forward-Forward法など):**  
  python scripts/demos/visual/run\_spiking\_ff\_demo.py

### **3\. 実験と学習**

scripts/experiments にあるスクリプトを使用して、科学的な仮説検証やモデルの学習を行います。

python scripts/experiments/learning/run\_continual\_learning\_experiment.py

## **📂 プロジェクト構造**

主要なディレクトリ構成は以下の通りです。

.  
├── app/                        \# Webインターフェース / API (FastAPI, Dashboard)  
├── configs/                    \# 実験やモデルのためのYAML設定ファイル  
├── doc/                        \# ドキュメント、研究ノート  
├── scripts/                    \# 実行可能なスクリプト群  
│   ├── agents/                 \# 自律エージェントの実行ランナー  
│   ├── benchmarks/             \# パフォーマンスとレイテンシのベンチマーク  
│   ├── demos/                  \# 特定機能のデモンストレーション  
│   ├── experiments/            \# 実験とデータ収集用スクリプト  
│   ├── training/               \# 学習ループとパイプライン  
│   └── visualization/          \# スパイク列や脳活動の可視化ツール  
├── snn\_research/               \# メインライブラリコード  
│   ├── core/                   \# SNNの基本要素 (ニューロン, シナプス, レイヤー)  
│   ├── cognitive\_architecture/ \# 高次脳機能モジュール (皮質, 海馬, 視床など)  
│   ├── models/                 \# ニューラルネットワークモデル (Spikformer, CNNなど)  
│   ├── learning\_rules/         \# 学習則 (STDP, BCMなど)  
│   ├── systems/                \# 統合されたエージェントシステム  
│   └── utils/                  \# ユーティリティ関数  
└── tests/                      \# ユニットテストと統合テスト  
