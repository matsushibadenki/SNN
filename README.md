# **SNN Project: The Artificial Brain**

**Next-Generation Neuromorphic AI & Neuro-Symbolic OS**

[🇺🇸 English](https://www.google.com/search?q=%23english) | [🇯🇵 日本語](https://www.google.com/search?q=%23japanese)



## **🚀 Vision: "The Brain as an Operating System"**

**"Realizing an Artificial Brain that learns, sleeps, and evolves."**

The **SNN Project** is an open-source initiative to build a comprehensive "Artificial Brain" architecture. We are moving beyond simple neural network models to create a **Neuromorphic OS** where multiple cognitive modules (Vision, Language, Motor control) compete for resources and consciousness, just like in biological brains.

Our goal is to overcome the limitations of traditional ANNs (massive energy consumption, catastrophic forgetting) by combining **Spiking Neural Networks (SNN)**, **Neuro-Symbolic AI**, and **Biologically Plausible Learning**.

### **🏆 Key Achievements & Features**

* **🧠 Cognitive Architecture:** Implements the Global Workspace Theory (GWT). Modules for perception, emotion, and memory compete for the "seat of consciousness."  
* **💤 Sleep & Consolidation:** A system where short-term memories (Hippocampus) are transferred to long-term synaptic weights (Cortex) during "sleep cycles," converting explicit knowledge into intuition.  
* **🧟 Spiking FrankenMoE:** A dynamic "Patchwork" architecture that integrates existing small models (MicroLMs) to act as a single massive brain.  
* **📚 GraphRAG Memory:** A neuro-symbolic memory system that allows for instant knowledge correction and logical reasoning via Knowledge Graphs.  
* **⚡ High-Efficiency SNN:** Achieved **5.38%** spike rate with **1.58-bit quantization** (BitNet), enabling extreme energy efficiency suitable for edge devices.

## **🛠️ Core Components**

### **1\. Artificial Brain Kernel**

* **Global Workspace:** The central hub where information is broadcasted.  
* **Astrocyte Network:** A resource manager that monitors energy (spike rates) and regulates module activity (Homeostasis).  
* **Self-Evolving Agent:** An autonomous agent that rewrites its own code and parameters based on meta-cognition.

### **2\. Learning & Plasticity**

* **Deep Bio-Calibration:** Translates ANN weights to SNNs for high-performance initialization.  
* **Causal Trace Learning:** A gradient-free learning rule that captures causal relationships in spike trains.  
* **Active Inference:** Agents minimize "Free Energy" (Surprise) to explore and adapt to the environment.

### **3\. Neuro-Symbolic Integration**

* **Symbol Grounding:** Connects neural patterns (attractors) to symbolic concepts in the Knowledge Graph.  
* **Neuro-Symbolic Feedback Loop:** A cycle where language interaction updates the graph, and sleep updates the neural weights.

## **📦 Installation**

\# Clone the repository  
git clone \[https://github.com/matsushibadenki/SNN.git\](https://github.com/matsushibadenki/SNN.git)  
cd SNN

\# Create a virtual environment  
python \-m venv .venv

\# Activate the virtual environment  
source .venv/bin/activate  \# Windows: .venv\\Scripts\\activate

\# Install dependencies  
pip install \-r requirements.txt

## **🚦 Quick Start**

### **1\. Launch the Artificial Brain (Simulation)**

Run a cognitive simulation where the brain perceives, thinks, and acts.

\# Register expert models (for demo purposes)  
python scripts/register\_bitnet\_expert.py  
python scripts/register\_demo\_experts.py

\# Create FrankenMoE configuration  
python scripts/manage\_models.py build-moe \--keywords "science,history,calculation" \--output my\_moe.yaml

\# Run the Brain Simulation  
python scripts/runners/run\_brain\_simulation.py \--model\_config configs/models/my\_moe.yaml \--prompt "Calculate 1+1"

### **2\. Experience GraphRAG & Consciousness**

Interact with the brain and observe its internal thought process.

python scripts/observe\_brain\_thought\_process.py \--model\_config configs/models/micro.yaml

\# Input example: "Cats are felines." (Knowledge will be stored in the graph)

## **🗺️ Roadmap (Summary)**

* ✅ **Phase 1-3:** Foundation, Cognitive Architecture, Scaling (Completed)  
* ✅ **Phase 4:** Autonomous Intelligence (Completed)  
* 🔄 **Phase 5: Neuro-Symbolic Evolution (Current)** \- Sleep Consolidation, Real-time Knowledge Editing.  
* 📅 **Phase 6:** Hardware Native Transition \- Moving from GPU simulation to Event-Driven Kernels.  
* 📅 **Phase 7:** The "Brain" OS \- Multi-agent competition and resource management.

See [doc/ROADMAP.md](https://www.google.com/search?q=doc/ROADMAP.md) for details.

## **📂 Directory Structure**

SNN/  
├── app/                    \# Application Layer (UI, DI Container)  
├── configs/                \# Configuration Files  
├── snn\_research/           \# Core Library  
│   ├── cognitive\_architecture/ \# Brain Modules (Cortex, Hippocampus, etc.)  
│   ├── core/                   \# SNN Kernels (Neurons, STDP, Attention)  
│   ├── agent/                  \# Autonomous Agents & Life Forms  
│   └── hardware/               \# Neuromorphic Compiler  
└── scripts/                \# Execution Scripts

## **🤝 Contributing**

We welcome contributions to build the future of AI\! Please refer to doc/SNN開発:プロジェクト機能テスト コマンド一覧.md for testing protocols.

## **📜 License**

MIT License

\<div align="right"\>

[⬆️ Back to Top](https://www.google.com/search?q=%23english)

\</div\>

\<a id="japanese"\>\</a\>

## **🚀 ビジョン: "OSとしての脳"**

**"学習し、眠り、進化する『人工脳』の実現"**

**SNNプロジェクト**は、単なるAIモデルの開発を超え、ヒトの脳のように複数の認知モジュール（視覚、言語、運動など）がリソースと意識の座を競い合う\*\*「ニューロモーフィックOS（脳型オペレーティングシステム）」\*\*の構築を目指すオープンソースプロジェクトです。

従来のANNが抱える「巨大な電力消費」と「再学習の困難さ」を克服するため、**スパイキングニューラルネットワーク(SNN)**、**ニューロシンボリックAI**、そして**生物学的学習則**を融合させた革新的なアーキテクチャを採用しています。

### **🏆 主な成果と特徴**

* **🧠 認知アーキテクチャ:** グローバルワークスペース理論(GWT)を実装。知覚・感情・記憶などのモジュールが「意識」を求めて競合します。  
* **💤 睡眠と記憶の固定化:** 短期記憶（海馬）を、睡眠サイクルを通じて長期記憶（大脳皮質のシナプス重み）に転送し、知識を直感へと変換します。  
* **🧟 Spiking FrankenMoE:** 既存の小さなモデル(MicroLM)を部品として動的に結合し、単一の巨大な脳として振る舞わせる「継ぎ接ぎ」アーキテクチャ。  
* **📚 GraphRAG記憶システム:** ベクトル検索とナレッジグラフを組み合わせ、文脈に沿った正確な記憶の想起と、対話による即座の知識修正を実現。  
* **⚡ 超高効率SNN:** **1.58bit量子化** (BitNet) とスパース性を組み合わせ、スパイク率 **5.38%** を達成。エッジデバイスでの自律動作を可能にします。

## **🛠️ コアコンポーネント**

### **1\. 人工脳カーネル**

* **Global Workspace:** 情報のブロードキャストを行う意識の中枢。  
* **Astrocyte Network:** エネルギー（スパイク率）を監視し、各モジュールの活動レベルを動的に調整（ホメオスタシス）するリソースマネージャ。  
* **Self-Evolving Agent:** メタ認知に基づいて、自身のソースコードやパラメータを書き換える自律エージェント。

### **2\. 学習と可塑性**

* **Deep Bio-Calibration:** ANNの重みをSNNの物理パラメータに変換し、高性能な初期状態を作成。  
* **Causal Trace Learning:** スパイクの因果連鎖に基づいて学習する、勾配計算不要（Gradient-free）の学習則。  
* **Active Inference (能動的推論):** 自由エネルギー（驚き）を最小化するように、自律的に環境を探索・行動します。

### **3\. 神経-記号融合 (Neuro-Symbolic)**

* **Symbol Grounding:** 神経活動パターン（アトラクタ）と、知識グラフ上の概念（シンボル）を動的に結びつけます。  
* **Neuro-Symbolic Feedback Loop:** 言語対話で知識グラフを更新し、睡眠でそれをSNNの重みに焼き付ける循環ループ。

## **📦 インストール**

\# リポジトリのクローン  
git clone \[https://github.com/matsushibadenki/SNN.git\](https://github.com/matsushibadenki/SNN.git)  
cd SNN

\# 仮想環境の作成  
python \-m venv .venv

\# 仮想環境の有効化  
source .venv/bin/activate  \# Windows: .venv\\Scripts\\activate

\# 依存関係のインストール  
pip install \-r requirements.txt

## **🚦 クイックスタート**

### **1\. 人工脳シミュレーションの起動**

脳が知覚し、思考し、行動する様子をシミュレートします。

\# エキスパートモデルの登録 (デモ用)  
python scripts/register\_bitnet\_expert.py  
python scripts/register\_demo\_experts.py

\# FrankenMoE構成ファイルの作成  
python scripts/manage\_models.py build-moe \--keywords "science,history,calculation" \--output my\_moe.yaml

\# 人工脳の起動 (計算タスクの依頼)  
python scripts/runners/run\_brain\_simulation.py \--model\_config configs/models/my\_moe.yaml \--prompt "1+1の計算をして"

### **2\. 思考プロセスの観察 (GraphRAG体験)**

人工脳と対話し、その思考や感情の変化、記憶の形成過程を観察します。

python scripts/observe\_brain\_thought\_process.py \--model\_config configs/models/micro.yaml

\# 入力例: "猫は猫科です" (知識がグラフに保存され、後の推論に影響します)

## **🗺️ ロードマップ (概要)**

* ✅ **Phase 1-3:** 基盤構築、認知アーキテクチャ、スケーリング (完了)  
* ✅ **Phase 4:** 自律知能 (完了)  
* 🔄 **Phase 5: Neuro-Symbolic Evolution (進行中)** \- 睡眠による記憶固定化、リアルタイム知識編集。  
* 📅 **Phase 6:** Hardware Native Transition \- GPUシミュレーションから、イベント駆動カーネルへの移行。  
* 📅 **Phase 7:** The "Brain" OS \- 複数エージェントの競合とリソース管理を行うOS化。

詳細は [doc/ROADMAP.md](https://www.google.com/search?q=doc/ROADMAP.md) をご覧ください。

## **📂 ディレクトリ構造**

SNN/  
├── app/                    \# アプリケーション層 (UI、DIコンテナ)  
├── configs/                \# 設定ファイル  
├── snn\_research/           \# コアライブラリ  
│   ├── cognitive\_architecture/ \# 脳モジュール (皮質、海馬など)  
│   ├── core/                   \# SNNカーネル (ニューロン、STDP、Attention)  
│   ├── agent/                  \# 自律エージェント、デジタル生命体  
│   └── hardware/               \# ニューロモーフィック・コンパイラ  
└── scripts/                \# 実行スクリプト

## **🤝 コントリビューション**

バグ報告、機能提案、プルリクエストを歓迎します！  
開発に参加される場合は、doc/SNN開発:プロジェクト機能テスト コマンド一覧.md を参照してテストを実施してください。

## **📜 ライセンス**

MIT License

## **📧 お問い合わせ**

質問やコラボレーションのお問い合わせは、Issueを開くか、メンテナーまでご連絡ください。

\<div align="right"\>

[⬆️ トップに戻る](https://www.google.com/search?q=%23japanese)

\</div\>
