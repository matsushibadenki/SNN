# **SNN Project: Humane Neuromorphic AGI**

**"The Brain as an Operating System" – Next-Generation Neuro-Symbolic Architecture**

[🇺🇸 English](https://www.google.com/search?q=%23english) | [🇯🇵 日本語](https://www.google.com/search?q=%23japanese)

## **🚀 Vision {https://www.google.com/search?q=%23english}**

**"Building a 'Humane' Artificial Brain that learns, sleeps, and evolves."**

The **SNN Project** is an open-source initiative to build a comprehensive "Artificial Brain" architecture beyond simple neural networks. We aim to create a **Neuromorphic OS** where multiple cognitive modules (Vision, Language, Motor Control) compete for resources and consciousness, governed by biological constraints and ethical guardrails.

Our goal is to realize **Humane Neuromorphic AGI**—AI that is energy-efficient, adaptable, and designed to coexist harmoniously with humans. We combine **Spiking Neural Networks (SNN)**, **Neuro-Symbolic AI**, and **Biologically Plausible Learning** to overcome the limitations of traditional ANNs.

### **🏆 Key Achievements (v16.0)**

* **⚡ SFormer (T=1) Backbone:** A scale-and-fire transformer that achieves ANN-level inference speed (T=1) with SNN energy efficiency.  
* **🧠 Cognitive Architecture:** Implements Global Workspace Theory (GWT). Modules compete for the "seat of consciousness" based on salience and energy budget.  
* **👁️ Unified Perception (SNN-DSA):** Dynamic Sparse Attention mechanism for processing multi-modal inputs (Image, Audio, Text, DVS) efficiently.  
* **🤔 System 2 Reasoning (GRPO \+ Verifier):** Multi-step reasoning engine with self-verification capabilities, distilled into efficient System 1 policies during "sleep."  
* **🛡️ Hyper-Robust Logic Gated SNN:** (New) A multiplication-free architecture using 1.58-bit weights. Achieved \>85% accuracy under 40% input noise conditions, demonstrating extreme robustness and biological plausibility.

## **📚 Core Architecture**

### **1\. The Neuromorphic Kernel**

At the heart of the system is the **Hybrid Neuromorphic Core**, which orchestrates:

* **Logic Gated SNN Layers:** Ultra-low power processing using ternary weights (-1, 0, 1\) and accumulators (no multiplications).  
* **Reservoir Computing (LSM):** Rich, high-dimensional projection of temporal data for robust pattern recognition.  
* **Thermodynamic Sampling:** Stochastic sampling (Langevin Dynamics) for creative generation and decision making under uncertainty.

### **2\. Cognitive Modules**

* **Prefrontal Cortex (Planner):** Long-term planning and executive function.  
* **Hippocampus (Memory):** Episodic memory storage and replay during sleep.  
* **Visual/Auditory Cortex:** Sensory processing pipelines.  
* **Amygdala (Emotion):** Value system that modulates learning rates and attention based on "pain" and "pleasure" signals.

### **3\. Advanced Learning Paradigms**

* **Deep Bio-Calibration:** High-fidelity conversion of ANN weights to SNN physical parameters (HSEO).  
* **Causal Trace Learning:** Gradient-free learning rule based on spike timing and causality.  
* **Distillation Pipeline:** Compresses complex "System 2" reasoning paths into fast "System 1" reflexes.  
* **Hyper-Robust Learning:** Training methodology involving variable noise injection (up to 45%) to force the model to learn structural invariants.

## **📦 Installation**

\# Clone the repository  
git clone \[https://github.com/matsushibadenki/SNN.git\](https://github.com/matsushibadenki/SNN.git)  
cd SNN

\# Create and activate virtual environment  
python \-m venv .venv  
source .venv/bin/activate  \# Windows: .venv\\Scripts\\activate

\# Install dependencies  
pip install \-r requirements.txt

## **🚦 Quick Start**

Use the unified CLI tool snn-cli.py to manage the lifecycle of the artificial brain.

### **1\. Launch Brain Simulation**

Simulate the brain perceiving, thinking, and acting in an interactive mode.

\# Start simulation with specific config  
python scripts/runners/run\_brain\_simulation.py \\  
    \--model\_config configs/models/small.yaml \\  
    \--mode interactive

### **2\. Run Hyper-Robustness Demo**

Demonstrate the Logic Gated SNN's ability to recognize patterns under extreme noise.

python scripts/run\_logic\_gated\_learning.py

### **3\. Health Check (System Diagnosis)**

Verify that all subsystems (learning, inference, memory, agents) are functioning correctly.

python snn-cli.py health-check

## **🗺️ Roadmap**

We are currently in **Phase 3: Cognitive Architecture Integration**.

* \[x\] **Phase 1: Foundation:** SFormer backbone, basic SNN layers.  
* \[x\] **Phase 2: Perception:** Multi-modal integration, DVS processing.  
* \[ \] **Phase 3: Cognition:** Global Workspace, System 2 Reasoning, Sleep consolidation.  
* \[ \] **Phase 4: Embodiment:** Robotics integration, real-world interaction.

See doc/ROADMAP.md for details.

## **🤝 Contributing**

Contributions are welcome\! Please read CONTRIBUTING.md (coming soon) for details on our code of conduct and the process for submitting pull requests.

## **📄 License**

\[License Information Here\]

# **SNN プロジェクト: Humane Neuromorphic AGI {https://www.google.com/search?q=%23japanese}**

**「脳をOSとして再発明する」 – 次世代ニューロ・シンボリック・アーキテクチャ**

## **🚀 ビジョン**

**「学習し、眠り、進化する『人間らしい』人工脳の構築」**

**SNNプロジェクト**は、単なるニューラルネットワークを超えた、包括的な「人工脳」アーキテクチャを構築するオープンソース・イニシアチブです。私たちは、生物学的な制約と倫理的なガードレールに基づき、複数の認知モジュール（視覚、言語、運動制御など）がリソースと意識の座を競い合う **Neuromorphic OS** の実現を目指しています。

私たちの目標は、エネルギー効率が高く、適応力があり、人間と調和して共存できる **Humane Neuromorphic AGI（人間性を持つニューロモルフィックAGI）** です。**スパイキングニューラルネットワーク(SNN)**、**ニューロ・シンボリックAI**、そして**生物学的妥当性を持つ学習則**を融合させ、従来のANNの限界を突破します。

### **🏆 主な成果 (v16.0)**

* **⚡ SFormer (T=1) Backbone:** SNNのエネルギー効率を保ちながら、ANNレベルの推論速度(T=1)を実現するScale-and-Fire Transformer。  
* **🧠 認知アーキテクチャ:** グローバルワークスペース理論(GWT)を実装。モジュール群が顕著性(Salience)とエネルギー予算に基づいて「意識の座」を競合します。  
* **👁️ 統合知覚 (SNN-DSA):** 画像、音声、テキスト、DVS(イベントカメラ)などのマルチモーダル入力を効率的に処理する動的スパース注意機構。  
* **🤔 System 2 推論 (GRPO \+ Verifier):** 自己検証機能を備えた多段階推論エンジン。深い思考プロセスは「睡眠」中に効率的なSystem 1（直感）ポリシーへと蒸留されます。  
* **🛡️ Logic Gated SNN (Hyper-Robust):** (New) 1.58ビット重みを採用した乗算フリーのアーキテクチャ。40%の入力ノイズ下でも85%以上の精度を維持する、極めて高い堅牢性と生物学的妥当性を実証しました。

## **📚 コア・アーキテクチャ**

### **1\. ニューロモルフィック・カーネル**

システムの中心には **Hybrid Neuromorphic Core** があり、以下を統合制御します：

* **Logic Gated SNN Layers:** 3値重み(-1, 0, 1)とアキュムレータのみを使用し、乗算を排除した超低消費電力処理。  
* **Reservoir Computing (LSM):** 入力データを高次元かつ豊かな時間的特徴空間へ投影し、堅牢なパターン認識を実現する液体状態マシン。  
* **Thermodynamic Sampling:** 不確実性下での意思決定や創造的生成のための、熱力学（ランジュバン動力学）に基づく確率的サンプリング。

### **2\. 認知モジュール群**

* **前頭前皮質 (Planner):** 長期計画、実行機能、抑制制御。  
* **海馬 (Memory):** エピソード記憶の保存と、睡眠中のリプレイ（記憶の定着）。  
* **視覚・聴覚野:** 感覚情報の処理パイプライン。  
* **扁桃体 (Emotion):** 「痛み」や「快感」の信号に基づき、学習率や注意の方向を調整する価値評価システム。  
* **ニューロ・シンボリック (概念):** シンボル接地問題(Symbol Grounding)に取り組み、神経活動と論理記号(概念)を動的に結びつけ、説明可能性を担保します。

### **3\. 高度な学習機能**

* **Deep Bio-Calibration:** ANNの重みをSNNの物理パラメータに変換・最適化(HSEO)する高忠実度パイプライン。  
* **Causal Trace Learning:** スパイクの因果連鎖に基づく、勾配計算不要(Gradient-free)の学習則。  
* **Distillation Pipeline:** System 2の深い思考過程を、軽量なSystem 1モデルに蒸留・圧縮します。  
* **Hyper-Robust Learning:** 可変ノイズ（最大45%）を注入する学習手法により、モデルに表面的なパターンではなく構造的な不変性を学習させます。

## **📦 インストール**

\# リポジトリのクローン  
git clone \[https://github.com/matsushibadenki/SNN.git\](https://github.com/matsushibadenki/SNN.git)  
cd SNN

\# 仮想環境の作成と有効化  
python \-m venv .venv  
source .venv/bin/activate  \# Windows: .venv\\Scripts\\activate

\# 依存関係のインストール  
pip install \-r requirements.txt

## **🚦 クイックスタート**

統合CLIツール snn-cli.py を使用して、人工脳のライフサイクル全体を管理できます。

### **1\. 人工脳シミュレーションの起動**

脳が知覚し、思考し、行動する様子を対話形式でシミュレートします。

\# 設定を指定してシミュレーションを開始  
python scripts/runners/run\_brain\_simulation.py \\  
    \--model\_config configs/models/small.yaml \\  
    \--mode interactive

### **2\. ハイパー・ロバスト性デモの実行**

Logic Gated SNNが極度のノイズ下でもパターンを認識できることを実証します。

python scripts/run\_logic\_gated\_learning.py

### **3\. ヘルスチェック (システム診断)**

学習、推論、記憶、エージェント機能など、全サブシステムが正常に動作しているか確認します。

python snn-cli.py health-check  
