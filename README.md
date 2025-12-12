# **SNN Project: Humane Neuromorphic AGI**

**"The Brain as an Operating System" – Next-Generation Neuro-Symbolic Architecture**

[🇺🇸 English](#english) | [🇯🇵 日本語](#japanese)



## **🚀 Vision** {#english}

**"Building a 'Humane' Artificial Brain that learns, sleeps, and evolves."**

The **SNN Project** is an open-source initiative to build a comprehensive "Artificial Brain" architecture beyond simple neural networks. We aim to create a **Neuromorphic OS** where multiple cognitive modules (Vision, Language, Motor Control) compete for resources and consciousness, governed by biological constraints and ethical guardrails.

Our goal is to realize **Humane Neuromorphic AGI**—AI that is energy-efficient, adaptable, and designed to coexist harmoniously with humans. We combine **Spiking Neural Networks (SNN)**, **Neuro-Symbolic AI**, and **Biologically Plausible Learning** to overcome the limitations of traditional ANNs.



### **🏆 Key Achievements (v16.0)**

- **⚡ SFormer (T=1) Backbone:** A scale-and-fire transformer that achieves ANN-level inference speed (T=1) with SNN energy efficiency.
- **🧠 Cognitive Architecture:** Implements Global Workspace Theory (GWT). Modules compete for the "seat of consciousness" based on salience and energy budget.
- **👁️ Unified Perception (SNN-DSA):** Dynamic Sparse Attention mechanism for processing multi-modal inputs (Image, Audio, Text, DVS) efficiently.
- **🤔 System 2 Reasoning (GRPO + Verifier):** Multi-step reasoning engine with self-verification loops, distilled into System 1 intuition via sleep consolidation.
- **🛡️ Ethical Guardrails:** A built-in "conscience" module that monitors thought processes and blocks unsafe actions at the neural level.
- **💤 Sleep & Consolidation:** Transfers short-term episodic memories (Hippocampus) to long-term synaptic weights (Cortex) via generative replay.



## **🛠️ Core Components**

### **1. Artificial Brain Kernel**

- **Global Workspace:** The central hub for conscious broadcasting.
- **Astrocyte Network (The OS):** A resource manager that monitors energy (spike rates) and regulates module activity via homeostasis.
- **Neuromorphic Scheduler:** Arbitrates task execution based on priority bidding and energy availability.

### **2. Neuro-Symbolic Integration**

- **GraphRAG Memory:** Combines vector search with knowledge graphs for precise retrieval and real-time knowledge correction.
- **Symbol Grounding:** Maps neural attractors to symbolic concepts, enabling explainable AI.

### **3. Advanced Learning**

- **Deep Bio-Calibration:** High-fidelity conversion of ANNs to SNNs using HSEO optimization.
- **Causal Trace Learning:** Gradient-free learning rule capturing causal relationships in spike trains.
- **Distillation Pipeline:** Compresses "System 2" reasoning traces (CoT) into efficient "System 1" SNN weights.



## **📦 Installation**

```bash
# Clone the repository
git clone https://github.com/matsushibadenki/SNN.git
cd SNN

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```



## **🚦 Quick Start**

We provide a unified CLI tool `snn-cli.py` to manage the entire lifecycle of the Artificial Brain.

### **1. Run the Brain Simulation**

Simulate the brain's cognitive cycle: perceiving, thinking, and acting.

```bash
# Run the brain simulation with a specific model configuration
python scripts/runners/run_brain_simulation.py \
    --model_config configs/models/small.yaml \
    --mode interactive
```

### **2. Health Check (System Verification)**

Verify that all subsystems (Learning, Memory, Reasoning) are functioning correctly.

```bash
python snn-cli.py health-check
```

### **3. Start Web UI (Unified Perception)**

Experience the "Hearing Colors" demo and interact with the brain via a web interface.

```bash
python snn-cli.py ui start
# Open http://127.0.0.1:7860
```



## **🗺️ Roadmap Status**

- ✅ **Phase 1-14:** Foundation, Cognitive Architecture, Brain OS (Completed)
- 🔄 **Phase 16 (Current): Stabilization & Evaluation** - Rigorous benchmarking, SNN-DSA integration, and safety stack implementation.
- 📅 **Phase 17:** Embodiment & Edge Deployment - Real-time operation on Jetson/Loihi.
- 📅 **Phase 18:** Social Implementation - Scaling laws and community challenges.

See [doc/ROADMAP.md](doc/ROADMAP.md) for details.



## **🤝 Contributing**

We welcome contributions! Please refer to the documentation in `doc/` for coding standards and testing protocols.



## **📜 License**

MIT License



[⬆️ Back to Top](#english)



---



## **🚀 ビジョン: "OSとしての脳"** {#japanese}

**"学習し、眠り、進化する『人道的な人工脳』の実現"**

**SNNプロジェクト**は、単なるAIモデルの開発を超え、ヒトの脳のように複数の認知モジュール(視覚、言語、運動など)がリソースと意識の座を競い合う**「ニューロモーフィックOS(脳型オペレーティングシステム)」**の構築を目指すオープンソースプロジェクトです。

私たちの目標は、省エネルギーで適応力が高く、人間と調和して共存できる**「優しいニューロモーフィックAGI」**を実現することです。従来型ANNの限界を克服するため、スパイキングニューラルネットワーク(SNN)、ニューロシンボリックAI、そして**生物学的学習則**を融合させています。



### **🏆 v16.0 の主な達成事項**

- **⚡ SFormer (T=1):** スパイク駆動でありながら、ANNと同等の推論速度(タイムステップ=1)と高精度を実現したバックボーンモデル。
- **🧠 認知アーキテクチャ:** グローバルワークスペース理論(GWT)を実装。知覚・感情・記憶などのモジュールが、エネルギー予算と重要度に基づいて「意識」を求めて競合します。
- **👁️ 統合知覚 (SNN-DSA):** 動的スパース注意機構 (Dynamic Sparse Attention) により、画像・音声・DVSなどのマルチモーダル入力を効率的に処理。
- **🤔 System 2 推論 (GRPO + Verifier):** 自己検証ループを持つ多段階推論エンジン。思考プロセスは睡眠サイクルを通じて直感(System 1)へと蒸留されます。
- **🛡️ 倫理的ガードレール:** 思考と行動をリアルタイムで監視し、危険なパターンを神経レベルで抑制する「良心」モジュール。
- **💤 睡眠と記憶固定化:** 海馬(短期記憶)から大脳皮質(長期記憶)へ、夢(Generative Replay)を通じて知識を転送・固定化します。



## **🛠️ コアコンポーネント**

### **1. 人工脳カーネル (Brain Kernel)**

- **Global Workspace:** 情報を放送し、意識を形成する中央ハブ。
- **Astrocyte Network (OS):** エネルギー(スパイク率)を監視し、恒常性(ホメオスタシス)に基づいて各モジュールの活動を制御するリソースマネージャ。
- **Neuromorphic Scheduler:** 優先度入札システムに基づき、タスクの実行順序を決定します。

### **2. 神経-記号融合 (Neuro-Symbolic)**

- **GraphRAG:** ベクトル検索とナレッジグラフを統合。対話による即座の知識修正と論理的推論を可能にします。
- **Symbol Grounding:** 神経活動パターン(アトラクタ)とシンボル(概念)を動的に結びつけ、説明可能性を担保します。

### **3. 高度な学習機能**

- **Deep Bio-Calibration:** ANNの重みをSNNの物理パラメータに変換・最適化(HSEO)する高忠実度パイプライン。
- **Causal Trace Learning:** スパイクの因果連鎖に基づく、勾配計算不要(Gradient-free)の学習則。
- **Distillation Pipeline:** System 2の深い思考過程を、軽量なSystem 1モデルに蒸留・圧縮します。



## **📦 インストール**

```bash
# リポジトリのクローン
git clone https://github.com/matsushibadenki/SNN.git
cd SNN

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```



## **🚦 クイックスタート**

統合CLIツール `snn-cli.py` を使用して、人工脳のライフサイクル全体を管理できます。

### **1. 人工脳シミュレーションの起動**

脳が知覚し、思考し、行動する様子を対話形式でシミュレートします。

```bash
# 設定を指定してシミュレーションを開始
python scripts/runners/run_brain_simulation.py \
    --model_config configs/models/small.yaml \
    --mode interactive
```

### **2. ヘルスチェック (システム診断)**

学習、推論、記憶、エージェント機能など、全サブシステムが正常に動作しているか確認します。

```bash
python snn-cli.py health-check
```

### **3. Web UIの起動**

ブラウザ上で「Hearing Colors(共感覚)」デモやチャット対話を体験できます。

```bash
python snn-cli.py ui start
# ブラウザで http://127.0.0.1:7860 にアクセス
```



## **🗺️ ロードマップ状況**

- ✅ **Phase 1-14:** 基盤構築、認知アーキテクチャ、脳型OSの確立 (完了)
- 🔄 **Phase 16 (現在): 安定化と評価** - ベンチマークの厳格化、SNN-DSAの統合、安全スタックの実装。
- 📅 **Phase 17:** 身体性とエッジ展開 - Jetson/Loihiでの実機動作とエネルギー測定。
- 📅 **Phase 18:** 社会実装 - スケーリング則の検証とコミュニティチャレンジ。

詳細は [doc/ROADMAP.md](doc/ROADMAP.md) をご覧ください。



## **🤝 コントリビューション**

バグ報告、機能提案、プルリクエストを歓迎します!  
開発に参加される場合は、`doc/` ディレクトリ内のガイドラインを参照してください。



## **📜 ライセンス**

MIT License



[⬆️ トップに戻る](#japanese)
