\<\!--  
Title: SNN5 Project README (English & Japanese)  
Description: 英語と日本語のセクションを単一ファイルに統合し、アンカーリンクでナビゲーション可能なREADMEです。  
\--\>  
\<div align="center"\>  
\<img src="https://www.google.com/search?q=https://via.placeholder.com/150%3Ftext%3DSNN5%2BLogo" alt="SNN5 Project Logo" width="120" height="120"\>  
\<h1\>SNN5 Project\</h1\>  
\<p\>  
\<strong\>Next-Generation High-Efficiency Spiking Neural Networks and Artificial Brain\</strong\>  
\</p\>  
\<p\>  
\<\!-- Language Navigation \--\>  
\<a href="\#english"\>\<strong\>🇺🇸 English\</strong\>\</a\> | \<a href="\#japanese"\>\<strong\>🇯🇵 日本語\</strong\>\</a\>  
\</p\>  
\</div\>  
\<\!-- English Section \--\>

\<a id="english"\>\</a\>

## **🚀 Project Overview**

**"Realizing next-generation AI that thinks, learns, and adapts autonomously with high efficiency, just like the human brain."**

The **SNN5 Project** is an open-source initiative to build an "Artificial Brain" that combines biological plausibility with advanced cognitive functions, based on energy-efficient Spiking Neural Networks (SNN).

To overcome the issues of "power consumption due to massive scaling" and "difficulty in re-learning" faced by traditional ANNs, we have adopted an innovative architecture that combines **FrankenMoE** (Patchwork Mixture of Experts), **1.58-bit quantization**, and **GraphRAG**.

### **🏆 Key Achievements**

* **Spiking FrankenMoE:** Established an architecture that dynamically combines existing small models (MicroLMs) as components to behave like a single massive model.  
* **Neuro-symbolic GraphRAG:** Extracts knowledge triples (Subject-Predicate-Object) from natural language and stores them in a structured knowledge graph. Allows for knowledge correction without re-training.  
* **1.58-bit Spiking RWKV:** An ultra-lightweight expert model that quantizes weights to $\\\\{-1, 0, 1\\\\}$, achieving sparsity and extreme energy efficiency.  
* **High-Efficiency SNN:** Achieved a spike rate of **5.38%** (clearing the SOTA standard of \< 7%).

## **🛠️ Key Features**

### **1\. FrankenMoE & Model Management**

* **Expert Integration:** Integrates MicroLMs with different areas of expertise, such as calculation, history, and science.  
* **Lifecycle Management:** manage\_models.py automatically organizes and selects models as they increase in number.  
* **1.58-bit Quantization:** Creates "disposable" ultra-lightweight models dedicated to specific tasks.

### **2\. Cognitive Architecture (Artificial Brain)**

* **Global Workspace Theory (GWT):** Modules for perception, emotion, and memory compete for the "seat of consciousness."  
* **GraphRAG Memory System:** Combines vector search and graph structures to realize accurate memory recall and modification aligned with context.  
* **Intrinsic Motivation:** Autonomously determines behavioral goals based on curiosity and boredom.

### **3\. High-Efficiency SNN Learning Infrastructure**

* **Predictive Coding SNN:** Unique layer structure incorporating predictive coding theory.  
* **Hybrid Learning:** Supports biological learning rules such as Surrogate Gradient Method, Knowledge Distillation, and STDP/BCM.

## **📦 Installation**

\# Clone the repository  
git clone \[https://github.com/matsushibadenki/SNN5.git\](https://github.com/matsushibadenki/SNN5.git)  
cd SNN5

\# Create a virtual environment  
python \-m venv .venv

\# Activate the virtual environment  
source .venv/bin/activate  \# Windows: .venv\\Scripts\\activate

\# Install dependencies  
pip install \-r requirements.txt

## **🚦 Quick Start**

### **1\. Building and Running FrankenMoE**

\# Register expert models (for demo purposes)  
python scripts/register\_bitnet\_expert.py  
python scripts/register\_demo\_experts.py

\# Create FrankenMoE configuration file  
python scripts/manage\_models.py build-moe \--keywords "science,history,calculation" \--output my\_moe.yaml

\# Launch the Artificial Brain (Request a calculation task)  
python scripts/runners/run\_brain\_simulation.py \--model\_config configs/models/my\_moe.yaml \--prompt "Calculate 1+1"

### **2\. Dialogue with the Artificial Brain (GraphRAG Experience)**

python scripts/observe\_brain\_thought\_process.py \--model\_config configs/models/micro.yaml

\# Input example: "Cats are felines." (Knowledge will be stored in the graph)

## **🗺️ Roadmap**

* ✅ **Phase 1: Foundation Building & Efficiency** (Completed)  
* ✅ **Phase 2: Cognitive Architecture & Knowledge Adaptation** (Completed) \- GraphRAG, Symbol Grounding  
* 🔄 **Phase 3: Scaling & Hybrid Intelligence** (In Progress) \- FrankenMoE, 1.58bit RWKV  
* 📅 **Phase 4: Practical Application & Ecosystem** (Planned)

## **📂 Directory Structure**

* app/: Application Layer (UI, DI Container)  
* configs/: Configuration Files (Model Definitions, MoE Configs)  
* snn\_research/: Core Library  
  * models/experimental/moe\_model.py: Implementation of FrankenMoE  
  * cognitive\_architecture/rag\_snn.py: Implementation of GraphRAG  
  * models/transformer/spiking\_rwkv.py: Implementation of 1.58bit RWKV  
* scripts/: Execution Scripts (Training, Management, Diagnostics)  
  * manage\_models.py: Model Lifecycle Management

## **🤝 Contribution**

Bug reports, feature suggestions, and pull requests are welcome\!  
If you wish to participate in development, please refer to doc/SNN開発：プロジェクト機能テスト コマンド一覧.md and perform tests.

## **📜 License**

MIT License

\<div align="right"\>  
\<a href="\#english"\>⬆️ Back to Top\</a\>  
\</div\>  
\<\!-- Japanese Section \--\>

\<a id="japanese"\>\</a\>

## **🚀 プロジェクト概要**

**"ヒトの脳のように高効率で、自律的に思考・学習・適応する次世代人工知能の実現"**

SNN5プロジェクトは、エネルギー効率に優れたスパイキングニューラルネットワーク (SNN) を基盤とし、生物学的妥当性と高度な認知機能を併せ持つ「人工脳」を構築するオープンソースプロジェクトです。

従来のANNが抱える「巨大化による電力消費」と「再学習の困難さ」を克服するため、FrankenMoE (継ぎ接ぎモデル) と 1.58bit量子化、そして GraphRAG を組み合わせた革新的なアーキテクチャを採用しています。

### **🏆 主な成果 (Current Achievements)**

* **Spiking FrankenMoE:** 既存の小さなモデル（MicroLM）を部品として動的に結合し、単一の巨大モデルのように振る舞わせるアーキテクチャを確立。  
* **ニューロシンボリック・GraphRAG:** 自然言語から知識トリプル（主語-述語-目的語）を抽出し、ナレッジグラフに構造化して保存。再学習なしで知識の修正が可能。  
* **1.58bit Spiking RWKV:** 重みを $\\\\{-1, 0, 1\\\\}$ の3値に量子化し、スパース性と極限のエネルギー効率を実現した超軽量エキスパートモデル。  
* **高効率SNN:** スパイク率 **5.38%** を達成 (SOTA基準 \< 7% をクリア)。

## **🛠️ 主要機能**

### **1\. FrankenMoE & モデル管理**

* **エキスパート統合:** 計算、歴史、科学など、異なる得意分野を持つMicroLMを統合。  
* **ライフサイクル管理:** 増え続けるモデルを自動で整理・選抜する manage\_models.py。  
* **1.58bit量子化:** 特定タスク専用の「使い捨て可能な」超軽量モデルの作成。

### **2\. 認知アーキテクチャ (Artificial Brain)**

* **Global Workspace Theory (GWT):** 知覚、感情、記憶などのモジュールが意識の座を競い合う。  
* **GraphRAG記憶システム:** ベクトル検索とグラフ構造を併用し、文脈に沿った正確な記憶の想起と修正を実現。  
* **内発的動機付け:** 好奇心や退屈に基づいて自律的に行動目標を決定。

### **3\. 高効率SNN学習基盤**

* **Predictive Coding SNN:** 予測符号化理論を取り入れた独自のレイヤー構造。  
* **ハイブリッド学習:** 代理勾配法、知識蒸留、STDP/BCMなどの生物学的学習則をサポート。

## **📦 インストール**

git clone \[https://github.com/matsushibadenki/SNN5.git\](https://github.com/matsushibadenki/SNN5.git)  
cd SNN5

python \-m venv .venv  
source .venv/bin/activate \# Windows: .venv\\Scripts\\activate

pip install \-r requirements.txt

## **🚦 クイックスタート**

### **1\. FrankenMoEの構築と実行**

\# エキスパートモデルの登録 (デモ用)  
python scripts/register\_bitnet\_expert.py  
python scripts/register\_demo\_experts.py

\# FrankenMoE構成ファイルの作成  
python scripts/manage\_models.py build-moe \--keywords "science,history,calculation" \--output my\_moe.yaml

\# 人工脳の起動 (計算タスクの依頼)  
python scripts/runners/run\_brain\_simulation.py \--model\_config configs/models/my\_moe.yaml \--prompt "1+1の計算をして"

### **2\. 人工脳との対話 (GraphRAG体験)**

python scripts/observe\_brain\_thought\_process.py \--model\_config configs/models/micro.yaml

\# 入力例: "猫は猫科です" (知識がグラフに保存されます)  
