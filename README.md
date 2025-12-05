<div align="center">
<img src="https://via.placeholder.com/150?text=SNN5+Logo" alt="SNN5 Project Logo" width="120" height="120">

# SNN5 Project

**Next-Generation High-Efficiency Spiking Neural Networks and Artificial Brain**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🇺🇸 English](#english) | [🇯🇵 日本語](#japanese)

</div>

---

<a id="english"></a>

## 🚀 Project Overview

**"Realizing next-generation AI that thinks, learns, and adapts autonomously with high efficiency, just like the human brain."**

The **SNN5 Project** is an open-source initiative to build an "Artificial Brain" that combines biological plausibility with advanced cognitive functions, based on energy-efficient Spiking Neural Networks (SNN).

To overcome the issues of "power consumption due to massive scaling" and "difficulty in re-learning" faced by traditional ANNs, we have adopted an innovative architecture that combines **FrankenMoE** (Patchwork Mixture of Experts), **1.58-bit quantization**, and **GraphRAG**.

### 🏆 Key Achievements

- **Spiking FrankenMoE:** Established an architecture that dynamically combines existing small models (MicroLMs) as components to behave like a single massive model.
- **Neuro-symbolic GraphRAG:** Extracts knowledge triples (Subject-Predicate-Object) from natural language and stores them in a structured knowledge graph. Allows for knowledge correction without re-training.
- **1.58-bit Spiking RWKV:** An ultra-lightweight expert model that quantizes weights to {-1, 0, 1}, achieving sparsity and extreme energy efficiency.
- **High-Efficiency SNN:** Achieved a spike rate of **5.38%** (clearing the SOTA standard of < 7%).

## 🛠️ Key Features

### 1. FrankenMoE & Model Management

- **Expert Integration:** Integrates MicroLMs with different areas of expertise, such as calculation, history, and science.
- **Lifecycle Management:** `manage_models.py` automatically organizes and selects models as they increase in number.
- **1.58-bit Quantization:** Creates "disposable" ultra-lightweight models dedicated to specific tasks.

### 2. Cognitive Architecture (Artificial Brain)

- **Global Workspace Theory (GWT):** Modules for perception, emotion, and memory compete for the "seat of consciousness."
- **GraphRAG Memory System:** Combines vector search and graph structures to realize accurate memory recall and modification aligned with context.
- **Intrinsic Motivation:** Autonomously determines behavioral goals based on curiosity and boredom.

### 3. High-Efficiency SNN Learning Infrastructure

- **Predictive Coding SNN:** Unique layer structure incorporating predictive coding theory.
- **Hybrid Learning:** Supports biological learning rules such as Surrogate Gradient Method, Knowledge Distillation, and STDP/BCM.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/matsushibadenki/SNN5.git
cd SNN5

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚦 Quick Start

### 1. Building and Running FrankenMoE

```bash
# Register expert models (for demo purposes)
python scripts/register_bitnet_expert.py
python scripts/register_demo_experts.py

# Create FrankenMoE configuration file
python scripts/manage_models.py build-moe --keywords "science,history,calculation" --output my_moe.yaml

# Launch the Artificial Brain (Request a calculation task)
python scripts/runners/run_brain_simulation.py --model_config configs/models/my_moe.yaml --prompt "Calculate 1+1"
```

### 2. Dialogue with the Artificial Brain (GraphRAG Experience)

```bash
python scripts/observe_brain_thought_process.py --model_config configs/models/micro.yaml

# Input example: "Cats are felines." (Knowledge will be stored in the graph)
```

## 🗺️ Roadmap

- ✅ **Phase 1: Foundation Building & Efficiency** (Completed)
- ✅ **Phase 2: Cognitive Architecture & Knowledge Adaptation** (Completed) - GraphRAG, Symbol Grounding
- 🔄 **Phase 3: Scaling & Hybrid Intelligence** (In Progress) - FrankenMoE, 1.58bit RWKV
- 📅 **Phase 4: Practical Application & Ecosystem** (Planned)

## 📂 Directory Structure

```
SNN/
├── app/                    # Application Layer (UI, DI Container)
├── configs/                # Configuration Files (Model Definitions, MoE Configs)
├── snn_research/           # Core Library
│   ├── models/
│   │   └── experimental/
│   │       └── moe_model.py         # FrankenMoE Implementation
│   ├── cognitive_architecture/
│   │   └── rag_snn.py               # GraphRAG Implementation
│   └── models/
│       └── transformer/
│           └── spiking_rwkv.py      # 1.58bit RWKV Implementation
└── scripts/                # Execution Scripts (Training, Management, Diagnostics)
    └── manage_models.py    # Model Lifecycle Management
```

## 🤝 Contributing

Bug reports, feature suggestions, and pull requests are welcome!

If you wish to participate in development, please refer to `doc/SNN開発:プロジェクト機能テスト コマンド一覧.md` and perform tests before submitting.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the maintainers.

<div align="right">

[⬆️ Back to Top](#english)

</div>

---

<a id="japanese"></a>

## 🚀 プロジェクト概要

**"ヒトの脳のように高効率で、自律的に思考・学習・適応する次世代人工知能の実現"**

**SNN5プロジェクト**は、エネルギー効率に優れたスパイキングニューラルネットワーク (SNN) を基盤とし、生物学的妥当性と高度な認知機能を併せ持つ「人工脳」を構築するオープンソースプロジェクトです。

従来のANNが抱える「巨大化による電力消費」と「再学習の困難さ」を克服するため、**FrankenMoE** (継ぎ接ぎモデル) と **1.58bit量子化**、そして **GraphRAG** を組み合わせた革新的なアーキテクチャを採用しています。

### 🏆 主な成果

- **Spiking FrankenMoE:** 既存の小さなモデル(MicroLM)を部品として動的に結合し、単一の巨大モデルのように振る舞わせるアーキテクチャを確立。
- **ニューロシンボリック・GraphRAG:** 自然言語から知識トリプル(主語-述語-目的語)を抽出し、ナレッジグラフに構造化して保存。再学習なしで知識の修正が可能。
- **1.58bit Spiking RWKV:** 重みを {-1, 0, 1} の3値に量子化し、スパース性と極限のエネルギー効率を実現した超軽量エキスパートモデル。
- **高効率SNN:** スパイク率 **5.38%** を達成 (SOTA基準 < 7% をクリア)。

## 🛠️ 主要機能

### 1. FrankenMoE & モデル管理

- **エキスパート統合:** 計算、歴史、科学など、異なる得意分野を持つMicroLMを統合。
- **ライフサイクル管理:** 増え続けるモデルを自動で整理・選抜する `manage_models.py`。
- **1.58bit量子化:** 特定タスク専用の「使い捨て可能な」超軽量モデルの作成。

### 2. 認知アーキテクチャ (人工脳)

- **Global Workspace Theory (GWT):** 知覚、感情、記憶などのモジュールが意識の座を競い合う。
- **GraphRAG記憶システム:** ベクトル検索とグラフ構造を併用し、文脈に沿った正確な記憶の想起と修正を実現。
- **内発的動機付け:** 好奇心や退屈に基づいて自律的に行動目標を決定。

### 3. 高効率SNN学習基盤

- **Predictive Coding SNN:** 予測符号化理論を取り入れた独自のレイヤー構造。
- **ハイブリッド学習:** 代理勾配法、知識蒸留、STDP/BCMなどの生物学的学習則をサポート。

## 📦 インストール

```bash
# リポジトリのクローン
git clone https://github.com/matsushibadenki/SNN5.git
cd SNN5

# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

## 🚦 クイックスタート

### 1. FrankenMoEの構築と実行

```bash
# エキスパートモデルの登録 (デモ用)
python scripts/register_bitnet_expert.py
python scripts/register_demo_experts.py

# FrankenMoE構成ファイルの作成
python scripts/manage_models.py build-moe --keywords "science,history,calculation" --output my_moe.yaml

# 人工脳の起動 (計算タスクの依頼)
python scripts/runners/run_brain_simulation.py --model_config configs/models/my_moe.yaml --prompt "1+1の計算をして"
```

### 2. 人工脳との対話 (GraphRAG体験)

```bash
python scripts/observe_brain_thought_process.py --model_config configs/models/micro.yaml

# 入力例: "猫は猫科です" (知識がグラフに保存されます)
```

## 🗺️ ロードマップ

- ✅ **フェーズ1: 基盤構築 & 効率化** (完了)
- ✅ **フェーズ2: 認知アーキテクチャ & 知識適応** (完了) - GraphRAG、シンボルグラウンディング
- 🔄 **フェーズ3: スケーリング & ハイブリッド知能** (進行中) - FrankenMoE、1.58bit RWKV
- 📅 **フェーズ4: 実用化 & エコシステム** (計画中)

## 📂 ディレクトリ構造

```
SNN5/
├── app/                    # アプリケーション層 (UI、DIコンテナ)
├── configs/                # 設定ファイル (モデル定義、MoE設定)
├── snn_research/           # コアライブラリ
│   ├── models/
│   │   └── experimental/
│   │       └── moe_model.py         # FrankenMoE実装
│   ├── cognitive_architecture/
│   │   └── rag_snn.py               # GraphRAG実装
│   └── models/
│       └── transformer/
│           └── spiking_rwkv.py      # 1.58bit RWKV実装
└── scripts/                # 実行スクリプト (学習、管理、診断)
    └── manage_models.py    # モデルライフサイクル管理
```

## 🤝 コントリビューション

バグ報告、機能提案、プルリクエストを歓迎します!

開発に参加される場合は、`doc/SNN開発:プロジェクト機能テスト コマンド一覧.md` を参照してテストを実施してください。

## 📜 ライセンス

本プロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 📧 お問い合わせ

質問やコラボレーションのお問い合わせは、Issueを開くか、メンテナーまでご連絡ください。

<div align="right">

[⬆️ トップに戻る](#japanese)

</div>
