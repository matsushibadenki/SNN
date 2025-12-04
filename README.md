# **SNN5 Project: 次世代高効率スパイキングニューラルネットワークと人工脳**

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

git clone

$$https://github.com/matsushibadenki/SNN5.git$$  
(https://github.com/matsushibadenki/SNN5.git)

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

## **🗺️ ロードマップ**

* ✅ **Phase 1: 基盤構築と高効率化** (完了)  
* ✅ **Phase 2: 認知アーキテクチャと知識適応** (完了) \- GraphRAG, Symbol Grounding  
* 🔄 **Phase 3: スケーリングとハイブリッド知能** (進行中) \- FrankenMoE, 1.58bit RWKV  
* 📅 **Phase 4: 実用化とエコシステム** (予定)

## **📂 ディレクトリ構造**

* app/: アプリケーション層 (UI, DIコンテナ)  
* configs/: 設定ファイル (モデル定義, MoE構成)  
* snn\_research/: コアライブラリ  
  * models/experimental/moe\_model.py: FrankenMoEの実装  
  * cognitive\_architecture/rag\_snn.py: GraphRAGの実装  
  * models/transformer/spiking\_rwkv.py: 1.58bit RWKVの実装  
* scripts/: 実行スクリプト (学習, 管理, 診断)  
  * manage\_models.py: モデルライフサイクル管理

## **🤝 コントリビューション**

バグ報告、機能提案、プルリクエストは大歓迎です！

開発に参加される方は、doc/SNN開発：プロジェクト機能テスト コマンド一覧.md を参照してテストを行ってください。

## **📜 ライセンス**

MIT ライセンス
