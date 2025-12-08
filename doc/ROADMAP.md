# **SNN Project Roadmap (v15.0: Convergence)**

## **🎯 プロジェクト目標: "Beyond ANN" \- 生きた人工脳の実用化**

本プロジェクトは、これまでに構築した認知アーキテクチャ、神経-記号融合、自律進化の機能を\*\*「使いやすく、高速で、実証された」単一のパッケージへと収束（Convergence）\*\*させ、世界的に影響力のあるOSSプロジェクトへと昇華させることを目指す。

最終目標は、現代のANN（人工ニューラルネットワーク）系AIが抱える課題（電力消費、壊滅的忘却、ブラックボックス性）を解決し、**「省電力・適応的・説明可能」な次世代AI標準**としての地位を確立することである。

### **🏆 コアKPI (Based on Objective.md)**

SNNがANNに対抗し、凌駕するための必須達成指標。

1. **Energy Efficiency:** ANN比 **1/50以下** の消費電力（推論時）。スパイク率 \< 5% を維持。  
2. **Accuracy:** 一般的なベンチマーク（CIFAR-10等）でANNのSOTAと拮抗（**95%以上**）。  
3. **Real-time Performance:** Pythonシミュレーションではなく、実用的な推論速度（**\< 10ms レイテンシ**）。  
4. **Adaptability:** 壊滅的忘却なしに新タスクを学習（継続学習精度 **95%以上** 維持）。

## **🚀 5つの戦略的柱 (Strategic Pillars)**

v15.0以降は、以下の5つの領域に注力し、研究段階から実用段階への移行を図る。

### **1\. Performance: 「シミュレーション」から「実用的な高速化」へ**

Pythonベースのシミュレーションから脱却し、ハードウェアの性能を極限まで引き出す実装へ移行する。

* **Custom CUDA/Triton Kernels:** EventDrivenSimulator や SDSA (Spike-Driven Self-Attention) をCUDA/Tritonカーネルで書き直し、GPU上で数百万ニューロン規模を爆速で動作させる。  
* **Massive Scaling:** 数Billionパラメータ規模での学習を実施し、SNNにおける **Scaling Law（規模則）** を実証する。

### **2\. Architecture: コードベースの「脱・モノリス化」**

機能追加により複雑化したシステムを整理し、他プロジェクトでも利用可能な形にする。

* **Neuro-Symbolic Engineの分離:** RAGSystem や SymbolGrounding を独立したライブラリ（例: neuro-rag）として切り出し、SNN以外でも利用可能にする。  
* **Plugin APIの整備:** NeuromorphicScheduler の入札（Bid）システムを標準化し、ユーザーが独自の「脳領域（モジュール）」を簡単に追加できるプラグイン機構を構築する。

### **3\. Validation: 「生物学的妥当性」と「工学的有用性」の証明**

「脳に似ている」だけでなく「工学的に優れている」ことを定量的に証明する。

* **Deep Bio-Calibrationの可視化:** HSEOによる最適化前後のエネルギー効率と精度のパレート曲線を可視化し、単なるファインチューニングとの違いを明確にする。  
* **ベンチマークの多様化:** DVS（Event Camera）データセットや長期時系列予測など、SNNが本質的に有利な領域でのリーダーボードを確立する。

### **4\. Experience: ドキュメントとデモの「魅せる化」**

誰もが直感的に「人工脳」の凄さを体験できる環境を整備する。

* **Interactive Web Demo:** snn-cli だけでなく、ブラウザ上で「眠り、進化する脳」と対話できるHugging Face Spacesデモを公開する。  
* **Step-by-Step Tutorials:** 「SNNとは？」から「独自の脳を作る」まで、段階的なJupyter Notebookチュートリアルを完備する。

### **5\. Real-world Impact: 身体性と実社会連携**

シミュレーション空間（GridWorld）から現実世界へ飛び出す。

* **Embodied AI:** ROS2ブリッジを実装し、実際のロボットアームやドローンをSNNで低遅延制御する。  
* **Edge AI OS:** Raspberry PiやJetsonのバッテリー残量や温度と連動し、思考レベルを動的に調整する「省エネ脳OS」を実機で動作させる。

## **🗓️ 実施スケジュール (Phase 8 \- 10\)**

### **📅 Phase 8: High-Performance Kernel (現在着手)**

**目標: CUDA/Triton実装による圧倒的な高速化とスケーリング実証**

* \[ \] **P8-1: Triton Spike Kernel:** EventDrivenSimulator のコアロジックをOpenAI Tritonで再実装。  
* \[ \] **P8-2: Optimized SDSA:** XNORベースのアテンション計算をCUDAカーネル化し、Transformerに対する速度優位性を証明。  
* \[ \] **P8-3: Large Scale Training:** configs/models/large.yaml を用いたWikiText-103完走と、SNN Scaling Lawのレポート公開。

### **📅 Phase 9: Modular & Plugin Architecture**

**目標:** エコシステムの拡大とユーザビリティの向上

* \[ \] **P9-1: Library Decoupling:** snn\_research を neuro-core, neuro-rag, neuro-agent 等のパッケージに分割。  
* \[ \] **P9-2: Brain Plugin System:** ユーザーが BrainModule クラスを継承して独自の入札ロジックを実装できるAPIの整備。  
* \[ \] **P9-3: Interactive Spaces Demo:** 学習・睡眠サイクルを可視化するWebデモの公開。

### **📅 Phase 10: Embodiment & Real-World Deployment**

**目標: 実世界での有用性証明（Killer Appの創出）**

* \[ \] **P10-1: ROS2 Integration:** snn\_research/io/actuator.py のROS2対応。  
* \[ \] **P10-2: Edge OS Deployment:** Jetson Orin Nano等での動作検証と、バッテリー連動型ホメオスタシスの実証。  
* \[ \] **P10-3: "Living AI" Release:** PCのバックグラウンドで常駐し、ユーザーの作業を学習・補佐するデスクトップアプリ版のリリース。

## **📊 成功指標 (KPIs \- ターゲット)**

| 評価軸 | 現在 (v14.1) | 目標 (v15.0+) | 達成手段 |
| :---- | :---- | :---- | :---- |
| **推論速度** | Python Loop依存 | **ANN比 5倍高速** | Custom CUDA/Triton Kernels |
| **エネルギー効率** | 推定値ベース | **実機計測で 1/50** | Edgeデバイスへのデプロイ & 電力計測 |
| **モデル規模** | Small (M parameters) | **Large (B parameters)** | Distributed Training & Scaling Law |
| **ユーザビリティ** | CLI/Script | **GUI / Web Demo** | Gradio Space & Desktop App |
| **適用範囲** | テキスト/画像分類 | **ロボ制御/実時間応答** | ROS2 & Event Camera Integration |

## **🔗 関連ドキュメント**

* [**doc/Objective.md**](https://www.google.com/search?q=doc/Objective.md)**:** プロジェクトの数値目標と設計思想の源流。  
* [**doc/Roadmap-history.md**](https://www.google.com/search?q=doc/Roadmap-history.md)**:** 過去のマイルストーン（v2.0 \- v14.1）の履歴。  
* \*\*\[doc/SNN開発：プロジェクト機能テスト コマンド一覧.md\](doc/SNN開発：プロジェクト機能テスト コ
