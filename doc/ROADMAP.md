# **SNN Project Roadmap (v15.1: Deep Evolution)**

## **🎯 プロジェクト目標: "Beyond ANN" \- 生きた人工脳の実用化**

本プロジェクトは、これまでに構築した認知アーキテクチャ、神経-記号融合、自律進化の機能を\*\*「使いやすく、高速で、実証された」単一のパッケージへと収束（Convergence）\*\*させ、世界的に影響力のあるOSSプロジェクトへと昇華させることを目指す。

v15.1では、**DeepSeek V3.2** で実証された最新の効率化・推論強化技術（DSA, GRPO, Distillation）をSNNアーキテクチャに統合し、「省電力」と「深い推論能力」の両立を加速させる。

### **🏆 コアKPI (Based on Objective.md)**

SNNがANNに対抗し、凌駕するための必須達成指標。

1. **Energy Efficiency:** ANN比 **1/50以下** の消費電力（推論時）。スパイク率 \< 5% を維持。  
2. **Accuracy:** 一般的なベンチマーク（CIFAR-10等）でANNのSOTAと拮抗（**95%以上**）。  
3. **Reasoning:** 複雑な論理パズルやコーディングタスクにおいて、思考プロセス（Chain of Thought）を生成し解決する能力。  
4. **Adaptability:** 壊滅的忘却なしに新タスクを学習（継続学習精度 **95%以上** 維持）。

## **🚀 6つの戦略的柱 (Strategic Pillars)**

v15.1以降は、以下の領域に注力し、研究段階から実用段階への移行を図る。

### **1\. Performance: 「シミュレーション」から「実用的な高速化」へ**

* **DeepSeek Sparse Attention (DSA) on SNN:** 従来の全結合的な注意機構を廃止。SNNのスパース性を活かし、「インデクサ」を用いて必要なニューロン（トークン）のみを活性化させる動的ルーティングを実装する。これにより計算量を $O(N^2)$ から $O(k \\cdot N)$ へ削減する。  
* **Custom CUDA/Triton Kernels:** EventDrivenSimulator や DSA をCUDA/Tritonカーネルで書き直し、GPU上で数百万ニューロン規模を爆速で動作させる。

### **2\. Intelligence: 「思考する」SNNへ (Reasoning Enhancement)**

* **GRPO (Group Relative Policy Optimization):** 自己進化エージェントの強化学習に導入。単一の正解を与えるのではなく、エージェントに複数の「思考の軌跡（Thought Trajectories）」を生成させ、それらの相対的な良さを評価することで、論理推論能力を飛躍的に向上させる。  
* **Thinking Process (Thinking in Tools):** 行動（ツール使用）の直前に、必ず「思考フェーズ（シミュレーション）」を挟むアーキテクチャを強制する。

### **3\. Architecture: コードベースの「脱・モノリス化」と統合**

* **Specialist Distillation (スペシャリスト蒸留):** FrankenMoE（継ぎ接ぎの専門家モデル群）を「教師」とし、その知識を単一の効率的な **SFormer (T=1)** モデルに圧縮・統合するパイプラインを確立する。  
* **Plugin API:** ユーザーが独自の脳領域モジュールを追加できるプラグイン機構。

### **4\. Validation: 「生物学的妥当性」と「工学的有用性」の証明**

* **Deep Bio-Calibrationの可視化:** HSEOによる最適化前後のエネルギー効率と精度のパレート曲線を可視化。  
* **ベンチマークの多様化:** DVSデータセットや、DeepSeekが強みを持つ「論理推論・コーディング」タスクでの評価を追加。

### **5\. Experience: ドキュメントとデモの「魅せる化」**

* **Interactive Web Demo:** ブラウザ上で「眠り、進化し、思考する脳」と対話できるHugging Face Spacesデモ。  
* **Step-by-Step Tutorials:** SNN構築のチュートリアル完備。

### **6\. Real-world Impact: 身体性と実社会連携**

* **Embodied AI:** ROS2ブリッジによるロボット制御。  
* **Edge AI OS:** バッテリー連動型ホメオスタシスの実機検証。

## **🗓️ 実施スケジュール (Phase 8 \- 10\)**

### **📅 Phase 8: High-Performance Kernel & Reasoning (現在着手)**

**目標: 計算効率の極大化と、深い推論能力の実装**

* $$ $$  
  **P8-1: Triton Spike Kernel:** EventDrivenSimulator のコアロジックをOpenAI Tritonで再実装。  
* $$ $$  
  **P8-2: SNN-DSA (Dynamic Sparse Attention):** XNORベースのSDSAに「インデクサ」を導入し、必要な情報のみにアクセスする超省エネ注意機構を実装。  
* $$ $$  
  **P8-3: GRPO for Logic:** ReinforcementLearnerAgent にGRPOロジックを組み込み、数学・論理パズルでの推論能力を強化。  
* $$ $$  
  **P8-4: Large Scale Training:** configs/models/ultra.yaml を用いた大規模学習と、SNN Scaling Lawの実証。

### **📅 Phase 9: Modular & Distillation Architecture**

**目標:** エコシステムの拡大と、巨大モデルから軽量モデルへの知識凝縮

* $$ $$  
  **P9-1: Library Decoupling:** snn\_research を neuro-core, neuro-rag, neuro-agent 等に分割。  
* $$ $$  
  **P9-2: Specialist Distillation Pipeline:** FrankenMoEで収集・結合した知識を、単一の軽量モデルに蒸留するフローの確立。  
* $$ $$  
  **P9-3: Brain Plugin System:** ユーザー定義モジュールの入札API整備。

### **📅 Phase 10: Embodiment & Real-World Deployment**

**目標: 実世界での有用性証明（Killer Appの創出）**

* $$ $$  
  **P10-1: ROS2 Integration:** snn\_research/io/actuator.py のROS2対応。  
* $$ $$  
  **P10-2: Edge OS Deployment:** Jetson Orin Nano等での動作検証。  
* $$ $$  
  **P10-3: "Living AI" Release:** ユーザーの作業を学習・補佐するデスクトップアプリ版リリース。

## **📊 成功指標 (KPIs \- ターゲット)**

| 評価軸 | 現在 (v14.1) | 目標 (v15.1+) | 達成手段 |
| :---- | :---- | :---- | :---- |
| **推論速度** | Python Loop依存 | **ANN比 5倍高速** | Custom CUDA/Triton \+ DSA |
| **エネルギー効率** | 推定値ベース | **実機計測で 1/50** | Edgeデバイス \+ Sparse Attention |
| **推論能力** | 単純応答 | **Chain of Thought** | GRPO \+ Thinking Process |
| **モデル統合** | FrankenMoE (結合) | **Distilled SFormer** | Specialist Distillation |
| **ユーザビリティ** | CLI/Script | **GUI / Web Demo** | Gradio Space & Desktop App |

## **🔗 関連ドキュメント**

* [**doc/Objective.md**](https://www.google.com/search?q=doc/Objective.md)**:** プロジェクトの数値目標と設計思想の源流。  
* [**doc/Roadmap-history.md**](https://www.google.com/search?q=doc/Roadmap-history.md)**:** 過去のマイルストーン（v2.0 \- v14.1）の履歴。  
* [**doc/test-command.md**](https://www.google.com/search?q=doc/test-command.md)**:** 統合機能テストコマンド一覧。
