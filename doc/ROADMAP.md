# **SNN Project Roadmap (v15.3: The Great Unification \- Achieved)**

## **🎯 プロジェクト目標: "Beyond ANN" \- 生きた人工脳の実用化**

本プロジェクトは、これまでに構築した認知アーキテクチャ、神経-記号融合、自律進化の機能を\*\*「使いやすく、高速で、実証された」単一のパッケージへと収束（Convergence）\*\*させ、世界的に影響力のあるOSSプロジェクトへと昇華させることを目指す。

v15.3では、v15.2で掲げた「五感を最初からスパイクとして混ぜ合わせる大統一アーキテクチャ」のコア実装を完了し、**実際に「音を聞いて色を感じる（Hearing Colors）」共感覚的連合学習の実証に成功**した。これより、プロジェクトはシミュレーション環境から実世界（身体性・エッジデバイス）への展開フェーズへと移行する。

### **🏆 コアKPI (Based on Objective.md)**

SNNがANNに対抗し、凌駕するための必須達成指標。

1. **Energy Efficiency:** ANN比 **1/50以下** の消費電力（推論時）。スパイク率 \< 5% を維持。  
2. **Accuracy:** 一般的なベンチマーク（CIFAR-10等）でANNのSOTAと拮抗（**95%以上**）。  
   * **Update:** SNN-DSA Transformerにより、小規模タスクで98.5%を達成。  
3. **Reasoning:** 複雑な論理パズルやコーディングタスクにおいて、思考プロセス（Chain of Thought）を生成し解決する能力。  
   * **Update:** GRPOにより、思考の軌跡の自己改善を確認。  
4. **Adaptability:** 壊滅的忘却なしに新タスクを学習（継続学習精度 **95%以上** 維持）。  
5. **Real-world Deployment:** 実機（Jetson, Loihi等）での動作実証と、measurableなエネルギー削減。  
6. **Cross-Modal Association:** 異なる感覚（音と映像など）を教師なしで連合学習し、片方の入力からもう片方を想起する能力。  
   * **Update:** **達成 (v15.3)** \- LACにより音声のみから視覚概念の想起に成功 (類似度 0.86+)。

## **🚀 8つの戦略的柱 (Strategic Pillars) \- 実用化への道筋**

v15.3以降は、確立されたコア技術を実世界に応用し、エコシステムを拡大させることに注力する。

### **1\. Performance: 「シミュレーション」から「実用的な高速化」へ**

**現状:** SNN-DSAの実装により、計算量の $O(N^2) \\to O(k \\cdot N)$ 削減を達成。

**解決策:**

* **Custom CUDA/Triton Kernels:** EventDrivenSimulator や DSA をCUDA/Tritonカーネルで書き直し、GPU上で数百万ニューロン規模を爆速で動作させる。  
* **Event Buffer Compression:** スパイクイベントをRun-Length Encoding (RLE) で圧縮し、メモリ帯域を削減。

**マイルストーン:**

* \[x\] Phase 8-2: SNN-DSA実装 (Dynamic Sparse Attention)  
* \[ \] Phase 8-1: Triton Spike Kernel (基本演算)  
* \[ \] Phase 8-4: 1M+ ニューロン規模での学習デモ

### **2\. Intelligence: 「思考する」SNNへ (Reasoning Enhancement)**

**現状:** GRPOの実装により、エージェントが複数の思考パスを生成・評価可能に。

**解決策:**

* **Thinking Process (Thinking in Tools):** 行動（ツール使用）の直前に、必ず「思考フェーズ（内的シミュレーション）」を挟むアーキテクチャを強制する。Working Memoryモジュールに「思考トークン」を追加。  
* **Multi-Step Verification:** 推論結果を別のモジュール（Verifier）で検証し、誤りを自己修正するループを実装。

**マイルストーン:**

* \[x\] Phase 8-3: GRPO for Logic (論理パズルでの探索能力向上を確認)  
* \[ \] Phase 9-4: Chain of Thought可視化ツール

### **3\. Architecture: コードベースの「脱・モノリス化」と統合**

**現状:** snn\_researchが肥大化しつつあるが、モジュール間の依存関係は整理されつつある。

**解決策:**

* **Specialist Distillation (スペシャリスト蒸留):** FrankenMoE（継ぎ接ぎの専門家モデル群）を「教師」とし、その知識を単一の効率的な **SFormer (T=1)** モデルに圧縮・統合するパイプラインを確立する。  
* **Library Decoupling:** ライブラリの分割とパッケージ化。

**マイルストーン:**

* \[ \] Phase 9-1: Library分割とpypi公開準備  
* \[ \] Phase 9-2: Distillation Pipeline実装  
* \[ \] Phase 9-3: Plugin API設計とドキュメント

### **4\. Validation: 「生物学的妥当性」と「工学的有用性」の証明**

**現状:** DVSデータセット対応と、基本的な学習能力の実証が完了。

**解決策:**

* **Deep Bio-Calibrationの可視化:** HSEOによる最適化前後のエネルギー効率と精度のパレート曲線を可視化。  
* **Energy Profiling on Real Hardware:** Jetson Orin Nano / Xavier NX でのリアルタイム消費電力測定。

**マイルストーン:**

* \[x\] Phase 8-5: DVS Dataset対応 (N-MNIST Pipeline)  
* \[ \] Phase 9-5: HumanEval/MBPP評価環境構築  
* \[ \] Phase 10-4: Jetson実機でのエネルギー測定レポート

### **5\. Experience: ドキュメントとデモの「魅せる化」**

**現状:** Gradioによる「Hearing Colors」デモが稼働中。

**解決策:**

* **Interactive Web Demo:** Hugging Face Spacesへのデプロイ。  
* **Tutorials:** SNN-DSAやGRPOの使い方を解説するNotebookの整備。

**マイルストーン:**

* \[x\] Phase 9-6: Gradio Web Demo v1.0 (Unified Perception)  
* \[ \] Phase 9-7: Tutorial動画 (英語/日本語)  
* \[ \] Phase 10-5: arXiv投稿

### **6\. Real-world Impact: 身体性と実社会連携**

**現状:** シミュレーション内完結。

**解決策:**

* **Embodied AI:** ROS2ブリッジによるロボット制御（TurtleBot, Fetch等）。  
* **Edge AI OS:** Jetson上でバッテリー残量をモニタリングし、Homeostatic Controllerがスパイク率を動的に調整。

**マイルストーン:**

* \[ \] Phase 10-1: ROS2連携とGridWorld → 物理ロボットへの移行  
* \[ \] Phase 10-2: Jetson Orin Nano実機デプロイ  
* \[ \] Phase 10-3: "Living AI Desktop App"

### **7\. Ecosystem & Community**

* **Model Zoo:** 事前学習済みSNN-DSAモデルの公開。  
* **Neuromorphic Hardware Partnerships:** Loihi / SpiNNaker へのポーティング。

**マイルストーン:**

* \[ \] Phase 9-8: Hugging Face Model Hub登録  
* \[ \] Phase 10-6: Intel Loihi SDKサポート検討

### **8\. 🆕 Unified Perception: 五感を切り離さない「液状」統合**

**現状:** LACの実装と共感覚デモの成功により、基本原理の実証が完了。

**解決策:**

* **Universal Spike Encoder:** あらゆる入力を統一フォーマットに変換。  
* **Liquid Association Cortex (LAC):** 異なるモダリティを統合し、STDPで連合学習。

**マイルストーン:**

* \[x\] Phase 8-6: Universal Spike Encoder (Image/Audio/Text/DVS to Spike)  
* \[x\] Phase 9-9: Liquid Association Cortex (LAC) 実装  
* \[x\] Phase 9-10: Cross-Modal Association Demo (Hearing Colors)

## **🗓️ 実施スケジュール (Current Status: Phase 9 Mid-Stage)**

### **✅ Phase 8: High-Performance Kernel & Reasoning (完了)**

**目標: 計算効率の極大化と、深い推論能力の実装**

* \[ \] **P8-1: Triton Spike Kernel:** (保留中: 最適化フェーズへ移行)  
* \[x\] **P8-2: SNN-DSA (Dynamic Sparse Attention):** 実装完了。Transformerモデルでの学習検証済 (Accuracy 98.5%)。  
* \[x\] **P8-3: GRPO for Logic:** ReinforcementLearnerAgentへの実装完了。思考軌跡の自己改善を確認。  
* \[ \] **P8-4: Large Scale Training:** (リソース依存のためスキップ/並行)  
* \[x\] **P8-5: DVS Dataset Integration:** N-MNISTパイプライン構築完了。  
* \[x\] **P8-6: Universal Spike Encoder:** 画像・音声・テキスト・DVSの統一エンコーダ実装完了。

**成果物:** snn\_research/core/layers/dsa.py, snn\_research/agent/reinforcement\_learner\_agent.py, snn\_research/io/universal\_encoder.py

### **🏃 Phase 9: Modular, Distillation & Unified Perception (進行中)**

**目標:** エコシステムの拡大、モデル圧縮、そして五感の統合

* \[ \] **P9-1: Library Decoupling:** パッケージ構成の整理。  
* \[ \] **P9-2: Specialist Distillation Pipeline:** 未着手。  
* \[ \] **P9-3: Brain Plugin System:** 未着手。  
* \[ \] **P9-4: Chain of Thought Visualization:** 未着手。  
* \[ \] **P9-5: Code Reasoning Benchmark:** 未着手。  
* \[x\] **P9-6: Gradio Web Demo v1.0:** app/unified\_perception\_demo.py 実装完了。  
* \[ \] **P9-7: Tutorial Series:** 計画中。  
* \[ \] **P9-8: Hugging Face Model Hub:** 計画中。  
* \[x\] **P9-9: Liquid Association Cortex (LAC):** マルチモーダルリザーバの実装完了。  
* \[x\] **P9-10: Cross-Modal Demo:** "Hearing Colors" デモ成功 (類似度改善 \+0.22)。

**成果物:** snn\_research/core/networks/liquid\_association\_cortex.py, app/unified\_perception\_demo.py

### **📅 Phase 10: Embodiment & Real-World Deployment (Next Focus)**

**目標: 実世界での有用性証明（Killer Appの創出）**

* **P10-1: ROS2 Integration:** snn\_research/io/actuator.py のROS2対応。物理ロボット制御デモ。  
* **P10-2: Edge OS Deployment:** Jetson Orin Nano等での動作検証。  
* **P10-3: "Living AI" Desktop App:** ユーザーの作業を学習・補佐するアプリ版リリース。  
* **P10-4: Real Hardware Energy Profiling:** Jetsonでの実測データ取得と論文化。  
* **P10-5: arXiv Paper Submission:** "Neuromorphic OS: A Unified Framework..."  
* **P10-6: Neuromorphic Hardware SDK Support:** Loihi 2対応検討  
* **P10-7: Community Challenge:** SNN Efficiency Challengeの企画・開催

### **📅 Phase 11: 🆕 Scientific Validation & Scaling Laws (Future)**

**目標: SNNのスケーリング則の発見と、科学的妥当性の確立**

* **P11-1: SNN Scaling Law Experiments:** モデルサイズと精度の関係調査。  
* **P11-2: Biological Plausibility Validation:** 神経科学データとの照合。  
* **P11-3: Catastrophic Forgetting Mitigation:** 継続学習ベンチマーク。  
* **P11-4: Multi-Agent Society Simulation:** マルチエージェント社会シミュレーション。  
* **P11-5: Ethical AI Framework:** 倫理的アライメント。

## **📊 成功指標 (KPIs \- Progress)**

| 評価軸 | 現在 (v15.3) | 目標 (v15.2+) | Phase 11終了時 | 達成手段 |
| :---- | :---- | :---- | :---- | :---- |
| **推論速度** | **改善 (DSA導入)** | ANN比 5倍高速 | 10倍高速 | Custom CUDA/Triton \+ DSA |
| **エネルギー効率** | 推定値ベース | 実機計測で 1/50 | 1/100 (Loihi) | Edgeデバイス \+ Sparse Attention |
| **推論能力** | **GRPO実装済** | Chain of Thought | GSM8K 80%+ | GRPO \+ Thinking Process |
| **継続学習** | 未評価 | CORe50 90%+ | 95%+ | Sleep Consolidation \+ EWC |
| **マルチモーダル** | **LAC実装済 (Hearing Colors)** | 早期統合 (Early Fusion) | 共感覚的想起 | Liquid Association Cortex |
| **モデル統合** | FrankenMoE (結合) | Distilled SFormer | Single Model SOTA | Specialist Distillation |
| **ユーザビリティ** | **Gradio Demo稼働** | GUI / Web Demo | 10k+ Downloads | Gradio Space & Desktop App |
| **コミュニティ** | 個人プロジェクト | GitHub 100+ Stars | 1k+ Stars, 論文引用 | Tutorials, Papers, Challenges |

## **🔗 関連ドキュメント**

* [**doc/Objective.md**](https://www.google.com/search?q=doc/Objective.md): プロジェクトの数値目標。  
* [**doc/Roadmap-history.md**](https://www.google.com/search?q=doc/Roadmap-history.md): 過去のマイルストーン履歴。  
* [**doc/test-command.md**](https://www.google.com/search?q=doc/test-command.md): 統合機能テストコマンド一覧。

## **✨ Next Milestone Focus**

Phase 10: Embodiment (身体性)  
シミュレーションから現実世界へ飛び出す時が来た。  
ROS2連携とJetsonへのデプロイにより、「SNNが実世界の身体を動かす」デモの実現を目指す。
