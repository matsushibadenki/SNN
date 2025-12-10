# **SNN Project Roadmap (v15.1: Deep Evolution + Pragmatic Breakthroughs)**

## **🎯 プロジェクト目標: "Beyond ANN" - 生きた人工脳の実用化**

本プロジェクトは、これまでに構築した認知アーキテクチャ、神経-記号融合、自律進化の機能を**「使いやすく、高速で、実証された」単一のパッケージへと収束（Convergence）**させ、世界的に影響力のあるOSSプロジェクトへと昇華させることを目指す。

v15.1では、**DeepSeek V3.2** で実証された最新の効率化・推論強化技術（DSA, GRPO, Distillation）をSNNアーキテクチャに統合し、「省電力」と「深い推論能力」の両立を加速させる。

### **🏆 コアKPI (Based on Objective.md)**

SNNがANNに対抗し、凌駕するための必須達成指標。

1. **Energy Efficiency:** ANN比 **1/50以下** の消費電力（推論時）。スパイク率 < 5% を維持。
2. **Accuracy:** 一般的なベンチマーク（CIFAR-10等）でANNのSOTAと拮抗（**95%以上**）。
3. **Reasoning:** 複雑な論理パズルやコーディングタスクにおいて、思考プロセス（Chain of Thought）を生成し解決する能力。
4. **Adaptability:** 壊滅的忘却なしに新タスクを学習（継続学習精度 **95%以上** 維持）。
5. **Real-world Deployment:** 実機（Jetson, Loihi等）での動作実証と、measurableなエネルギー削減。

---

## **🚀 7つの戦略的柱 (Strategic Pillars) - 実用化への道筋**

v15.1以降は、以下の領域に注力し、研究段階から実用段階への移行を図る。

### **1. Performance: 「シミュレーション」から「実用的な高速化」へ**

**現状の課題:** Python-based Event Loop は柔軟だが、大規模化で律速になる。

**解決策:**
- **DeepSeek Sparse Attention (DSA) on SNN:** 従来の全結合的な注意機構を廃止。SNNのスパース性を活かし、「インデクサ」を用いて必要なニューロン（トークン）のみを活性化させる動的ルーティングを実装する。これにより計算量を $O(N^2)$ から $O(k \cdot N)$ へ削減する。
- **Custom CUDA/Triton Kernels:** EventDrivenSimulator や DSA をCUDA/Tritonカーネルで書き直し、GPU上で数百万ニューロン規模を爆速で動作させる。
- **Event Buffer Compression:** スパイクイベントをRun-Length Encoding (RLE) で圧縮し、メモリ帯域を削減。

**マイルストーン:**
- Phase 8-1: Triton Spike Kernel (基本演算)
- Phase 8-2: SNN-DSA実装
- Phase 8-4: 1M+ ニューロン規模での学習デモ

---

### **2. Intelligence: 「思考する」SNNへ (Reasoning Enhancement)**

**現状の課題:** SNNは反射的な応答は得意だが、複数ステップの推論が苦手。

**解決策:**
- **GRPO (Group Relative Policy Optimization):** 自己進化エージェントの強化学習に導入。単一の正解を与えるのではなく、エージェントに複数の「思考の軌跡（Thought Trajectories）」を生成させ、それらの相対的な良さを評価することで、論理推論能力を飛躍的に向上させる。
- **Thinking Process (Thinking in Tools):** 行動（ツール使用）の直前に、必ず「思考フェーズ（内的シミュレーション）」を挟むアーキテクチャを強制する。Working Memoryモジュールに「思考トークン」を追加。
- **Multi-Step Verification:** 推論結果を別のモジュール（Verifier）で検証し、誤りを自己修正するループを実装。

**マイルストーン:**
- Phase 8-3: GRPO for Logic (GSM8K, MATH等での評価)
- Phase 9-4: Chain of Thought可視化ツール

---

### **3. Architecture: コードベースの「脱・モノリス化」と統合**

**現状の課題:** snn_researchが肥大化し、依存関係が複雑化。

**解決策:**
- **Specialist Distillation (スペシャリスト蒸留):** FrankenMoE（継ぎ接ぎの専門家モデル群）を「教師」とし、その知識を単一の効率的な **SFormer (T=1)** モデルに圧縮・統合するパイプラインを確立する。これにより、推論時のルーティングコストを削減。
- **Plugin API:** ユーザーが独自の脳領域モジュール（カスタムニューロン、学習則）を追加できるプラグイン機構。`snn_research/plugins/` ディレクトリで管理。
- **Library Decoupling:** 
  - `neuro-core`: SNN基本演算（ニューロン、STDP、シミュレータ）
  - `neuro-rag`: GraphRAG、Symbol Grounding
  - `neuro-agent`: Active Inference、Self-Evolving Agent
  - `neuro-hw`: Hardware Compiler、Neuromorphic Backend

**マイルストーン:**
- Phase 9-1: Library分割とpypi公開準備
- Phase 9-2: Distillation Pipeline実装
- Phase 9-3: Plugin API設計とドキュメント

---

### **4. Validation: 「生物学的妥当性」と「工学的有用性」の証明**

**現状の課題:** 理論は美しいが、実測データが不足している。

**解決策:**
- **Deep Bio-Calibrationの可視化:** HSEOによる最適化前後のエネルギー効率と精度のパレート曲線を可視化。ANN→SNN変換の「品質保証」を数値化。
- **ベンチマークの多様化:** 
  - DVSデータセット（N-MNIST, DVS-Gesture）での精度
  - DeepSeekが強みを持つ「論理推論・コーディング」タスク（HumanEval, MBPP）での評価
  - 継続学習ベンチマーク（CORe50, Permuted-MNIST）
- **Energy Profiling on Real Hardware:** 
  - Jetson Orin Nano / Xavier NX でのリアルタイム消費電力測定
  - Intel Loihi 2 / BrainChip Akida での動作実証（可能なら）
- **Ablation Studies:** 各コンポーネント（Sleep, Astrocyte, STDP）の個別貢献度を定量化。

**マイルストーン:**
- Phase 8-5: DVS Dataset対応
- Phase 9-5: HumanEval/MBPP評価環境構築
- Phase 10-4: Jetson実機でのエネルギー測定レポート

---

### **5. Experience: ドキュメントとデモの「魅せる化」**

**現状の課題:** 技術的には高度だが、新規ユーザーがキャッチアップしにくい。

**解決策:**
- **Interactive Web Demo:** 
  - Hugging Face Spaces / Gradioで「眠り、進化し、思考する脳」と対話できるデモを公開
  - スパイクラスター、Knowledge Graph、Attention Map をリアルタイム可視化
  - "Ask the Brain" チャットボット（GraphRAGベース）
- **Step-by-Step Tutorials:** 
  1. "15分でSNN入門" - MNIST分類
  2. "Active Inferenceエージェントの構築"
  3. "あなただけのエキスパートモデルの追加"
  4. "睡眠サイクルで知識を固定化する"
- **YouTube解説動画:** アーキテクチャの哲学、各モジュールの役割を視覚的に解説
- **論文執筆:** arXivへの投稿（"Neuromorphic OS: A Unified Framework for SNN, Neuro-Symbolic AI, and Autonomous Evolution"）

**マイルストーン:**
- Phase 9-6: Gradio Demo v1.0リリース
- Phase 9-7: Tutorial動画 (英語/日本語)
- Phase 10-5: arXiv投稿

---

### **6. Real-world Impact: 身体性と実社会連携**

**現状の課題:** シミュレーション内完結で、物理世界への影響がない。

**解決策:**
- **Embodied AI:** 
  - ROS2ブリッジによるロボット制御（TurtleBot, Fetch等）
  - Active Inferenceエージェントが、カメラ（DVS）とLiDARで環境を知覚し、ナビゲーション
- **Edge AI OS:** 
  - Jetson上でバッテリー残量をモニタリングし、Homeostatic Controllerがスパイク率を動的に調整
  - "省電力モード" では推論精度をトレードオフ
- **産業応用のPoC (Proof of Concept):**
  - 異常検知（工場の音響データからの故障予兆検出、SNNのリアルタイム性を活用）
  - ウェアラブルデバイス（脳波・心拍データからの感情推定、継続学習で個人適応）
  - ドローン制御（低遅延な視覚処理とモーター制御）

**マイルストーン:**
- Phase 10-1: ROS2連携とGridWorld → 物理ロボットへの移行
- Phase 10-2: Jetson Orin Nano実機デプロイ
- Phase 10-3: "Living AI Desktop App" (ユーザーの作業を学習し補佐)

---

### **7. 🆕 Ecosystem & Community: エコシステムの構築**

**新規追加柱:** プロジェクトの持続可能性と影響力の拡大のために。

**解決策:**
- **Model Zoo & Pre-trained Weights:** 
  - Hugging Faceに事前学習済みモデルをアップロード（snn-vit-base, snn-gpt-nano等）
  - ユーザーがFine-tuneできる形式で提供
- **Neuromorphic Hardware Partnerships:** 
  - Intel (Loihi), IBM (TrueNorth), BrainChip (Akida) 等とのコラボレーション打診
  - 彼らのSDKへの対応レイヤーを開発
- **Community Challenges:** 
  - "SNN Efficiency Challenge" - 最も省エネな推論パイプラインの構築コンテスト
  - "Neuro-Symbolic Reasoning Benchmark" - 新しい評価指標の提案
- **Academic Collaboration:** 
  - 神経科学者、認知科学者とのジョイント研究
  - 実際の脳データ（fMRI, EEG）とSNNの活動パターンの比較

**マイルストーン:**
- Phase 9-8: Hugging Face Model Hub登録
- Phase 10-6: Intel Loihi SDKサポート検討
- Phase 10-7: SNN Efficiency Challenge開催

---

## **🗓️ 実施スケジュール (Phase 8 - 11)**

### **📅 Phase 8: High-Performance Kernel & Reasoning (現在着手 - 3ヶ月)**

**目標: 計算効率の極大化と、深い推論能力の実装**

- **P8-1: Triton Spike Kernel:** EventDrivenSimulator のコアロジックをOpenAI Tritonで再実装。
- **P8-2: SNN-DSA (Dynamic Sparse Attention):** XNORベースのSDSAに「インデクサ」を導入し、必要な情報のみにアクセスする超省エネ注意機構を実装。
- **P8-3: GRPO for Logic:** ReinforcementLearnerAgent にGRPOロジックを組み込み、数学・論理パズルでの推論能力を強化。
- **P8-4: Large Scale Training:** configs/models/ultra.yaml を用いた大規模学習と、SNN Scaling Lawの実証。
- **P8-5: DVS Dataset Integration:** N-MNIST, DVS-Gestureでの精度評価パイプライン構築。

**成果物:** 高速化レポート、GSM8K精度、DVSベンチマーク結果

---

### **📅 Phase 9: Modular & Distillation Architecture (3-6ヶ月)**

**目標:** エコシステムの拡大と、巨大モデルから軽量モデルへの知識凝縮

- **P9-1: Library Decoupling:** snn_research を neuro-core, neuro-rag, neuro-agent 等に分割。
- **P9-2: Specialist Distillation Pipeline:** FrankenMoEで収集・結合した知識を、単一の軽量モデルに蒸留するフローの確立。
- **P9-3: Brain Plugin System:** ユーザー定義モジュールの入札API整備。
- **P9-4: Chain of Thought Visualization:** 推論プロセスの可視化ツール（Attention Flow, Spike Raster）
- **P9-5: Code Reasoning Benchmark:** HumanEval, MBPP対応
- **P9-6: Gradio Web Demo v1.0:** Interactive Brain Experience
- **P9-7: Tutorial Series:** 動画 + Jupyter Notebook
- **P9-8: Hugging Face Model Hub:** Pre-trained Weights公開

**成果物:** pypiパッケージ、Distillationパイプライン、Web Demo、チュートリアル動画

---

### **📅 Phase 10: Embodiment & Real-World Deployment (6-9ヶ月)**

**目標: 実世界での有用性証明（Killer Appの創出）**

- **P10-1: ROS2 Integration:** snn_research/io/actuator.py のROS2対応。物理ロボット制御デモ。
- **P10-2: Edge OS Deployment:** Jetson Orin Nano等での動作検証。
- **P10-3: "Living AI" Desktop App:** ユーザーの作業を学習・補佐するアプリ版リリース。
- **P10-4: Real Hardware Energy Profiling:** Jetsonでの実測データ取得と論文化。
- **P10-5: arXiv Paper Submission:** "Neuromorphic OS: A Unified Framework..."
- **P10-6: Neuromorphic Hardware SDK Support:** Loihi 2対応検討
- **P10-7: Community Challenge:** SNN Efficiency Challengeの企画・開催

**成果物:** ロボットデモ動画、Edgeデプロイガイド、デスクトップアプリ、論文、コンテスト

---

### **📅 Phase 11: 🆕 Scientific Validation & Scaling Laws (9-12ヶ月)**

**目標: SNNのスケーリング則の発見と、科学的妥当性の確立**

- **P11-1: SNN Scaling Law Experiments:** モデルサイズ（ニューロン数）、データセット規模、学習時間と精度・エネルギーの関係を系統的に調査。"Scaling Laws for SNNs" として論文化。
- **P11-2: Biological Plausibility Validation:** 
  - fMRI/EEGデータとSNN活動の相関分析
  - 神経科学者とのコラボレーション
  - "Brain-Inspired ≠ Brain-Equivalent" の明確化
- **P11-3: Catastrophic Forgetting Mitigation:** 
  - Sleep Consolidation + Elastic Weight Consolidation (EWC) のハイブリッド
  - 継続学習ベンチマーク (CORe50) で95%以上の精度維持を実証
- **P11-4: Multi-Agent Society Simulation:** 
  - 複数のSelf-Evolving Agentが協調・競合する環境の構築
  - 創発的知能の観察と定量化
- **P11-5: Ethical AI Framework:** 
  - 倫理的選好 (Ethical Preferences) の理論的基盤を強化
  - アライメント問題への貢献

**成果物:** Scaling Law論文、神経科学共同研究、継続学習レポート、マルチエージェントデモ

---

## **📊 成功指標 (KPIs - ターゲット)**

| 評価軸 | 現在 (v14.1) | 目標 (v15.1+) | Phase 11終了時 | 達成手段 |
|:---|:---|:---|:---|:---|
| **推論速度** | Python Loop依存 | **ANN比 5倍高速** | **10倍高速** | Custom CUDA/Triton + DSA |
| **エネルギー効率** | 推定値ベース | **実機計測で 1/50** | **1/100 (Loihi)** | Edgeデバイス + Sparse Attention |
| **推論能力** | 単純応答 | **Chain of Thought** | **GSM8K 80%+** | GRPO + Thinking Process |
| **継続学習** | 未評価 | **CORe50 90%+** | **95%+** | Sleep Consolidation + EWC |
| **モデル統合** | FrankenMoE (結合) | **Distilled SFormer** | **Single Model SOTA** | Specialist Distillation |
| **ユーザビリティ** | CLI/Script | **GUI / Web Demo** | **10k+ Downloads** | Gradio Space & Desktop App |
| **コミュニティ** | 個人プロジェクト | **GitHub 100+ Stars** | **1k+ Stars, 論文引用** | Tutorials, Papers, Challenges |

---

## **🎯 Critical Success Factors (成功の鍵)**

1. **"Show, Don't Tell":** 理論だけでなく、動くデモと実測データで証明する。
2. **Incremental Validation:** 各Phaseで小さな成功を積み重ね、コミュニティの信頼を獲得。
3. **Hardware Partnership:** Neuromorphic Hardware企業との連携が、真の省エネ実証の鍵。
4. **User-Centric Design:** 研究者だけでなく、エンジニアが「使いたい」と思うAPIとドキュメント。
5. **Scientific Rigor:** 論文発表とピアレビューによる客観的評価の獲得。

---

## **🔗 関連ドキュメント**

- [**doc/Objective.md**](doc/Objective.md): プロジェクトの数値目標と設計思想の源流。
- [**doc/Roadmap-history.md**](doc/Roadmap-history.md): 過去のマイルストーン（v2.0 - v14.1）の履歴。
- [**doc/test-command.md**](doc/test-command.md): 統合機能テストコマンド一覧。
- [**doc/SCALING_LAW.md**](doc/SCALING_LAW.md) (新規): SNNのスケーリング則に関する実験計画と結果。
- [**doc/HARDWARE_GUIDE.md**](doc/HARDWARE_GUIDE.md) (新規): Neuromorphic Hardware対応ガイド。

---

## **🚧 既知のリスクと対策**

| リスク | 影響度 | 対策 |
|:---|:---|:---|
| **Hardware非依存での省エネ実証困難** | 高 | Jetsonでの実測 + Loihi Partnership |
| **Scaling Lawが非線形** | 中 | 系統的実験 + 理論モデル構築 |
| **コミュニティ形成の遅れ** | 中 | 積極的なSNS発信 + Challengeイベント |
| **学習効率のボトルネック** | 高 | GRPO + Distillation で軽減 |
| **ドキュメント整備の工数** | 中 | Phase毎に段階的に整備 |

---

## **✨ 最終ビジョン: "The Brain as an OS"**

2026年末までに、SNNプロジェクトが以下の状態になることを目指す：

1. **Technical Excellence:** ANN比1/100のエネルギーで、SOTAに匹敵する精度。
2. **Usability:** pip install一発で動作し、チュートリアルが充実。
3. **Real-world Impact:** 少なくとも1つの産業応用（異常検知 or ロボティクス）でのPoC成功。
4. **Scientific Recognition:** Top-tier Conference (NeurIPS, ICLR) での論文採択。
5. **Community:** 1000+ GitHub Stars、複数の外部コントリビューター、活発なDiscussions。

**"ANNを超える"のは、単なる数値目標ではなく、新しいAIパラダイムの提示である。**
