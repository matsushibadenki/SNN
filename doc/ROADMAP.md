# **SNN Project Roadmap (v14.0)**

## **🎯 プロジェクト目標: "Artificial Brain" の実現**

人間の脳の動作原理（スパイク、可塑性、階層構造、睡眠、意識）を工学的に模倣し、従来のANN（誤差逆伝播法）への依存を排除した、超低消費電力かつ自律的に進化し続ける次世代AIアーキテクチャを実現する。

コアフィロソフィー: "From Simulation to Embodiment"  
シミュレーション上の知能から、物理的制約（エネルギー、時間、身体性）を持つ実存的な知能への転換を目指す。

## **🚀 戦略的柱 (Strategic Pillars)**

1. **Deep Bio-Calibration (深層生物学的校正):**  
   * ANNの高度な性能をSNNの初期状態として移植し、生物学的可塑性（STDP/BCM）で環境に適応させる「いいとこ取り」戦略。  
2. **Neuro-Symbolic Feedback Loop (神経-記号還流):**  
   * 言語的知識（GraphRAG）を睡眠フェーズでシナプス重み（SNN）に「コンパイル」し、説明可能な知識を直感的な反射へと昇華させる。  
3. **Neuromorphic OS (脳型オペレーティングシステム):**  
   * 単なるモデルではなく、複数の認知モジュール（視覚、言語、運動）のリソースと競合を管理するOSとしての脳アーキテクチャ。

## **🗓️ 実施スケジュールとマイルストーン**

### **✅ Phase 1: Foundation & Efficiency (完了)**

* **成果:** 代理勾配法による学習基盤、ANN-SNN変換、DVSデータ対応。  
* **主要技術:** train.py, ann\_to\_snn\_converter.py

### **✅ Phase 2: Cognitive Architecture (完了)**

* **成果:** 認知コンポーネント（海馬、扁桃体、前頭前野）の統合、意識のブロードキャスト（GWT）。  
* **主要技術:** ArtificialBrain, GlobalWorkspace

### **✅ Phase 3: Scaling & Hybrid Intelligence (完了)**

* **成果:** $T=1$ 高速推論 (SFormer)、FrankenMoEによるモデル統合、1.58bit量子化。  
* **主要技術:** SFormer, SpikingFrankenMoE, BitSpikingRWKV

### **✅ Phase 4: Autonomous Intelligence (完了/最適化中)**

* **成果:** 能動的推論 (Active Inference) による自律行動、HSEOによる自己進化、倫理的選好。  
* **主要技術:** ActiveInferenceAgent, SelfEvolvingAgentMaster, HSEO

### **🔄 Phase 5: Neuro-Symbolic Evolution (現在 \- 6ヶ月)**

**目標:** 言語的知識と神経的直感の双方向ループを完成させ、再学習なしで「教えれば賢くなる」脳を実現する。

* **\[ \] Sleep Consolidation System (睡眠時記憶固定化)**  
  * GraphRAGに蓄積された知識トリプルを、睡眠中にSNNへの入力として再生（Replay）する。  
  * Causal Trace Learning (V2) を用い、エピソード記憶を長期的なシナプス重みに焼き付ける。  
* **\[ \] Neuro-Symbolic Grounding (深層記号接地)**  
  * SymbolGrounding を強化し、SNNの隠れ層の活動パターン（アトラクタ）と、GraphRAGの概念ノードを動的にリンクさせる。  
  * 「赤い」という言葉を聞くだけで、視覚野のV4エリアが発火するようなトップダウン信号の実装。  
* **\[ \] Real-time Knowledge Editing**  
  * 対話による訂正が即座にGraphRAGに反映され、次回の睡眠サイクルでSNNの振る舞い（バイアス）を修正するパイプラインの確立。

### **📅 Phase 6: Hardware Native Transition (6ヶ月 \- 1年)**

**目標:** GPUシミュレーションの限界（速度・電力）を突破するため、計算基盤をニューロモーフィックハードウェア仕様に完全移行する。

* **\[ \] Deep Bio-Calibration Pipeline**  
  * ECL (Error Compensation Learning) を拡張し、大規模ANN（Llama/Mistralクラス）の重みを、スパイキングニューロンの物理パラメータ（膜抵抗、閾値）に高精度にマッピングする自動校正システム。  
* **\[ \] Event-Driven Kernels**  
  * PyTorch/CUDAへの依存を減らし、スパイクイベントが発生した時のみ計算を行うカスタムCUDAカーネルまたはFPGAロジックへの書き換え。  
* **\[ \] On-Chip Plasticity**  
  * 推論（Forward）中にローカルメモリ内で重みを更新する、ハードウェアフレンドリーな学習則（R-STDPの簡易版）の実装。

### **📅 Phase 7: The "Brain" OS (1年以降)**

**目標:** 複数のAIエージェントやモジュールが、単一のハードウェアリソースを共有・競合しながら動作する「脳型OS」の構築。

* **\[ \] Neuromorphic Scheduler (Astrocyte Manager)**  
  * AstrocyteNetwork をOSのスケジューラに昇格。  
  * 各領域（視覚野、言語野）のエネルギー消費（スパイク率）を監視し、グルコース（計算リソース）を動的に配分・制限する。  
* **\[ \] Multi-Agent Competition**  
  * 「視覚野エージェント」「言語野エージェント」「運動野エージェント」が独立したプロセスとして並列動作。  
  * Global Workspace を介した通信のみで協調し、全体として一つの人格を形成するマイクロサービス・アーキテクチャ。  
* **\[ \] Self-Hosted Evolution**  
  * AI自身が自分のソースコード（または回路構成）を理解し、リコンパイルして自己修正する完全自律ループ。

## **🛠️ 技術スタック要件 (Future)**

* **Core:** PyTorch \-\> **Lava / Norse / Custom CUDA Kernels**  
* **Memory:** FAISS/NetworkX \-\> **Hyperdimensional Computing (HDC) Vector Stores**  
* **Optimization:** Optuna \-\> **Evolutionary Strategies (ES) on Hardware**  
* **Interface:** Gradio \-\> **Direct Neural Interface (Spike Streams)**

## **📊 成功指標 (KPIs)**

1. **Knowledge Retention:** 睡眠サイクル後のタスク遂行精度が、学習前より向上していること。  
2. **Energy Efficiency:** 同等のタスクにおいて、GPUベースのANNと比較して **100倍** 以上のエネルギー効率。  
3. **Adaptability:** 未知のタスクに対し、数回の提示（Few-shot）と睡眠を経て適応できること。
