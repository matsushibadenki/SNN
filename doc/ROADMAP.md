# **SNN Roadmap v20.4 — *Brain v20: The Bit-Spike Convergence***

## **Humane Neuromorphic AGI (Async Event-Driven Architecture)**

**目的**: 人間とロボットが共存し、互いに尊重し合い、豊かな日常を作るための"優しい"ニューロモーフィックAI（SNNベース）を、実装可能な工程に落とし込む。生物学的一貫性・工学的有用性・倫理設計を同時に満たすこと。

**v20.4 現在の到達点 (Current Status)**:

* ✅ **1.58bit BitSpike Architecture**: 推論コストを極限まで削減する BitSpikeMamba の実装完了。  
* ✅ **SpikingVLM**: 視覚 (SpikingCNN) と言語 (Transformer) を統合し、画像からテキストを生成する能力を獲得。  
* ✅ **Self-Evolution Loop**: System 2 (熟考) の解法を System 1 (直感) に蒸留する自律進化サイクルの実証完了。  
* ✅ **Sleep Consolidation**: 睡眠による記憶整理とGenerative Replayの実装。  
* ✅ **Reliability & QA**: 網羅的なテストマニュアルと自動テストスイートの整備による品質保証体制の確立。

**Next Step (v20.4)**: 実世界環境（Web/ロボットシミュレータ）での自律動作テストと、モデルの大規模化（Scale-Up）。

## **目次**

1. [ビジョンと原則](https://www.google.com/search?q=%231-%E3%83%93%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%A8%E5%8E%9F%E5%89%87)  
2. [主要なマイルストーン (Phase 1 \- Phase 4\)](https://www.google.com/search?q=%232-%E4%B8%BB%E8%A6%81%E3%81%AA%E3%83%9E%E3%82%A4%E3%83%AB%E3%82%B9%E3%83%88%E3%83%BC%E3%83%B3-phase-1---phase-4)  
3. [技術的詳細 (Architecture & Stack)](https://www.google.com/search?q=%233-%E6%8A%80%E8%A1%93%E7%9A%84%E8%A9%B3%E7%B4%B0-architecture--stack)  
4. [倫理と安全性のガードレール](https://www.google.com/search?q=%234-%E5%80%AB%E7%90%86%E3%81%A8%E5%AE%89%E5%85%A8%E6%80%A7%E3%81%AE%E3%82%AC%E3%83%BC%E3%83%89%E3%83%AC%E3%83%BC%E3%83%AB)  
5. [Implementation Status (v20.4)](https://www.google.com/search?q=%235-implementation-status-v204)

## **1\. ビジョンと原則**

### **Core Philosophy**

* **Bio-Plausible**: 脳の構造（スパイク、可塑性、睡眠、ゆらぎ）を模倣するだけでなく、その工学的利点（省電力、適応性）を活かす。  
* **Human-Centric**: 人間の代替ではなく、人間の幸福を最大化するパートナーとしてのAGI。  
* **Thermodynamic Efficient**: 熱力学的な効率（最小作用の原理）を計算の基礎に置く。

### **Design Pillars**

1. **Bit-Spike Hybrid**: 重い行列演算は1.58bit量子化で、通信と発火はスパイクで行うハイブリッド構成。  
2. **System 2 to System 1**: 言語モデル(System 2)の論理的推論を、SNN(System 1)の直感的反射回路へ蒸留する。  
3. **Sleep & Dream**: オンライン学習のカタストロフィ・忘却を防ぐため、オフライン（睡眠）での記憶定着を行う。

## **2\. 主要なマイルストーン (Phase 1 \- Phase 4\)**

### **Phase 1: Foundation & Efficiency (v20.0 \- v20.2) \[完了\]**

* **目標**: 既存のANN（Transformer/Mamba）をSNNへ変換し、推論コストを1/10以下にする。  
* **成果物**:  
  * BitSpikeMamba: 1.58bit重み \+ スパイク活性化関数の基本モデル。  
  * SpikingCNN: 視覚野の効率的実装。  
  * Neuro-Symbolic Bridge: 言語による制御インターフェース。

### **Phase 2: Cognitive Cycle & Adaptation (v20.3 \- v20.4) \[現在\]**

* **目標**: 学習・推論・睡眠・進化のサイクルを回し、自律的に賢くなるシステムを作る。  
* **成果物**:  
  * SleepConsolidator: 睡眠時の記憶再生モジュール。  
  * ThoughtDistillation: 推論プロセス（Chain of Thought）のSNNへの蒸留。  
  * Multimodal Integration: 視覚と言語の融合 (SpikingVLM)。  
  * Quality Assurance: 網羅的テストと品質保証体制の確立。

### **Phase 3: Embodiment & Real-World (v21.0 \- ) \[次回\]**

* **目標**: 身体性を持ち、物理環境またはWeb環境でタスクをこなす。  
* **予定**:  
  * Web Agent: ブラウザを操作し、情報を収集・判断するSNN。  
  * Robotic Control: アームや移動ロボットの低レイテンシ制御。  
  * Active Inference: 能動的推論による環境探索。

### **Phase 4: Collective Intelligence (v22.0 \- ) \[将来\]**

* **目標**: 複数のSNNエージェントが協調し、社会的な知性を形成する。  
* **予定**:  
  * Liquid Democracy Protocol: エージェント間の流動的な合意形成。  
  * Shared Knowledge Graph: 分散型知識共有。

## **3\. 技術的詳細 (Architecture & Stack)**

### **A. Neural Engine (Core)**

* **Language**: Python 3.10+ (PyTorch, JAX optional)  
* **Spike Model**: LIF (Leaky Integrate-and-Fire) with Adaptive Threshold  
* **Learning Rule**:  
  * **STDP (Spike-Timing-Dependent Plasticity)**: ミリ秒単位の局所学習。  
  * **R-STDP (Reward-Modulated)**: 強化学習的要素。  
  * **BPTT (Backpropagation Through Time)**: オフラインでの教師あり学習。

### **B. Cognitive Architecture (Brain Modules)**

1. **Prefrontal Cortex (System 2\)**:  
   * BitNet b1.58 ベースの推論エンジン。複雑な計画立案を担当。  
2. **Basal Ganglia (Action Selection)**:  
   * ドーパミン（報酬予測誤差）に基づく行動選択ゲート。  
3. **Hippocampus (Memory)**:  
   * 短期記憶バッファ。睡眠時に長期記憶（大脳皮質）へ転送。  
4. **Visual Cortex (Perception)**:  
   * Spiking ResNet / Vision Mamba による特徴抽出。

## **4\. 倫理と安全性のガードレール**

* **Asimov-aligned Guardrails**: ロボット工学三原則を現代的に解釈したハードコードされた制約。  
* **Entropy Monitoring**: システムのエントロピー（混乱度）を監視し、異常時に緊急停止するサーキットブレーカー。  
* **Value Alignment**: 「優しさ」「協力」を報酬関数として明示的に組み込む。

## **5\. Implementation Status (v20.4)**

### **A. Core Models**

* ✅ **BitSpikeMamba** \* ✅ 1.58bit Weight Quantization  
  * ✅ Spiking Activation Function  
  * ✅ Mamba (SSM) Backbone  
* ✅ **Spiking Vision** \* ✅ SpikingCNN 最適化  
  * ✅ AsyncVisionAdapter (Real Input)

### **B. Cognitive Functions (System 2\)**

* ✅ **World Model** \* ✅ SpikingWorldModel (Latent Dynamics)  
  * ✅ 脳内シミュレーションによる行動計画  
* ✅ **Meta-Cognition** \* ✅ エントロピー監視によるSystem 1/2切り替え  
  * ✅ ThoughtDistillationManager (System 2 \-\> System 1 蒸留)

### **C. Biological Lifecycle**

* ✅ **Sleep & Memory** \* ✅ SleepConsolidator (Generative Replay)  
  * ✅ 睡眠サイクルの自律スケジューリング  
* ✅ **Homeostasis** \* ✅ OnChipSelfCorrector (推論時適応)

### **D. Embodiment & Environment (Next Focus)**

* \[ \] **Web Agent** \* \[ \] 自律Webブラウジング機能の統合 (Playwright/Selenium)  
  * \[ \] 検索結果からの知識抽出と学習  
* \[ \] **Physical Agent** \* \[ \] GridWorldから連続値制御環境 (MuJoCo/PyBullet) への移行  
  * \[ \] マルチモーダルセンサー統合 (Audio/Tactile)

### **E. Tools & Scripts**

* ✅ **Unified Demos** \* ✅ run\_brain\_evolution.py (完全統合デモ)  
  * ✅ run\_vlm\\\_sleep.py (睡眠デモ)  
* ✅ **Quality Assurance**  
  * ✅ Comprehensive Test Manual (doc/comprehensive\_test\_manual.md)  
  * ✅ Automated Test Suite (scripts/run\_all\_tests.py)  
* \[ \] **Dashboard** \* \[ \] 脳内状態のリアルタイム可視化 (Web UI)

## **6\. 開発フローと貢献**

開発は main ブランチで行わず、機能ごとの feature/ ブランチで行い、Pull Requestベースでマージする。  
コミットメッセージは Conventional Commits に準拠すること。
