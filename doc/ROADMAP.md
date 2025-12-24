# **SNN Roadmap v20.4 — *Brain v20: The Bit-Spike Convergence***

## **Humane Neuromorphic AGI (Async Event-Driven Architecture)**

**目的**: 人間とロボットが共存し、互いに尊重し合い、豊かな日常を作るための"優しい"ニューロモーフィックAI（SNNベース）を、実装可能な工程に落とし込む。生物学的一貫性・工学的有用性・倫理設計を同時に満たすこと。

**v20.4 現在の到達点 (Current Status)**:

* ✅ **1.58bit BitSpike Architecture**: 推論コストを極限まで削減する BitSpikeMamba の実装完了。  
* ✅ **SpikingVLM**: 視覚 (SpikingCNN) と言語 (Transformer) を統合し、画像からテキストを生成する能力を獲得。  
* ✅ **Self-Evolution Loop**: System 2 (熟考) の解法を System 1 (直感) に蒸留する自律進化サイクルの実証完了。  
* ✅ **Sleep Consolidation**: 睡眠による記憶整理とGenerative Replayの実装。

**Next Step (v20.4)**: 実世界環境（Web/ロボットシミュレータ）での自律動作テストと、モデルの大規模化（Scale-Up）。

## **目次**

1. [ビジョンと原則](https://www.google.com/search?q=%231-%E3%83%93%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%A8%E5%8E%9F%E5%89%87)  
2. [目標（KPI）と受け入れ基準](https://www.google.com/search?q=%232-%E7%9B%AE%E6%A8%99kpi%E3%81%A8%E5%8F%97%E3%81%91%E5%85%A5%E3%82%8C%E5%9F%BA%E6%BA%96)  
3. [開発フェーズ詳細](https://www.google.com/search?q=%233-%E9%96%8B%E7%99%BA%E3%83%95%E3%82%A7%E3%83%BC%E3%82%BA%E8%A9%B3%E7%B4%B0)  
4. [アーキテクチャ設計 (Brain v20)](https://www.google.com/search?q=%234-%E3%82%A2%E3%83%BC%E3%82%AD%E3%83%86%E3%82%AF%E3%83%81%E3%83%A3%E8%A8%AD%E8%A8%88-brain-v20)  
5. [実装タスクリスト](https://www.google.com/search?q=%235-%E5%AE%9F%E8%A3%85%E3%82%BF%E3%82%B9%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%88)  
6. [勉強すべき技術・必読論文リスト](https://www.google.com/search?q=%236-%E5%8B%89%E5%BC%B7%E3%81%99%E3%81%B9%E3%81%8D%E6%8A%80%E8%A1%93%E3%83%BB%E5%BF%85%E8%AA%AD%E8%AB%96%E6%96%87%E3%83%AA%E3%82%B9%E3%83%88)  
7. [開発ルール・注意事項](https://www.google.com/search?q=%237-%E9%96%8B%E7%99%BA%E3%83%AB%E3%83%BC%E3%83%AB%E3%83%BB%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A0%85)

## **1\. ビジョンと原則**

### **Core Philosophy**

1. **Be Kind (優しくあれ)**: 全ての判断基準に「他者への貢献と害の回避」を組み込む。  
2. **Be Efficient (効率的であれ)**: 1.58bit量子化とスパイク通信により、脳のような圧倒的なエネルギー効率を目指す。  
3. **Be Autonomous (自律的であれ)**: 外部からの指示を待つのではなく、内発的動機（好奇心）と代謝（バッテリー管理）に基づいて行動する。

## **2\. 目標（KPI）と受け入れ基準**

### **KPIs (Key Performance Indicators)**

* **推論効率**: 従来ANN比で **1/100** のエネルギー消費（BitNet \+ SNNスパース性）。  
* **適応速度**: 未知のタスクに対し、5回以内の試行錯誤（Few-shot）で適応可能であること。  
* **自律稼働**: 人間の介入なしに **24時間以上**、Web空間またはシミュレータ内で活動し続けること。

## **3\. 開発フェーズ詳細**

### **Phase 20.1 \- 20.3: The Foundation (Completed)**

* **テーマ**: 1.58bit化と自律進化サイクルの確立  
* **期間**: 2025/12  
* **成果**:  
  * BitSpikeLinear / BitSpikeMamba の実装  
  * SpikingVLM (Vision-Language) の統合  
  * MetaCognitiveSNN (System 1/2 Switching)  
  * SleepConsolidator (Generative Replay)

### **Phase 20.4: Real-World Embodiment (Current)**

* **テーマ**: 身体性と実環境への適応  
* **期間**: 2026/01 \-  
* **目標**:  
  * **Web Browsing**: Headless Chromium を用いた自律的な情報収集と学習。  
  * **Embodiment**: ロボットアームや移動ロボット（シミュレータ）の制御。  
  * **Long-Term Memory**: RAG (Retrieval-Augmented Generation) とエピソード記憶の完全統合。

### **Phase 21: The Social Brain (Future)**

* **テーマ**: 社会性と他者理解（Theory of Mind）  
* **目標**:  
  * 複数のエージェント間での協力・対話。  
  * 人間の感情推定と共感的な応答生成。

## **4\. アーキテクチャ設計 (Brain v20)**

### **System 1 (Fast / Intuitive)**

* **Model**: BitSpikeMamba (1.58bit Recurrent SNN)  
* **Role**: 無意識的・反射的な推論。視覚認識、運動制御、日常会話。  
* **Characteristics**: 低遅延、低消費電力、並列処理。

### **System 2 (Slow / Logical)**

* **Model**: SpikingWorldModel \+ Tree of Thoughts  
* **Role**: 意識的・論理的な思考。計画立案、未知の問題解決、自己修正。  
* **Characteristics**: 逐次処理、シミュレーションベース、高精度。

### **Kernel (OS Layer)**

* **AsyncArtificialBrain**: asyncio ベースの非同期イベントバス。  
* **Astrocyte Network**: エネルギー（トークン/電力）の管理とグローバルな抑制制御。

## **5\. 実装タスクリスト**

### **A. Core Architecture (System 1\)**

* ✅ **BitNet (1.58bit) Integration**  
  * ✅ BitSpikeLinear 実装  
  * ✅ BitSpikeMamba 実装  
* ✅ **Vision Integration**  
  * ✅ SpikingCNN 最適化  
  * ✅ AsyncVisionAdapter (Real Input)

### **B. Cognitive Functions (System 2\)**

* ✅ **World Model**  
  * ✅ SpikingWorldModel (Latent Dynamics)  
  * ✅ 脳内シミュレーションによる行動計画  
* ✅ **Meta-Cognition**  
  * ✅ エントロピー監視によるSystem 1/2切り替え  
  * ✅ ThoughtDistillationManager (System 2 \-\> System 1 蒸留)

### **C. Biological Lifecycle**

* ✅ **Sleep & Memory**  
  * ✅ SleepConsolidator (Generative Replay)  
  * ✅ 睡眠サイクルの自律スケジューリング  
* ✅ **Homeostasis**  
  * ✅ OnChipSelfCorrector (推論時適応)

### **D. Embodiment & Environment (Next Focus)**

* \[ \] **Web Agent**  
  * \[ \] 自律Webブラウジング機能の統合 (Playwright/Selenium)  
  * \[ \] 検索結果からの知識抽出と学習  
* \[ \] **Physical Agent**  
  * \[ \] GridWorldから連続値制御環境 (MuJoCo/PyBullet) への移行  
  * \[ \] マルチモーダルセンサー統合 (Audio/Tactile)

### **E. Tools & Scripts**

* ✅ **Unified Demos**  
  * ✅ run\_brain\_evolution.py (完全統合デモ)  
  * ✅ run\_vlm\_sleep.py (睡眠デモ)  
* \[ \] **Dashboard**  
  * \[ \] 脳内状態のリアルタイム可視化 (Web UI)

## **6\. 勉強すべき技術・必読論文リスト**

1. **BitNet b1.58**: *The Era of 1-bit LLMs* (Microsoft, 2024\) \- **実装完了**  
2. **World Models**: *Recurrent World Models Facilitate Policy Evolution* (Ha & Schmidhuber) \- **実装完了**  
3. **Active Inference**: 自由エネルギー原理と能動的推論  
4. **Liquid Neural Networks**: 時定数が適応的に変化するSNN

## **7\. 開発ルール・注意事項**

### **コーディング規約**

1. **非同期ファースト**: asyncio を活用し、ブロッキング処理を避ける。  
2. **型安全性**: mypy による型チェックを通過すること。  
3. **ドキュメント**: 新機能には必ずデモスクリプト (scripts/runners/) を同梱する。

**合言葉**: *"Make it wake up, make it dream, make it kind."*