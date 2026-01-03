# **SNN Roadmap v22.0 — *Brain v22: The Collective Mind***

## **Humane Neuromorphic AGI (Async Event-Driven Architecture)**

**目的**: 人間とロボットが共存し、互いに尊重し合い、豊かな日常を作るための"優しい"ニューロモーフィックAI（SNNベース）。生物学的一貫性・工学的有用性・倫理設計を同時に満たすこと。

v22.0 コンセプト:  
「個」としての知能（Brain v21）を確立した次のステップとして、複数のエージェントが経験と知識を共有し、高度な意思決定を行う「集合知（Collective Intelligence）」を構築する。従来の「単一アーキテクチャ」から脱却し、脳の機能局在に基づいたハイブリッド構成（Spikformer \+ Mamba）を基盤に、社会性を持つAIへと進化する。

## **目次**

1. [ビジョンと原則 (Design Pillars)](https://www.google.com/search?q=%231-%E3%83%93%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%A8%E5%8E%9F%E5%89%87-design-pillars)  
2. [技術的詳細 (Architecture & Stack)](https://www.google.com/search?q=%232-%E6%8A%80%E8%A1%93%E7%9A%84%E8%A9%B3%E7%B4%B0-architecture--stack)  
3. [主要なマイルストーン (Phase 1 \- Phase 5\)](https://www.google.com/search?q=%233-%E4%B8%BB%E8%A6%81%E3%81%AA%E3%83%9E%E3%82%A4%E3%83%AB%E3%82%B9%E3%83%88%E3%83%BC%E3%83%B3-phase-1---phase-5)  
4. [実装ガイドライン (Implementation Guide)](https://www.google.com/search?q=%234-%E5%AE%9F%E8%A3%85%E3%82%AC%E3%82%A4%E3%83%89%E3%83%A9%E3%82%A4%E3%83%B3-implementation-guide)  
5. [倫理と安全性のガードレール](https://www.google.com/search?q=%235-%E5%80%AB%E7%90%86%E3%81%A8%E5%AE%89%E5%85%A8%E6%80%A7%E3%81%AE%E3%82%AC%E3%83%BC%E3%83%89%E3%83%AC%E3%83%BC%E3%83%AB)

## **1\. ビジョンと原則 (Design Pillars)**

本プロジェクトの設計思想は、脳の構造的特徴を工学的メリット（省電力・高速性・適応力）に変換することにある。

### **Design Pillars**

1. **Spatio-Temporal Hybrid (Mamba-Transformer Synergy)**:  
   * **空間 (Space / Visual Cortex)**: 視覚野には **Spiking Transformer (Spikformer)** を採用。画像内の大域的な文脈（Global Context）や物体間の関係性をSelf-Attention機構で高精度に捉える。  
   * **時間 (Time / Prefrontal Cortex)**: 前頭前野には **BitSpikeMamba** を採用。無限に続く時系列データからの推論、長期記憶の圧縮、文脈の維持を線形計算量 $O(N)$ で低コストに行う。  
2. **Collective Intelligence (Swarm of Brains)**:  
   * 単体の知能だけでなく、複数のエージェントが協調する群知能を重視。  
   * **Liquid Democracy**: 自信がないタスクは信頼できるエキスパートに委任する流動的な意思決定。  
3. **System 2 to System 1**:  
   * 言語モデル(System 2)による論理的・分析的な推論結果を、SNN(System 1)の直感的・反射的な回路へと「蒸留」する。  
4. **Sleep & Dream**:  
   * オンライン学習に伴う「破滅的忘却」を防ぐため、オフライン（睡眠）時に記憶の再生（Replay）と整理（Consolidation）を行う。

## **2\. 技術的詳細 (Architecture & Stack)**

### **A. Neural Engine (Core)**

* **Language**: Python 3.10+ (PyTorch, **SpikingJelly** based)  
* **Spike Model**: **DA-LIF (Dual Adaptive Leaky Integrate-and-Fire)**  
  * 学習可能な時定数（膜電位・電流）を持ち、タスクに応じて保持時間を最適化。  
* **Encoding**: **Hybrid Temporal-8-Bit Coding**  
  * 8bit画素値をビットプレーン分解し、時間軸に展開。M4チップ上で15ms以下の超低レイテンシ推論を実現。  
* **Collective Protocol**: **Liquid Democracy**  
  * Reputation（評判）スコアに基づく重み付き投票と、動的な委任メカニズム。

### **B. Cognitive Architecture (Brain Modules)**

#### **1\. Visual Cortex (Perception)**

* **Architecture**: Spiking Transformer (Spikformer)  
* **Role**: 視覚入力の処理と特徴抽出。

#### **2\. Prefrontal Cortex (Reasoning)**

* **Architecture**: BitSpikeMamba  
* **Role**: 長期記憶の保持と時系列推論。

#### **3\. Swarm Interface (Communication)**

* **Architecture**: Liquid Democracy Protocol  
* **Role**: 他のエージェントとの提案共有、投票、委任の管理。

## **3\. 主要なマイルストーン (Phase 1 \- Phase 5\)**

### **Phase 1: Foundation & Efficiency (v20.0 \- v20.2)**

$$完了$$

* 既存のANNをSNNへ変換し、基礎的な効率化を達成。  
  * BitSpikeMamba プロトタイプ作成。  
  * SpikingCNN による基礎視覚の実装。

### **Phase 2: Cognitive Cycle & Adaptation (v20.3 \- v20.4)**

$$完了$$

* 自律的に賢くなるサイクルの確立。  
  * SleepConsolidator (睡眠学習)。  
  * Multimodal Integration (SpikingVLM)。

### **Phase 3: Embodiment & Real-World (v21.0 \- v21.5)**

$$完了$$

* **成果**:  
  * **High-Fidelity Vision**: CNNからSpikformerへの完全移行完了。  
  * **Reflexive Control**: Hybrid Temporal-8-Bit Codingにより、M4チップ(MPS)上で **15.08ms** の超低レイテンシ動作を達成。  
  * **Brain v21**: 視覚-思考-行動ループの統合完了。

### **Phase 4: Collective Intelligence (v22.0 \- )**

$$現在$$

* **目標**: 複数のSNNエージェント間の協調、知識共有、および民主的な意思決定プロトコルの確立。  
* **主な実装項目**:  
  * **Liquid Democracy Protocol**:  
    * エージェントが自身の「専門性」と「自信」に基づいて投票を行う。  
    * 自信がない場合は、信頼できるエキスパートエージェントに投票権を動的に「委任」する。  
  * **Knowledge Distillation Network**:  
    * 経験豊富なエージェント（Teacher）から新規エージェント（Student）への知識蒸留。  
  * **Swarm Deployment**:  
    * エッジデバイス群（ドローン、ロボット）での分散協調動作シミュレーション。

### **Phase 5: Neuromorphic OS (v23.0 \- )**

$$将来$$

* ハードウェアレベルでのリソース管理と、OSとしてのSNNカーネルの実装。  
* ニューロモーフィックチップへの直接実装。

## **4\. 実装ガイドライン (Implementation Guide)**

### **Step 1: 集合知プロトコルの構築 (Liquid Democracy)**

1. snn\_research/collective/liquid\_democracy.py を作成。  
2. Proposal, Vote, Delegation のデータ構造を定義。  
3. 評判（Reputation）スコアの更新ロジックを実装。

### **Step 2: スワームシミュレーション (The Hive Mind)**

1. scripts/runners/run\_collective\_intelligence.py を作成。  
2. 複数のBrain Agent（Generalist, Expert, Newbie）を生成。  
3. 難易度の高いタスクに対して、個々の限界を「委任」によって乗り越えるデモを実施する。

### **Step 3: 知識の共有と蒸留 (Knowledge Sharing)**

1. エキスパートエージェントの重みや経験バッファを、ニュービーエージェントへ効率的に転送・蒸留するパイプラインを構築する。

## **5\. 倫理と安全性のガードレール**

* **Democracy Guardrails**: 投票操作や悪意ある結託（Collusion）を防ぐための評判監視システム。  
* **Asimov-aligned Guardrails**: ロボット工学三原則を現代的に解釈したハードコードされた制約。  
* **Value Alignment**: 「優しさ」「協力」を報酬関数として明示的に組み込む。