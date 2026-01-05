# **SNN Project Roadmap History**

このドキュメントは、プロジェクトの主要なマイルストーン、バージョンアップ、および戦略的転換点の履歴を記録する。

## **目次**

1. **v22.0: The Collective Mind (Current)**  
2. **v21.0: The Spatio-Temporal Hybrid**  
3. **v20.4: The Bit-Spike Convergence**  
4. **v20.3: Cognitive Cycle & Adaptation**  
5. **v20.0: Foundation & Efficiency**  
6. **v14.1: 人工脳OSの確立と神経-記号進化**  
7. **v14.0: 脳型OSとハードウェアネイティブへの移行**  
8. **v13.0: 自律的デジタル生命体と倫理的進化**  
9. **v12.0: 能動的知能と身体性への転換**  
10. **v11.0: スケーリングと** $T=1$ **統合**  
11. v10.0: 修正可能な適応的知能の実現  
    ... (以下、過去のバージョン)

## **v22.0: The Collective Mind (Current)**

Date: 2025-12-31  
Status: Phase 4 Active / Phase 3 Completed

### **概要**

個体としての知能（Brain v21）の確立を受け、開発の焦点を「集合知（Collective Intelligence）」へと移行。複数のエージェントが協調し、民主的に意思決定を行うための基盤を整備した。

### **主な変更点**

* **Phase 3 (Embodiment) 完了**:  
  * **超低レイテンシ達成**: M4チップ(MPS)およびHybrid Temporal-8-Bit Codingにより、推論レイテンシ **15.08ms** を記録。  
  * **視覚野の刷新**: CNNベースからSpikformerへの完全移行を完了。  
* **Phase 4 (Collective Intelligence) 開始**:  
  * **Liquid Democracy Protocol**: 専門性と自信に基づく流動的な投票・委任プロトコルを策定・実装。  
  * **Swarm Simulation**: 難易度の高いタスクにおける集団的合意形成のデモ run\_collective\_intelligence.py を実装。

## **v21.0: The Spatio-Temporal Hybrid**

Date: 2025-12-01  
Status: Phase 3 Active

### **概要**

従来の単一アーキテクチャから、脳の機能局在に基づいたハイブリッド構成（Transformer \+ Mamba）へ進化。物理世界やWeb環境での「即応性」と「認識力」の両立を目指した。

### **主な変更点**

* **アーキテクチャ刷新**:  
  * **Visual Cortex**: Spiking Transformer (Spikformer) を採用し、大域的な文脈理解を強化。  
  * **Prefrontal Cortex**: BitSpikeMamba を採用し、低コストな長期記憶を実現。  
* **ニューロンモデル**: 学習可能な時定数を持つ **DA-LIF (Dual Adaptive LIF)** を導入。  
* **符号化方式**: レイテンシを劇的に短縮する **Hybrid Temporal-8-Bit Coding** を考案。

## **v20.4: The Bit-Spike Convergence**

Date: 2024-2025  
Status: Phase 2 Extended

### **概要**

（詳細情報取得不可のためタイトルのみ記載。詳細はバージョン管理履歴を参照してください。）

## **v20.3: Cognitive Cycle & Adaptation**

Date: 2025-10-15  
Status: Phase 2 Completed

### **概要**

「学習し続ける脳」を実現するため、睡眠サイクルとマルチモーダル統合機能を実装。

### **主な変更点**

* **Sleep & Dream**: オンライン学習の破滅的忘却を防ぐための、睡眠時の記憶整理（Consolidation）プロセスを実装。  
* **Multimodal Integration**: 視覚情報と言語情報を統合する SpikingVLM のプロトタイプを作成。

## **v20.0: Foundation & Efficiency**

Date: 2025-08-01  
Status: Phase 1 Completed

### **概要**

プロジェクトの立ち上げ。既存のANN資産をSNNへ変換し、ニューロモーフィック計算による効率化（省電力・高速化）の基礎を固めた。

### **主な変更点**

* **BitSpikeMamba**: 1.58bit量子化とスパイクSNNを組み合わせた基本モデルの実装。  
* **SpikingJelly導入**: PyTorchベースのSNNフレームワークを用いた開発環境の構築。  
* **ベンチマーク**: CIFAR-10 / MNIST におけるANN対比での効率性検証。

## **v14.1: 人工脳OSの確立と神経-記号進化 (完了)**

* **ステータス:** 完了。

### **1\. 概要**

（詳細情報取得不可のためプレースホルダー）

## **v11.0: スケーリングと $T=1$ 統合 (完了)**

* **ステータス:** 完了。大規模言語モデル (LLM) の知見を取り入れたスケーリング則の検証と、超高速推論 ($T=1$) の実現。

### **1\. 概要**

SNNの課題であった「時間ステップ数による推論遅延」を解消するため、時間ステップ $T=1$ でも動作する直接符号化 (Direct Coding) 技術を確立。また、TransformerアーキテクチャをSNNに移植し、大規模化への道を拓きました。

### **2\. 実現された主要機能**

* **Spiking Transformer / RWKV:** アテンション機構のスパイク化と、RNN形式の線形Transformerの実装。  
* **Direct Coding (**$T=1$**):** 入力を時間的に展開せず、一度のパスで推論を行う高速化技術。

## **v10.0: 修正可能な適応的知能の実現 (完了)**

* **ステータス:** 完了。メタ認知と自己修正メカニズムのプロトタイプ実装。

### **1\. 概要**

学習後のモデルに対しても、特定のエラーを修正したり、新しい知識を追加したりするための「適応的メカニズム」を導入しました。

### **2\. 実現された主要機能**

* **Error Detection & Correction:** 予測ミスを検知し、局所的に可塑性を高めるメタ学習。