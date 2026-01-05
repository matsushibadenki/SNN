# **SNN Project: ANN-SNN高忠実度変換 & Deep Bio-Calibration パイプライン (v14.0)**

## **🚀 プロジェクト概要**

本プロジェクトは、既存の高性能なANNモデル（LLMやCNN）を、エネルギー効率に優れたスパイクニューラルネットワーク（SNN）モデルに変換するための**高忠実度自動変換パイプライン**を提供します。

v14.0では、単なる重みの変換にとどまらず、**Deep Bio-Calibration（深層生物学的校正）** の概念を導入しました。これは、ANNの性能を「初期状態」としてSNNに移植し、その後、生物学的可塑性（STDP/BCM/Causal Trace）を用いて環境やハードウェア制約に適応させる、**ハイブリッド進化戦略**の中核を担う技術です。

### **🏆 主要な機能強化**

| 機能 | 目的 | 実装技術 |
| :---- | :---- | :---- |
| **Deep Bio-Calibration** | ANNの精度とSNNの適応性の完全融合 | **ECL (誤差補償学習)** の拡張、膜電位ダイナミクスの自動調整 |
| **高忠実度変換** | ANNと同等の精度をSNNで達成 | Scale-and-Fireニューロン（SFN）、$T=1$ 推論最適化 |
| **動的最適化** | モデルサイズとエネルギー消費の削減 | スパイク頻度に基づく**動的プルーニング**、**量子化メタデータ** |
| **スパイク注意機構** | LLM/TransformerのSNN化 | Softmax代替**Spiking Attention**モジュール |
| **Logic Gated SNN** | **超堅牢性・超低消費電力** (New) | 1.58ビット重みによる乗算フリー演算、**ハイパー・ロバスト学習** |
| **Hardware Native Ready** | ニューロモーフィックチップへの展開 | Loihi/TrueNorth向けの中間表現出力、イベント駆動シミュレーション |

## **🧠 パイプラインの全体像 (7つのフェーズ)**

このパイプラインは、以下の7つのフェーズを経て、標準的なANNモデルを**生物学的制約を満たす自律的なSNNエージェント**へと進化させます。

1. **Phase 1: Model Ingestion (モデル取り込み)**  
   * HuggingFaceやPyTorch HubからANNモデルをロード。  
   * 計算グラフを解析し、SNN互換レイヤーへのマッピング計画を策定。  
2. **Phase 2: Hybrid Quantization & Pruning (ハイブリッド量子化・プルーニング)**  
   * 重みを低ビット幅（4bit/8bit）に量子化。  
   * **Logic Gated Layer** への変換を選択した場合、重みを1.58ビット（-1, 0, 1）に圧縮。  
   * スパイク発火に寄与しない冗長な結合を枝刈り（Pruning）。  
3. **Phase 3: SNN Conversion & Activation Matching (SNN変換と活性化マッチング)**  
   * ANNの活性化関数（ReLUなど）をLIF（Leaky Integrate-and-Fire）ニューロンやSFN（Scale-and-Fire Neuron）に置換。  
   * データセットの一部を用いて、ANNとSNNの出力スパイク頻度を同期させる（Calibration）。  
4. **Phase 4: Deep Bio-Calibration (深層生物学的校正 \- HSEO)**  
   * **HSEO (Hybrid Spike-based Evolutionary Optimization)** を実行。  
   * 膜電位の閾値、時定数、リーク率などの物理パラメータを、タスク性能を最大化するように微調整。  
5. **Phase 5: Hyper-Robust Learning (ハイパー・ロバスト学習)** (New)  
   * **Logic Gated SNN** 向けの高負荷トレーニング。  
   * 入力データに**0%〜45%の可変ノイズ**を注入し、モデルに「表面的なパターン」ではなく「構造的な不変性」を強制的に学習させる。  
   * これにより、実環境のセンサーノイズや欠損に対する極めて高い堅牢性を獲得。  
6. **Phase 6: Hardware Native Compilation (ハードウェアネイティブコンパイル)**  
   * ターゲットデバイス（CPU, GPU, FPGA, Neuromorphic Chip）に合わせた最適化コード（CUDAカーネル、非同期イベントストリーム）を生成。  
7. **Phase 7: Neuro-Symbolic Evolution (ニューロ・シンボリック進化)**  
   * 変換されたSNNを「エージェント」として環境に配置。  
   * STDP（スパイクタイミング依存可塑性）や強化学習を通じて、変換後も継続的に自己改善を行う。

## **🛠️ 使用方法 (Quick Start)**

### **1\. 変換スクリプトの実行**

\# 基本的な変換とキャリブレーション  
python scripts/convert\_model.py \\  
    \--model\_name "resnet18" \\  
    \--dataset "cifar10" \\  
    \--output\_dir "artifacts/converted\_snn"

### **2\. Logic Gated SNN (ハイパー・ロバスト) のトレーニング**

変換や学習をゼロから行う場合（特に堅牢性を重視する場合）は、専用スクリプトを使用します。

\# 1.58ビット重みとノイズ注入学習による堅牢なモデル構築  
python scripts/run\_logic\_gated\_learning.py

### **3\. 設定ファイル (config.yaml) のカスタマイズ**

変換の詳細な挙動は、configs/templates/base\_config.yaml をベースにした設定ファイルで制御します。

conversion:  
  method: "scale\_and\_fire"  \# または "logic\_gated"  
  target\_timesteps: 4       \# 推論のタイムステップ数 (T)  
  enable\_bio\_calibration: true

bio\_calibration:  
  algorithm: "hseo"  
  population\_size: 20  
  generations: 10

robustness:  
  enable\_noise\_injection: true  
  noise\_range: \[0.0, 0.45\]  \# 0%〜45%のノイズを注入

## **📊 パフォーマンス指標 (Logic Gated SNN)**

最新のベンチマーク（v16.0時点）において、Logic Gated SNNアーキテクチャは以下の性能を実証しました。

* **Clean Accuracy:** 100.0% (Synthetic Pattern)  
* **Robust Accuracy (Noise 40%):** **85.2%**  
* **Theoretical Limit (Noise 50%):** \~10% (Random Guess) \- 物理的に正しい挙動  
* **Energy Efficiency:** 乗算回数 **0回** (全結合層において加算のみで処理)

## **⚠️ 制限事項と技術的制約**

1. **初期精度の低下**: 変換直後はANNに比べて精度が低下する場合があります。Deep Bio-Calibration (ECL) と**Surrogate Gradient**を用いた微調整（Fine-tuning）を推奨します。  
2. **メモリ使用量**: 時間方向への展開（BPTT）を行う場合、学習時のVRAM使用量が増加します。T=1 設定や slayer（Spike Layer）のチェックポイント機能を活用してください。  
3. **ハイパー・ロバスト学習のコスト**: ノイズ耐性を高める学習は、収束までに通常より多くのエポック数（1.5倍〜2倍）を必要とします。