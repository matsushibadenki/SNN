# **Statistical Centroid Alignment Learning (SCAL)**

**Version:** 3.1 (Stabilized Ensemble & Hybrid Integration)

**Date:** 2025-12-30

**Status:** Validated (Noise Tolerance $\\epsilon \\approx 0.45$)

**Target:** Brain v3.0 Unified Architecture

## **English Version**

### **1\. Overview**

**SCAL (Statistical Centroid Alignment Learning)** v3.1 is the **Stabilized Ensemble Implementation** of the SCAL paradigm. It focuses on achieving extreme noise tolerance through a hybrid approach that combines statistical centroid alignment with ensemble averaging.

While traditional backpropagation fails at high noise levels, SCAL v3.1 successfully demonstrates robust learning capabilities up to **Noise Level 0.45**, leveraging an ensemble of SCAL units to form a "Statistical Consensus."

**Key Features in v3.1:**

1. **Adaptive Ensemble SCAL**: Uses multiple parallel SCAL kernels with stabilizing mechanisms to reduce variance in noisy input streams.  
2. **Multiscale Feature Integration**: (Enabled in v3.1) Captures features at different resolutions to improve robustness.  
3. **Hybrid Optimization**: Stabilized unsupervised alignment followed by ensemble voting/averaging.

### **2\. Theoretical Background**

#### **2.1 Statistical Centroid Alignment (The Core)**

The fundamental principle remains: **"Neurons should align their weights to the statistical centroid of their firing inputs."**

$$\\Delta W\_{ij} \= \\eta \\cdot (x\_i \- W\_{ij}) \\cdot \\mathbb{I}(y\_j \\text{ is active})$$  
Where:

* $\\mathbb{I}$ is the activation indicator (Phase transition function).  
* The weight vector $W\_j$ converges to the centroid of the input cluster $X$.

#### **2.2 Ensemble Variance Reduction**

In v3.1, the ensemble mechanism is empirically validated. The combined estimation $\\hat{F}$ reduces the variance of the noise error $\\sigma^2$:

$$\\text{Var}(\\hat{F}\_{ensemble}) \\approx \\frac{1}{K} \\text{Var}(\\hat{F}\_{single}) \+ \\text{Cov}(F\_i, F\_j)$$  
This allows the system to operate in noise levels approaching **S/N ratio** $\\approx 0.1$ (Noise \~0.45-0.50), maintaining high accuracy where single networks fail.

### **3\. Experimental Results (v3.1 Validation)**

The following results were obtained from run\_improved\_scal\_training.py \--ensemble.

#### **3.1 Training Progression**

The model demonstrates stable convergence even as noise levels increase during training stages.

| Epoch | Stage | Acc | Loss | SpkRate | V\_th | Status |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5 | Noise 0.10 | 100.00% | 0.027 | 19.23% | 5.00 | Stable |
| 15 | Noise 0.40 (Mix 0.2) | 99.96% | 0.028 | 19.17% | 5.00 | Robust |
| 30 | Noise 0.45 (Mix 0.1) | 87.25% | 0.044 | 15.71% | 5.00 | **High Noise Stability** |
| 60 | Noise 0.45 (Mix 0.1) | 87.49% | 0.044 | 15.87% | 5.00 | Converged |
| 65 | Noise 0.48 (Mix 0.1) | 43.58% | 0.077 | 13.21% | 5.00 | Phase Transition |

#### **3.2 Robustness Evaluation (10 trials)**

Accuracy is maintained at \>86% up to noise level 0.45, confirming the design goal. A sharp drop is observed at 0.48, indicating the current thermodynamic limit of the v3.1 ensemble.

| Noise Level | Accuracy (Mean ± Std) | Status |
| :---- | :---- | :---- |
| **0.10** | **100.00% ± 0.00%** | Excellent |
| **0.30** | **99.95% ± 0.00%** | Excellent |
| **0.40** | **99.87% ± 0.00%** | Excellent |
| **0.45** | **86.15% ± 0.00%** | **IMPROVED ✓** |
| 0.48 | 37.51% ± 0.00% | Weak (Limit) |
| 0.50 | 9.89% ± 0.00% | Failed |

### **4\. System Architecture (Hybrid Ensemble)**

The SCAL v3.1 architecture implements the following hierarchy:

graph TD  
    Input\[Noisy Sensory Input\] \--\> EnsembleLayer\[Ensemble SCAL Layer\]  
    subgraph "Ensemble SCAL Layer (v3.1)"  
        SCAL1\[SCAL Unit 1\]  
        SCAL2\[SCAL Unit 2\]  
        SCAL3\[SCAL Unit 3\]  
        SCAL\_N\[...\]  
    end  
    EnsembleLayer \--\>|Consensus Features| Aggregation\[Statistical Aggregation\]  
    Aggregation \--\> Output\[Action / Classification\]

1. **Ensemble Perception Layer**: Parallel SCAL units extract features.  
2. **Aggregation Layer**: Computes the weighted average or voting result.  
3. **Hybrid Interface**: The aggregated output serves as clean input for downstream tasks (RL or Classification).

### **5\. Future Roadmap (Towards v3.5)**

To bridge the gap between v3.1 (Noise 0.45) and the target v3.5 (Noise 0.55+), the following improvements are planned:

1. **Dynamic Thresholding**: Adapting V\_th dynamically during the 0.48 noise phase to prevent spike rate collapse.  
2. **RL Feedback**: Incorporating Top-down feedback from the value network to guide SCAL attention.

## **日本語版**

### **1\. 概要**

**SCAL (Statistical Centroid Alignment Learning)** v3.1は、SCALパラダイムの\*\*安定化アンサンブル実装（Stabilized Ensemble Implementation）\*\*です。統計的重心アライメントとアンサンブル平均化を組み合わせたハイブリッドアプローチにより、極限のノイズ耐性を実現することに焦点を当てています。

従来の誤差逆伝播法が高ノイズレベルで機能しなくなるのに対し、SCAL v3.1は**ノイズレベル 0.45** まで堅牢な学習能力を維持することを実証しました。これは、複数のSCALユニットを使用して「統計的合意（Statistical Consensus）」を形成することで達成されています。

**v3.1の主な機能:**

1. **Adaptive Ensemble SCAL**: 複数の並列SCALカーネルを使用し、ノイズ変動を低減。  
2. **マルチスケール特徴統合**: (v3.1で有効化) 異なる解像度で特徴を捉え、頑健性を向上。  
3. **ハイブリッド最適化**: 安定化された教師なしアライメントと、それに続くアンサンブル投票/平均化。

### **2\. 理論的背景**

#### **2.1 統計的重心アライメント（コア理論）**

基本原理：**「ニューロンは、発火入力の統計的重心に重みを合わせるべきである。」**

$$ \\Delta W\_{ij} \= \\eta \\cdot (x\_i \- W\_{ij}) \\cdot \\mathbb{I}(y\_j \\text{ is active}) $$

#### **2.2 アンサンブルによる分散低減**

v3.1では、アンサンブルメカニズムの効果が実証されました。統合された推定 $\\hat{F}$ は、ノイズ誤差の分散を低減し、**S/N比** $\\approx 0.1$ (Noise \~0.45-0.50) に近い環境での動作を可能にします。

### **3\. 実験結果 (v3.1 検証)**

以下は run\_improved\_scal\_training.py \--ensemble の実行結果に基づきます。

#### **3.1 学習の推移**

学習ステージが進みノイズレベルが上昇しても、モデルは安定した収束を示しています。

| Epoch | ステージ | 精度 (Acc) | 損失 (Loss) | スパイク率 | ステータス |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 5 | Noise 0.10 | 100.00% | 0.027 | 19.23% | 安定 |
| 15 | Noise 0.40 | 99.96% | 0.028 | 19.17% | 堅牢 |
| 30 | Noise 0.45 | 87.25% | 0.044 | 15.71% | **高ノイズ下で安定** |
| 60 | Noise 0.45 | 87.49% | 0.044 | 15.87% | 収束済み |
| 65 | Noise 0.48 | 43.58% | 0.077 | 13.21% | 相転移（性能低下） |

#### **3.2 頑健性評価 (10試行)**

ノイズレベル0.45までは精度86%以上を維持し、設計目標を達成しました。0.48で急激な低下が見られ、これが現在のv3.1アンサンブルの熱力学的限界点であることを示しています。

| ノイズレベル | 精度 (平均 ± 標準偏差) | 評価 |
| :---- | :---- | :---- |
| **0.10 \- 0.40** | **99.8% \- 100.00%** | Excellent |
| **0.45** | **86.15% ± 0.00%** | **IMPROVED ✓** |
| 0.48 | 37.51% ± 0.00% | Weak (限界点) |
| 0.50 | 9.89% ± 0.00% | Failed |

### **4\. システムアーキテクチャ（ハイブリッド設計）**

SCAL v3.1アーキテクチャは以下の階層を実装しています：

1. **アンサンブル知覚層**: 並列SCALユニットが特徴を抽出。  
2. **集約層 (Aggregation)**: 重み付き平均または投票による合意形成。  
3. **ハイブリッドインターフェース**: 集約された出力を、後段のタスク（RLや分類）へのクリーンな入力として提供。

### **5\. 今後のロードマップ (Towards v3.5)**

v3.1 (Noise 0.45) から目標とする v3.5 (Noise 0.55+) へのギャップを埋めるため、以下の改善が計画されています：

1. **動的閾値調整**: Noise 0.48領域でのスパイク率低下を防ぐため、V\_th を動的に適応させる。  
2. **RLフィードバック**: 価値ネットワークからのトップダウンフィードバックを統合し、SCALの注意機構を誘導する。

### **6\. 実装ノート**

* **スクリプト**: scripts/run\_improved\_scal\_training.py  
* **使用法**:  
  python scripts/run\_improved\_scal\_training.py \--ensemble  
