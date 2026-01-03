# **SNN Core Technology Specifications v22.0**

## **Overview**

本ドキュメントは、SNN Project v22.0 "The Collective Mind" において開発・実装された中核技術の詳細仕様を定義する。これらの技術は、従来のSNN（Spiking Neural Networks）が抱える「推論レイテンシ」「表現力の限界」「協調動作の欠如」という課題を解決するために独自に設計されたものである。

## **1\. Hybrid Temporal-8-Bit Coding (HT8B)**

### **1.1 概念と目的**

従来のレートコーディング（Rate Coding）は情報を伝達するために多数のタイムステップ（$T=50 \\sim 100$）を要し、リアルタイム制御における致命的なボトルネックとなっていた。  
HT8B は、デジタル信号処理における「ビットプレーン分解」の概念をSNNの時間軸に適用し、わずか $T=4 \\sim 8$ ステップでの高精度な情報伝達を実現する。

### **1.2 アルゴリズム**

入力画素値 $x \\in \[0, 255\]$ (8bit整数) をビット列 $b\_7 b\_6 ... b\_0$ に分解し、各ビットをタイムステップ $t$ にマッピングする。

**数式定義:**

$$x(t) \= \\text{bit}(x, 7-t) \\quad \\text{for } t \\in \\{0, ..., 7\\}$$  
ここで $\\text{bit}(x, k)$ は整数 $x$ の $k$ ビット目の値（0または1）を返す関数である。

* $t=0$: MSB (Most Significant Bit, $2^7$) \- 最も重要な大まかな明暗情報。  
* $t=1$: $2^6$ の情報。  
* ...  
* $t=7$: LSB (Least Significant Bit, $2^0$) \- 微細なノイズレベルの情報。

### **1.3 実装仕様**

* **Input Tensor**: (Batch, Channels, Height, Width)  
* **Output Spike Tensor**: (Batch, Time, Channels, Height, Width)  
* **Time Steps (**$T$**)**:  
  * **High Precision Mode**: $T=8$ (全ビット使用)  
  * **Low Latency Mode**: $T=4$ (上位4ビットのみ使用。下位ビットは切り捨て)  
  * **Reflex Mode**: $T=1 \\sim 2$ (MSBのみによる超高速・粗視化推論)

### **1.4 性能評価 (M4 Chip)**

* 従来のレートコーディング ($T=50$): レイテンシ \~120ms  
* **HT8B (**$T=4$**)**: レイテンシ **15.08ms** (90%削減)

## **2\. Dual Adaptive Leaky Integrate-and-Fire (DA-LIF)**

### **2.1 概念と目的**

標準的なLIFニューロンは固定された時定数（$\\tau$）を持つが、タスクや入力の動的な変化に対応できない。  
DA-LIF は、「膜電位の減衰（忘却）」と「入力電流の減衰（持続）」をそれぞれ独立した学習可能パラメータとして定義し、ニューロンが自律的に情報の保持期間を最適化できるようにした。

### **2.2 ニューロンダイナミクス**

連続時間における微分方程式を、オイラー法を用いて離散化した更新式は以下の通りである。

1. シナプス電流更新 (Synaptic Current Update):  
   $$I\[t\] \= I\[t-1\] \\cdot \\sigma(w\_s) \+ X\[t\]$$  
   * $w\_s$: 電流減衰パラメータ（学習対象）。$\\sigma(\\cdot)$ はSigmoid関数。  
   * 入力スパイク $X\[t\]$ の影響がどれだけ長く残るかを制御する。  
2. 膜電位更新 (Membrane Potential Update):  
   $$V\[t\] \= V\[t-1\] \\cdot \\sigma(w\_m) \+ I\[t\]$$  
   * $w\_m$: 膜電位減衰パラメータ（学習対象）。  
   * ニューロンが発火に至るまでの「積分能力」を制御する。  
3. 発火とリセット (Fire & Reset):  
   $$S\[t\] \= \\Theta(V\[t\] \- V\_{th})$$$$V\[t\] \\leftarrow V\[t\] \\cdot (1 \- S\[t\]) \+ V\_{reset} \\cdot S\[t\]$$  
   * $\\Theta(\\cdot)$: Heavisideステップ関数（学習時はSurrogate Gradientを使用）。

### **2.3 特徴**

* **短期記憶**: $w\_m$ が大きい（1に近い）場合、過去の入力を長く保持し、時系列パターン検出器として機能する。  
* **一致検出**: $w\_m$ が小さい（0に近い）場合、同時入力のみに反応するCoincidence Detectorとして機能する。  
* これらが同一ネットワーク内に混在することで、多様な時間スケールの情報を処理可能となる。

## **3\. Liquid Democracy Protocol (LDP)**

### **3.1 概念と目的**

複数の自律エージェント（Swarm）が協調して意思決定を行うためのプロトコル。「1人1票」の硬直的な民主主義ではなく、専門知識を持つエージェントに投票権を委ねる「流動的（Liquid）」な合意形成を目指す。

### **3.2 プロトコル構成要素**

#### **A. Reputation Score (評判スコア)**

各エージェント $A\_i$ は、過去の貢献度に基づく評判スコア $R\_i$ を持つ（$R\_i \\ge 0.1$, 初期値 1.0）。

* 成功した提案に賛成: $R\_i \\leftarrow R\_i \+ \\alpha$  
* 失敗した提案に賛成: $R\_i \\leftarrow R\_i \- \\beta$

#### **B. Confidence (自信度)**

現在のタスクに対するエージェントの内部信頼度 $C\_i \\in \[0, 1\]$。  
ニューラルネットワークの出力確率のエントロピーや、Spike発火率の分散から算出される。

#### **C. Delegation (委任)**

自身の $C\_i$ が閾値 $\\theta\_{del}$ (例: 0.4) を下回る場合、エージェントは自身より高い $R\_j$ を持つエージェント $A\_j$ に投票権を委任できる。

### **3.3 集計アルゴリズム**

提案 $P\_k$ に対するスコア $S\_k$ は以下のように計算される。

$$S\_k \= \\sum\_{i \\in \\text{Voters}} (\\text{Power}\_i \\times C\_i \\times \\text{Direction}\_i)$$

* $\\text{Direction}\_i$: 賛成(+1) / 反対(-1)  
* Voting Power:  
  $$\\text{Power}\_i \= R\_i \+ \\sum\_{j \\in \\text{Delegators}(i)} R\_j$$

  委任を受けたエージェントは、委任元の評判スコアを自身の票に加算して行使する。

### **3.4 利点**

* **専門知の活用**: 知識のない多数決（衆愚政治）を防ぎ、そのタスクのエキスパートの意見が重み付けされる。  
* **スケーラビリティ**: 全員が全てのタスクを詳細に検討する必要がなく、信頼できるノードに処理をオフロードできる。

## **4\. Appendix: Implementation Reference**

| Technology | Path | Core Class |
| :---- | :---- | :---- |
| **HT8B** | snn\_research/io/spike\_encoder.py | HybridTemporal8BitEncoder |
| **DA-LIF** | snn\_research/core/neurons/da\_lif\_node.py | DualAdaptiveLIFNode |
| **LDP** | snn\_research/collective/liquid\_democracy.py | LiquidDemocracyProtocol |

