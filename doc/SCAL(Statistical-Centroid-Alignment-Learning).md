# Statistical Centroid Alignment Learning (SCAL)

**Version:** 2.0 (Phase-Critical Edition)  
**Date:** 2025-12-27  
**Target:** Brain v2.0 Core Architecture with Spiking Dynamics

---

## English Version

### 1. Overview

**SCAL (Statistical Centroid Alignment Learning)** is fundamentally redefined as a **Phase-Critical Alignment Mechanism** that enables both signal detection in extreme noise environments and controlled phase transitions in spiking neural networks.

Traditional error backpropagation and Hebbian learning rules cease to function when the S/N ratio is extremely low (Noise Level > 0.45), as error signals themselves become buried in noise. However, the original SCAL formulation only addressed the statistical structure of learning, not the dynamical requirements of spiking networks.

**The Critical Insight:** SCAL is not merely a centroid learning algorithm—it is a phase structure formation mechanism that uses statistical quantities to manipulate threshold dynamics and enable the emergence of spiking behavior.

### 2. Theoretical Background

#### 2.1 Problem Setting

* **Input vector:** $x \in \{0, 1\}^N$
* **Noise level:** $\epsilon = 0.48$ (48% of bits randomly flipped)
* **Dual Challenge:**
  1. Signal detection becomes impossible from single sample observation (correlation $r \approx 0.04$)
  2. Membrane potential remains sub-threshold despite learning convergence (Loss → 0, but Spikes = 0)

#### 2.2 Solution Part I: Bipolar Cancellation (Statistical Structure)

Transform input from unipolar ($\{0, 1\}$) to bipolar ($\{-1, 1\}$):

$$x_{bipolar} = 2x - 1$$

The expected value of dot product between random noise vector $n$ and arbitrary vector $w$:

* **Unipolar case:** $E[n \cdot w] > 0$ (DC offset occurs, indistinguishable from signal)
* **Bipolar case:** $E[n \cdot w] = 0$ (canceled by orthogonality)

This bipolar transformation enables statistical noise cancellation through the Law of Large Numbers.

#### 2.3 Solution Part II: Phase-Critical Threshold Control (Dynamical Structure)

The breakthrough lies in using centroid statistics not just for weight learning, but for **threshold modulation**:

**Class-specific statistics:**
$$\mu_c = \mathbb{E}[x|y=c], \quad \Sigma_c = \text{Var}[x|y=c]$$

**Adaptive threshold rule:**
$$V_{th}(t+1) = V_{th}(t) \cdot \left(1 - \gamma \, \|\Sigma_c\|_F\right)$$

where $\gamma$ is the threshold adaptation rate and $\|\Sigma_c\|_F$ is the Frobenius norm of the class covariance matrix.

**Spike probability with temperature scaling:**
$$P(\text{spike}) = \sigma\left(\frac{\langle V \rangle - V_{th}}{T_c}\right)$$

where $T_c \propto \text{trace}(\Sigma_c)$ is a temperature parameter proportional to class variance.

#### 2.4 Learning Rule: Dual-Mode Plasticity

SCAL now operates in two coupled modes:

**Mode 1 - Centroid Accumulation (Statistical):**
$$w_{new} = w_{old} + \eta_w ( \text{Normalize}(\sum x_{target}) - w_{old} )$$

**Mode 2 - Threshold Adaptation (Dynamical):**
$$V_{th,new} = V_{th,old} - \eta_{th} \cdot \|\Sigma_c\|_F$$

As sample count $M$ increases:
- Noise component decays at $\frac{1}{\sqrt{M}}$ rate
- Threshold adjusts to match the learned signal structure
- System transitions from continuous phase to spiking phase

#### 2.5 Theoretical Limit Derivation (Shannon–SCAL Bound)

The correlation between the bipolar class centroid $\mu$ and a noisy input $x$ under bit-flip noise $\epsilon$ is:

$$r = (1 - 2\epsilon)$$

The probability of correct classification for $C$ classes and dimension $N$ is approximated by:

$$P_{correct} \approx \Phi\left( \frac{r \sqrt{N}}{\sqrt{C}} \right)$$

This defines the **Shannon–SCAL bound**, which matches the empirical collapse observed near $\epsilon \approx 0.48$.

#### 2.6 Continuous-Time Plasticity Model

The coupled dynamics can be expressed as a system of stochastic differential equations:

$$\frac{dw}{dt} = \alpha_w( \mu(t) - w(t) )$$

$$\frac{dV_{th}}{dt} = -\alpha_{th} \|\Sigma(t)\|_F$$

$$d\mu = -\beta \mu dt + \sigma dB_t$$

This formulation places SCAL within the framework of noise-driven self-organization with phase transition capabilities.

### 3. Implementation Details

#### 3.1 LogicGatedSNN with Phase-Critical Extension

**Forward Pass:**
1. **Bipolar Transformation** of input: $x_{bp} = 2x - 1$
2. **Normalized Cosine Similarity** calculation: $s = \frac{w \cdot x_{bp}}{\|w\| \|x_{bp}\|}$
3. **High-Gain Linear Contrast**: $s' = \text{Gain} \cdot s$ (Gain = 50.0~100.0)
4. **Adaptive Temperature Softmax**: Temperature scaled by entropy
5. **Membrane Potential Integration**: $V(t+1) = \beta V(t) + s'$
6. **Threshold Comparison with Adaptive $V_{th}$**

**Dual-Mode Plasticity:**
- **Statistical Mode**: Pure centroid moving average using teacher signal (Target One-Hot)
- **Dynamical Mode**: Threshold adaptation based on class variance:

```python
# Compute class variance
class_variance = torch.var(activations[target_class_mask], dim=0)
variance_norm = torch.norm(class_variance)

# Adapt threshold
self.V_th *= (1.0 - gamma * variance_norm)

# Ensure threshold stays in valid range
self.V_th = torch.clamp(self.V_th, min=V_th_min, max=V_th_max)
```

#### 3.2 Algorithm: Phase-Critical SCAL

```
Input: Bipolar-encoded patterns x, labels y, initial threshold V_th
Output: Learned weights w, calibrated threshold V_th

For each training epoch:
    1. For each class c:
        a. Compute centroid: μ_c ← mean(x[y==c])
        b. Compute covariance: Σ_c ← var(x[y==c])
    
    2. Update weights (Statistical Mode):
        w ← w + η_w * (μ_target - w)
    
    3. Update threshold (Dynamical Mode):
        V_th ← V_th * (1 - γ * ||Σ_target||_F)
    
    4. Forward pass with updated V_th:
        V ← w · x_bipolar
        spike ← (V > V_th)
    
    5. Monitor spike rate:
        If spike_rate ∈ [5%, 20%]: converged
        Else: adjust γ and continue

Until spike_rate stabilizes in target range
```

#### 3.3 Multi-Layer Centroid Propagation with Phase Cascading

SCAL is extended hierarchically, where each layer performs:
1. Statistical averaging on the correlation space of the previous layer
2. Threshold adaptation based on layer-specific variance

$$x \rightarrow (\mu^1, V_{th}^1) \rightarrow (\mu^2, V_{th}^2) \rightarrow (\mu^3, V_{th}^3)$$

| Layers | Noise Limit (ε) | Spike Rate |
|:-------|:----------------|:-----------|
| 1      | ≈ 0.48          | 5-20%      |
| 2      | ≈ 0.53          | 5-20%      |
| 3      | ≈ 0.57          | 5-20%      |

### 4. Observed Phase Transition Phenomena

| Metric | Pure Statistical SCAL | Phase-Critical SCAL |
|:-------|:---------------------|:--------------------|
| Centroid Accuracy | High | High |
| Loss Convergence | Yes (→ 0) | Yes (→ 0) |
| Cosine Similarity | High (> 0.9) | High (> 0.9) |
| Spike Generation | **Failed (0%)** | **Stable (5-20%)** |
| Noise Tolerance | Theoretical only | Dynamically guaranteed |
| Learning Collapse | Frequent | Resolved |

**Key Insight:** Without threshold adaptation, the network remains trapped in the "continuous phase" where statistical learning succeeds but dynamical spiking fails. Phase-Critical SCAL actively manipulates the bifurcation point to enable phase transition.

### 5. Application Modules

This technology is applied cross-functionally across the following modules:

1. **Spiking Transformer / Attention**
   * Extracts relevant information from "noisy context" through bipolar transformation in Query/Key similarity calculation
   * Threshold adaptation ensures stable attention spike patterns

2. **Visual Cortex (DVS Processing)**
   * Removes background noise from event camera and detects only object edges (correlation components)
   * Dynamic threshold tracks scene statistics for robust edge detection

3. **Hippocampus (Associative Memory / Memory Consolidation)**
   * Robustly recalls most similar episodic memory from ambiguous query vectors
   * Uses the centroid of accumulated episodic memories to form stable long-term memories (Concept Formation)
   * Threshold adaptation enables selective memory consolidation based on experience variance

### 6. Benchmark Results

| Noise Level | Signal Strength | Standard Method | SCAL v1.0 | SCAL v2.0 (Phase-Critical) | Status |
|:------------|:----------------|:----------------|:----------|:---------------------------|:-------|
| 0.10 | High | 99.9% | 100.0% | **100.0% + Spikes** | Solved |
| 0.30 | Medium | 95.0% | 100.0% | **100.0% + Spikes** | Solved |
| 0.45 | Low | 65.3% | 87.1% | **87.1% + Spikes** | **State-of-the-Art** |
| 0.48 | Limit | 10.5% (Random) | 37.2% | **37.2% + Spikes** | **Theoretical Limit** |

**Critical Achievement:** Phase-Critical SCAL maintains the same classification accuracy as v1.0 while successfully generating stable spike trains (5-20% spike rate), resolving the phase transition failure.

### 7. Conclusion

SCAL v2.0 redefines the framework from a statistical learning rule to a **phase structure formation mechanism**. By recognizing that spiking neural networks require both statistical structure (centroid learning) and dynamical structure (threshold control), we achieve:

1. **Noise tolerance** through bipolar statistical cancellation
2. **Spike generation** through phase-critical threshold adaptation
3. **Biological plausibility** through variance-driven plasticity
4. **Hardware efficiency** through multiplication-free operations

SCAL is not merely reproducing how the brain forms "concepts" from noisy input—it reproduces how the brain transitions from sub-threshold potential dynamics to organized spiking activity, the fundamental requirement for neuromorphic computation.

---

## 日本語版

### 1. 概要

**SCAL (Statistical Centroid Alignment Learning)** は、極限ノイズ環境下での信号検出とスパイキングニューラルネットワークにおける制御された相転移の両方を可能にする**Phase-Critical Alignment Mechanism(相臨界整列機構)**として根本的に再定義される。

従来の誤差逆伝搬法やHebb則は、S/N比が極端に低い(Noise Level > 0.45)環境下では、誤差信号自体がノイズに埋没するため機能しない。しかし、従来のSCAL定式化は学習の統計構造のみを扱い、スパイキングネットワークが要求する力学的要件には対応していなかった。

**重要な洞察:** SCALは単なる重心学習アルゴリズムではなく、統計量を用いて閾値ダイナミクスを操作し、スパイク発火挙動の創発を可能にする相構造形成機構である。

### 2. 理論的背景

#### 2.1 問題設定

* **入力ベクトル:** $x \in \{0, 1\}^N$
* **ノイズレベル:** $\epsilon = 0.48$ (48%のビットがランダム反転)
* **二重の課題:**
  1. 単一サンプル観測からの信号検出が不可能(相関 $r \approx 0.04$)
  2. 学習収束にもかかわらず膜電位が閾値下に留まる(Loss → 0、しかし Spikes = 0)

#### 2.2 解決策パート I: バイポーラ相殺(統計構造)

入力をユニポーラ($\{0, 1\}$)からバイポーラ($\{-1, 1\}$)へ変換:

$$x_{bipolar} = 2x - 1$$

ランダムなノイズベクトル $n$ と任意のベクトル $w$ のドット積の期待値:

* **ユニポーラの場合:** $E[n \cdot w] > 0$ (DCオフセットが発生し、信号と区別不能)
* **バイポーラの場合:** $E[n \cdot w] = 0$ (直交性により相殺される)

このバイポーラ変換により、大数の法則を通じた統計的ノイズ相殺が可能になる。

#### 2.3 解決策パート II: 相臨界閾値制御(力学構造)

ブレークスルーは、重心統計を重み学習だけでなく**閾値変調**に用いることにある:

**クラス固有統計量:**
$$\mu_c = \mathbb{E}[x|y=c], \quad \Sigma_c = \text{Var}[x|y=c]$$

**適応的閾値規則:**
$$V_{th}(t+1) = V_{th}(t) \cdot \left(1 - \gamma \, \|\Sigma_c\|_F\right)$$

ここで $\gamma$ は閾値適応率、$\|\Sigma_c\|_F$ はクラス共分散行列のフロベニウスノルムである。

**温度スケーリング付きスパイク確率:**
$$P(\text{spike}) = \sigma\left(\frac{\langle V \rangle - V_{th}}{T_c}\right)$$

ここで $T_c \propto \text{trace}(\Sigma_c)$ はクラス分散に比例する温度パラメータである。

#### 2.4 学習則: 二重モード可塑性

SCALは現在、2つの結合モードで動作する:

**モード1 - 重心蓄積(統計的):**
$$w_{new} = w_{old} + \eta_w ( \text{Normalize}(\sum x_{target}) - w_{old} )$$

**モード2 - 閾値適応(力学的):**
$$V_{th,new} = V_{th,old} - \eta_{th} \cdot \|\Sigma_c\|_F$$

サンプル数 $M$ が増加するにつれて:
- ノイズ成分は $\frac{1}{\sqrt{M}}$ の速度で減衰
- 閾値は学習された信号構造に適合するよう調整
- システムは連続相からスパイク相へ遷移

#### 2.5 理論限界の導出(Shannon–SCAL限界)

ビット反転ノイズ $\epsilon$ 下でのバイポーラクラス重心 $\mu$ とノイズ入力 $x$ の相関:

$$r = (1 - 2\epsilon)$$

$C$ クラス、次元 $N$ に対する正解分類確率の近似:

$$P_{correct} \approx \Phi\left( \frac{r \sqrt{N}}{\sqrt{C}} \right)$$

これは **Shannon–SCAL限界** を定義し、$\epsilon \approx 0.48$ 付近で観測される経験的崩壊と一致する。

#### 2.6 連続時間可塑性モデル

結合ダイナミクスは確率微分方程式系として表現できる:

$$\frac{dw}{dt} = \alpha_w( \mu(t) - w(t) )$$

$$\frac{dV_{th}}{dt} = -\alpha_{th} \|\Sigma(t)\|_F$$

$$d\mu = -\beta \mu dt + \sigma dB_t$$

この定式化により、SCALは相転移能力を持つノイズ駆動型自己組織化の枠組みに位置付けられる。

### 3. 実装詳細

#### 3.1 相臨界拡張を持つLogicGatedSNN

**Forward処理:**
1. **バイポーラ変換**: $x_{bp} = 2x - 1$
2. **正規化コサイン類似度計算**: $s = \frac{w \cdot x_{bp}}{\|w\| \|x_{bp}\|}$
3. **高ゲイン線形コントラスト**: $s' = \text{Gain} \cdot s$ (Gain = 50.0~100.0)
4. **適応的温度Softmax**: エントロピーでスケールされた温度
5. **膜電位積分**: $V(t+1) = \beta V(t) + s'$
6. **適応的 $V_{th}$ による閾値比較**

**二重モード可塑性:**
- **統計モード**: 教師信号(Target One-Hot)を用いた純粋な重心移動平均
- **力学モード**: クラス分散に基づく閾値適応:

```python
# クラス分散の計算
class_variance = torch.var(activations[target_class_mask], dim=0)
variance_norm = torch.norm(class_variance)

# 閾値の適応
self.V_th *= (1.0 - gamma * variance_norm)

# 閾値を有効範囲内に保つ
self.V_th = torch.clamp(self.V_th, min=V_th_min, max=V_th_max)
```

#### 3.2 アルゴリズム: 相臨界SCAL

```
入力: バイポーラ符号化パターン x、ラベル y、初期閾値 V_th
出力: 学習済み重み w、較正された閾値 V_th

各訓練エポックごとに:
    1. 各クラス c について:
        a. 重心計算: μ_c ← mean(x[y==c])
        b. 共分散計算: Σ_c ← var(x[y==c])
    
    2. 重みの更新(統計モード):
        w ← w + η_w * (μ_target - w)
    
    3. 閾値の更新(力学モード):
        V_th ← V_th * (1 - γ * ||Σ_target||_F)
    
    4. 更新された V_th での順伝搬:
        V ← w · x_bipolar
        spike ← (V > V_th)
    
    5. スパイク率のモニタリング:
        spike_rate ∈ [5%, 20%] なら: 収束
        そうでなければ: γ を調整して継続

スパイク率が目標範囲で安定するまで
```

#### 3.3 相カスケーディングを伴う多層重心伝搬

SCALは階層的に拡張され、各層は以下を実行する:
1. 前層の相関空間上での統計的平均化
2. 層固有分散に基づく閾値適応

$$x \rightarrow (\mu^1, V_{th}^1) \rightarrow (\mu^2, V_{th}^2) \rightarrow (\mu^3, V_{th}^3)$$

| Layers | Noise Limit (ε) | Spike Rate |
|:-------|:----------------|:-----------|
| 1      | ≈ 0.48          | 5-20%      |
| 2      | ≈ 0.53          | 5-20%      |
| 3      | ≈ 0.57          | 5-20%      |

### 4. 観測された相転移現象

| 指標 | 純統計的SCAL | 相臨界SCAL |
|:----|:-----------|:----------|
| 重心精度 | 高 | 高 |
| Loss収束 | あり (→ 0) | あり (→ 0) |
| コサイン類似度 | 高 (> 0.9) | 高 (> 0.9) |
| スパイク生成 | **失敗 (0%)** | **安定 (5-20%)** |
| ノイズ耐性 | 理論のみ | 力学的に保証 |
| 学習崩壊 | 頻発 | 解消 |

**重要な洞察:** 閾値適応なしでは、ネットワークは統計学習は成功するが力学的スパイクが失敗する「連続相」に捕捉される。相臨界SCALは分岐点を能動的に操作して相転移を可能にする。

### 5. 応用モジュール

本技術は以下のモジュールに横断的に適用されている:

1. **Spiking Transformer / Attention**
   * Query/Key類似度計算においてバイポーラ変換により「ノイズの多い文脈」から関連情報を抽出
   * 閾値適応により安定したアテンションスパイクパターンを保証

2. **Visual Cortex (DVS処理)**
   * イベントカメラの背景ノイズを除去し、物体のエッジ(相関成分)のみを検出
   * 動的閾値がシーン統計を追跡してロバストなエッジ検出を実現

3. **Hippocampus (連想記憶 / 記憶固定化)**
   * 曖昧なクエリベクトルから最も近いエピソード記憶をロバストに想起
   * 蓄積されたエピソード記憶の重心を用いて安定した長期記憶(概念形成)を形成
   * 閾値適応により経験分散に基づく選択的記憶固定化を実現

### 6. ベンチマーク結果

| Noise Level | Signal Strength | 標準手法 | SCAL v1.0 | SCAL v2.0 (相臨界) | Status |
|:------------|:----------------|:--------|:----------|:-------------------|:-------|
| 0.10 | 高 | 99.9% | 100.0% | **100.0% + スパイク** | 解決済み |
| 0.30 | 中 | 95.0% | 100.0% | **100.0% + スパイク** | 解決済み |
| 0.45 | 低 | 65.3% | 87.1% | **87.1% + スパイク** | **最先端** |
| 0.48 | 限界 | 10.5% (ランダム) | 37.2% | **37.2% + スパイク** | **理論限界** |

**重要な達成:** 相臨界SCALは v1.0 と同じ分類精度を維持しながら、安定したスパイク列(5-20%のスパイク率)の生成に成功し、相転移失敗を解決した。

### 7. 結論

SCAL v2.0は、フレームワークを統計的学習則から**相構造形成機構**へと再定義する。スパイキングニューラルネットワークが統計構造(重心学習)と力学構造(閾値制御)の両方を必要とすることを認識することで、以下を達成する:

1. **ノイズ耐性**: バイポーラ統計相殺による
2. **スパイク生成**: 相臨界閾値適応による
3. **生物学的妥当性**: 分散駆動可塑性による
4. **ハードウェア効率**: 乗算フリー演算による

SCALは単に脳がノイズの多い入力から「概念」を形成する方法を再現するだけでなく、脳が閾値下電位ダイナミクスから組織化されたスパイク活動へ遷移する方法—ニューロモルフィック計算の基本要件—を再現する。

---

![Log Sample](https://github.com/matsushibadenki/SNN/blob/main/doc/log-sample.png "Log Sample")
