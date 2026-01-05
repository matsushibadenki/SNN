# **次世代ニューロモーフィック・コンピューティング：松芝電気SNNプロジェクトの戦略的進化に関する包括的研究報告書**

## **1\. エグゼクティブサマリー**

2024年から2025年にかけて、ニューロモーフィック・コンピューティングの分野、特にスパイキングニューラルネットワーク（SNN）の研究領域は、かつてないほどの技術的転換期を迎えています。従来のSNNは、生物学的妥当性を追求するあまり、計算効率や推論精度において人工ニューラルネットワーク（ANN）に遅れをとっていましたが、近年の「第3世代」SNNアーキテクチャの登場により、そのパラダイムは劇的に変化しました。本報告書は、松芝電気（Matsushiba Denki）がGitHub上で公開しているSNNプロジェクト（[https://github.com/matsushibadenki/SNN](https://github.com/matsushibadenki/SNN)）を対象に、現在の学術界における最先端技術（State-of-the-Art, SOTA）を適用し、プロジェクトを世界トップレベルの水準へと引き上げるための包括的な戦略提案書です。

現状のプロジェクト構成は、標準的なLeaky Integrate-and-Fire（LIF）ニューロンモデルと畳み込みニューラルネットワーク（CNN）アーキテクチャ、そしてレートコーディングに基づく情報符号化を採用していると推察されます。これはSNNの基礎学習としては適切ですが、産業応用やSOTAとの競争力という観点からは、「精度・レイテンシ・エネルギー効率」のトリレンマに直面することになります。

本報告書では、以下の3つの核心的な技術革新を導入することを提案します。

1. **ニューロンモデルの刷新**：静的なLIFモデルから、**Dual Adaptive Leaky Integrate-and-Fire (DA-LIF)** モデルへの移行。これにより、膜電位と入力電流の時定数を学習可能にし、時間的・空間的な特徴抽出能力を飛躍的に向上させます 1。  
2. **アーキテクチャの進化**：CNNベースのバックボーンから、**Spiking Transformer (Spikformer V2 / SpiLiFormer)** への転換。自己注意機構（Self-Attention）をスパイクベースで実装することで、局所的な特徴だけでなく大域的な文脈情報を捉え、ImageNet等の大規模データセットにおいて80%を超える精度を実現します 3。  
3. **情報符号化の高度化**：非効率なレートコーディングを廃止し、**Hybrid Temporal-8-Bit Spike Coding** を採用。入力情報のビットプレーン分解と時間的符号化を組み合わせることで、極めて低いレイテンシ（$T \\le 4$）での高精度推論を可能にします 5。

これらの技術要素を、PyTorchベースの最新フレームワークである**SpikingJelly**を用いて実装・統合することで、松芝電気のプロジェクトは単なるプロトタイプから、次世代AIハードウェアの中核を担う実用的なシステムへと進化を遂げることが可能です。以下、各技術の詳細な分析と実装戦略について論じます。

## ---

**2\. 背景と現状分析：SNNの技術的変遷と課題**

### **2.1 第3世代ニューラルネットワークの台頭**

人工知能の歴史において、SNNは「第3世代のニューラルネットワーク」として位置づけられています。第2世代である現在のANN（Deep Learning）が連続的な浮動小数点演算（MAC演算）に依存しているのに対し、SNNは生物学的脳と同様に、離散的な「スパイク（パルス）」によるイベント駆動型の情報処理を行います。

2024年から2025年にかけての研究動向を概観すると、SNNの主要な利点である「省電力性」を維持しつつ、課題であった「学習の困難さ」と「推論精度」を克服するためのブレイクスルーが多数報告されています。特に、サロゲート勾配（Surrogate Gradient）を用いた直接学習法の洗練化と、TransformerアーキテクチャのSNNへの移植が成功したことは、この分野における特異点と言えます 7。

### **2.2 現行プロジェクトのボトルネック分析**

松芝電気の既存プロジェクト（Githubリポジトリの標準的な構成を想定）における技術的課題は、以下の表に示すように、現在のSOTA技術との間に明確なギャップとして存在しています。

| 技術要素 | 現行プロジェクト（推定ベースライン） | 最先端技術（2025 SOTA） | 課題と影響 |
| :---- | :---- | :---- | :---- |
| **ニューロンモデル** | **Static LIF** (時定数 $\\tau$ が固定) | **DA-LIF / Parametric LIF** (時定数が学習可能・適応的) | 固定された時定数は、多様な時間スケールを持つ実世界のデータに対応できず、表現力が制限される 1。 |
| **アーキテクチャ** | **Spiking CNN** (VGG / ResNet) | **Spiking Transformer** (Spikformer V2 / SpiLiFormer) | CNNの局所受容野では大域的な依存関係を捉えきれず、大規模画像認識で精度が頭打ちになる 3。 |
| **情報符号化** | **Rate Coding** (ポアソン分布等) | **Hybrid Temporal-8-Bit** (ビットプレーン符号化) | レートコーディングは情報の伝達に多数のタイムステップを要し、低レイテンシ化（高速化）を阻害する 5。 |
| **学習手法** | **ANN-to-SNN Conversion** (変換法) | **Direct Training** (サロゲート勾配法) | 変換法は変換ロスが発生しやすく、特に短いタイムステップでの精度維持が困難である 8。 |

この分析から明らかなように、単なるパラメータ調整や層の追加といった対症療法的な改善では、プロジェクトを「次のレベル」へ引き上げることは不可能です。根底にある数理モデルとアーキテクチャパラダイムそのものを刷新する必要があります。

## ---

**3\. 戦略提案 I：ニューロンダイナミクスの革新 (DA-LIF)**

SNNの計算能力を規定する最小単位はニューロンモデルです。従来のLIFモデルは計算が単純である反面、生物学的脳が持つ柔軟な適応能力を欠いています。本プロジェクトにおいては、**Dual Adaptive Leaky Integrate-and-Fire (DA-LIF)** モデルの導入を強く推奨します。

### **3.1 均質性からの脱却：学習可能な時定数の重要性**

従来のLIFモデルでは、膜電位の減衰率（Decay Factor）はハイパーパラメータとして固定されています。しかし、入力データの特徴（例えば、画像のテクスチャや物体の動き）によって、最適な情報の保持時間は異なります。

DA-LIFモデルは、膜電位の減衰（時間的要素）と入力電流の減衰（空間的要素）をそれぞれ独立した学習可能パラメータとして定義します。これにより、ネットワークは「どの情報を長く保持し、どの情報を即座に忘却すべきか」をデータから学習することが可能になります。これは、生物学におけるグリア細胞による神経変調作用を数理的に模倣したものであり、SNNに高い表現力を付与します 1。

### **3.2 DA-LIFモデルの数理的定式化**

DA-LIFモデルの離散時間における更新式は、以下のように導出されます。

まず、連続時間におけるLIFモデルの微分方程式は以下の通りです。

$$\\tau\_m \\frac{du(t)}{dt} \= \-(u(t) \- u\_{rest}) \+ R\_m I(t)$$

ここで、$\\tau\_m$ は膜時定数、$u(t)$ は膜電位、$I(t)$ は入力電流です。  
DA-LIFでは、これを離散化し、さらに減衰項を学習可能なパラメータ $\\alpha, \\beta$ に置き換えます。時刻 $t$、レイヤー $n$ におけるニューロンの膜電位 $V^{t,n}$ の更新式は以下の差分方程式で表されます。

$$V^{t,n} \= \\beta\_n H^{t-1,n} \+ \\alpha\_n X^{t,n}$$  
ここで、各変数の定義は以下の通りです。

* $V^{t,n}$: 現在のタイムステップにおける膜電位（発火判定前）。  
* $H^{t-1,n}$: 前のタイムステップにおける、リセット後の膜電位。  
* $X^{t,n}$: 現在のタイムステップにおける入力電流（前層からのスパイク入力の重み付き和）。  
* $\\beta\_n$: **時間的減衰パラメータ (Temporal Decay)**。膜電位の履歴をどれだけ保持するかを制御します。  
* $\\alpha\_n$: **空間的減衰パラメータ (Spatial Decay)**。現在の入力信号をどれだけ強く受け入れるかを制御します。

#### **3.2.1 パラメータの学習機構**

$\\alpha\_n$ と $\\beta\_n$ が固定値であれば、従来のLIFと変わりません。DA-LIFの革新性は、これらをシグモイド関数等を用いて $$ の範囲に制約しつつ、誤差逆伝播法（Backpropagation）によって最適化する点にあります 1。

$$\\alpha\_n \= \\text{sigmoid}(\\tau\_{a,n}), \\quad \\beta\_n \= \\text{sigmoid}(\\tau\_{b,n})$$  
ここで、$\\tau\_{a,n}, \\tau\_{b,n}$ は学習可能な実数値パラメータです。この機構により、ネットワークの層ごとに、あるいはチャンネルごとに異なる時間特性を獲得させることができます。例えば、浅い層では速い変化に対応するために小さな $\\beta$ を、深い層では文脈を保持するために大きな $\\beta$ を自動的に獲得するといった挙動が期待されます。

### **3.3 実装上の利点とインパクト**

このモデルを松芝電気のプロジェクトに導入することで、以下の効果が見込まれます。

1. **収束速度の向上**: ニューロン自体がデータに適応するため、学習の収束が早まります。  
2. **精度の向上**: CIFAR-10やImageNetなどのベンチマークにおいて、静的LIFと比較して数パーセントの精度向上が報告されています 1。  
3. **計算効率**: 複雑な微分方程式を解く必要なく、単純な積和演算と加算のみで実装できるため、ハードウェア実装時のコストも低く抑えられます。

## ---

**4\. 戦略提案 II：Spiking Transformerアーキテクチャへの移行**

現在のディープラーニングにおいて、画像認識のデファクトスタンダードはCNNからVision Transformer (ViT) へと移行しています。SNNにおいても同様の潮流があり、**Spikformer** や **SpiLiFormer** と呼ばれるアーキテクチャが、従来のSpiking ResNetを超える性能を叩き出しています。

### **4.1 従来のSpiking CNNの限界**

CNNは「局所受容野」と「重み共有」を前提としており、画像内の局所的な特徴抽出には優れています。しかし、画像全体にまたがる大域的な関係性（Global Context）を捉える能力には限界があります。SNNの場合、情報がスパース（疎）であるため、局所的な情報のみに頼ると、特徴を見落とすリスクがさらに高まります。

### **4.2 Spiking Self-Attention (SSA) の導入**

Transformerの核となるのは自己注意機構（Self-Attention）ですが、オリジナルのAttention機構はSNNにそのまま適用できません。

$$\\text{Attention}(Q, K, V) \= \\text{Softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$  
この式には2つの致命的な問題があります。

1. **Softmax関数の非互換性**: 指数関数と除算を含むSoftmaxは計算コストが高く、整数演算や加算を基本とするSNNハードウェアには不向きです。  
2. **浮動小数点演算**: $Q, K, V$ が連続値をとる場合、行列積はMAC演算となり、SNNの省電力性を損ないます。

#### **4.2.1 Spikformer V2における解決策**

この問題を解決するために、**Spiking Self-Attention (SSA)** 3 を採用します。SSAでは、Query ($Q$), Key ($K$), Value ($V$) をすべてスパイク（0または1）として生成します。

$$Q \= \\text{SN}(XW\_Q), \\quad K \= \\text{SN}(XW\_K), \\quad V \= \\text{SN}(XW\_V)$$  
ここで $\\text{SN}$ はスパイキングニューロン層です。さらに、Softmaxを除去し、スパイクベースの演算のみでAttention Mapを計算する手法を導入します。

$$A \= \\text{Scale}(QK^T)$$

$$\\text{Output} \= \\text{SN}(AV)$$  
$Q$ と $K$ がバイナリであるため、$QK^T$ の計算は論理積（AND）と加算（PopCount）に置き換えることができ、エネルギー効率が飛躍的に向上します。また、Softmaxを除去しても、スパイクの希薄性（Sparsity）自体が注意機構として機能するため、精度が維持されることが実証されています 3。

### **4.3 SpiLiFormer：側抑制によるロバスト性の向上**

さらなる精度向上策として、**SpiLiFormer** 4 の概念を統合することを提案します。これは、生物の網膜における「側抑制（Lateral Inhibition）」メカニズムをAttention機構に組み込むものです。

具体的には、Attention Mapにおいて、あるトークンが強く発火した場合、その近傍のトークンの発火を抑制するようなリカレント結合を導入します。これにより、ノイズに対するロバスト性が高まり、注目すべき特徴がより鮮明になります。ImageNetにおける実験では、SpiLiFormerは従来のSpikformerと比較して、より少ないパラメータ数で高い精度を達成しています。

## ---

**5\. 戦略提案 III：ハイブリッド・テンポラル・8ビット符号化**

入力層における情報の損失とレイテンシは、SNNの永遠の課題です。松芝電気のプロジェクトがポアソン符号化（レートコーディング）を使用している場合、画像のピクセル値を正確に伝達するために、数百ステップのシミュレーションが必要になります。これはリアルタイム処理において致命的です。

### **5.1 レートコーディングの非効率性**

レートコーディングは、画素の輝度を発火頻度（確率）に変換します。例えば、輝度0.5を表現するには、100ステップ中50回発火する必要があります。しかし、これでは「いつ」発火するかという時間情報がランダムであり、情報密度が極めて低くなります。

### **5.2 Bit-Plane Coding (ビットプレーン符号化) の採用**

ここで提案するのは、**Hybrid Temporal-8-Bit Spike Coding** 5 です。これは、デジタル画像の各ピクセル（通常8ビット整数）をビットプレーンに分解し、それを時間軸に沿って入力する手法です。

#### **5.2.1 アルゴリズムのメカニズム**

画像 $I(x, y)$ の各画素値 $P$ は以下のように表現されます。

$$P \= \\sum\_{k=0}^{7} b\_k \\cdot 2^k$$

ここで $b\_k \\in \\{0, 1\\}$ は $k$ ビット目の値です。  
この手法では、SNNのタイムステップ $t$ に応じて、上位ビットから順に入力を行います。

* **$t=1$**: 最上位ビット（MSB, $b\_7$）を入力。  
* **$t=2$**: 次のビット（$b\_6$）を入力。  
* ...  
* **$t=8$**: 最下位ビット（LSB, $b\_0$）を入力。

#### **5.2.2 導入のメリット**

1. **完全な情報伝達**: わずか8ステップで、画像の持つ8ビット情報を完全にロスレスでネットワークに入力できます。従来のレートコーディングでは不可能な情報密度です。  
2. **Time-to-First-Spike (TTFS) との整合性**: 最上位ビット（画像の主要な構造情報）が最初に入力されるため、ネットワークは早い段階で大まかな推論を開始できます。これは生物学的妥当性が高く、低レイテンシ推論（Early Exit）の実装にも有利です 12。  
3. **微分可能性**: この符号化プロセスは決定論的であり、サロゲート勾配を用いた学習において、入力層から直接勾配を計算することが可能です（従来のポアソン生成は確率的であるため、勾配の推定が困難でした）6。

この符号化方式を採用することで、推論に必要なタイムステップ数 $T$ を、従来の $T=50 \\sim 100$ から、理論上の最小値に近い $T=4 \\sim 8$ まで短縮することが可能になります。

## ---

**6\. 実装ロードマップ：SpikingJellyを用いた統合**

上述の3つの提案（DA-LIF、Spikformer、ハイブリッド符号化）を具現化するためには、堅牢かつ柔軟なソフトウェアフレームワークが必要です。ここでは、PyTorchエコシステム上で動作するオープンソースライブラリ **SpikingJelly** 14 を用いた実装戦略を提示します。

### **6.1 SpikingJellyの選定理由**

SpikingJellyは、現在のSNN研究においてデファクトスタンダードとなりつつあるフレームワークです。以下の特徴が本プロジェクトに最適です。

* **CUDA最適化**: cupy バックエンドを利用することで、Pythonのオーバーヘッドを回避し、GPU上で高速な学習が可能です。  
* **Surrogate Gradientの豊富さ**: 微分不可能なスパイク発火関数の勾配近似として、ATan（逆正接関数）やSigmoidなどを容易に切り替え可能です。  
* **カスタムニューロンの定義**: BaseNode クラスを継承することで、前述のDA-LIFのような複雑なダイナミクスを持つニューロンを簡単に実装できます。

### **6.2 DA-LIFニューロンの実装コード例**

以下に、DA-LIFニューロンをSpikingJellyで実装する際の概念コードを示します。このコードは、時定数を学習可能パラメータとして定義し、フォワードパスで差分方程式を計算する構造を持っています。

Python

import torch  
import torch.nn as nn  
from spikingjelly.activation\_based import neuron, surrogate

class DALIFNode(neuron.BaseNode):  
    def \_\_init\_\_(self, channels: int, v\_threshold: float \= 1.0,   
                 surrogate\_function=surrogate.ATan()):  
        super().\_\_init\_\_(v\_threshold=v\_threshold,   
                         v\_reset=0.0,   
                         surrogate\_function=surrogate\_function,  
                         detach\_reset=True)  
          
        \# 学習可能な時定数パラメータの初期化  
        \# シグモイド関数を通して0\~1の減衰率に変換するため、初期値は逆算して設定  
        self.tau\_a \= nn.Parameter(torch.zeros(channels)) \# 空間的減衰用  
        self.tau\_b \= nn.Parameter(torch.zeros(channels)) \# 時間的減衰用

    def forward(self, x: torch.Tensor):  
        \# シグモイド関数で減衰率alpha, betaを計算  
        alpha \= torch.sigmoid(self.tau\_a)  
        beta \= torch.sigmoid(self.tau\_b)  
          
        \# 膜電位の更新（差分方程式の実装）  
        \# self.v は前のステップの膜電位 H\[t-1\] に相当  
        self.v\_float \= beta \* self.v \+ alpha \* x  
          
        \# 発火判定（Heavisideステップ関数）  
        spike \= self.surrogate\_function(self.v\_float \- self.v\_threshold)  
          
        \# リセット処理（Hard Reset or Soft Reset）  
        self.v \= self.v\_float \* (1\. \- spike) \# Hard resetの例  
          
        return spike

### **6.3 アーキテクチャの構築手順**

プロジェクトの進化は以下のフェーズで進めることを推奨します。

1. **Phase 1: ベースラインの移行 (Week 1-4)**  
   * 既存のTensorFlowや独自のNumPyコードをSpikingJelly (PyTorch) に移植します。  
   * まずは標準的なResNet-18 \+ LIF構成で、CIFAR-10データセットにおいて93%程度の精度再現を目指します。  
2. **Phase 2: ニューロンと符号化のアップグレード (Week 5-8)**  
   * LIFノードを上記のDA-LIFノードに置換します。  
   * 入力パイプラインにBit-Plane符号化を実装します。これにより、タイムステップ数を $T=4$ まで削減しても精度が維持されることを確認します。  
3. **Phase 3: Spikformerの実装 (Week 9-12)**  
   * CNNバックボーンを撤廃し、Spiking Self-Attentionブロックを構築します。  
   * パッチ分割（Patch Embedding）層の実装と、Q/K/V生成層のスパイク化を行います。  
   * ImageNetなどの大規模データセットでの学習を開始します。

## ---

**7\. ベンチマークと期待される成果**

本提案の実装により達成が期待される定量的指標を、現在のSOTA研究 3 に基づいて設定します。

### **7.1 目標精度 (Accuracy Metrics)**

以下の表は、ベースライン（一般的なSNN）と、本提案（DA-LIF \+ Spikformer \+ Hybrid Coding）導入後の目標値の比較です。

| データセット | 指標 | ベースライン (ResNet-18 LIF) | 目標値 (DA-LIF Spikformer) | 備考 |
| :---- | :---- | :---- | :---- | :---- |
| **CIFAR-10** | Top-1 Accuracy | \~93.5% | **\> 96.5%** | DA-LIFによる表現力向上で限界突破を目指す。 |
| **CIFAR-100** | Top-1 Accuracy | \~72.0% | **\> 82.0%** | クラス数が多い場合、Transformerの大域的特徴抽出が有利 4。 |
| **ImageNet** | Top-1 Accuracy | \~65.0% | **\> 80.0%** | SNNにとって80%の壁は長らくの課題だったが、Spikformer V2により到達可能 3。 |
| **DVS128 Gesture** | Top-1 Accuracy | \~95.0% | **\> 98.5%** | イベントカメラデータにおいて、時間的適応ニューロンは極めて強力に作用する。 |

### **7.2 エネルギー効率とレイテンシ**

SNNの真価はエネルギー効率にあります。本提案では、レイテンシ（タイムステップ数 $T$）を削減することで、総演算量とエネルギー消費を最小化します。

* **レイテンシ**: 従来 $T=20 \\sim 50$ $\\rightarrow$ **目標 $T=4$**  
* エネルギー消費 (SOPs):  
  SNNのエネルギーは、シナプス演算回数（SOPs: Synaptic Operations）で概算されます。

  $$E\_{total} \\approx N\_{spike} \\times E\_{AC}$$

  ここで $E\_{AC}$（加算エネルギー）は $E\_{MAC}$（積和演算エネルギー）の約1/5〜1/10です。  
  スパイクのスパース性（Sparsity）を高める学習則（正則化項の導入など）を併用することで、ANNと比較して 50%〜70% のエネルギー削減 を目指します 10。

### **7.3 検証プロトコル**

成果を正当に評価するため、以下の厳密な検証プロトコルを採用すべきです。

1. **同一条件での比較**: バッチサイズ、Optimizer（AdamW推奨）、Augmentation（RandAugment, MixUp）をANNのベースラインと統一する 16。  
2. **シード値の固定**: 少なくとも3回の試行の平均と分散を報告し、結果の再現性を保証する。  
3. **理論エネルギーの算出**: 単なる精度だけでなく、推論時の平均発火率（Firing Rate）を計測し、ハードウェア実装時の消費電力を推定する。

## ---

**8\. 結論：松芝電気SNNプロジェクトの未来**

本報告書で行った調査と分析は、松芝電気のSNNプロジェクトが現在直面しているであろう「精度の壁」と「実用性の欠如」が、決してSNNという技術自体の限界ではなく、採用しているアーキテクチャとニューロンモデルの旧態化に起因することを示唆しています。

学術論文に基づく以下の3つの柱：

1. **DA-LIFによる「時間」の学習** 1  
2. **Spiking Transformerによる「文脈」の理解** 3  
3. **Hybrid Temporal Codingによる「情報」の高速伝達** 5

これらを統合することで、本プロジェクトは単なる実験的なコードベースから脱却し、エッジAIやニューロモーフィックハードウェアの社会実装を牽引する、世界水準のテクノロジーへと昇華されます。特に、ImageNetにおける80%超えの精度と、CIFAR-100におけるSOTAの達成は、SNNがもはやANNの劣った代替品ではなく、エネルギー制約下における優れた選択肢であることを証明するマイルストーンとなるでしょう。

今こそ、静的なCNNの時代に別れを告げ、動的で適応的な次世代SNNの開発へと舵を切るべき時です。

### ---

**引用文献 (Inline Citations Source Mapping)**

本報告書内で参照された文献IDは以下の通りです。

* 7  
  : SNN Architecture Search Survey  
* 8  
  : Deep SNN Training Advancements  
* 11  
  : SpikedAttention Mechanism  
* 3  
  : Spikformer V2 / SpiLiFormer & Benchmarks  
* 1  
  : DA-LIF (Dual Adaptive LIF) Model details  
* 5  
  : Hybrid Temporal-8-Bit Spike Coding  
* 14  
  : SpikingJelly Framework & Documentation  
* 16  
  : CIFAR-100 Training Strategies & Noisy Student

#### **引用文献**

1. DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for Deep Spiking Neural Networks, 12月 30, 2025にアクセス、 [https://arxiv.org/html/2502.10422v1](https://arxiv.org/html/2502.10422v1)  
2. \[論文評述\] DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for ..., 12月 30, 2025にアクセス、 [https://www.themoonlight.io/tw/review/da-lif-dual-adaptive-leaky-integrate-and-fire-model-for-deep-spiking-neural-networks](https://www.themoonlight.io/tw/review/da-lif-dual-adaptive-leaky-integrate-and-fire-model-for-deep-spiking-neural-networks)  
3. Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket \- arXiv, 12月 30, 2025にアクセス、 [https://arxiv.org/html/2401.02020v1](https://arxiv.org/html/2401.02020v1)  
4. SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition \- CVF Open Access, 12月 30, 2025にアクセス、 [https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng\_SpiLiFormer\_Enhancing\_Spiking\_Transformers\_with\_Lateral\_Inhibition\_ICCV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.pdf)  
5. Hybrid Temporal-8-Bit Spike Coding for Spiking Neural Network Surrogate Training \- arXiv, 12月 30, 2025にアクセス、 [https://arxiv.org/html/2512.03879v1](https://arxiv.org/html/2512.03879v1)  
6. Hybrid Temporal-8-Bit Spike Coding for Spiking Neural Network Surrogate Training, 12月 30, 2025にアクセス、 [https://www.researchgate.net/publication/398312481\_Hybrid\_Temporal-8-Bit\_Spike\_Coding\_for\_Spiking\_Neural\_Network\_Surrogate\_Training](https://www.researchgate.net/publication/398312481_Hybrid_Temporal-8-Bit_Spike_Coding_for_Spiking_Neural_Network_Surrogate_Training)  
7. Spiking Neural Network Architecture Search: A Survey \- arXiv, 12月 30, 2025にアクセス、 [https://arxiv.org/html/2510.14235v1](https://arxiv.org/html/2510.14235v1)  
8. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 12月 30, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
9. DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for Deep Spiking Neural Networks \- arXiv, 12月 30, 2025にアクセス、 [https://arxiv.org/pdf/2502.10422](https://arxiv.org/pdf/2502.10422)  
10. \[Quick Review\] Spikformer: When Spiking Neural Network Meets Transformer \- Liner, 12月 30, 2025にアクセス、 [https://liner.com/review/spikformer-when-spiking-neural-network-meets-transformer](https://liner.com/review/spikformer-when-spiking-neural-network-meets-transformer)  
11. SpikedAttention: Training-Free and Fully Spike-Driven Transformer-to-SNN Conversion with Winner-Oriented Spike Shift for Softmax \- NIPS papers, 12月 30, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/7c9341ad0263428b5057d92f4d88dfa0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c9341ad0263428b5057d92f4d88dfa0-Paper-Conference.pdf)  
12. Our proposed coding method. | Download Scientific Diagram \- ResearchGate, 12月 30, 2025にアクセス、 [https://www.researchgate.net/figure/Our-proposed-coding-method\_fig1\_388439042](https://www.researchgate.net/figure/Our-proposed-coding-method_fig1_388439042)  
13. Improvement of Spiking Neural Network With Bit Planes and Color Models \- IEEE Xplore, 12月 30, 2025にアクセス、 [https://ieeexplore.ieee.org/iel8/6287639/10820123/11261679.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/11261679.pdf)  
14. spikingjelly \- PyPI, 12月 30, 2025にアクセス、 [https://pypi.org/project/spikingjelly/](https://pypi.org/project/spikingjelly/)  
15. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence arXiv:2310.16620v1 \[cs.NE\], 12月 30, 2025にアクセス、 [https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)  
16. FII ATNN 2025 \- Project \- Noisy CIFAR100 \- Kaggle, 12月 30, 2025にアクセス、 [https://www.kaggle.com/competitions/fii-atnn-2025-project-noisy-cifar-100](https://www.kaggle.com/competitions/fii-atnn-2025-project-noisy-cifar-100)  
17. SpikedAttention: Training-Free and Fully Spike-Driven Transformer ..., 12月 30, 2025にアクセス、 [https://papers.nips.cc/paper\_files/paper/2024/hash/7c9341ad0263428b5057d92f4d88dfa0-Abstract-Conference.html](https://papers.nips.cc/paper_files/paper/2024/hash/7c9341ad0263428b5057d92f4d88dfa0-Abstract-Conference.html)