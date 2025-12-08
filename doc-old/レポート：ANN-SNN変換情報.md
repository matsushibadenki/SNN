

# **ANN-SNN変換：アーキテクチャ分析と技術**

## **I. 序論：ANN-SNN変換のフロンティアと本レポートの目的**

### **A. 背景：LLMにおけるエネルギー効率の追求**

Gemma 3やLlama 4に代表される最先端の大規模言語モデル（LLM）は、その卓越した性能の代償として、巨大なパラメータ数と推論時の膨大な計算要求を抱えています 1。この計算コストは、特に推論フェーズにおけるエネルギー消費と運用コストの増大という深刻な問題を引き起こしています。

この課題に対する根本的な解決策として、スパイキングニューラルネットワーク（SNN）が注目されています。SNNは、生物学的な神経細胞の挙動を模倣し、「スパイク」と呼ばれる離散的なイベント信号を用いて情報を処理します 3。このイベント駆動型（event-driven）およびスパース（sparse）な計算原理により、SNNは特にニューロモルフィック・ハードウェア上で実行された場合、従来型の人工ニューラルネットワーク（ANN）と比較して桁違いのエネルギー効率を達成する潜在能力を持ちます 4。

高性能な事前学習済みANN-LLMをSNNに変換する技術（ANN-to-SNN conversion）は、ゼロからSNNを学習させる高額な再学習コストを回避しつつ、エネルギー効率の高い推論を実現するための最重要研究領域の一つとなっています 2。

### **B. 課題：SOTAアーキテクチャの複雑性**

初期のANN-SNN変換技術は、主にReLU活性化関数や標準的なLayerNormを用いた比較的単純なアーキテクチャを対象としていました 7。しかし、Gemma 3およびLlama 4のアーキテクチャは、これらの初期モデルとは比較にならないほど高度に最適化されています。

これらのモデルは、Grouped-Query Attention (GQA) 9、SwiGLUやGeGLUといったGLU系活性化関数 11、RMSNorm 9 といったコンポーネントを標準搭載しています。さらに、Gemma 3はQK-norm 9、Llama 4はMixture-of-Experts (MoE) 13 やiRoPE 14 といった、SNNの離散的なスパイク・ベースの計算モデルへ直接マッピングすることが極めて困難な、独自の非線形演算や動的ルーティング機構を導入しています。

### **C. 本レポートの構成と目的**

本レポートは、Gemma 3およびLlama 4モデルをSNNに変換するために必要な技術情報を網羅的に分析・整理することを目的とします。具体的には、研究開発者が両モデルのSNN変換プロジェクトを計画・実行する上で必要となる、詳細な「アーキテクチャ分解」と「コンポーネント別SNN変換戦略」を提供します。

そのために、(1) SNN変換の観点から両モデルのアーキテクチャを徹底的に比較分析し、(2) 各コンポーネント（アテンション、正規化、活性化、位置エンコーディング、MoE）のSNN実装に関する最新のSOTA（State-of-the-Art）研究を特定・詳述し、(3) 変換における固有の課題と実現可能性を評価します。

## **II. SNN変換のためのアーキテクチャ分析：Gemma 3 vs Llama 4**

SNN変換の対象となる主要コンポーネントを特定するため、両モデルのアーキテクチャを詳細に分解します。

### **A. Gemma 3：メモリ効率と安定化への特化**

Gemma 3は、Gemma 2のハイブリッド・アテンションをさらに発展させ、特に推論時のKVキャッシュのメモリ削減とトレーニングの安定性に焦点を当てたアーキテクチャを採用しています 9。

* **アテンション機構:** 「5-to-1 Interleaved Attention」と呼ばれる構造を採用しています 16。これは、5層のローカルアテンション（スライディングウィンドウ方式、ウィンドウスパン1024トークン 9）と1層のグローバルアテンションを交互に配置する設計です 9。この設計の主目的は、長いコンテキスト長（128Kトークン 9）でのKVキャッシュの爆発的なメモリ消費を劇的に削減することです 16。  
* **位置エンコーディング (PE):** デュアル周波数RoPE (Rotary Positional Embeddings) を使用します 9。長距離依存性を捉えるグローバルアテンション層には1MのRoPEベース周波数を、局所的な文脈を捉えるローカルアテンション層には10kのベース周波数を適用します 9。  
* **活性化関数:** Gemma 2で採用された「approximated GeGLU (Gated GELU)」 11 を継承している可能性が極めて高いと推測されます 18。  
* **正規化層 (1) \- RMSNorm:** Transformerブロックの標準正規化として、pre-normおよびpost-normにRMSNorm (Root Mean Square Normalization) を使用します 9。  
* **正規化層 (2) \- QK-norm:** Gemma 3の最も特徴的かつ重要な新規コンポーネントです。Gemma 2のソフトキャッピング機構に代わり「QK-norm」を導入しました 9。これは、アテンションスコアを計算する直前に、Query (Q) ベクトルとKey (K) ベクトルのそれぞれをL2ノルム（ユークリッド長）で正規化する手法です 20。

### **B. Llama 4：計算効率とパラメータスケーリングへの特化**

Llama 4は、MoE（Mixture-of-Experts）アーキテクチャを全面的に採用し、モデルの総パラメータ数を数兆規模にスケールさせつつ、推論時の活性化パラメータ（実質的な計算コスト）を抑制することに焦点を当てています 12。

* **基幹構造 (FFN):** Mixture-of-Experts (MoE) を採用しています 13。例えば、Llama 4 Maverickモデルは128の専門家（routed experts）と1つの共有専門家（shared expert）を持ちます 13。各トークンは、共有専門家と、ルーターによって動的に選ばれた1つの専門家にのみ送られます 13。  
* **アテンション機構:** Grouped-Query Attention (GQA) を採用しています 10。これはLlama 2の大型モデルで導入された技術 23 を継承しており、複数のQueryヘッドが単一のKey/Valueヘッドを共有することで、KVキャッシュのサイズを削減します 25。  
* **位置エンコーディング (PE):** 「iRoPE (interleaved RoPE)」と呼ばれる革新的な構造を採用しています 13。これは、従来のRoPEを使用する層と、位置エンコーディングを全く使用しない層（NoPE \- No Positional Encoding）を交互に配置する（interleave）アーキテクチャです 10。  
* **活性化関数:** SwiGLU (Gated SiLU) を使用しています 12。これはLlamaファミリーの標準的な活性化関数です 23。  
* **正規化層:** RMSNormを使用しています 12。これもLlamaファミリーの標準コンポーネントです 24。

### **C. SNN変換のための主要コンポーネント対照表**

Gemma 3とLlama 4のアーキテクチャをSNN変換の観点から比較した結果を、以下の表1にまとめます。

**表1：SNN変換のための主要コンポーネント対照表**

| アーキテクチャ要素 | Gemma 3 | Llama 4 | SNN変換における主要課題 |
| :---- | :---- | :---- | :---- |
| 基幹構造 | Dense Transformer | Mixture-of-Experts (MoE) 13 | MoEのスパイクベース・ルーティングの実装 30 |
| アテンション | 5:1 Interleaved Local (SWA) / Global 9 | Grouped-Query Attention (GQA) 10 | SNN-SWAの実装 31。GQAのK/V共有マッピング 32 |
| 正規化（主） | RMSNorm 9 | RMSNorm 12 | SNNによる二乗・平方根の近似 33 |
| 正規化（新規） | **QK-norm** 9 | なし | \*\*最難関。\*\*L2ノルムのスパイクベース実装 20 |
| 活性化関数 | GeGLU (推定) 11 | SwiGLU 12 | Gated Linear Unitのスパイク近似 33 |
| 位置エンコーディング | Dual-Frequency RoPE 9 | **iRoPE** (Interleaved RoPE/NoPE) 13 | RoPEの回転行列の近似 35。iRoPEの動的相互作用 |

### **D. アーキテクチャの方向性とSNN変換の親和性**

Gemma 3とLlama 4のアーキテクチャは、それぞれ異なる最適化のベクトル（メモリ効率 vs 計算効率）を向いています。Gemma 3の「QK-norm」 9 や「デュアル周波数RoPE」 9 は、トレーニングの安定化や連続値空間での精度維持（特に長距離依存性）に寄与するために導入された、複雑な*連続値*演算です。

対照的に、Llama 4の「MoE」 13 や「GQA」 10 は、計算グラフ自体を*構造的*にスパース化（疎）または単純化するアプローチです。SNNは、本質的にイベント駆動型であり、計算的スパース性（活性化したニューロンのみが計算を行う）を持つモデルです 3。

このSNNの計算的スパース性は、Llama 4の構造的スパース性と非常に高い親和性を持ちます。特にMoEのルーティング機構は、SNNのスパースな活性化を利用した動的なルーティング 30 に置き換えることが可能であり、これは単なる「近似」を超えた「相補的な設計」とさえ言えます。

一方で、Gemma 3の「QK-norm」は、SNNの離散的なスパイクドメインにおいて、高コストな連続値L2ノルム演算を*シミュレート*することを要求します。これはSNNの本来のエネルギー効率を著しく損なう*負債*となる可能性が高いです。

したがって、**Llama 4のアーキテクチャは、Gemma 3よりもSNNへの変換（またはSNNネイティブな再設計）に対して、原理的により親和性が高い**と推論されます。

## **III. Transformer SNN変換の基礎理論と最先端パラダイム**

Gemma 3とLlama 4の固有コンポーネントを議論する前に、Transformer SNN変換の基盤となる最新の汎用技術（SOTAパラダイム）を確立します。これらの手法は、両モデルの標準コンポーネント（FFN、RMSNorm）変換の基盤となります。

### **A. Transformer変換における中核的課題**

TransformerのSNN変換には、以下の3つの中核的課題が存在します。

1. **非線形関数の近似:** Transformerは、Softmax、GELU/SiLU、LayerNorm/RMSNormといった、SNNの単純なIFニューロン（ReLUと等価 7）では直接表現できない複雑な非線形関数に強く依存しています 7。  
2. **活性化外れ値 (Activation Outliers):** LLMは、一部のニューロンが極端に大きな活性化値（外れ値）を持つことが知られています 36。従来のSNN変換（レートコーディング）では、これらの外れ値を表現するためにニューロンの閾値を極端に大きく設定する必要があり、結果として他の微細な情報（通常の活性化値）がスパイクに変換されず、深刻な情報損失を引き起こします 36。  
3. **時間ステップ (T) と遅延:** 従来のレートコーディング 37 では、ANNの活性化値をSNNの発火率で近似するために、多数の時間ステップ（T）が必要でした 38。Tが小さいと近似誤差が増大し 39、Tが大きいとSNNの利点である低遅延性が失われます 38。

### **B. SOTAパラダイム1：LAS (Loss-less ANN-SNN Conversion)**

LAS 36 は、特にLLMの外れ値問題（課題2）と非線形関数（課題1）に対処するために設計された、高精度な変換手法です。

* **OAT (Outlier-Aware Threshold) ニューロン:** 課題2への解答。入力電位を「通常」と「外れ値」に分離する2つのマルチ閾値（MT）サブニューロンで構成されます 36。外れ値は高い閾値（$\\theta\_{out}$）で、通常値は低い閾値（$\\theta\_{nor}$）で処理されます 36。これにより、外れ値の情報を保持しつつ、通常値の解像度も維持します。  
* **HG (Hierarchically Gated) ニューロン:** 課題1への解答。GELUやLayerNormなどの複雑な非線形関数を近似するために設計された、階層的なゲートメカニズムを持つN個のFS（Few Spikes）サブニューロンで構成されます 36。関数を区分的に線形近似し、各区分を個別のFSニューロンが担当します 36。  
* **Gemma 3/Llama 4への適用:** LASは、両モデルのFFN（GeGLU/SwiGLU）およびRMSNormの非線形性を高精度に近似するための強力な基盤技術となります。

### **C. SOTAパラダイム2：One-Timestep (T=1) 変換**

このアプローチ 40 は、課題3である遅延問題を根本的に解決することを目指します。

* **理論 \- 時間-空間等価性 (Temporal-to-Spatial Equivalence):** 本手法は、マルチタイムステップ（T\>1）のIFニューロン（時間的蓄積）が、シングルタイムステップ（T=1）のマルチ閾値ニューロン（MTN）（空間的閾値）と*等価*であることを理論的に証明します 40。  
* **SFN (Scale-and-Fire Neuron):** この理論に基づき、SFNはT=1で動作します。従来のレートコーディングが「Tステップかけて発火率（平均）を計算する」のに対し、SFNは「1ステップで複数の閾値を参照し、複数のスパイクを出力する」（空間的コーディング）ことで、ANNの活性化値を瞬時に表現します 40。  
* **Gemma 3/Llama 4への適用:** T=1変換は、LLMの推論遅延を最小化する上で究極的な目標です。実際の変換では、LASのOATニューロンによる外れ値処理 36 と、SFNによるT=1コーディング 40 の長所を組み合わせた、ハイブリッド・ニューロンモデルの設計が求められる可能性があります。

## **IV. コンポーネント別・SNN変換戦略詳論**

セクションIIで特定した各コンポーネントを、セクションIIIの基礎理論に基づき、SNNに変換する具体的な戦略と課題を詳述します。

### **A. 正規化層 (Normalization Layers)**

#### **1\. SNN-RMSNorm (Gemma 3 & Llama 4\)**

* **ANN定義:** RMSNormは、LayerNormから平均（re-centering）項を除外し、二乗平均平方根（RMS）のみで正規化する手法です 12。計算が単純で効率的です 43。  
* **SNN変換戦略:** RMSNormは「二乗（Square）」と「平方根（Square Root）」という2つの非線形演算を含みます。この変換に関しては、"BrainTransformers" 33 がSNNRMSNormの実装を提案しています。  
* **実装:** この手法では、SquareApproximatorとSqrtApproximatorというカスタム・スパイキング・ニューロン・モジュールを設計します 33。これらは、LASのHGニューロン 36 と同様に、区分的線形近似を用いて二乗関数と平方根関数をスパイクドメインで実装します。  
* **結論:** SNN-RMSNormは、SOTA研究 33 に基づいて実装可能な、**解決済みの課題**です。

#### **2\. SNN-QK-norm (Gemma 3\)**

* **ANN定義:** Gemma 3の新規コンポーネント 9 であり、アテンションスコアを計算する直前に、QベクトルとKベクトルのそれぞれをL2ノルムが1になるように正規化します 20。  
* **SNN変換戦略:** QK-normをSNNで実装したという先行研究は存在しません。これは\*\*本変換における最大の技術的難関（フロンティア）\*\*です。  
* **設計提案:** QK-normの実装は、SNN-RMSNorm 33 の技術応用として設計可能です。  
  1. QK-normは、ベクトル$x$を $x / \\sqrt{\\sum(x^2) \+ \\epsilon}$ でスケーリングする演算です。  
  2. $\\sum(x^2)$ の部分は、SNN-RMSNormのSquareApproximator 33 を各ベクトル要素に適用し、それらのスパイク出力を時間的または空間的に*加算*（積分）することで計算できます。  
  3. $\\sqrt{\\dots}$ の部分は、SNN-RMSNormのSqrtApproximator 33 をステップ(2)の結果（スカラ値）に適用することで計算できます。  
  4. $x / \\dots$ の除算は、SNNにおいて最も困難な演算の一つです。しかし、ここでは$x$（スパイク列）に $1 / \\sqrt{\\dots}$（スケーリング係数）を乗算すると解釈できます。このスケーリング係数を、ニューロンの*閾値*または*膜電位の減衰係数*を動的に調整するパラメータとして使用することで、乗算的（抑制的）効果を近似できる可能性があります。  
* **結論:** SNN-QK-normは、SNNRMSNorm 33 の技術を基盤とした、高度なカスタム・スパイキング・オペレータの設計を必要とする**高難易度の未解決課題**です。（注記: QKFormer 44 は、"Q-K Attention"という*異なる*アテンション*機構*を提案するものであり、Gemma 3のQK-*norm*（正規化層）とは無関係です 44。）

### **B. 活性化関数 (Activation Functions)**

#### **1\. SNN-SwiGLU (Llama 4\)**

* **ANN定義:** SwiGLU(x, W, V) \= SiLU(xW) ⊗ xV 23。ここでSiLU(x) \= x \* sigmoid(x) です 29。  
* **SNN変換戦略:** "BrainTransformers" 33 は、SNNSiLUの実装を主要な貢献の一つとして挙げています。  
* **実装:** SiLUは乗算とS字曲線を含みます。これをSNNで近似するには、34 のような区分的近似や、LASのHGニューロン 36 のような汎用非線形近似器が使用されます。Gated Linear Unit (GLU) の構造（$A \\otimes B$）自体は、2つの並列なスパイク列の要素ごとのAND演算（乗算のバイナリ近似）として実装可能です。  
* **結論:** SNN-SwiGLUは、SOTA研究 33 に基づき、**実装可能な課題**です。

#### **2\. SNN-GeGLU (Gemma 3\)**

* **ANN定義:** GeGLU(x, W, V) \= GELU(xW) ⊗ xV 11。  
* **SNN変換戦略:** SwiGLUと同様ですが、SiLUの代わりにGELUを近似する必要があります。  
* **実装:** LASのHGニューロン 36 は、GELUのような複雑な非線形関数を近似する目的で明示的に設計されています 36。  
* **結論:** SNN-GeGLUは、LAS 36 のような汎用近似器を用いて、**実装可能な課題**です。

### **C. アテンション機構 (Attention Mechanisms)**

#### **1\. Spiking GQA (Llama 4\)**

* **ANN定義:** GQAは、MHA（Multi-Head Attention）とMQA（Multi-Query Attention）の中間的な手法です 25。QヘッドをGグループに分割し、各グループが単一のK/Vヘッドを共有します 26。  
* **SNN変換戦略:** GQAのSNN変換は、MHAのSNN変換 49 に対する*構造的変更*として扱われます。  
* **実装:** SNN-MHAでは、Q, K, Vそれぞれが個別のヘッドを持ちます。SNN-GQAでは、Qヘッドの数はH個（例えば32個）のままですが、KヘッドとVヘッドの数をG個（例えば8個）に減らし、対応するQヘッドグループ（この例では4つ）が同じSNN-K/Vヘッドの出力を参照するように計算グラフを変更します 32。  
* **結論:** これはSNNのニューロン・レベルでの新しい近似を必要とせず、*ネットワーク・トポロジーの変更*で対応可能です。**中難易度の課題**です（実装は容易ですが、ANNと同様の品質/速度トレードオフをSNNドメインで再検証する必要があるため 51）。

#### **2\. Spiking Interleaved Attention (Gemma 3\)**

* **ANN定義:** 5層のローカルアテンション（スライディングウィンドウ, SWA）と1層のグローバルアテンションの繰り返しです 9。  
* **SNN変換戦略:** これは2つの既知のSNNアテンション技術のハイブリッドです。  
* **実装 (Local):** SWAのSNN変換は、"SpikingBrain"モデル 31 で実証されています。これは、アテンションマップをスパースなスライディングウィンドウに制限することで、計算量を削減する手法です 31。SNNのスパースな計算と親和性が高いです。  
* **実装 (Global):** グローバルアテンション層は、標準的なSNN-Self-Attention 5 として実装されます。  
* **結論:** SNN-SWA 31 とSNN-Global Attention 5 という2つの既存技術をGemma 3の5:1の比率で交互に積層することで実装できます。**中難易度の課題**です（個々の技術は存在するが、そのハイブリッド構造のSNNドメインでの安定性・性能は未知数）。

### **D. 位置エンコーディング (Positional Encodings)**

#### **1\. SNN-RoPE (Gemma 3 & Llama 4\)**

* **ANN定義:** RoPEは、絶対位置情報を回転行列としてQとKのベクトルに適用し、アテンション計算において暗黙的に相対位置情報をエンコードする手法です 56。  
* **SNN変換戦略:** RoPEの核となる「回転行列の乗算」は、SNNで直接実行することが非常に困難です。したがって、特性を*近似*する別のアプローチが必要です。  
* **実装:** 2つのアプローチが考えられます。  
  1. **近似:** 59 では、Gray Code（グレイコード）を利用してRoPEの特性（相対位置情報）を*近似*するSNN向けRPE（Relative Positional Encoding）手法を提案しています。  
  2. **条件付き生成:** 35 では、SNNが浮動小数点形式のPEをサポートしない問題に対し、「条件付き位置エンベディング生成器」を用いて*スパイク形式*のRPEを生成し、入力パッチに加算する手法を提案しています。  
* **結論:** RoPEの忠実な変換は困難であり、35 や 59 のような*近似的*または*生成的*なSNNネイティブ手法で代替する必要があります。**高難易度の課題**です。

#### **2\. SNN-iRoPE (Llama 4\)**

* **ANN定義:** RoPE層とNoPE（位置エンコーディングなし）層の交互配置です 10。  
* **SNN変換戦略:** これは本レポートにおける最も深刻な*未解決の研究課題*の一つです。ANNにおけるNoPE層は、先行する層（RoPE層など）の表現から位置情報を*暗黙的に学習*・伝達できるという知見に基づいています 60。  
* **iRoPEの変換パラドックス:**  
  1. SNN-iRoPEを実装する場合、RoPE層は上記D.1で述べたような*近似的*なSNN-RoPE 35 に置き換えられます。このSNN-RoPEは、ANNのRoPEと比較して必ず*近似誤差*（情報の量子化・損失）を含みます。  
  2. 続くNoPE層（これはSNN変換が容易。単にPEモジュールを省略するだけ）は、この「ノイズが多く、量子化された」スパイクベースの位置情報を受け取ることになります。  
  3. ANNのNoPE層が機能した「暗黙的な位置学習」メカニズムが、SNNの離散的・確率的なダイナミクス、および先行するSNN-RoPE層からの*近似誤差*の存在下でも同様に機能するという保証は全くありません。  
* **結論:** iRoPEの「RoPE層とNoPE層の相互作用」そのものが、ANN-SNN変換によって崩壊するリスクがあります。SNN-iRoPEの変換は、個々の層の変換を単純に組み合わせるだけでは不十分であり、ANNの暗黙的な学習特性がSNNドメインで維持されるかを検証する、**極めて高度な基礎研究を必要とするフロンティア課題**です。

### **E. 基幹構造：Spiking MoE (Llama 4\)**

* **ANN定義:** Llama 4のFFN層はMoEで構成されます 13。従来のMoEは、入力トークンをどの専門家（Expert）に送るかを決めるために、SoftmaxとTopKを用いたゲート（ルーター）を使用します 30。このルーティング計算自体が、特にSoftmaxにおいて非線形かつ高コストです 62。  
* **SNN変換戦略:** Llama 4のMoE変換には、**決定的なSOTA研究が存在します**。NIPS 2024で発表された「**SEMM (Spiking Experts Mixture Mechanism)**」です 30。  
* **実装 (SEMM):** SEMMは、ANN-MoEのSoftmax/TopKルーティングを、SNNの特性を利用したスパイクベースのルーティングに置き換えます 30。  
  1. **スパイク駆動:** ルーター(R)と専門家(E)の両方がスパイクシーケンスを出力します 30。  
  2. **乗算回避:** 専門家の選択は、ルーターのスパイク列(R)と専門家のスパイク列(E)の間の要素ごとのアダマール積（Hadamard product, $\\odot$）と加算で実行されます 30。スパイク（0か1）のアダマール積は、実質的な乗算を必要としない単純なANDゲート（マスク処理）となり、SNNと非常に親和性が高いです 30。  
  3. **動的スパース性:** ANNのTopK（固定的なスパース化）とは異なり、SEMMはSNN固有の「スパースな発火」自体をルーティングに利用します 30。発火しなかったニューロン（トークン）は、どの専門家にも送られず、計算が動的にスキップされます。  
* **結論:** Spiking MoEは、SEMM 30 という明確な実装指針が存在します。これはLlama 4のアーキテクチャとSNNの計算原理が*相乗効果*を生む稀有な例であり、**実装可能性が非常に高い中核的機会**です。

## **V. 統合的変換パスウェイと実装に関する考察**

### **A. ANNコンポーネントからSNN等価物への変換戦略マトリクス**

本レポートの分析結果を、SNN変換のための実用的なロードマップとして表2に集約します。

**表2：ANNコンポーネントからSNN等価物への変換戦略マトリクス**

| ANNコンポーネント | ターゲットモデル | SNN等価手法 / SOTA参考文献 | 変換難易度 |
| :---- | :---- | :---- | :---- |
| RMSNorm | 両方 | SNNRMSNorm (Square/Sqrt Approximator) / "BrainTransformers" 33 | **中** (SOTA実装は存在するが複雑) |
| QK-norm | Gemma 3 | *設計提案:* SNNRMSNorm 33 の応用によるカスタムL2ノルム・オペレータ / 20 | **高** (SOTA実装が存在しない) |
| SwiGLU | Llama 4 | SNNSiLU \+ Spiking GLU / "BrainTransformers" 33 | **中** (SOTA実装が存在する) |
| GeGLU | Gemma 3 | SNN-GELU (HG Neuron) \+ Spiking GLU / "LAS" 36 | **中** (SOTA実装が存在する) |
| GQA | Llama 4 | SNN-MHAの構造的変更（K/Vヘッド共有） / 26 | **低** (トポロジー変更で対応可) |
| Interleaved Attn | Gemma 3 | SNN-SWA \+ SNN-Global Attn のハイブリッド / "SpikingBrain" 5 | **中** (既存技術の組み合わせ) |
| RoPE | 両方 | 近似RPE（Gray Code）または生成的RPE / 35 | **高** (忠実な変換は不可。近似が必要) |
| iRoPE | Llama 4 | SNN-RoPEとSNN-NoPE 60 の交互配置 | **フロンティア** (暗黙的学習のSNNでの機能性が未知数) |
| MoE | Llama 4 | SEMM (Spiking Experts Mixture Mechanism) / NIPS 2024 30 | **中** (決定的なSOTA実装が存在する) |

### **B. Gemma 3 変換のための統合パイプライン考察**

Gemma 3の変換は、「QK-norm」 9 という単一だが非常に困難なカスタム・スパイキング・オペレータの開発にその成否が依存します。  
提案されるパイプラインは以下の通りです。

1. **標準コンポーネントの変換:** RMSNormをSNNRMSNorm 33 に、GeGLUをSNN-GeGLU 36 に変換します。  
2. **アテンション変換:** 5:1の比率でSNN-SWA 31 とSNN-Global Attention 5 を実装します。  
3. **PE変換:** デュアル周波数RoPE 9 を、その周波数特性を近似したSNN-RPE 35 に置き換えます。  
4. **最難関:** SNN-QK-norm 20 モジュールを（IV.A.2の設計提案に基づき）新規に開発し、アテンション・ブロックに挿入します。

### **C. Llama 4 変換のための統合パイプライン考察**

Llama 4の変換は、個々のコンポーネント（MoE, GQA）に明確なSOTAソリューション（SEMM 30, 32）が存在する一方で、「iRoPE」 13 というシステム・レベルでの深刻なリスクを抱えています。  
提案されるパイプラインは以下の通りです。

1. **標準コンポーネントの変換:** RMSNormをSNNRMSNorm 33 に、SwiGLUをSNNSiLU 33 に変換します。  
2. **アテンション変換:** GQA 10 のトポロジー（K/Vヘッド共有）をSNN-Attentionに実装します 32。  
3. **最重要:** FFN層をSEMM 30 を用いたSpiking MoEブロックに置き換えます。  
4. **最大リスク:** iRoPE 14 を、SNN-RoPE 35 層とSNN-NoPE層の交互配置として実装し、ANNの暗黙的位置学習がSNNドメインでも維持されるかを徹底的に評価・検証します。

### **D. 実装フレームワークと「SOTAツーリング・ギャップ」**

Gemma 3およびLlama 4のSNN変換を実行する上で、既存のSNNフレームワークは現状では不十分です。

SpikingJellyのann2snn.Converter 63 は、単純なCNN/FeedforwardネットのReLU-to-IF変換に焦点を当てており、Max Poolingさえ理想的な解がないと述べています 63。Gemma 3/Llama 4の複雑なコンポーネント（QK-norm, MoE, RMSNorm）は、そのサポート範囲を完全に超えています。

同様に、IntelのLAVAフレームワークのlava.lib.dl.bootstrap 64 も、ANN-SNN*訓練*アプローチであり、Gemma 3/Llama 4のような巨大な*事前学習済み*モデルの*変換*を直接サポートする機能は記載されていません 64。

この「SOTAツーリング・ギャップ」は、Gemma 3 / Llama 4のSNN変換が、既存のライブラリを単に利用するだけでは実行不可能であることを意味します。本レポートのセクションIVで詳述した、SNNRMSNorm 33、SEMM 30、SNNSiLU 33、SNN-RPE 35、そしてカスタムSNN-QK-normといったコンポーネントすべてを、PyTorchやTensorFlowといった基盤フレームワーク上で、SOTA論文（36等）を参照しながら**フルカスタムで手動実装**する必要があります。

したがって、本レポートは、その「フルカスタム実装」のための技術的な青写真として機能します。変換に必要な情報は、既存のツール・ドキュメントには存在せず、本レポートで分析・統合された最先端のアカデミックなSOTA研究群そのものです。

#### **引用文献**

1. Survey and Evaluation of Converging Architecture in LLMs based on Footsteps of Operations \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2410.11381v1](https://arxiv.org/html/2410.11381v1)  
2. Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications, 11月 9, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2025/papers/Bu\_Inference-Scale\_Complexity\_in\_ANN-SNN\_Conversion\_for\_High-Performance\_and\_Low-Power\_Applications\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.pdf)  
3. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 11月 9, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
4. \[2508.07710\] Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformer \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2508.07710](https://arxiv.org/abs/2508.07710)  
5. Introducing Accurate Addition-Only Spiking Self-Attention for Transformer \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2503.00226v1](https://arxiv.org/html/2503.00226v1)  
6. Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2409.02111v1](https://arxiv.org/html/2409.02111v1)  
7. LAS: Loss-less ANN-SNN Conversion for Fully Spike-Driven Large Language Models, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2505.09659v1](https://arxiv.org/html/2505.09659v1)  
8. IC-SNN: Optimal ANN2SNN Conversion at Low Latency \- MDPI, 11月 9, 2025にアクセス、 [https://www.mdpi.com/2227-7390/11/1/58](https://www.mdpi.com/2227-7390/11/1/58)  
9. Gemma 3 Technical Report \- Googleapis.com, 11月 9, 2025にアクセス、 [https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)  
10. 5 Techniques in Llama 4 That Improve Performance and Efficiency \- ApX Machine Learning, 11月 9, 2025にアクセス、 [https://apxml.com/posts/llama-4-model-efficiency-performance](https://apxml.com/posts/llama-4-model-efficiency-performance)  
11. Gemma 2: Improving Open Language Models at a Practical Size \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2408.00118v1](https://arxiv.org/html/2408.00118v1)  
12. Llama (language model) \- Wikipedia, 11月 9, 2025にアクセス、 [https://en.wikipedia.org/wiki/Llama\_(language\_model)](https://en.wikipedia.org/wiki/Llama_\(language_model\))  
13. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation, 11月 9, 2025にアクセス、 [https://ai.meta.com/blog/llama-4-multimodal-intelligence/](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)  
14. Llama 4's Architecture Deconstructed: MoE, iRoPE, and Early Fusion Explained \- Medium, 11月 9, 2025にアクセス、 [https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067](https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067)  
15. \[2503.19786\] Gemma 3 Technical Report \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786)  
16. Gemma explained: What's new in Gemma 3 \- Google Developers Blog, 11月 9, 2025にアクセス、 [https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)  
17. Essential Highlights from Gemma 2: Improving Open Language Models at a Practical Size, 11月 9, 2025にアクセス、 [https://medium.com/@imabhi1216/essential-highlights-from-gemma-2-improving-open-language-models-at-a-practical-size-4acd315caaa1](https://medium.com/@imabhi1216/essential-highlights-from-gemma-2-improving-open-language-models-at-a-practical-size-4acd315caaa1)  
18. Transformer Activation Functions and their Details \- JoeLogs, 11月 9, 2025にアクセス、 [https://sathvikjoel.github.io/posts/tech/05032024\_activationfunctions/](https://sathvikjoel.github.io/posts/tech/05032024_activationfunctions/)  
19. Understanding MatFormer — Matryoshka Nested Transformers For Compute Efficient Inference | by Bhavin Jawade, Ph.D, 11月 9, 2025にアクセス、 [https://bhavinjawade.medium.com/understanding-matformer-0b5cb3a500e2](https://bhavinjawade.medium.com/understanding-matformer-0b5cb3a500e2)  
20. Normalization (machine learning) \- Wikipedia, 11月 9, 2025にアクセス、 [https://en.wikipedia.org/wiki/Normalization\_(machine\_learning)](https://en.wikipedia.org/wiki/Normalization_\(machine_learning\))  
21. Normalization Techniques in Transformer-Based LLMs: LayerNorm, RMSNorm, and Beyond, 11月 9, 2025にアクセス、 [https://sushant-kumar.com/blog/normalization-in-transformer-based-llms](https://sushant-kumar.com/blog/normalization-in-transformer-based-llms)  
22. Meta Llama \- Hugging Face, 11月 9, 2025にアクセス、 [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)  
23. The Llama Model Family: Architecture Evolution from Llama (1) to Llama 4 | by Bahadır AKDEMİR | Oct, 2025 | Medium, 11月 9, 2025にアクセス、 [https://medium.com/@akdemir\_bahadir/the-llama-model-family-architecture-evolution-from-llama-1-to-llama-4-76eb405e5f91](https://medium.com/@akdemir_bahadir/the-llama-model-family-architecture-evolution-from-llama-1-to-llama-4-76eb405e5f91)  
24. A Comprehensive Comparison of Meta's Llama Series and the Classic Transformer Architecture \- A.H. \- Allen Shaing, 11月 9, 2025にアクセス、 [https://allenshaing.com/blog/llama-series-transformer/](https://allenshaing.com/blog/llama-series-transformer/)  
25. What is grouped query attention (GQA)? \- IBM, 11月 9, 2025にアクセス、 [https://www.ibm.com/think/topics/grouped-query-attention](https://www.ibm.com/think/topics/grouped-query-attention)  
26. Grouped Query Attention (GQA) \- Intel, 11月 9, 2025にアクセス、 [https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-1/grouped-query-attention-gqa.html](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-1/grouped-query-attention-gqa.html)  
27. Demystifying GQA — Grouped Query Attention for Efficient LLM Pre-training \- Medium, 11月 9, 2025にアクセス、 [https://medium.com/data-science/demystifying-gqa-grouped-query-attention-3fb97b678e4a](https://medium.com/data-science/demystifying-gqa-grouped-query-attention-3fb97b678e4a)  
28. Aman's AI Journal • Models • LLaMA, 11月 9, 2025にアクセス、 [https://aman.ai/primers/ai/LLaMA/](https://aman.ai/primers/ai/LLaMA/)  
29. Exploring SwiGLU : The Activation Function Powering Modern LLMs | by Selssabil | Medium, 11月 9, 2025にアクセス、 [https://medium.com/@s\_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7](https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7)  
30. Spiking Transformer with Experts Mixture \- NIPS papers, 11月 9, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/137101016144540ed3191dc2b02f09a5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/137101016144540ed3191dc2b02f09a5-Paper-Conference.pdf)  
31. SpikingBrain Technical Report: Spiking Brain-inspired Large Models \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)  
32. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2305.13245v3](https://arxiv.org/html/2305.13245v3)  
33. BrainTransformers: SNN-LLM \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2410.14687v2](https://arxiv.org/html/2410.14687v2)  
34. BrainTransformers: SNN-LLM \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2410.14687v1](https://arxiv.org/html/2410.14687v1)  
35. Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2401.02020v1](https://arxiv.org/html/2401.02020v1)  
36. LAS: Loss-less ANN-SNN Conversion for Fully Spike-Driven ... \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2505.09659](https://arxiv.org/abs/2505.09659)  
37. ICML Poster Differential Coding for Training-Free ANN-to-SNN Conversion, 11月 9, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/45408](https://icml.cc/virtual/2025/poster/45408)  
38. Differential Coding for Training-Free ANN-to-SNN Conversion \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2503.00301v1](https://arxiv.org/html/2503.00301v1)  
39. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2510.23383v1](https://arxiv.org/html/2510.23383v1)  
40. Achieving High-performance ANN-to-SNN Conversion via ... \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2510.23383](https://arxiv.org/abs/2510.23383)  
41. \[1910.07467\] Root Mean Square Layer Normalization \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)  
42. Root Mean Square Layer Normalization \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/pdf/1910.07467](https://arxiv.org/pdf/1910.07467)  
43. Root Mean Square Layer Normalization \- OpenReview, 11月 9, 2025にアクセス、 [https://openreview.net/references/pdf?id=S1qBAf6rr](https://openreview.net/references/pdf?id=S1qBAf6rr)  
44. QKFormer: Hierarchical Spiking Transformer using QK Attention, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2403.16552](https://arxiv.org/abs/2403.16552)  
45. GLU Variants Improve Transformer \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/pdf/2002.05202](https://arxiv.org/pdf/2002.05202)  
46. \[2410.14687\] BrainTransformers: SNN-LLM \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2410.14687](https://arxiv.org/abs/2410.14687)  
47. Performance evaluation of BrainTransformer on various tasks \- ResearchGate, 11月 9, 2025にアクセス、 [https://www.researchgate.net/figure/Performance-evaluation-of-BrainTransformer-on-various-tasks\_tbl1\_385108209](https://www.researchgate.net/figure/Performance-evaluation-of-BrainTransformer-on-various-tasks_tbl1_385108209)  
48. Understanding, Using, and Finetuning Gemma \- Lightning AI, 11月 9, 2025にアクセス、 [https://lightning.ai/lightning-ai/studios/understanding-using-and-finetuning-gemma](https://lightning.ai/lightning-ai/studios/understanding-using-and-finetuning-gemma)  
49. Attention Is All You Need \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/1706.03762v7](https://arxiv.org/html/1706.03762v7)  
50. \[2305.13245\] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)  
51. Grouped-Query Attention (GQA) \- Sebastian Raschka, 11月 9, 2025にアクセス、 [https://sebastianraschka.com/llms-from-scratch/ch04/04\_gqa/](https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/)  
52. SpikingBrain Technical Report: Spiking Brain-inspired Large Models (2509.05276v1) \- Emergent Mind, 11月 9, 2025にアクセス、 [https://www.emergentmind.com/papers/2509.05276](https://www.emergentmind.com/papers/2509.05276)  
53. SpikingBrain: a revolutionary brain-inspired Chatgpt made in China \- Buonaiuto@Work, 11月 9, 2025にアクセス、 [https://buonaiuto.work/spikingbrain-a-revolutionary-brain-inspired-chatgpt-made-in-china/](https://buonaiuto.work/spikingbrain-a-revolutionary-brain-inspired-chatgpt-made-in-china/)  
54. Daily Papers \- Hugging Face, 11月 9, 2025にアクセス、 [https://huggingface.co/papers?q=norm-aware%20spikiness](https://huggingface.co/papers?q=norm-aware+spikiness)  
55. Accelerating Attention with Stochastic Computing in Spiking Networks \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/pdf/2402.09109](https://arxiv.org/pdf/2402.09109)  
56. \[2104.09864\] RoFormer: Enhanced Transformer with Rotary Position Embedding \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)  
57. RoFormer: Enhanced Transformer with Rotary Position Embedding \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/pdf/2104.09864](https://arxiv.org/pdf/2104.09864)  
58. Round and Round We Go\! What makes Rotary Positional Encodings useful? \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/abs/2410.06205](https://arxiv.org/abs/2410.06205)  
59. Toward Relative Positional Encoding in Spiking Transformers \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2501.16745v1](https://arxiv.org/html/2501.16745v1)  
60. The Impact of Positional Encoding on Length Generalization in Transformers \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/pdf/2305.19466](https://arxiv.org/pdf/2305.19466)  
61. MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts \- arXiv, 11月 9, 2025にアクセス、 [https://arxiv.org/html/2402.00893v1](https://arxiv.org/html/2402.00893v1)  
62. SpikedAttention: Training-free and fully spike-driven transformer-to-SNN conversion with winner-oriented spike shift for softmax operation \- NIPS papers, 11月 9, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/7c9341ad0263428b5057d92f4d88dfa0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/7c9341ad0263428b5057d92f4d88dfa0-Paper-Conference.pdf)  
63. ANN2SNN — spikingjelly alpha 文档, 11月 9, 2025にアクセス、 [https://spikingjelly.readthedocs.io/zh-cn/latest/activation\_based\_en/ann2snn.html](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/ann2snn.html)  
64. Deep Learning — Lava documentation \- Lava framework, 11月 9, 2025にアクセス、 [https://lava-nc.org/dl.html](https://lava-nc.org/dl.html)