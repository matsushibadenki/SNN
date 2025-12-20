# **ポスト・フォン・ノイマン知能：逆伝播法と行列演算を超越する生体模倣的・熱力学的計算パラダイム**

## **1\. 序論：エネルギーの壁とシリコンの限界**

現代の人工知能（AI）、とりわけ大規模言語モデル（LLM）の急速な進歩は、計算能力の指数関数的な増大に依存しています。しかし、この成長曲線は物理的かつ経済的な限界、いわゆる「エネルギーの壁」に直面しつつあります。GPT-3のようなモデルの学習には、数千メガワット時（MWh）の電力を消費し、これは小規模な都市の1日分の消費電力に匹敵します1。一方、汎用的な知能を具現化している唯一の存在である人間の脳は、わずか20ワット程度の消費電力で稼働しています2。この6〜9桁にも及ぶ効率の格差は、現在の主流であるディープラーニングのアプローチ――高精度な浮動小数点演算、密な行列積、大域的な誤差逆伝播法（Backpropagation）、そしてフォン・ノイマン型アーキテクチャ――が、本質的に非効率であることを示唆しています。

本報告書は、この現状を打破し、人間の脳の効率に並び、最終的にはそれを凌駕するための技術的ロードマップを提示します。その核心は、**「大域的な誤差逆伝播法からの脱却（Local Learning）」**、**「行列演算の排除（Accumulation-based / Analog Compute）」**、そして\*\*「熱力学的・確率的計算（Thermodynamic Computing）」\*\*の融合にあります。私たちは、計算を論理的な操作としてではなく、物理的な緩和過程として再定義する必要があります。

本稿では、誤差逆伝播法に代わる生物学的妥当性の高い学習アルゴリズム（Forward-Forwardアルゴリズム、平衡伝播法、予測符号化）を詳細に分析し、それらを支える次世代ハードウェア（アナログ・インメモリコンピューティング、スパイキングニューラルネットワーク、熱力学チップ）との統合可能性を論じます。これにより、シリコンの限界を超え、熱力学的な極限（ランダウアーの限界）に迫る「超・生物的効率（Super-Biological Efficiency）」を持つ知能システムの構築可能性を探求します。

## ---

**2\. アルゴリズムの革命：誤差逆伝播法（Backpropagation）の呪縛からの解放**

現在のディープラーニングの成功は、誤差逆伝播法（BP）による功績が大きいですが、BPは生物学的な脳の学習機構とは大きく異なり、かつハードウェア実装上の重大なボトルネックを抱えています。BPは、出力層から入力層に向かって誤差勾配を伝播させるために、順伝播時の重み行列の転置行列（$W^T$）を必要としますが、生物学的シナプスにおいては前向きの結合と後ろ向きの結合が物理的に別個であり、重みの対称性を保証する機構が存在しません（重み輸送問題）4。また、BPは順伝播の活性化値をメモリに保持し、逆伝播が完了するまで重みを更新できないため、メモリ消費量が膨大になり、層ごとの並列処理を阻害します（Update Locking）6。

### **2.1 Forward-Forwardアルゴリズム（FF）：推論のみによる学習**

ジェフリー・ヒントンによって提案されたForward-Forward（FF）アルゴリズムは、BPの制約を打破するための最も有力な候補の一つです。FFは、逆伝播（Backward Pass）を完全に排除し、2つの異なる順伝播（Forward Pass）のみを用いてネットワークを学習させます8。

#### **メカニズムと「Goodness」関数**

FFの核となるアイデアは、各層が局所的な目的関数、すなわち「Goodness（良さ）」を持つことです。通常、Goodnessはニューロンの活性化の二乗和（エネルギー）として定義されます。学習は以下の2つのフェーズで行われます。

1. ポジティブ・パス（Positive Pass）:  
   ネットワークに「実データ（Positive Data）」を入力します。教師あり学習の場合、これは正しいラベルが埋め込まれた画像やテキストシーケンスです。このパスでは、各層の重みは、その層のGoodnessを閾値以上に高めるように更新されます9。  
2. ネガティブ・パス（Negative Pass）:  
   ネットワークに「偽データ（Negative Data）」を入力します。これは、誤ったラベルを持つデータや、生成モデルによって作られた不自然なデータです。このパスでは、各層の重みは、Goodnessを閾値以下に下げるように更新されます8。

数理的には、ある層のニューロン活動を $y\_j$ とすると、入力がポジティブである確率はシグモイド関数を用いて $p(\\text{positive}) \= \\sigma(\\sum\_j y\_j^2 \- \\theta)$ とモデル化されます。FFは、ポジティブデータに対してはこの値を高く、ネガティブデータに対しては低くするように、層ごとに独立して勾配降下法を適用します12。

#### **BPに対する優位性とRNNへの応用**

FFの最大の利点は、**層間の独立性**にあります。第$L$層の更新は、第$L+1$層からの誤差信号を待つ必要がありません。データが層を通過した直後に、その層の重みを更新し、活性化メモリを解放することができます。これにより、パイプライン処理が可能になり、メモリ効率が劇的に向上します13。

さらに、FFは\*\*リカレントニューラルネットワーク（RNN）の学習において革新的な意味を持ちます。従来のBPTT（Backpropagation Through Time）では、時間方向に展開されたネットワーク全体にわたって誤差を逆流させる必要がありましたが、FFを用いれば、各タイムステップごとに局所的に学習を行うことが可能です。これは、途切れることのないストリームデータ（動画や継続的なテキスト）を、一時停止することなく学習し続ける「オンライン学習」を実現する鍵となります14。最近の研究では、FFをRNNに適用したSelf-Contrastive Forward-Forward (SCFF)\*\*などが提案されており、時系列データ処理における有効性が示されています16。

### **2.2 平衡伝播法（Equilibrium Propagation）：物理法則による勾配計算**

平衡伝播法（Equilibrium Propagation: EP）は、エネルギーベースモデル（EBM）とバックプロパゲーションのギャップを埋めるアルゴリズムとして、Yoshua Bengioらによって提案されました。EPは、システムがエネルギー関数を最小化する状態へ「緩和（Relaxation）」する物理現象を利用します17。

#### **物理的緩和による学習プロセス**

EPは、動的システム（リカレントネットワークなど）が固定点（平衡状態）に収束する性質を利用します。学習は以下の2相で行われます。

1. **自由相（Free Phase）:** 入力を固定（クランプ）し、ネットワークを自由に時間発展させ、エネルギー最小状態（平衡状態 $s^0$）に到達させます。この状態が「予測」に相当します18。  
2. **ナッジ相（Nudged Phase）:** 出力層に対して、正解ラベルに近づくような微小な力（Nudge）を加えます。これにより、システムは新たな平衡状態 $s^\\beta$ に緩和します。この状態は、予測誤差がわずかに減少した状態です19。

重みの更新則は非常に単純で、これら2つの平衡状態における局所的な神経活動の差異に基づきます。

$$\\Delta W \\propto (s^\\beta\_i s^\\beta\_j \- s^0\_i s^0\_j)$$

驚くべきことに、ナッジの強さがゼロに近づく極限において、この更新則はバックプロパゲーションによって計算される勾配と数学的に一致することが証明されています17。

#### **アナログハードウェアとの親和性**

EPの真価は、デジタルシミュレーションではなく、**アナログ回路や物理系**での実装において発揮されます。抵抗ネットワークやスピン系などの物理システムは、キルヒホッフの法則やハミルトニアンに従って、計算コストをかけることなく自然に平衡状態へと緩和します。EPを用いれば、複雑な微分計算を行うことなく、物理系が「勝手に」到達した状態を測定するだけで学習が可能になります。これは、GPUによる行列演算を完全に排除し、物理現象そのものを計算リソースとして利用する道を開きます20。最近の研究では、EPを量子系に拡張した**Quantum Equilibrium Propagation**も提案されており、オンサーガーの相反定理を利用して量子ハミルトニアンのパラメータを効率的に学習させる可能性が示されています22。

### **2.3 予測符号化（Predictive Coding）：脳の階層的推論**

予測符号化（Predictive Coding: PC）は、神経科学における主要な理論であり、脳を「予測マシン」としてモデル化します。この枠組みでは、上位の層が下位の層の活動を予測し、その「予測誤差」のみが上位へ伝播されます23。

#### **局所的推論と学習**

PCネットワーク（PCN）では、推論と学習が明確に定義されたエネルギー最小化プロセスとして記述されます。

* **推論（Inference）:** 重みを固定した状態で、各ニューロンの活動値を更新し、予測誤差（自由エネルギー）を最小化します。これは高速な緩和過程です24。  
* **学習（Learning）:** ニューロンの活動値が平衡に達した後、残留する予測誤差をさらに減らすようにシナプス重みを更新します。

BPとは異なり、PCの計算は完全に局所的です。各ニューロンは、自身が受け取った予測誤差と、自身が生成した予測のみを知っていればよく、大域的な情報のやり取りを必要としません。さらに、PCはフィードバック結合を本質的に含んでいるため、再帰的な構造を持つ生成モデルとして機能します25。

#### **スケーラビリティとTransformerへの応用**

従来のPCは深いネットワークでの学習が不安定であるという課題がありましたが、最近の研究（$\\mu$PCやIncremental PCなど）により、100層を超えるディープニューラルネットワークや、Transformerのような複雑なアーキテクチャの学習が可能になりつつあります24。PCのエネルギー効率の鍵は**スパース性**にあります。予測が正確であれば誤差信号はゼロに近くなり、情報の伝播（およびそれに伴うエネルギー消費）が発生しません。これは、驚き（予測誤差）のみに反応する生物学的ニューロンの省エネ戦略を模倣しています24。

## ---

**3\. アーキテクチャの革新：行列演算の終焉と「加算」への回帰**

LLMの計算コストの大部分（90%以上）は、TransformerのAttention機構とFeed-Forward層における巨大な行列積（GEMM: General Matrix Multiply）が占めています。脳の効率を超えるためには、このGEMMを物理的に、あるいはアルゴリズム的に排除する必要があります。

### **3.1 1-bitおよび3値（Ternary）LLM：乗算の排除**

行列演算におけるエネルギー消費の主犯は「浮動小数点の乗算」です。最近の研究である**BitNet b1.58**は、重みの精度を極限まで下げることで、この問題を根底から解決しようとしています27。

#### **1.58ビットの衝撃**

BitNet b1.58は、すべての重みパラメータを $\\{-1, 0, 1\\}$ の3値（Ternary）に制約します。3つの状態を持つため、情報量は $\\log\_2 3 \\approx 1.58$ ビットとなります。  
このアーキテクチャの革命的な点は、行列積 $W \\cdot X$ における乗算を完全に排除できることです。重みが $-1, 0, 1$ のいずれかであれば、計算は単純な加算（Accumulation）と減算、あるいは $0$ の場合のスキップに置き換わります。

* **エネルギー削減:** 浮動小数点の乗算（FP16 Mul）に比べ、整数の加算（Int8 Add）はエネルギー消費が桁違いに小さく、さらに $0$ 重みによるスパース性を活用すれば計算自体を省略できます。  
* **メモリ帯域:** 重みを1.58ビット（実装上は2ビット等）で表現できるため、FP16に比べてメモリ転送量が約1/10に激減します。LLMの推論速度はメモリ帯域律速（Memory Wall）であることが多いため、これは劇的な高速化をもたらします29。

#### **スケーリング則と性能**

驚くべきことに、BitNet b1.58は、3Bパラメータ以上の規模において、フル精度（FP16）のLLaMAモデルと同等のPerplexity（予測性能）とタスク精度を達成しています27。これは、「知能の創発には高精度な数値計算が必要である」という従来の常識を覆し、「パラメータの数（結合の複雑さ）」こそが重要であり、個々の結合の重みは粗い離散値で十分であることを示唆しています。この特性は、乗算器を持たない極めて単純なハードウェアや、後述するメモリ内演算（CIM）との相性が抜群です30。

### **3.2 スパイキングニューラルネットワーク（SNN）：時間的スパース性**

SNNは、脳のニューロンと同様に、離散的なパルス（スパイク）を用いて情報を伝達します。常時値を出力する人工ニューロン（ANN）とは異なり、SNNは「イベント駆動型」であり、重要な情報がある時だけスパイクを発火します。

#### **Spiking TransformerとSpikingMamba**

TransformerのSelf-Attention機構は $O(N^2)$ の計算量を持ちますが、これをSNNに適用する試みが進んでいます。**Spike-Driven Self-Attention (SDSA)** では、Query、Key、Value行列をスパイク列に変換し、行列積をスパイクの論理演算（AND操作と加算）に置き換えます。これにより、乗算を排除するだけでなく、スパイクのスパース性（多くの要素が0）を利用して計算量を劇的に削減します31。

さらに、Transformerの代替として注目される**Mamba（State Space Models）とSNNを組み合わせたSpikingMamba**も提案されています。Mambaは線形計算量 $O(N)$ で長いシーケンスを処理できますが、SNN化することで、そのエネルギー効率をさらに高めることができます。SpikingMambaは、従来のMambaと比較して4.76倍のエネルギー効率を達成しつつ、精度劣化を最小限に抑えています33。

### **3.3 樹状突起計算（Dendritic Computation）：単一ニューロンの多層化**

脳の効率性を語る上で見落とされがちなのが、ニューロンの形態、特に\*\*樹状突起（Dendrite）\*\*の役割です。従来のニューラルネットワークでは、ニューロンは入力を線形に加算する「点（Point Neuron）」としてモデル化されてきました。しかし、実際の生物学的ニューロンは、樹状突起上で局所的な非線形演算を行っています。

#### **樹状突起による計算能力の拡張**

最近の研究では、樹状突起の非線形性を導入したDendritic Neural Networksが提案されています。これにより、単一のニューロンが従来の2層ニューラルネットワークに相当する計算能力（例えばXOR問題の解決など）を持つことができます34。  
Dendritic Localized Learning (DLL) は、この構造を利用して学習アルゴリズムを局所化します。感覚入力は基底樹状突起（Basal Dendrite）へ、教師信号やフィードバックは尖頭樹状突起（Apical Dendrite）へと分離して入力され、細胞体（Soma）で統合されます。この分離により、誤差逆伝播を用いずに、局所的な電位差のみでシナプス強度を調整することが可能になります5。これは、ハードウェア実装において配線を簡素化し、素子数を削減する上で極めて有効です35。

## ---

**4\. ハードウェアの物理的基盤：計算とメモリの融合**

アルゴリズムの革新だけでは不十分です。フォン・ノイマン型アーキテクチャにおけるデータ移動（メモリとプロセッサ間の往復）は、計算そのものよりも多くのエネルギーを消費します。脳の効率に迫るには、**Compute-in-Memory (CIM)**、すなわちメモリの中で計算を行うアプローチが不可欠です。

### **4.1 アナログ・インメモリコンピューティング（AIMC）：キルヒホッフの法則による行列積**

AIMCは、物理法則を利用して行列積を瞬時に行います。抵抗変化型メモリ（ReRAM）や相変化メモリ（PCM）のクロスバーアレイを用います。

* **原理:** 行列の重み $W\_{ij}$ をメモリセルのコンダクタンス（電気伝導度）$G\_{ij}$ として物理的に記憶させます。入力ベクトル $x\_j$ を電圧 $V\_j$ として行線に印加します。  
* **計算:** オームの法則（$I \= G \\cdot V$）により、各セルで乗算が行われます。そして、キルヒホッフの電流則により、列線上で電流が自然に加算されます（$I\_i \= \\sum\_j G\_{ij} V\_j$）。  
* **効果:** これにより、クロックサイクルを回してデジタル論理回路で計算するのではなく、物理的な電流の流れとして一瞬で行列ベクトル積が得られます。データの移動は発生しません36。

#### **IBM NorthPoleとIntel Loihi 2**

* **IBM NorthPole:** デジタル回路でありながら、脳のようにメモリと演算器を微細に混在させたアーキテクチャです。外部メモリへのアクセスを排除し、チップ内部ですべての推論を完結させることで、GPUと比較して25倍のエネルギー効率を達成しています。3BパラメータのLLMの動作も実証されています38。  
* **Intel Loihi 2:** 非同期スパイキングニューラルネットワークチップであり、スパースなイベント駆動処理に特化しています。**Hala Point**システム（Loihi 2を1152個搭載）は、人間の脳の1/5000程度の規模（11.5億ニューロン）を持ちながら、最適化問題において従来型コンピュータの100倍の効率を示しています39。

### **4.2 光コンピューティング：光速・低遅延・低発熱**

電子ではなく光子（フォトン）を用いる光コンピューティングは、究極の低遅延と低消費電力を約束します。

* **マッハ・ツェンダー干渉計（MZI）:** 光の干渉を利用して行列演算を行います。光は抵抗を受けないため、演算自体には熱が発生せず、理論上のエネルギー効率は極めて高くなります41。  
* **シリコンフォトニクス:** チップ間の通信に光配線を用いることで、帯域幅のボトルネックを解消します。これは、兆パラメータ級のモデルを多数のチップに分散させて学習する際に、通信遅延をほぼゼロにするために不可欠です42。

### **4.3 熱力学的コンピューティング（Thermodynamic Computing）：ノイズを資源にする**

従来のコンピュータは、熱ノイズを排除するために多大なエネルギーを費やして「0」と「1」を維持しています。しかし、**熱力学的コンピューティング**は、このノイズを積極的に計算資源として利用します。

* **確率的サンプリング:** 生成AIの本質は、高次元の確率分布からのサンプリングです。熱力学的コンピュータ（TSU: Thermodynamic Sampling Unit）は、熱揺らぎを持つ物理素子（p-bitsなど）を用いることで、ボルツマン分布やエネルギーベースモデルからのサンプリングを、デジタル計算による擬似乱数生成なしに、物理的に行います44。  
* **ランジュバン動力学:** システムの物理的な時間発展（拡散プロセス）そのものを推論プロセスとして利用します。これにより、GPUが膨大な計算ステップを費やして行う処理を、自然な物理現象として瞬時に完了させることが可能になります45。Extropic社などのスタートアップがこの分野を開拓しており、生成AIの推論における圧倒的なエネルギー効率（GPU比で数桁向上）を目指しています46。

## ---

**5\. 理論的統合：生物学的限界を超えるための「スーパー・チューリング」アプローチ**

人間の脳の効率（約20W）は驚異的ですが、物理法則が許容する限界（ランダウアーの限界）にはまだ遠く及びません。脳を超える効率を実現するためには、生物学的模倣を超え、物理学の極限に挑む必要があります。

### **5.1 ランダウアーの限界と可逆計算**

ランダウアーの原理によれば、情報の1ビットを消去（不可逆操作）するために必要な最小エネルギーは $k\_B T \\ln 2$ （室温で約 $3 \\times 10^{-21}$ ジュール）です。現在のデジタルコンピュータは、この理論限界の数億倍から数十億倍のエネルギーを浪費しています47。  
脳を超えるためには、計算プロセスを可能な限り\*\*可逆（Reversible）\*\*にする必要があります。熱力学的コンピューティングや平衡伝播法は、平衡状態への緩和という物理プロセスを利用するため、準静的過程に近づけることで、エネルギー散逸を理論限界まで抑え込む潜在能力を持っています49。

### **5.2 能動的推論（Active Inference）と自由エネルギー原理**

学習の効率化には、データを受動的に処理するだけでなく、能動的に情報を獲得するエージェントが必要です。\*\*自由エネルギー原理（FEP）\*\*に基づく能動的推論は、知覚と行動を統一的に説明する枠組みです。エージェントは、自身の内部モデルの予測誤差（変分自由エネルギー）を最小化するように、環境に働きかけ（行動）、内部状態を更新（知覚）します50。  
LLMに能動的推論の層を組み込むことで、静的なプロンプトに依存するのではなく、情報の不確実性を減らすために自律的に探索や質問を行う「適応的エージェント」を構築できます。これにより、学習に必要なデータ量を劇的に削減し、効率的な知識獲得が可能になります52。

### **5.3 統合された「ニューロモルフィック・スタック」の提案**

以上の分析から、人間の脳を超える効率を実現するための具体的なシステム構成（スタック）が導き出されます。

| レイヤー | 従来の技術 (Current Paradigm) | 提案される未来の技術 (Post-Von Neumann Paradigm) | 効率向上の要因 |
| :---- | :---- | :---- | :---- |
| **アルゴリズム** | 誤差逆伝播法 (BP) | **平衡伝播法 (EP) / Forward-Forward (FF)** | 大域同期の排除、物理的勾配計算 |
| **データ表現** | FP16 / FP32 (密行列) | **1.58-bit (Ternary) / スパイク (Event-driven)** | 乗算の排除、時間的・空間的スパース性 |
| **アーキテクチャ** | Transformer (Global Attention) | **Spiking Transformer / Mamba / 予測符号化** | 線形計算量、局所的結合 |
| **ハードウェア** | GPU (Von Neumann) | **アナログCIM / 熱力学チップ (Non-Von Neumann)** | データ移動ゼロ、物理法則による計算 |
| **理論基盤** | 損失最小化 | **自由エネルギー最小化 (FEP) / ランダウアー限界** | 熱力学的最適化、能動的学習 |

このスタックは、各階層で相乗効果を生み出します。

1. **BitNetの3値重み**は、アナログCIMの課題である「精度の低さ（ノイズ）」に対するロバスト性を提供します。重みが粗くて良いため、アナログ素子のバラつきが許容されます。  
2. \*\*平衡伝播法（EP）\*\*は、アナログ回路の物理的緩和を利用して学習するため、回路の特性（非線形性やノイズ）を「バグ」ではなく「機能」として取り込みます。  
3. **スパイキング動作**は、回路の待機電力を極小化し、熱力学的チップの確率的挙動と整合します。

## ---

**6\. 結論**

「誤差逆伝播法やGPU、行列計算に頼らないLLM」の探求は、単なる代替手段の模索ではなく、知能機械の必然的な進化の方向性を示しています。本調査の結果、以下の結論が得られました。

1. **アルゴリズムの転換:** 大域的な逆伝播は、**Forward-Forward**や**平衡伝播法**といった局所学習則に置き換えるべきです。これにより、物理ハードウェア上での直接学習が可能になります。  
2. **演算の簡素化:** 行列積は不要です。**1.58ビット（3値）重み**と**スパイク駆動型注意機構**を採用することで、すべての計算を加算（Accumulation）と論理演算に還元できます。  
3. **物理計算の採用:** デジタル論理の抽象化を捨て、**アナログ・インメモリ**や**熱力学的コンピューティング**を採用することで、データ移動のエネルギーを排除し、物理現象（緩和、熱揺らぎ）そのものを計算として利用すべきです。

これらの技術を統合した「ニューロモルフィック・スタック」は、現在のGPUベースのAIと比較して**1万倍以上のエネルギー効率**（$10^4 \\times$ efficiency gain）を実現する潜在能力を秘めています。これは、人間の脳のエネルギー効率（約20W）に到達するだけでなく、長期的には可逆計算の原理を取り入れることで、生物学的制約に縛られない**超・高効率な人工知能**へと至る唯一の道筋です。

我々は今、計算機科学と物理学、そして神経科学が真に融合する「熱力学的AI」の時代の入り口に立っています。

#### **引用文献**

1. LightFF: Lightweight Inference for Forward-Forward Algorithm \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2404.05241v5](https://arxiv.org/html/2404.05241v5)  
2. The human brain runs on about 20 watts of power \- Hacker News, 12月 16, 2025にアクセス、 [https://news.ycombinator.com/item?id=44030154](https://news.ycombinator.com/item?id=44030154)  
3. Energy Efficiency in Artificial and Biological Intelligence \- Neurozone® Resources, 12月 16, 2025にアクセス、 [https://blog.neurozone.com/energy-efficiency-in-artificial-and-biological-intelligence](https://blog.neurozone.com/energy-efficiency-in-artificial-and-biological-intelligence)  
4. Feedback alignment in deep convolutional net- works \- Ashok Litwin-Kumar \- Columbia University, 12月 16, 2025にアクセス、 [http://lk.zuckermaninstitute.columbia.edu/pdf/moskovitz\_feedback\_2019.pdf](http://lk.zuckermaninstitute.columbia.edu/pdf/moskovitz_feedback_2019.pdf)  
5. Dendritic Localized Learning: Toward Biologically Plausible Algorithm \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2501.09976v1](https://arxiv.org/html/2501.09976v1)  
6. The Forward-Forward Algorithm: Some Preliminary Investigations \- Department of Computer Science, 12月 16, 2025にアクセス、 [https://www.cs.toronto.edu/\~hinton/absps/FFXfinal.pdf](https://www.cs.toronto.edu/~hinton/absps/FFXfinal.pdf)  
7. Brains@Bay Meetup \- Alternatives to Backpropagation in Neural Networks (Nov 18, 2020), 12月 16, 2025にアクセス、 [https://www.youtube.com/watch?v=oXyQU0aScq0](https://www.youtube.com/watch?v=oXyQU0aScq0)  
8. Going Forward-Forward in Distributed Deep Learning \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2404.08573v1](https://arxiv.org/html/2404.08573v1)  
9. FFCL: Forward-Forward Net with Cortical Loops, Training and Inference on Edge Without Backpropagation \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2405.12443v1](https://arxiv.org/html/2405.12443v1)  
10. Back-Propagation No More? Breaking Down the Forward-Forward Algorithm | by Dev Shah, 12月 16, 2025にアクセス、 [https://medium.com/demistify/back-propagation-no-more-breaking-down-the-forward-forward-algorithm-6e49f2f0507](https://medium.com/demistify/back-propagation-no-more-breaking-down-the-forward-forward-algorithm-6e49f2f0507)  
11. A new way to train neural networks: the Forward-Forward algorithm | Quantdare, 12月 16, 2025にアクセス、 [https://quantdare.com/a-new-way-to-train-neural-networks-the-forward-forward-algorithm/](https://quantdare.com/a-new-way-to-train-neural-networks-the-forward-forward-algorithm/)  
12. The Forward-Forward Algorithm with a Spiking Neural Network \- snnTorch \- Read the Docs, 12月 16, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_forward\_forward.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_forward_forward.html)  
13. Forward-Forward algorithm \- Learning without backpropagation, 12月 16, 2025にアクセス、 [https://pages.mini.pw.edu.pl/\~mandziukj/2023-04-19.pdf](https://pages.mini.pw.edu.pl/~mandziukj/2023-04-19.pdf)  
14. Scalable Forward-Forward Algorithm \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2501.03176v1](https://arxiv.org/html/2501.03176v1)  
15. \[2212.13345\] The Forward-Forward Algorithm: Some Preliminary Investigations \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/2212.13345](https://arxiv.org/abs/2212.13345)  
16. Self-Contrastive Forward-Forward Algorithm \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2409.11593v1](https://arxiv.org/html/2409.11593v1)  
17. \[1602.05179\] Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/1602.05179](https://arxiv.org/abs/1602.05179)  
18. Scaling Equilibrium Propagation to Deeper Neural Network Architectures This work was supported in part by the RISC-V Knowledge Centre of Excellence (RKCoE), sponsored by the Ministry of Electronics and Information Technology (MeitY). \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2509.26003v1](https://arxiv.org/html/2509.26003v1)  
19. Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing Its Gradient Estimator Bias \- NIH, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC7930909/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7930909/)  
20. Research Proves End-to-End Analog Chips for AI Computation Possible \- EE Times, 12月 16, 2025にアクセス、 [https://www.eetimes.com/research-breakthrough-promises-end-to-end-analog-chips-for-ai-computation/](https://www.eetimes.com/research-breakthrough-promises-end-to-end-analog-chips-for-ai-computation/)  
21. Training deep resistive networks with equilibrium propagation, 12月 16, 2025にアクセス、 [https://mcmahon.aep.cornell.edu/aspen/2024/slides/Scellier.pdf](https://mcmahon.aep.cornell.edu/aspen/2024/slides/Scellier.pdf)  
22. Quantum equilibrium propagation for efficient training of quantum systems based on Onsager reciprocity \- PMC \- PubMed Central, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12271321/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12271321/)  
23. A Survey on Brain-inspired Deep Learning via Predictive Coding \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2308.07870v2](https://arxiv.org/html/2308.07870v2)  
24. Towards Scaling Deep Neural Networks with Predictive Coding: Theory and Practice \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/2510.23323](https://arxiv.org/abs/2510.23323)  
25. Inspires effective alternatives to backpropagation: predictive coding helps understand and build learning \- PMC \- NIH, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11881729/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881729/)  
26. A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks, 12月 16, 2025にアクセス、 [https://openreview.net/forum?id=RyUvzda8GH](https://openreview.net/forum?id=RyUvzda8GH)  
27. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2402.17764v1](https://arxiv.org/html/2402.17764v1)  
28. Embracing the Era of 1-Bit LLMs: Microsoft & UCAS's BitNet b1.58 Redefines Efficiency, 12月 16, 2025にアクセス、 [https://syncedreview.com/2024/02/29/embracing-the-era-of-1-bit-llms-microsoft-ucass-bitnet-b1-58-redefines-efficiency/](https://syncedreview.com/2024/02/29/embracing-the-era-of-1-bit-llms-microsoft-ucass-bitnet-b1-58-redefines-efficiency/)  
29. TerEffic: Highly Efficient Ternary LLM Inference on FPGA \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2502.16473v1](https://arxiv.org/html/2502.16473v1)  
30. A Quick Look at Microsoft's BitNet b1.58 2B4T: Tiny but Mighty \- Apidog, 12月 16, 2025にアクセス、 [https://apidog.com/blog/microsoft-bitnet-2b/](https://apidog.com/blog/microsoft-bitnet-2b/)  
31. \[2503.00226\] Spiking Transformer:Introducing Accurate Addition-Only Spiking Self-Attention for Transformer \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/2503.00226](https://arxiv.org/abs/2503.00226)  
32. Spike-driven Transformer, 12月 16, 2025にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)  
33. \[2510.04595\] SpikingMamba: Towards Energy-Efficient Large Language Models via Knowledge Distillation from Mamba \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/2510.04595](https://arxiv.org/abs/2510.04595)  
34. Impact of dendritic non-linearities on the computational capabilities of neurons \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2407.07572v1](https://arxiv.org/html/2407.07572v1)  
35. \[2505.01635\] Dendritic Computing with Multi-Gate Ferroelectric Field-Effect Transistors, 12月 16, 2025にアクセス、 [https://arxiv.org/abs/2505.01635](https://arxiv.org/abs/2505.01635)  
36. An Overview of Compute-in-Memory Architectures for Accelerating Large Language Model Inference | alphaXiv, 12月 16, 2025にアクセス、 [https://www.alphaxiv.org/overview/2406.08413](https://www.alphaxiv.org/overview/2406.08413)  
37. Demonstration of transformer-based ALBERT model on a 14nm analog AI inference chip, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12485056/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485056/)  
38. IBM's NorthPole achieves new speed and efficiency milestones, 12月 16, 2025にアクセス、 [https://research.ibm.com/blog/northpole-llm-inference-results](https://research.ibm.com/blog/northpole-llm-inference-results)  
39. Intel Builds World's Largest Neuromorphic System to Enable More Sustainable AI, 12月 16, 2025にアクセス、 [https://www.intc.com/news-events/press-releases/detail/1691/intel-builds-worlds-largest-neuromorphic-system-to](https://www.intc.com/news-events/press-releases/detail/1691/intel-builds-worlds-largest-neuromorphic-system-to)  
40. Next-Level Neuromorphic Computing: Intel Lab's Loihi 2 Chip, 12月 16, 2025にアクセス、 [https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.html](https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.html)  
41. What is next for LLMs? Pushing the boundaries of next-gen AI computing hardware with photonic chips \- NIH, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12592636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12592636/)  
42. Optically Connected Multi-Stack HBM Modules for Large Language Model Training and Inference \- Computer Systems Laboratory, 12月 16, 2025にアクセス、 [https://www.csl.cornell.edu/\~cbatten/pdfs/ou-optical-mem-llm-cal2025.pdf](https://www.csl.cornell.edu/~cbatten/pdfs/ou-optical-mem-llm-cal2025.pdf)  
43. Accelerating Large Language Model Training with In-Package Optical Links for Scale-Out Systems | IEEE Conference Publication, 12月 16, 2025にアクセス、 [https://ieeexplore.ieee.org/document/10682729/](https://ieeexplore.ieee.org/document/10682729/)  
44. Thermodynamic Computing: From Zero to One \- Extropic, 12月 16, 2025にアクセス、 [https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one)  
45. thermox: The First Thermodynamic Computing Simulator, 12月 16, 2025にアクセス、 [https://www.normalcomputing.com/blog/thermox-the-first-thermodynamic-computing-simulator](https://www.normalcomputing.com/blog/thermox-the-first-thermodynamic-computing-simulator)  
46. The 10000x Energy Revolution: How Thermodynamic Computing Could Rewrite the Rules of AI | by Pedro Miguel Lourenço \- Medium, 12月 16, 2025にアクセス、 [https://medium.com/@pedromlourenco/the-10-000x-energy-revolution-how-thermodynamic-computing-could-rewrite-the-rules-of-ai-4dc17e413347](https://medium.com/@pedromlourenco/the-10-000x-energy-revolution-how-thermodynamic-computing-could-rewrite-the-rules-of-ai-4dc17e413347)  
47. Landauer's principle \- Wikipedia, 12月 16, 2025にアクセス、 [https://en.wikipedia.org/wiki/Landauer%27s\_principle](https://en.wikipedia.org/wiki/Landauer%27s_principle)  
48. Minimum Energy of Computing, Fundamental Considerations \- Semantic Scholar, 12月 16, 2025にアクセス、 [https://pdfs.semanticscholar.org/834c/17928d749dfd1c73710a2397d84ecf462b00.pdf](https://pdfs.semanticscholar.org/834c/17928d749dfd1c73710a2397d84ecf462b00.pdf)  
49. Does computation actually require no energy? : r/AskPhysics \- Reddit, 12月 16, 2025にアクセス、 [https://www.reddit.com/r/AskPhysics/comments/1okgona/does\_computation\_actually\_require\_no\_energy/](https://www.reddit.com/r/AskPhysics/comments/1okgona/does_computation_actually_require_no_energy/)  
50. The Free Energy Principle for Perception and Action: A Deep Learning Perspective \- NIH, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/)  
51. Generalised free energy and active inference \- PMC \- PubMed Central, 12月 16, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/)  
52. Active Inference for Self-Organizing Multi-LLM Systems: A Bayesian Thermodynamic Approach to Adaptation \- arXiv, 12月 16, 2025にアクセス、 [https://arxiv.org/html/2412.10425v2](https://arxiv.org/html/2412.10425v2)