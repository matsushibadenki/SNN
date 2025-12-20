# **次世代計算パラダイムの包括的研究：1ビットLLM、ニューロモルフィック、および物理ベースコンピューティングの技術エコシステム**

## **1\. 序論：ムーアの法則後の計算効率の追求**

人工知能（AI）の進化は、長らく「スケール」によって定義されてきた。モデルパラメータ数の増大、データセットの巨大化、そしてそれを支える計算リソースの指数関数的な拡張が、GPT-4やGeminiといった現代の基盤モデルの能力を決定づけてきた。しかし、この成長曲線は物理的な限界、すなわち「エネルギーの壁」に直面している。データセンターの消費電力は持続不可能なレベルに達しつつあり、エッジデバイスでの高度な推論はメモリ帯域幅と熱設計電力（TDP）によって厳しく制限されている。

従来のフォン・ノイマン型アーキテクチャ上で、浮動小数点演算（FLOPs）を大量に消費するバックプロパゲーション（誤差逆伝播法）を用いたディープラーニングを行うという現在のアプローチは、非効率性の極みにあると言っても過言ではない。この閉塞感を打破するために、現在、ハードウェアとソフトウェアのスタックを根本から再構築する複数の技術潮流が同時多発的に発生している。

本レポートでは、従来のフル精度・同期型・デジタル計算のパラダイムに挑戦する技術群を包括的に調査・分析する。具体的には、Microsoftが提唱する「1ビットLLM（BitNet）」による乗算器の排除、脳のスパイク発火を模倣する「スパイキングニューラルネットワーク（SNN）」、大域的な同期を必要としない「Forward-Forwardアルゴリズム」や「平衡伝播（Equilibrium Propagation）」、そして熱雑音やアナログ素子の物理特性を計算資源として利用する「熱力学・アナログコンピューティング」である。これらの技術は単なる理論研究の域を出て、具体的なライブラリや実装プロジェクトとして結実しつつある。本稿では、これらの技術開発を行っているプロジェクトやライブラリを詳細に調査し、その実装の詳細、パフォーマンス、そして将来的な含意について、15,000語規模の深度を持って論じる。

## ---

**2\. 1ビットおよび三値LLMの革命：BitNet b1.58とそのエコシステム**

現在のLLM推論における最大のボトルネックは、計算量ではなくメモリ帯域幅である。従来のFP16（16ビット浮動小数点）やINT8（8ビット整数）の重みは、依然として大量のデータ転送を必要とする。この問題に対する最も急進的な解答が、モデルの重みを極限まで量子化し、計算プロセスそのものを変革する「1ビットLLM」である。

### **2.1 BitNet b1.58の理論的基盤とアーキテクチャ**

Microsoft Researchによって発表されたBitNet b1.58は、従来のバイナリニューラルネットワーク（BNN）の限界を克服し、フル精度のLLMと同等の性能を維持しながら、劇的な効率化を実現するアーキテクチャである。

#### **2.1.1 1.58ビットの意味と三値重み**

BitNet b1.58の中核的な革新は、重みを $\\{-1, 0, 1\\}$ の三値（Ternary）に制限することにある。従来の1ビットBNNは重みを $\\{-1, 1\\}$ の二値に制限していたが、これは「0（無効）」の状態を表現できないため、モデルが特徴量を選択的に無視（フィルタリング）する能力を著しく欠いていた。BitNet b1.58では「0」を導入することで、モデルに明示的なスパース性（疎性）を持たせている。情報理論的に、三値は $\\log\_2(3) \\approx 1.58$ ビットの情報量を持つため、「1.58ビット」と呼称される 1。

#### **2.1.2 乗算フリーの行列演算**

このアーキテクチャの最大の利点は、行列演算（MatMul）から高コストな浮動小数点乗算を排除できる点にある。重みが $\\{-1, 0, 1\\}$ に限定されるため、入力アクティベーションベクトル $x$ と重み行列 $W$ の積 $y \= Wx$ は、単純な加算と減算のみで計算可能となる。

* 重みが $+1$ の場合：アクティベーションを加算。  
* 重みが $-1$ の場合：アクティベーションを減算。  
* 重みが $0$ の場合：演算なし（スキップ）。

これにより、シリコン上での実装面積と消費電力が大幅に削減される。報告によれば、70Bパラメータクラスのモデルにおいて、従来のFP16ベースラインと比較してエネルギー消費を最大70〜80%削減し、レイテンシを大幅に改善することが示されている 4。

### **2.2 公式実装 bitnet.cpp の詳細分析**

Microsoftは、この理論を実用的な推論エンジンとして具現化した bitnet.cpp を公開している。これは llama.cpp のフレームワークをベースにしつつ、1.58ビットモデルに特化したカーネルを実装したものである。

#### **2.2.1 最適化カーネルの設計：I2\_S, TL1, TL2**

bitnet.cpp の性能を支えているのは、CPUアーキテクチャごとに高度にチューニングされたカーネル群である。これらは、三値の重みを効率的にメモリから読み出し、レジスタ上で演算を行うための低レベルな工夫が凝らされている 4。

* **I2\_S カーネル（2-bit Storage, Unpacked Computation）**  
  * **概要**: 重みを2ビットでストレージに格納し、演算時にオンザフライで展開（Unpack）して計算する方式。  
  * **特徴**: 汎用性が高く、x86（AVX2/AVX-512）およびARM（NEON）の両方で動作する。2ビットで表現された重み（00, 01, 10, 11のようなビットパターン）を、SIMD命令を用いて高速に $\\{-1, 0, 1\\}$ の数値として解釈し、アキュムレータに加算する。  
  * **適用範囲**: 主に小規模から中規模のモデル、あるいは特殊な命令セットを持たない環境でのベースラインとして機能する。  
* **TL1 カーネル（T-MACベース, Lookup Table Type 1）**  
  * **概要**: T-MAC（Ternary Matrix Multiplication Accumulation）の技術を応用し、ルックアップテーブル（LUT）を活用する方式。  
  * **メカニズム**: 重みのパターンを事前にインデックス化し、計算結果をテーブル参照によって取得することで、加減算の回数すら削減する。  
  * **ターゲット**: 主にARMアーキテクチャ（Apple Silicon等）で効果を発揮し、メモリ帯域幅の利用効率を最大化する。  
  * **性能**: ARM CPUにおいて、1.37倍から5.07倍の高速化を実現する主要因となっている 4。  
* **TL2 カーネル（Lookup Table Type 2）**  
  * **概要**: TL1の改良版またはx86向けの特化版であり、より大きなブロックサイズや特定のSIMD命令セットに最適化されている。  
  * **性能**: x86 CPUにおいて、2.37倍から6.17倍という驚異的な高速化を記録している 4。

これらのカーネルにより、bitnet.cpp は100Bパラメータクラスの超巨大モデルであっても、単一のCPU上で人間が読める速度（毎秒5-7トークン）での推論を可能にしている 4。これは、従来のFP16モデルでは数百GBのメモリ帯域を必要とし、GPUクラスタなしでは不可能であった領域である。

#### **2.2.2 導入と変換フロー**

bitnet.cpp の利用には、Hugging Faceからモデルをダウンロードし、専用の形式に変換するプロセスが必要である。提供されている setup\_env.py スクリプトは、Hugging Face CLIを用いてモデル（例：1bitLLM/bitnet\_b1\_58-large や Llama3-8B-1.58）をダウンロードし、量子化スクリプトを用いて .gguf 互換の形式へ変換する 4。

### **2.3 llama.cpp における統合と技術的課題**

オープンソースLLM推論のデファクトスタンダードである llama.cpp においても、BitNetのサポートに向けた活発な開発が行われている。しかし、汎用的な推論エンジンに特異な三値モデルを統合することは、多くの技術的課題を伴っている。

#### **2.3.1 パッキングフォーマットの戦い：TQ1\_0 vs TQ2\_0**

三値の重み（$-1, 0, 1$）をデジタルメモリ上に効率的に配置する方法（パッキング）は自明ではない。2ビットを使えば簡単に表現できるが（$2^2=4$ 通り）、情報理論的な限界である1.58ビットに近づけるには高度な圧縮が必要となる。llama.cpp の開発コミュニティでは、以下のフォーマットが議論・実装されている 7。

* **TQ2\_0（Ternary Quantization 2.0 bits）**  
  * **構造**: 各重みを単純に2ビットで表現する。1バイトに4つの重みを格納できる。  
  * **利点**: デコードが極めて高速。ビットマスクとシフト演算だけで値を取り出せるため、SIMD化が容易であり、計算速度（トークン生成速度）を最優先する場合に適している。  
  * **欠点**: 1.58ビットに対して冗長性があり、モデルサイズが理論上の最小値より大きくなる。  
* **TQ1\_0（Ternary Quantization \~1.6 bits）**  
  * **構造**: 複数の重み（例えば5つ）をまとめて一つの数値として扱い（$3^5 \= 243 \< 256$、つまり1バイトに5つの重みを詰め込むような工夫）、情報密度を高める。  
  * **利点**: モデルサイズを極限まで小さくできる。メモリ帯域幅が最大のボトルネックである環境で有利。  
  * **欠点**: 展開（Unpack）の計算コストが高い。複雑なビット演算が必要となり、推論速度がCPUの演算性能によって律速されるリスクがある。

#### **2.3.2 汎用バックエンドとの摩擦**

llama.cpp は本来、FP16やINT8/4の行列演算を前提に設計されている。BitNetのような「乗算フリー」のアーキテクチャをサポートするためには、単に重みをロードするだけでなく、行列演算カーネルそのものを差し替える必要がある。現状の llama.cpp の実装（メインラインへのマージ状況）では、専用の三値カーネル（Ternary Dot Product）が部分的に導入されているものの、一部の処理では依然としてFP16やINT8へのキャスト（展開）が行われており、bitnet.cpp ほどの劇的なエネルギー効率向上には至っていないという報告もある 9。特に、GPU（CUDA）バックエンドにおいては、Tensor Coreが三値演算をネイティブサポートしていないため、INT8のTensor Coreを利用するためのエミュレーションが必要となり、実装の難易度を高めている 9。

### **2.4 PyTorchエコシステムにおける研究開発**

C++による推論エンジンと並行して、Python/PyTorch上での研究用実装も多数公開されている。これらは主にモデルの学習（Training）やファインチューニングを目的としている。

* **kevbuh/bitnet**: BitNet b1.58の純粋なPyTorch実装。学習時の挙動再現に主眼を置いている。ここでは、重みの量子化関数として w\_quant \= round(w / (w.abs().mean() \+ epsilon)) という「AbsMean」方式が採用されており、勾配の近似にはStraight-Through Estimator (STE) が用いられていることがコードから確認できる 1。  
* **kyegomez/BitNet**: よりモジュール化された実装で、既存のTransformersライブラリ（Hugging Face）のモデルに対し、nn.Linear を BitLinear に置換するためのユーティリティを提供している。また、BitMGQA（Multi-Grouped Query Attention）のような、BitNetに特化したアテンション機構の実装も含まれている 11。  
* **Oxen-AI/BitNet-1.58-Instruct**: BitNetの指示追従能力（Instruction Following）を検証するプロジェクト。SQuADやRedPajamaデータセットを用いたファインチューニングを行い、三値モデルであっても複雑な言語タスクを学習可能であることを実証している。学習率や重み減衰（Weight Decay）のスケジューリングに関する知見（高めの学習率が必要など）が共有されている 12。

## ---

**3\. ニューロモルフィック・コンピューティング：SNNフレームワークの全貌**

1ビットLLMが「計算精度の極限」を追求するのに対し、スパイキングニューラルネットワーク（SNN）は「時間的スパース性」を追求する。脳のニューロンのように、情報の伝達を離散的なパルス（スパイク）で行うことで、イベントが発生しない時間は電力を消費しないという究極の省エネ性を目指す。しかし、スパイク発火という不連続な動作は微分不可能であるため、従来のバックプロパゲーションを直接適用できないという課題があった。この課題を解決するために開発されたのが、以下の主要なソフトウェアフレームワークである。

### **3.1 snnTorch: 代理勾配法によるSNNの民主化**

snnTorch は、PyTorchのエコシステムをそのまま利用しながらSNNを学習可能にするライブラリであり、その親しみやすさから研究者や学生に広く利用されている 13。

#### **3.1.1 代理勾配（Surrogate Gradient）の魔法**

SNNの学習における最大の障壁は「デッドニューロン問題」である。スパイク発火関数（Heavisideステップ関数）の微分は、発火閾値以外で常に0になるため、勾配が伝播せず学習が進まない。snnTorch は snntorch.surrogate モジュールを通じて、この問題に対する洗練された解法を提供する。

* **メカニズム**: 順伝播（Forward）では正確なステップ関数を用いてスパイクを生成するが、逆伝播（Backward）ではその関数を滑らかなシグモイド関数や逆正接関数（ATan）、あるいは三角形関数に「置き換え」て勾配を計算する。  
* **実装**: ユーザーは spike\_grad \= surrogate.fast\_sigmoid() のように関数を定義し、それをニューロンモデルに渡すだけでよい。これにより、PyTorchの強力な自動微分機構（Autograd）を騙して、SNNの学習を可能にしている 16。

#### **3.1.2 データ符号化とニューロンモデル**

snnTorch は、静的な画像データなどを時間的なスパイク列に変換するための snntorch.spikegen モジュールを備えている。

* **レートコーディング**: ピクセル強度を発火頻度（レート）に変換する。  
* **レイテンシコーディング**: ピクセル強度を「最初のスパイクまでの時間」に変換する。  
* デルタ変調: 時間的な変化分のみをスパイク化する。  
  ニューロンモデルとしては、Leaky Integrate-and-Fire (LIF) モデルをはじめ、Synaptic ConductanceモデルやAlphaニューロンなど、生物学的妥当性の異なる多様なモデルが再帰的なユニット（RNNセルに近い形式）として実装されている 17。

### **3.2 SpikingJelly: 高性能バックエンドとフルスタック開発**

SpikingJelly は、北京大学とPeng Cheng Laboratoryによって開発された、よりパフォーマンス指向の強いフレームワークである。snnTorch が使いやすさを重視する一方、SpikingJelly は大規模な学習と高速な推論に焦点を当てている 18。

#### **3.2.1 CUDAバックエンドとCuPyの活用**

Pythonのループ処理で時間ステップごとのニューロン状態を更新するのは遅い。SpikingJelly は、マルチステップのニューロン更新を高速化するために、CUDAベースのカスタムカーネルを CuPy 経由で提供している。これにより、純粋なPyTorch実装と比較して、大規模なネットワークの学習時間を大幅に短縮できる。これは、SNNを実用的なサイズ（ResNet-18やResNet-50クラス）にスケールさせる上で不可欠な機能である 19。

#### **3.2.2 ANN-SNN変換のツールキット**

SNNを一から学習するのは収束が難しいため、事前に学習済みのANN（通常のCNNなど）の重みをSNNに移植し、アクティベーション関数をスパイク生成関数に置換する「ANN-SNN変換」が広く行われている。SpikingJelly はこのための専用モジュールを備えており、変換後の精度劣化を最小限に抑えるための正規化技術などが組み込まれている 19。

### **3.3 Norse: ディープラーニング互換性の追求**

Norse は、ドイツのハイデルベルク大学やKTHの研究者らによって開発されているライブラリで、「ディープラーニング互換」を強く意識している。SNNを「特殊なシミュレータ」としてではなく、「PyTorchのプリミティブの一部」として扱う設計哲学を持つ 21。

* **モジュラー設計**: LIFCell や LIFParameters といったコンポーネントは、PyTorchの標準的なレイヤーとシームレスに混在させることができる。これにより、CNNの特徴抽出層の後にSNNの処理層を配置するといったハイブリッドな構成が容易である。  
* **JITコンパイルへの対応**: 最近のアップデートでは、torch.compile への対応が進められており、演算の融合（Fusion）によってPythonのオーバーヘッドを隠蔽し、JAXやカスタムCUDAカーネルに迫る速度を実現しつつある 24。

### **表1: SNNフレームワークの比較**

| フレームワーク | 主要開発元 | 特徴的な強み | バックエンド技術 | ユースケース |
| :---- | :---- | :---- | :---- | :---- |
| **snnTorch** | UCSC | 直感的なAPI、豊富な教育資料、多様な代理勾配 | PyTorch (Pure) | 研究入門、プロトタイピング |
| **SpikingJelly** | 北京大学/PCL | 高速性、CUDAカーネル、ANN-SNN変換 | PyTorch \+ CuPy | 大規模モデル学習、高性能推論 |
| **Norse** | ハイデルベルク大他 | PyTorchとの深い統合、モジュール性 | PyTorch (JIT) | ハイブリッドモデル、既存DLへの統合 |

## ---

**4\. バックプロパゲーションを超えて：局所学習アルゴリズムの実装**

ディープラーニングの成功を支えてきたバックプロパゲーションだが、生物学的な妥当性の欠如（脳内には大域的な誤差信号の伝達経路が見当たらない）や、メモリ効率の悪さ（計算グラフ全体の保持が必要）が指摘されている。これに代わる「局所学習（Local Learning）」アルゴリズムの実装が進んでいる。

### **4.1 Forward-Forwardアルゴリズム：推論のみによる学習**

ジェフリー・ヒントンが2022年に提唱したForward-Forward（FF）アルゴリズムは、フォワードパス（順伝播）とバックワードパス（逆伝播）の代わりに、2回のフォワードパスを行うことでネットワークを学習させる 25。

#### **4.1.1 アルゴリズムのメカニズム**

FFアルゴリズムは、各層において「Goodness（良さ）」という指標を最大化することを目指す。

* **Positive Pass（正のパス）**: 実際のデータ（例：画像と正しいラベルの組み合わせ）を入力し、各層のアクティビティ（ニューロン出力の二乗和など）を増加させるように重みを更新する。  
* **Negative Pass（負のパス）**: 「偽のデータ」（例：画像と誤ったラベル、あるいは生成された不自然な画像）を入力し、各層のアクティビティを減少させる（閾値以下に抑える）ように重みを更新する。

このアプローチにより、各層は前後の層の状態や大域的な誤差を知ることなく、局所的な情報のみで学習が可能となる。これは、ネットワークの並列化や、メモリ制約の厳しいエッジデバイスでのオンデバイス学習において極めて有利である。

#### **4.1.2 実装の現状とベンチマーク**

* **loeweX/Forward-Forward**: ヒントンの論文（MNIST実験）を忠実に再現したPyTorch実装。教師あり学習の設定において、ラベル情報を画像に埋め込む（Overlay）手法や、Goodness関数の詳細（$\\sum x^2$）が実装されている。MNISTにおいて1.45%のエラー率を達成しており、公式のMatlab実装と同等の性能を示している 25。  
* **mpezeshki/pytorch\_forward\_forward**: より汎用的な実装を目指しており、CIFAR-10などの複雑なデータセットへの適用や、畳み込み層への拡張（FF\_CNN）が試みられている。ベンチマーク結果によれば、単純な全結合層ではバックプロパゲーションに近い精度が出るものの、深い畳み込みネットワークへのスケーリングには、正解/不正解データの生成方法（教師なしモードでのマスク処理など）に高度な工夫が必要であることが示唆されている 26。

### **4.2 平衡伝播（Equilibrium Propagation）：物理法則との融合**

平衡伝播（EqProp）は、エネルギーベースモデル（EBM）の学習則であり、ニューラルネットワークのダイナミクスが「エネルギー最小化」に向かう性質を利用する。

#### **4.2.1 2つの相：FreeとClamped**

学習は2つのフェーズで行われる。

1. **Free Phase**: 入力だけを与え、ネットワークを自由に緩和させ、エネルギーが最小となる平衡状態に到達させる。  
2. Clamped Phase (Teaching Phase): 出力層を正解ラベルの方へわずかに「引っ張る（Nudge）」。この状態で再び新たな平衡状態へ到達させる。  
   この2つの平衡状態間のわずかな変化をもとに、重みの更新量を計算する。この計算は局所的であり、物理的な回路（アナログ回路）の動作と整合性が高い 29。

#### **4.2.2 JAXによる実装とホロモルフィック拡張**

* **laborieux-axel/holomorphic\_eqprop**: JAXを用いた実装。ここでは「正則平衡伝播（Holomorphic Equilibrium Propagation）」という拡張が提案されている。従来のEqPropは数値的な微分に頼る部分があったが、複素平面上での解析的な性質を利用することで、より正確な勾配推定を可能にしている。JAXの自動微分機能（jax.grad, jax.vmap）を活用し、不動点探索（Fixed-point iteration）を効率的に並列化している点が特徴である 31。

## ---

**5\. 能動的推論（Active Inference）：知覚と行動の統一理論**

能動的推論は、カール・フリストンらが提唱する理論であり、エージェントが外界の状態を推論すること（知覚）と、外界に働きかけて期待する感覚入力を得ようとすること（行動）を、単一の「変分自由エネルギー最小化」のプロセスとして記述する。強化学習とは異なり、報酬の最大化だけでなく「不確実性の解消（情報の獲得）」が本質的に組み込まれている。

### **5.1 pymdp: 離散状態空間エージェントのためのライブラリ**

pymdp は、この複雑な数理モデルをPython上で手軽に扱えるようにしたライブラリであり、特に部分観測マルコフ決定過程（POMDP）に焦点を当てている 32。

#### **5.1.1 オブジェクト配列と因子化された分布**

pymdp の内部データ構造における最大の特徴は、NumPyのオブジェクト配列（Object Arrays）の活用である。複雑な環境をモデル化する際、状態空間は複数の「因子（Factors）」（例：場所、空腹度、天気）に分解され、観測も複数の「モダリティ（Modalities）」（例：視覚、聴覚）に分解される。  
これらを単一の巨大なテンソルで表現すると組み合わせ爆発を起こすため、pymdp では $P(s) \= \\prod P(s\_i)$ のように分布を因子化し、それぞれの周辺分布をオブジェクト配列の要素として保持する。これにより、メモリ効率と計算効率を劇的に改善している 35。

#### **5.1.2 生成モデルの行列：A, B, C, D**

ユーザーは以下の行列を定義することでエージェントを設計する 36。

* **A行列（尤度）**: $P(o|s)$。隠れ状態と観測の関係。  
* **B行列（遷移）**: $P(s\_{t+1}|s\_t, u\_t)$。行動による状態の変化。  
* **C行列（選好）**: $P(o)$。エージェントが「好ましい」と感じる観測（事前分布）。  
* **D行列（事前信念）**: $P(s\_0)$。初期状態の信念。

#### **5.1.3 認識的連鎖（Epistemic Chaining）の実装**

pymdp の強力な機能の一つが、認識的連鎖のシミュレーションである。これは、エージェントが最終的な報酬を得るために、まず「情報を得るための行動」を連鎖的に行う振る舞いである（例：宝箱を開ける鍵の場所を知るために、まず地図を探す）。infer\_policies() メソッド内で計算される期待自由エネルギー（EFE）には、リスク（選好からの逸脱）と共に「曖昧さ（Ambiguity）」の項が含まれており、これを最小化しようとすることで、エージェントは自律的に探索行動を生成する 34。

## ---

**6\. ハードウェアとソフトウェアの融合：アナログ・熱力学・NPU**

ソフトウェアアルゴリズムの進化に対し、それを物理レベルで支えるハードウェア、およびそのための低レベルソフトウェアスタックも急速に進化している。

### **6.1 熱力学コンピューティング：Extropicと THRML**

Extropic社は、「熱ノイズ」を計算資源として利用する熱力学コンピューティング（Thermodynamic Computing）を提唱している。従来のコンピュータがノイズを排除するために電力を消費するのに対し、彼らの「熱力学サンプリングユニット（TSU）」は、ノイズを利用して確率分布からのサンプリングを行う 38。

#### **6.1.1 THRML ライブラリの役割**

この新しいハードウェアを利用するために開発されたのが THRML（Thermodynamic Hypergraphical Model Library）である。

* **JAXベースの設計**: THRML はGoogleのJAX上に構築されている。これは、JAXが持つベクトル化（vmap）やJITコンパイル（jit）の機能が、確率モデルの並列サンプリングシミュレーションに最適であるためである 40。  
* **ブロックギブスサンプリング**: TSUの動作原理である、物理的な相互作用を通じた確率変数の更新をシミュレートするために、効率的なブロックギブスサンプリングアルゴリズムが実装されている。  
* **エネルギーベースモデル（EBM）**: ユーザーはイジングモデルのようなEBMを定義し、それをTSU上で（あるいは現在はGPUシミュレーション上で）緩和させることで、最適化問題の解や生成モデルのサンプルを得ることができる 41。

### **6.2 アナログAIのシミュレーション：IBM aihwkit**

IBM Researchが開発する aihwkit は、相変化メモリ（PCM）や抵抗変化メモリ（ReRAM/RRAM）を用いたアナログ・インメモリコンピューティングのためのPyTorchツールキットである 42。

* **非理想性のシミュレーション**: アナログ素子は、書き込みノイズ、経年劣化（ドリフト）、温度依存性といった「非理想性」を持つ。aihwkit は、AnalogLinear などのレイヤーを提供し、これらの物理的特性を学習プロセス（Forward/Backward）に注入する。これにより、実際のチップにデプロイしても性能が劣化しない「Hardware-Aware Training」が可能となる。  
* **RPU構成**: ユーザーは抵抗処理ユニット（RPU）の構成を細かく定義でき、パルス更新の粒度やデバイスの非対称性などを設定ファイルで制御できる。  
* **AIHW Composer**: クラウド上でこれらの実験をノーコードで行えるGUI環境も提供されており、ハードウェアの専門知識がない研究者でもアナログAIの実験が可能になっている 44。

### **6.3 NPUソフトウェアスタック：Ryzen AIと Riallto**

コンシューマー向けPCへのNPU（Neural Processing Unit）搭載が進む中、AMDはRyzen AIのためのソフトウェアスタックを整備している。

* **Ryzen AI Software**: Vitis AIとONNX Runtimeをベースにしており、PyTorchやTensorFlowで学習したモデルを量子化（INT8など）し、XDNAアーキテクチャを持つNPU向けにコンパイルするフローを提供する 45。  
* **Riallto フレームワーク**: より低レベルな開発者のために、NPUの内部挙動（空間データフロー）を可視化・制御するための実験的フレームワーク。  
  * **データフローグラフ**: NPUはCPUのような命令実行型ではなく、データがタイル間を流れることで計算が進むデータフロー型である。Riallto はこのデータの流れをPython上で定義・可視化し、メモリバッファの割り当てやカーネル間の同期を細かく制御することを可能にする 46。  
* **Rain AI**: Rain AIはデジタル・インメモリコンピューティングのタイル構造を持つNPUを開発しており、独自のSDKを通じて超低遅延・高エネルギー効率のエッジAIワークロードをサポートする計画である 47。

### **表2: ハードウェア指向AIライブラリの比較**

| ライブラリ | 開発元 | ターゲットハードウェア | 実装言語/ベース | 主な機能 |
| :---- | :---- | :---- | :---- | :---- |
| **THRML** | Extropic | 熱力学サンプリングユニット (TSU) | Python / JAX | EBMの定義、ギブスサンプリング、ハードウェアシミュレーション |
| **aihwkit** | IBM | アナログメモリ (PCM, ReRAM) | Python / PyTorch, C++ | アナログ素子のノイズ・ドリフトシミュレーション、Hardware-Aware学習 |
| **Riallto** | AMD | Ryzen AI NPU (XDNA) | Python, C++ | データフローグラフの可視化、低レベルカーネル制御 |

## ---

**7\. 結論と展望**

本調査から明らかになったのは、AI技術の重心が「単なるアルゴリズムの改良」から「計算パラダイムの再定義」へと移行しているという事実である。

1. **BitNet** は、1.58ビットという極限の量子化によって、現在のデジタル半導体（CMOS）上で即座に利用可能な劇的な効率化を提供する。bitnet.cpp や llama.cpp の進化は、この技術が実験室を出て実用段階に入ったことを示している。  
2. **SNN** は、時間的なスパース性を利用し、イベント駆動型の処理を実現する。SpikingJelly や Norse といった成熟したフレームワークの登場により、ディープラーニングとの統合が容易になり、ニューロモルフィックハードウェアのポテンシャルを引き出しつつある。  
3. **Forward-Forward** や **Active Inference** は、学習と推論のアルゴリズム自体を見直し、大域的な同期を不要にしたり、不確実性の解消を行動原理に組み込んだりすることで、より自律的で効率的なエージェントの実現を目指している。  
4. **熱力学・アナログコンピューティング** は、物理現象そのものを計算に利用するという究極のアプローチであり、THRML や aihwkit といったシミュレータが、将来のハードウェアへの架け橋となっている。

これらの技術は相互に排他的なものではない。例えば、BitNetの三値重みをアナログメモリで実装する、あるいはActive Inferenceの制御ループをSNNで実装するといった融合領域にこそ、次世代のブレイクスルーが潜んでいると考えられる。エンジニアや研究者にとって、これらの「非標準」的なツールチェーンを習得することは、ムーアの法則後の世界でイノベーションを牽引するための必須条件となるだろう。

---

引用:

38

#### **引用文献**

1. kevbuh/bitnet: pure pytorch implementation of Microsoft's BitNet b1.58 2B4T \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/kevbuh/bitnet](https://github.com/kevbuh/bitnet)  
2. microsoft/bitnet-b1.58-2B-4T \- Hugging Face, 12月 17, 2025にアクセス、 [https://huggingface.co/microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)  
3. 1-Bit LLM and the 1.58 Bit LLM- The Magic of Model Quantization | by Dr. Nimrita Koul, 12月 17, 2025にアクセス、 [https://medium.com/@nimritakoul01/1-bit-llm-and-the-1-58-bit-llm-the-magic-of-model-quantization-ee47697c561a](https://medium.com/@nimritakoul01/1-bit-llm-and-the-1-58-bit-llm-the-magic-of-model-quantization-ee47697c561a)  
4. microsoft/BitNet: Official inference framework for 1-bit LLMs \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)  
5. 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs \- Microsoft, 12月 17, 2025にアクセス、 [https://www.microsoft.com/en-us/research/publication/1-bit-ai-infra-part-1-1-fast-and-lossless-bitnet-b1-58-inference-on-cpus/](https://www.microsoft.com/en-us/research/publication/1-bit-ai-infra-part-1-1-fast-and-lossless-bitnet-b1-58-inference-on-cpus/)  
6. A Practitioner's Guide on Inferencing over 1-bit LLMs using bitnet.cpp, 12月 17, 2025にアクセス、 [https://adasci.org/a-practitioners-guide-on-inferencing-over-1-bit-llms-using-bitnet-cpp/](https://adasci.org/a-practitioners-guide-on-inferencing-over-1-bit-llms-using-bitnet-cpp/)  
7. 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs \- arXiv, 12月 17, 2025にアクセス、 [https://arxiv.org/html/2410.16144v1](https://arxiv.org/html/2410.16144v1)  
8. ggml-quants : ternary packing for TriLMs and BitNet b1.58 \#8151 \- SemanticDiff, 12月 17, 2025にアクセス、 [https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/8151/overview](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/8151/overview)  
9. llama.cpp merges support for TriLMs and BitNet b1.58 : r/LocalLLaMA \- Reddit, 12月 17, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1fa3ryv/llamacpp\_merges\_support\_for\_trilms\_and\_bitnet\_b158/](https://www.reddit.com/r/LocalLLaMA/comments/1fa3ryv/llamacpp_merges_support_for_trilms_and_bitnet_b158/)  
10. 1.58 BitNets \- a new opportunities for llamafile? \#313 \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/Mozilla-Ocho/llamafile/discussions/313](https://github.com/Mozilla-Ocho/llamafile/discussions/313)  
11. Implementation of "BitNet: Scaling 1-bit Transformers for Large Language Models" in pytorch \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/kyegomez/BitNet](https://github.com/kyegomez/BitNet)  
12. Oxen-AI/BitNet-1.58-Instruct \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/Oxen-AI/BitNet-1.58-Instruct](https://github.com/Oxen-AI/BitNet-1.58-Instruct)  
13. snnTorch Documentation — snntorch 0.9.4 documentation, 12月 17, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/](https://snntorch.readthedocs.io/en/latest/)  
14. snnTorch \- Open Neuromorphic, 12月 17, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/snntorch/](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/snntorch/)  
15. snnTorch Documentation — snntorch 0.9.4 documentation, 12月 17, 2025にアクセス、 [https://snntorch.readthedocs.io/](https://snntorch.readthedocs.io/)  
16. snntorch.surrogate \- Read the Docs, 12月 17, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html)  
17. snntorch.spikegen \- Read the Docs, 12月 17, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html)  
18. SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)  
19. spikingjelly \- PyPI, 12月 17, 2025にアクセス、 [https://pypi.org/project/spikingjelly/](https://pypi.org/project/spikingjelly/)  
20. SpikingJelly \- Open Neuromorphic, 12月 17, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/)  
21. norse.torch.module.snn. \- GitHub Pages, 12月 17, 2025にアクセス、 [https://norse.github.io/norse/auto\_api/norse.torch.module.snn.html](https://norse.github.io/norse/auto_api/norse.torch.module.snn.html)  
22. Spiking neural networks with Norse — Norse Tutorial Notebook, 12月 17, 2025にアクセス、 [https://norse.github.io/notebooks/intro\_norse.html](https://norse.github.io/notebooks/intro_norse.html)  
23. norse/norse: Deep learning with spiking neural networks (SNNs) in PyTorch. \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/norse/norse](https://github.com/norse/norse)  
24. Spiking Neural Network (SNN) Library Benchmarks \- Open Neuromorphic, 12月 17, 2025にアクセス、 [https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)  
25. Reimplementation of Geoffrey Hinton's Forward-Forward Algorithm \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/loeweX/Forward-Forward](https://github.com/loeweX/Forward-Forward)  
26. On Advancements of the Forward-Forward Algorithm \- arXiv, 12月 17, 2025にアクセス、 [https://arxiv.org/html/2504.21662v1](https://arxiv.org/html/2504.21662v1)  
27. THE TRIFECTA: THREE SIMPLE TECHNIQUES FOR TRAINING DEEPER FORWARD-FORWARD NETWORKS \- OpenReview, 12月 17, 2025にアクセス、 [https://openreview.net/pdf/4151184ec274e5bfb19ef0f9600ff0ef1e4963f7.pdf](https://openreview.net/pdf/4151184ec274e5bfb19ef0f9600ff0ef1e4963f7.pdf)  
28. mpezeshki/pytorch\_forward\_forward: Implementation of Hinton's forward-forward (FF) algorithm \- an alternative to back-propagation \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/mpezeshki/pytorch\_forward\_forward](https://github.com/mpezeshki/pytorch_forward_forward)  
29. CoderPat/eqprop: Implementation of Equilibrium Propagation and related papers in JAX, 12月 17, 2025にアクセス、 [https://github.com/CoderPat/eqprop](https://github.com/CoderPat/eqprop)  
30. zeligism/eqprop: Equilibrium Propagation \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/zeligism/eqprop](https://github.com/zeligism/eqprop)  
31. Laborieux-Axel/holomorphic\_eqprop: Repository to reproduce the results of the paper "Holomorphic Equilibrium Propagation Computes Exact Gradients Through Finite Size Oscillations" \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/Laborieux-Axel/holomorphic\_eqprop](https://github.com/Laborieux-Axel/holomorphic_eqprop)  
32. (PDF) pymdp: A Python library for active inference in discrete state spaces \- ResearchGate, 12月 17, 2025にアクセス、 [https://www.researchgate.net/publication/357765758\_pymdp\_A\_Python\_library\_for\_active\_inference\_in\_discrete\_state\_spaces](https://www.researchgate.net/publication/357765758_pymdp_A_Python_library_for_active_inference_in_discrete_state_spaces)  
33. \[2201.03904\] pymdp: A Python library for active inference in discrete state spaces \- arXiv, 12月 17, 2025にアクセス、 [https://arxiv.org/abs/2201.03904](https://arxiv.org/abs/2201.03904)  
34. infer-actively/pymdp: A Python implementation of active inference for Markov Decision Processes \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/infer-actively/pymdp](https://github.com/infer-actively/pymdp)  
35. pymdp Fundamentals — pymdp 0.0.7.1 documentation \- pymdp's documentation\!, 12月 17, 2025にアクセス、 [https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp\_fundamentals.html](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp_fundamentals.html)  
36. Tutorial 1: Active inference from scratch \- pymdp's documentation\! \- Read the Docs, 12月 17, 2025にアクセス、 [https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active\_inference\_from\_scratch.html](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html)  
37. Extropic's 10,000x AI energy breakthrough \- The Rundown AI, 12月 17, 2025にアクセス、 [https://www.therundown.ai/p/extropics-10-000x-ai-energy-breakthrough](https://www.therundown.ai/p/extropics-10-000x-ai-energy-breakthrough)  
38. Thermodynamic Computing: From Zero to One \- Extropic, 12月 17, 2025にアクセス、 [https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one)  
39. Extropic | Home, 12月 17, 2025にアクセス、 [https://extropic.ai/](https://extropic.ai/)  
40. extropic-ai/thrml: Thermodynamic Hypergraphical Model Library in JAX \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)  
41. Software | Extropic, 12月 17, 2025にアクセス、 [https://extropic.ai/software](https://extropic.ai/software)  
42. IBM/aihwkit: IBM Analog Hardware Acceleration Kit \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/IBM/aihwkit](https://github.com/IBM/aihwkit)  
43. Welcome to IBM Analog Hardware Acceleration Kit's documentation\! — IBM Analog Hardware Acceleration Kit 1.0.0 documentation, 12月 17, 2025にアクセス、 [https://aihwkit.readthedocs.io/](https://aihwkit.readthedocs.io/)  
44. Analog AI Cloud Composer Overview, 12月 17, 2025にアクセス、 [https://aihwkit.readthedocs.io/en/latest/composer\_overview.html](https://aihwkit.readthedocs.io/en/latest/composer_overview.html)  
45. Ryzen AI Software — Ryzen AI Software 1.6.1 documentation \- AMD, 12月 17, 2025にアクセス、 [https://ryzenai.docs.amd.com/](https://ryzenai.docs.amd.com/)  
46. Exploration Software Framework — Riallto \- An exploration framework for Ryzen AI, 12月 17, 2025にアクセス、 [https://riallto.ai/notebooks/4\_1\_software\_framework.html](https://riallto.ai/notebooks/4_1_software_framework.html)  
47. Rain Neuromorphics Tapes Out Demo Chip for Analog AI \- UF Innovate, 12月 17, 2025にアクセス、 [https://innovate.research.ufl.edu/rain-neuromorphics-analog-ai/](https://innovate.research.ufl.edu/rain-neuromorphics-analog-ai/)  
48. Products — Rain AI, 12月 17, 2025にアクセス、 [https://rain.ai/products](https://rain.ai/products)  
49. rain-neuromorphics/energy-based-learning \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/rain-neuromorphics/energy-based-learning](https://github.com/rain-neuromorphics/energy-based-learning)  
50. TSU 101: An Entirely New Type of Computing Hardware | Extropic, 12月 17, 2025にアクセス、 [https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware)  
51. Llama.cpp now supports BitNet\! : r/LocalLLaMA \- Reddit, 12月 17, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1dmt4v7/llamacpp\_now\_supports\_bitnet/](https://www.reddit.com/r/LocalLLaMA/comments/1dmt4v7/llamacpp_now_supports_bitnet/)