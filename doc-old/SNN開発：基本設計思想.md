

# **検証レポート：スパイキングニューラルネットワーク（SNN）が人工ニューラルネットワーク（ANN）を超えるための技術的妥当性評価 — 基本設計思想に基づくロードマップの必須工程（2024-2025年次レビュー）**

提出先： AI戦略研究コンソーシアム・技術評価委員会  
作成者： 主任研究員、ニューロモルフィック・コンピューティング部門  
文書番号： TDC-25-1089  
日付： 2025年10月28日  
機密度： 高（戦略的R\&D指針）

---

## **I. 序論：命題の再定義 — SNNの「基本設計思想」と「ANNを超える」ことの真の意味**

本レポートは、スパイキングニューラルネットワーク（SNN）ベースのAIが、現行の人工ニューラルネットワーク（ANN）系を超えるために策定された研究開発ロードマップ（roadmap.md）の技術的妥当性を検証するフレームワークを提供する。ユーザー要求に基づき、本検証は「基本設計思想」の整合性を最重要視する。これは、SNNがANNの単なる高効率な模倣に留まらず、その本質的な計算パラダイムを活用し、質的に異なる、あるいはANNでは達成不可能な優位性を確立できるかという戦略的評価を意味する。

roadmap.mdの具体的内容が提示されていないため、本レポートは、2024年から2025年にかけての最新の学術論文および技術的実証に基づき、SNNがANNを超えるために**不可欠な技術的要件、克服すべき中核的障壁、および戦略的岐路**を網羅的に分析・提示する「マスター検証フレームワーク」として機能する。提示されたロードマップは、本フレームワークに照らし合わせて評価されるべきである。

### **1.1. SNNの核心的パラダイム：ANNとの根本的断絶**

SNNは「第3世代のニューラルネットワーク」と呼称される 1。この呼称は単なる世代更新を意味するのではなく、ANN（第2世代）との計算モデルにおける根本的な断絶を示している。SNNの「基本設計思想」は、生物学的な脳の動作原理に触発された以下の3つの柱に集約される 3。

1. **イベント駆動（Event-Driven）:** ANNがフレーム（静止画など）に対し、クロック同期ですべてのニューロンを計算する「密（dense）」な処理を行うのに対し、SNNは情報（入力）が到着した（＝イベントが発生した）ニューロンのみが非同期に計算を実行する。  
2. **スパース性（Sparsity）:** SNNの情報伝達は、離散的なバイナリ信号である「スパイク」によって行われる 8。情報はスパイクの有無とそのタイミングに符号化され、ネットワーク全体の活性は極めて「疎（sparse）」に保たれる。  
3. **時間的情報処理（Temporal Processing）:** ANNのニューロンが本質的に「状態を持たない（stateless）」のに対し、SNNのニューロン（例：Leaky Integrate-and-Fire, LIF）は「状態を持つ（stateful）」 4。膜電位という形で過去の入力を時間的に統合し、そのダイナミクス自体が計算の一部を担う。これにより、SNNは本質的に時間的な情報を扱う能力を持つ 3。その本質は、静的なANNのみならず、時間を離散ステップで模倣するRNNとも異なる 10。

### **1.2. 「ANNを超える」ことの多次元的定義**

ロードマップの妥当性を検証する上で、最初の重要なステップは「ANNを超える」という目標の定義である。ANN、特にTransformerベースの大規模モデル（LLM）は、現在「精度（Accuracy）」において支配的な地位を確立している。SNNがこれを超えるとは、単一の指標ではなく、以下の多次元的なベンチマークにおいて優位性を示すことである。

* A) 精度（Accuracy）:  
  従来、SNNはANNの精度に及ばないとされてきた 11。しかし、このギャップは急速に縮小している。学習アルゴリズム（後述）とアーキテクチャの進展により、2024年の研究（SGLFormer）では、SNNが大規模データセットImageNet-1kにおいてTop-1精度83.73%を達成した 13。これは、数年前の高性能ANNに匹敵する領域であり、SNNが大規模タスクでANNと競合可能であることを示している。  
* B) エネルギー効率（Energy Efficiency）:  
  これはSNNの最大の約束であり、「基本設計思想」の直接的な帰結である 7。優位性の源泉は2点ある。第一に、計算がスパースであるため、活性化したニューロンのみがエネルギーを消費する 3。第二に、ANNで支配的な高コストな浮動小数点乗加算（MAC）演算が、SNNではバイナリ・スパイクによる低コストな加算（AC）演算で置き換えられるためである 13。  
* C) レイテンシ（Latency）:  
  イベント駆動型であるため、SNNは最初の入力スパイクを受け取った瞬間から処理を開始できる 3。これにより、特に時間的な応答が求められるタスク（例：ロボティクス、音声認識）において、ANN（フレーム全体をバッファリングする必要がある）と比較して、原理的に超低レイテンシでの推論が可能となる 6。  
* D) 新規能力（Novel Capability）:  
  SNNの真の優位性は、ANNが本質的に苦手とする領域にある。それは、DVS（Dynamic Vision Sensor）カメラ、RF信号、あるいは生体信号（EEG, ECoG）など、非同期的でノイズが多く、時間的構造が重要な実世界のセンサーデータを直接扱う能力である 16。

### **1.3. 効率性の誤謬：von Neumannアーキテクチャ（GPU）という罠**

ロードマップの成否を分ける**最も根本的な前提条件**は、SNNを実行する「計算基盤」の定義である。

現行のAI開発で主流のGPUやCPUは、クロック同期型で高密度の行列演算に最適化された「von Neumann型」アーキテクチャである。SNNの「基本設計思想」（非同期、スパース、イベント駆動）は、このGPUの設計思想と**根本的にミスマッチ**である 12。

GPU上でSNNを実行することは、本質的に非同期なプロセスを、高密度な行列演算を用いてクロック同期で「シミュレート」することに他ならない。このシミュレーションは極めて非効率であり、SNNが持つ本来のエネルギー効率の利点を**完全に相殺、あるいは逆転させる**。

この事実は、2024年の研究 20 によって明確に実証されている。商用ハードウェア（GPU）上でSNNとCNN（ANNの一種）の画像分類タスクを比較したところ、SNNはCNNと比較して**142%多い電力**と**128%多いメモリ**をトレーニング中に消費した。

したがって、ロードマップがSNNの実行基盤としてGPUを前提とし、GPU上でのANNとの性能比較を主軸に置いている場合、そのロードマップは「基本設計思想」から逸脱しており、**戦略的に根本的な誤り**を犯している。SNNの優位性は、後述する「ニューロモルフィック・ハードウェア」の設計思想と不可分である 21。

### **1.4. 真のベンチマーク：スパースANN vs. SNN on Neuromorphic**

「ANN vs SNN」という比較は、多くの場合、実行基盤の違いにより不公平（Apple-to-Orange）なものとなっている 8。SNNの真の価値を問うためには、より厳密な比較が求められる。

2024年の研究 8 は、この「公正な比較」を実行した。研究チームは、SNNと同様にイベント駆動処理が可能なニューロモルフィック・プロセッサ「SENECA」を開発した。その上で、同程度にスパース化（活性化率約5%）された「スパースANN」と「SNN」を、同一のハードウェア上で比較した。

実験結果は決定的であった。イベントベースのオプティカルフロー推定タスクにおいて、SNNはスパースANNと比較して、**75.2%のエネルギー消費**（24.8%削減）および**62.5%の処理時間**（37.5%削減）で処理を完了した 8。

この優位性の源泉は、従来主張されてきた「MAC演算の削減」8 だけでなく、より深いレベルにあることが示唆された。SNNは、ANN（66.5%）と比較してピクセルワイズのスパイク密度が低かった（43.5%）。これは、SNNがニューロンの状態（膜電位など）を保持・更新するために必要な**メモリアクセス操作が少ない**ことを意味する 8。

この分析がロードマップに示唆する内容は極めて重要である。SNNがANNを超えるために最適化すべき指標は、単なる計算量（FLOPs）ではなく、「**スパース性の維持**」と「**メモリアクセスコストの最小化**」である。

### **1.5. 提案テーブル(1)：SNN vs. ANN：「超越」の多角的ベンチマークと実行基盤の依存性**

ロードマップの前提条件を検証するため、以下のテーブルは「SNNの優位性がいかに実行基盤に依存するか」を明確にする。これは、ロードマップが「ハードウェア協調設計」を必須工程とせねばならない根拠となる。

| 評価指標 | ANN (on GPU) | SNN (on GPU) | SNN (on Neuromorphic Hardware) |
| :---- | :---- | :---- | :---- |
| **精度 (ImageNet)** | **SOTA (例: 86.7%)** \[22\] | **高性能 (例: 83.73%)** 13 (SOTA ANNに匹敵しつつある) | **高性能 (例: 83.73%)** 13 (アルゴリズムは同一) |
| **エネルギー効率** | 基準 (高消費電力) | **ANNより大幅に悪化** (CNN比 142%の電力消費) 20 | **ANNより大幅に向上** (スパースANN比 75.2%の消費) 8 |
| **レイテンシ** | 基準 (フレームレート依存) | **ANNより悪化** (シミュレーション・オーバーヘッド) 20 | **ANNより大幅に向上** (スパースANN比 62.5%の時間) 8 |
| **基本設計思想** | 高密度・同期的・状態なし | **非互換** (非同期・スパース処理を同期・高密度ハードでシミュレート) | **完全互換** (非同期・スパース・イベント駆動) |

---

## **II. 最大の障壁：「学習」と「時間的信用分配問題（TCA）」の解体**

SNNがANNを超えるためのロードマップにおいて、最もリソースを要し、かつ最も根本的な「工程」は、効率的かつスケーラブルな学習アルゴリズムの確立である。

### **2.1. SNN学習の根本的困難：非微分可能性**

SNNの学習が歴史的に困難であった理由は、その「基本設計思想」に起因する。SNNニューロンの発火（スパイク）は、「オール・オア・ナッシング」の不連続なイベントである 13。これは数学的にはヘヴィサイドのステップ関数としてモデル化され、その導関数（勾配）はほとんど全ての点でゼロであり、閾値でのみ無限大となる。

つまり、SNNの発火イベントは**微分不可能**である 23。

これにより、ANNの爆発的な成功を支えたバックプロパゲーション（誤差逆伝播法）と勾配降下法（SGD）という、勾配情報に基づく最適化手法が**SNNに直接適用できない** 23。この問題が、SNNの高性能化を長らく妨げてきた最大の技術的障壁であった。

### **2.2. 真の課題：時間的信用分配問題（Temporal Credit Assignment, TCA）**

しかし、「非微分可能性」は問題の表層に過ぎない。SNNの学習における真の課題は、より深く、より困難な「**時間的信用分配（Temporal Credit Assignment, TCA）**」の問題である 26。

前述の通り、SNNは本質的に時間的ダイナミクスを持つ再帰的なシステム（Recurrent System）である 10。TCA問題とは、「ある時点 $t$ で観測されたエラー（＝望ましい結果との差）に対して、$t$ よりも（遠い）過去のどのスパイク（＝行動）が、どれだけ寄与したのか」を特定する問題である 26。

ANN（非再帰型）の信用分配は「空間的」（どのニューロンがエラーの原因か）である。RNNの信用分配は「時間的」だが、SNNのTCAは、RNNのそれよりも遥かに複雑である。なぜなら、SNNの情報は「発火したか、しなかったか」というバイナリかつスパースなイベントに凝縮されており、RNNの連続値の活性化と比較して、勾配情報が本質的に乏しいためである。

TCAの解決は、SNNが単なる「レートコーダー」（発火頻度のみを情報として使い、ANNを模倣する）を超え、その「基本設計思想」の核心である*スパイクの精密なタイミング* 31 に意味を持たせるための**絶対的な前提条件**である。

### **2.3. ベンチマークの罠：我々のSNNは本当に「時間」を使っているか？**

ロードマップを検証する上で、極めて重大な論点が存在する。それは、「開発中のSNNは、その核心的利点である**時間的処理能力を本当に使っているのか？**」という問いである。

この問いに対して、2025年2月に発表された研究 4 は、現在のSNN研究コミュニティ全体に衝撃を与える分析結果を提示した。この研究は、「分離的時間プローブ（Segregated Temporal Probe, STP）」という新しい分析ツールを提案し、SNNの学習における時空間的な情報経路を意図的に遮断・分離した。

具体的には、SNNの学習を以下の3つのコンポーネントに分解した。

1. **STBP（Spatio-Temporal Backpropagation）:** SNNの完全な時空間バックプロパゲーション（標準的なSNN学習）。  
2. **SDBP（Spatial Domain Backpropagation）:** 順伝播は時間的に行うが、逆伝播（学習）では時間的な勾配（過去への依存）を遮断し、空間的な勾配のみを伝播する。  
3. **NoTD（No Temporal Domain）:** 順伝播・逆伝播の両方で時間的な依存関係を遮断し、各タイムステップを独立したものとして扱う（＝時間処理能力を完全に排除）。

衝撃的な結果:  
CIFAR10-DVSやN-Caltech101といった、SNNのテストに広く使われている「イベントベースの視覚データセット」において、NoTD（時間処理なし）が、STBP（完全なSNN）とほぼ同等の分類性能を達成した 4。  
この結果が意味すること:  
これは、これらの広く使われているベンチマーク・タスクが、SNNの核心的利点である「時間的処理能力」を全く利用していなくても解けてしまうことを意味する。言い換えれば、現在SOTA（State-of-the-Art）とされている多くのSNNは、TCA問題を解決することなく、各タイムステップを独立した静止画のように処理する「レートコーダー」（ANNの非効率な亜種 32）として振る舞っているだけで、その性能が達成されている可能性が極めて高い。  
ロードマップへの示唆:  
ロードマップの「工程」と「マイルストーン」は、この「ベンチマークの罠」を回避するものでなければならない。単にCIFAR10-DVSのような既存のデータセットで高精度を出すことを目標にしてはならない。  
ロードマップは、TCA問題を*本当に*テストできるベンチマーク（例：4が提案する新しいベンチマークスイートや、高次な時間的相関を必要とするタスク）を導入し、そこでSDBPやNoTDを圧倒する性能を実証することを、必須の「工程」として定義しなければならない。さもなければ、開発されるSNNは「設計思想」から逸脱したものとなり、「ANNを超える」という目標は達成不可能である。

---

## **III. 現行の主要「工程」の分析：SNN学習方法論の批判的評価**

SNNがANNを超えるためのロードマップは、前述の「学習」と「TCA」という根本課題を解決するため、以下の3つの主要な学習「工程」のいずれか（またはその組み合わせ）に依存することになる。各工程の技術的妥当性、最新の進捗（2024-2025年）、および「基本設計思想」との整合性を批判的に評価する。

### **3.1. 工程A：ANN-SNN変換（"移植"）**

* 原理:  
  SNNの学習の困難さを回避する、最も実用的なアプローチ。まず、性能が実証された標準的なANN（通常、活性化関数ReLUを使用）を通常通り訓練する。その後、その学習済み重みを、構造的に対応するSNNに移植（コンバート）する 12。この際、ANNの連続的な活性値（ReLUの出力）を、SNNの平均発火頻度（Firing Rate）が近似するようにパラメータ（発火閾値など）を調整する。  
* 利点:  
  ANNの成熟したエコシステム、強力なアーキテクチャ（ResNet, VGGなど）、および高性能な学習済みモデルを直接活用できる 25。歴史的に、SNNで高精度を達成する最も手軽な手法であった。  
* 致命的欠陥（レイテンシ）:  
  この手法の根本的な欠陥は、「レイテンシと精度のトレードオフ」である。ANNの連続的な活性値をSNNの発火頻度（レート）で正確に近似するためには、SNNが非常に多くの時間ステップ（タイムステップ）にわたってスパイクを観測・積分する必要がある 12。  
  これは、推論に長時間を要すること（＝高レイテンシ）を意味し、SNNの「基本設計思想」である低レイテンシという利点と真っ向から対立する。  
* 原因:  
  この性能劣化の原因は、ANNの連続値とSNNの離散スパイク間の変換プロセスで生じる「変換誤差」である。これには、膜電位が閾値を超えないことによる「クリッピング誤差」、スパイクの離散性による「量子化誤差」36、そしてスパイクが時間的に均一に到着しないことによる「時間的不整合（temporal misalignment）」42 が含まれる。  
* 最新技術（2024-2025）による克服の試み:  
  この致命的欠陥を克服するため、2024年から2025年にかけて「超低レイテンシ変換」技術の研究が爆発的に進展した。  
  1. **画像分類（CNN）:** 2025年のICML（国際機械学習会議）で発表された研究 39 は、この変換誤差を補償するための新しい学習手法（学習可能な閾値クリッピング、デュアル閾値ニューロンなど）を提案した。その結果、CIFAR-10データセットにおいて、ResNet-18ベースのSNNが**わずか2タイムステップ**という超低レイテンシで94.75%の精度を維持することに成功したと報告している。  
  2. **Transformer変換:** ANN-SNN変換は、ReLUベースのCNNに限定されていた。TransformerモデルにはLayerNormやGELUといった非線形モジュールが含まれるため、変換が困難であった 43。しかし、2024年のACM MMで発表された研究 2 は、これらの非線形モジュールを扱うための「期待値補償モジュール（Expectation Compensation Module）」を提案した。これにより、Spiking Transformerへの変換が可能となり、**わずか4タイムステップ**で元のANNの精度から-1%の低下（88.60%）に抑えることに成功した。  
* 戦略的評価:  
  ANN-SNN変換は、SNNを迅速にハードウェア展開するための「橋渡し」技術（Bridging Technology） 44 としては依然として有効である。特に最新の超低レイテンシ変換技術 39 は、その実用性を大幅に高めた。  
  しかし、このアプローチが「ANNを超える」ための最終的なロードマップにはなり得ない。その本質はあくまでANNのレートコーディングの「模倣」であり、SNN固有の時間的潜在能力（Temporal Coding）を引き出すものではない 34。ロードマップ上では、「短期的製品化トラック」として位置づけるのが妥当である。

### **3.2. 工程B：代理勾配（Surrogate Gradient）による直接学習（"SNNネイティブ"）**

* 原理:  
  SNNの「非微分可能性」問題に対する、現在最も主流かつ強力な解決策であり、SNNの学習における「デファクトスタンダード」である 3。  
  この手法の核心は、「順伝播と逆伝播の分離」にある。  
  1. **順伝播（Forward Pass）:** 通常通り、SNNの厳密なダイナミクス（LIFモデルなど）と微分不可能なステップ関数を用いて、スパイクを生成・伝播させる 47。  
  2. **逆伝播（Backward Pass）:** 勾配降下法のための勾配計算（誤差逆伝播）の際、微分不可能なステップ関数の導関数の代わりに、数学的に「滑らか（differentiable）」な\*\*「代理（Surrogate）」\*\*の関数（例：シグモイド関数の導関数や矩形関数）で置き換える 13。

SNNは時間的ダイナミクスを持つため、この代理勾配（SG）は通常、RNNで用いられるBackpropagation Through Time (BPTT) と組み合わせて使用される 12。

* 利点:  
  SNNを（ANN-SNN変換のような）ANNの制約から解放し、SNNネイティブなアーキテクチャをエンドツーエンドで直接学習できる 13。柔軟性が非常に高く、近年のSNNのSOTA性能（例：47）の多くは、このSG-BPTT手法によって達成されている。  
* **課題（ロードマップ上のリスク）:**  
  1. **計算コストとメモリ:** BPTTは、SNNの計算グラフを時間軸に沿って展開（Unroll）する必要がある。タスクが必要とする時間ステップが長くなると（例：TCA問題）、展開されたグラフが巨大になり、**膨大なメモリ消費と計算コスト**を要する 3。  
  2. **スケーラビリティ:** 上記の理由により、ANNの基盤モデル（Foundation Models）のような超大規模ネットワークへのスケーリングには、BPTTがボトルネックとなる 3。  
  3. **勾配ミスマッチ（Gradient Mismatch）:** 代理勾配は、あくまで「真の勾配」の「近似」である。この不一致（ミスマッチ）が、学習の不安定性や、最適解への収束を妨げる（局所最適解に陥る）リスクを常にはらんでいる 50。

### **3.3. 代理勾配は「時間」を学習できるか？（"レートコーディングの罠"）**

代理勾配（SG）はSNNの学習を可能にしたが、それは*何を*学習しているのか？ この問いは、ロードマップが「基本設計思想」に沿っているかを判断する上で決定的である。これは、セクションIIで提示した「ベンチマークの罠」4 と表裏一体の問題である。

* 証拠A（レートコーディングの罠）:  
  2024年のNeurIPSで発表された研究 32 は、SGベースのBPTTが捉える情報は、主にANNと同様の「レートコーディング」（発火頻度）であると結論付けた。この知見に基づき、同研究は「レートベース・バックプロパゲーション」という新しい手法を提案している。これは、あえて詳細な時間的導関数を無視し、平均化されたダイナミクス（＝レート）のみに焦点を当てることで、BPTTの計算コストを削減する手法である。  
  この研究の存在自体が、SG-BPTTがデフォルトではSNNの「基本設計思想」である精密な時間情報を活用しきれておらず、「レートコーディングの罠」にはまっている可能性を強く示唆している 52。  
* 証拠B（時間符号化の潜在能力）:  
  一方で、SGは時間符号化を学習する「潜在能力」を秘めていることも、最新の研究によって示されつつある。  
  この疑問に答えるため、2025年に発表された一連の研究 31 は、意図的にレート情報を排除し、スパイクの精密なタイミングのみ（例：ニューロン内のスパイク間隔（ISI）、ニューロン間の発火同期（Coincidence））に依存する合成タスクを設計した。  
* 重要な発見:  
  これらのタスクにおいて、SGで訓練されたSNNは、時間符号化を必要とするタスクを学習可能であった。一方、レート情報のみに依存するモデルは、これらのタスクの解決に失敗した 31。  
  さらに決定的な証拠として、学習済みのSNNに対し、入力スパイク列を「時間反転」させて入力したところ、性能が著しく低下した 31。これは、SNNが単なるスパイクの統計（レート）ではなく、「時間的因果性」（Aの後にBが来る）を学習していたことを強力に証明するものである。  
* ロードマップへの示唆:  
  代理勾配（SG）を採用する「工程」は、ロードマップ上では「正しい」選択である。しかし、単にSG-BPTTを適用するだけでは、自動的に時間的処理能力が獲得されるわけではなく、「レートコーディングの罠」32 に陥る。  
  したがって、健全なロードマップは、SG-BPTTと並行し、SNNに時間符号化の学習を「強制」するメカニズムを開発するサブタスクを必須工程として含まねばならない。具体的には、「時間的情報に焦点を当てた損失関数の設計」13、「スパイク遅延の学習」31、そして「TCAを真に問うベンチマークでの評価」4 が含まれる。

### **3.4. 工程C：生物学的妥当性を持つ学習（"次世代"）**

* 原理:  
  BPTT（グローバルなエラー情報を全ニューロンに逆伝播させる）が生物学的にあり得ない（non-plausible）という批判に基づき、脳の学習則（シナプス可塑性）にヒントを得た、よりローカルな情報のみで学習するルール群である。これらは「基本設計思想」に最も忠実であり、ニューロモルフィック・ハードウェアへの実装に極めて親和性が高い 56。  
* STDP (Spike-Timing-Dependent Plasticity):  
  最も有名で生物学的に広く観測されている学習則 58。プリニューロンの発火（A）がポストニューロンの発火（B）の直前に起これば結合が強まり（LTP）、順序が逆なら弱まる（LTD）。  
  * *利点:* 非教師あり学習 59、完全ローカル（必要な情報がシナプス近傍に揃う）57、ハードウェア実装が容易 60。  
  * *課題:* STDP単体では、タスク全体のグローバルな目的（例：画像分類のエラー最小化）を解くことができず、性能が限定的である 61。大規模ネットワークへのスケールも困難であった 3。  
  * *評価:* STDPは神経科学的なモデルとしては重要だが、ANNを超える「高性能」を目指す工学的ロードマップの**主流からは外れている** 59。  
* e-prop (Eligibility Propagation):  
  ロードマップがTCA問題（セクションII）を根本的に解決し、かつSG-BPTTの計算コスト問題（セクション3.2）を回避し、さらにハードウェア親和性をも目指すならば、e-prop 63 の研究開発が\*\*最も重要な次世代「工程」\*\*となる。  
  * *原理:* e-propは、ローカルなSTDPとグローバルなBPTTの「ハイブリッド」と見なすことができる 63。  
    1. 各シナプスは、STDPのようにローカルな情報（プリ/ポストニューロンの発火タイミング）に基づき、「**適格性トレース（eligibility trace）**」と呼ばれる短期的な「記憶」を計算・保持する 64。これは「このシナプスが、最近のネットワークの活動にどれだけ寄与したか」の痕跡である。  
    2. 学習の最後に（あるいは断続的に）、タスクのグローバルなエラー信号（あるいは報酬信号 65）がネットワークにブロードキャストされる。  
    3. 各シナプスは、保持していたローカルな「適格性トレース」と、受信したグローバルな「エラー信号」を乗算し、自身の重みを更新する。  
  * *優位性:* このメカニズムにより、BPTTのように時間を通じて勾配を逆伝播させる必要がなくなり、計算コストを大幅に削減できる。それにもかかわらず、BPTTと同等の性能でTCA問題を解くことができる 63。  
  * *最新技術（2025）:* 2025年のICMLで発表予定の研究 66 は、e-propがタスク精度においてBPTTに匹敵するだけでなく、学習されたネットワークの内部表現が、実際の**生物学的な神経データとの「類似性」においてもBPTTに匹敵する**ことを示した。  
  * *戦略的評価:* e-propは、高性能（BPTT）、ハードウェア親和性（ローカルルール）、および「基本設計思想」（TCA解決）の3つを最もバランス良く満たす可能性のある、**最有力な次世代学習アルゴリズム**である。

### **3.5. 提案テーブル(2)：SNN主要学習方法論の批判的評価（2024-2025）**

ロードマップが直面する戦略的選択（どの学習工程にリソースを投下するか）は、以下のトレードオフとして要約できる。

| 方法論 | 基本原理 | SOTA精度 / レイテンシ (2024-2025) | 主要ボトルネック | 「基本設計思想」との整合性 |
| :---- | :---- | :---- | :---- | :---- |
| **ANN-SNN変換** | ANN(ReLU)を学習後、SNN(Rate-coding)に重みを移植。 | **高精度・超低レイテンシ** (CIFAR10: 94.75% @ 2-steps) 39 (Transformer: \-1% acc @ 4-steps) \[2\] | **ANNの模倣** 非ReLU（GELU等）の変換が困難 43。時間的符号化の潜在能力を無視 34。 | **低い** (レートコーディングに依存し、SNN固有の時間処理能力を活用しない) |
| **代理勾配(SG) BPTT** | 微分不可能なスパイクの勾配を、滑らかな関数で「代理」し、BPTTで学習。 | **SNNネイティブSOTA** (ImageNet: 83.73%) 13 (時間符号化の学習能力を実証) 31 | **計算コストとスケーラビリティ** BPTTのメモリ消費が膨大 3。大規模モデルへの適用が困難。 | **中〜高** (時間符号化を学習する「潜在能力」を持つが 31、デフォルトでは「レートコーディングの罠」にはまる 32) |
| **STDP** | 生物学的なローカル学習則（発火タイミングの前後関係）。 | 低〜中精度（限定的） | **性能とスケーラビリティの限界** グローバルなタスク最適化が困難 61。高性能化の主流ではない 59。 | **高い** (ハードウェア親和性) **低い** (高性能化) |
| **e-prop** | ローカルな「適格性トレース」とグローバルな「エラー信号」を組み合わせるハイブリッド学習。 | **BPTTに匹敵** 66 (TCA問題を解く) | **研究開発途上** SG-BPTTほど成熟していない。 | **非常に高い** (TCA問題を解き 63、かつハードウェア親和性（ローカル性）を持つ) |

---

## **IV. レート符号化を超えて：真の時空間コンピューティングのためのアーキテクチャ設計**

学習アルゴリズム（工程）が「How（いかに学ぶか）」であるならば、アーキテクチャは「What（何を学ぶか）」を規定する。ANNのアーキテクチャ（例：ResNet, Transformer）を単純に模倣するだけでは、SNNの「基本設計思想」を活かすことはできず、ANNを超えることはできない。

### **4.1. 情報符号化の責務：時間符号化 vs. レート符号化**

SNNの性能と効率を決定づける最初の設計選択は、「情報をどのようにスパイクに変換（符号化）するか」である。

* レート符号化（Rate Coding）:  
  ANNのReLU活性値の模倣である。情報の値（例：ピクセルの輝度）をスパイクの「頻度」に変換する 9。これは生物学的にも観測されるが、情報を伝達するのに時間ウィンドウを必要とするため、遅く、非効率である 9。セクションIIIで論じた「罠」の根源である。  
* 時間符号化（Temporal Coding）:  
  SNNの「基本設計思想」の核心 52。情報はスパイクの「頻度」ではなく、「タイミング」に符号化される。  
  * **TTFS (Time-to-First-Spike):** 最もシンプルで効率的な時間符号化の一つ 12。情報の値が強い（例：輝度が高い）ほど、「最初」のスパイクが「早く」発火する 9。  
  * **相対的タイミング:** スパイク間の相対的な時間差（ISI）や、異なるニューロン間の同期性に情報が符号化される 31。

SNNの効率性は、レート符号化から時間符号化へ移行することによって飛躍的に向上する。2024年12月に発表された研究 69 では、時間符号化（W-TCRL）とSTDPを組み合わせた表現学習により、他のSNNベースの手法と比較して\*\*最大900倍のスパース性（＝超高効率）\*\*を達成しつつ、高い再構成精度を実証した。

ロードマップへの示唆:  
ロードマップは、「入力符号化スキームの最適化」を独立した重要な「工程」として認識しなければならない。単に固定の符号化（例：レートコーディング）を前提とするのではなく、タスクに最適な符号化方法を選択、あるいは設計する必要がある。2025年9月の研究 70 が示すように、最先端のアプローチは、この符号化プロセス自体を（代理勾配を用いて）\*\*学習可能（differentiable）\*\*にし、ネットワーク全体でエンドツーエンドに最適化することである。

### **4.2. ニューロンモデルの進化：LIFから適応型・再帰型へ**

アーキテクチャの基本単位は「ニューロンモデル」である。どのモデルを選択するかが、ネットワークが処理できる時間的ダイナミクスの豊かさを決定する。

* LIF (Leaky Integrate-and-Fire):  
  最も一般的で、計算コストが低い標準モデル 47。膜電位が時間と共に「漏洩（Leaky）」する単純な積分器である。  
* ALIF (Adaptive LIF):  
  LIFに「適応（Adaptive）」メカニズムを追加したモデル。例えば、ニューロンが一度発火すると、その発火閾値が一時的に上昇し、徐々に元に戻る、といった動的な特性を持つ。  
  2025年7月の研究 72 は、このALIFニューロンが、単純なLIFよりも優れた時空間処理能力を持つことを理論的・実証的に示した。さらに、2025年9月の研究 73 は、ALIFが持つ適応ダイナミクスが、ネットワークの学習において時間的信用分配（TCA）を促進する（＝学習を助ける）という重要な役割を果たすことを明らかにしている。  
* Izhikevich Model:  
  LIFよりもわずかに複雑だが、バースト発火（連続したスパイク群）、チャタリング（高速なバースト）、正則発火など、生物学的なニューロンが示す多様な発火パターンを、計算効率よく再現できる 74。タスクの要求に応じてこれらの多様なダイナミクスを使い分けることで、LIFベースのネットワークよりも高い性能と効率を達成できる可能性がある 17。  
* RSNN (Recurrent SNN):  
  SNNはニューロン内部に「状態」（膜電位）を持つため、本質的に再帰的である。しかし、アーキテクチャレベルで明示的な「再帰結合（Recurrent Connection）」を追加することで、その時間的情報処理能力をさらに強化できる 15。  
  2025年9月の研究 77 は、RSNNにおいて、再帰結合における「伝達遅延（Delay）」が時間的処理に不可欠な役割を果たすことを指摘した。さらに、この遅延時間を固定値とするのではなく、ネットワークがタスクに応じて\*\*遅延自体を学習可能（DelRec）\*\*にすることで、時間的タスクの性能が大幅に向上することを示している。

ロードマップへの示唆:  
ロードマップの「工程」には、単純なLIFモデルからスタートし、タスクの要求（特に複雑な時間的依存関係）に応じて、ALIF（TCAの促進）72 や、学習可能な遅延を持つRSNN 77 へと、ニューロンモデルを高度化するステップが含まれるべきである。

### **4.3. スケーラビリティの最前線(1)：Spiking Transformer**

SNNがANNを超えるには、現在のANNの成功を支える最先端アーキテクチャ、すなわち「Transformer」78 の領域で競合する必要がある。2024年から2025年にかけて、SNNコミュニティはこの課題に正面から取り組み、目覚ましい成果を上げている 12。

* 根本的課題:  
  標準的なTransformerのSelf-Attention（自己注意）機構は、SNNの「基本設計思想」と致命的に非互換である。  
  1. **高密度な演算:** Attentionは、Query (Q), Key (K), Value (V) の3つの高密度なテンソル間の行列積（Dot Product）を必要とする 80。  
  2. **非SNN的演算:** 計算途中でSoftmax関数（指数関数と除算）が必要であり、これはSNNの加算ベースのイベント駆動型計算とは相容れない 80。  
* 解決策（"Spiking Self-Attention" の発明）:  
  ロードマップは、「Attention機構自体をSNNネイティブに再設計する」という「工程」を含まなければならない。  
  * Spikformer 82: 2023年に提案された先駆的なモデル。Q, K, Vをスパイク形式（バイナリ）で扱い、Softmaxを回避する「Spiking Self Attention (SSA)」を提案した。これにより、高コストな乗算を回避し、スパースな計算（スパイクが発生した箇所のみの計算）を実現した 82。  
  * **最新の進展（2025）:**  
    * 2025年のCVPR（コンピュータビジョンとパターン認識のトップ会議）で発表予定の研究 83 は、トークン間の類似度計算（QとKの内積）を、SNNと親和性の高いバイナリ演算である「**XNOR**」（排他的論理和の否定）に置き換えるアプローチを提案している。  
    * 2025年1月の研究 84 は、SNNのAttention機構に「ゲート（Gating）」メカニズム（ANNのLSTMやGRUで使われる技術）を導入し、Attentionをより動的に制御する「SGSAFormer」を提案している。  
* 性能ベンチマーク:  
  これらのSNNネイティブなアプローチにより、SNNはTransformer領域でもANNに匹敵する性能を達成し始めた。  
  * SGLFormer (2024) は、Transformerと畳み込み構造をSNN内で融合し、ImageNet-1kで\*\*83.73%\*\*という画期的な精度を達成した 13。  
  * SNN-ViT (2025) 85 や Spike-driven Transformer v2 13 など、SOTA性能を更新するSpiking Transformerが次々と発表されている。  
* 戦略的岐路（Insight 8）:  
  Spiking Transformerには2つのアプローチが存在する。  
  1. **ANN-SNN変換（工程A）:** 訓練済みのANN TransformerをSNNに変換する 2。  
  2. **SNNネイティブ設計（工程B）:** Spikformer 82 や SGLFormer 13 のように、Attention機構自体をSNNネイティブに再設計する 80。

「ANNを超える」という「基本設計思想」の観点からは、後者（SNNネイティブ設計）が本質的なロードマップであり、リソースを集中投下すべき本命のアプローチである。

### **4.4. スケーラビリティの最前線(2)：SNNによる生成モデル**

SNNがANNを超えるための戦場は、もはや画像分類のような「識別モデル（Discriminative Model）」に限定されない。

2024年から2025年にかけて、SNNが「**生成モデル（Generative Model）**」という、ANNの最も高度な領域（LLMや拡散モデル）に到達したことは、ロードマップの最終目標を再定義する上で極めて重要である。

* SpikeGPT (大規模言語モデル, LLM):  
  2024年に発表された「SpikeGPT」88 は、SNNによる初の\_大規模\_言語モデル（最大2億1600万パラメータ）である。  
  * *アーキテクチャ:* ANNのLLMであるRWKVアーキテクチャをベースにしている。RWKVは、TransformerのSelf-Attention（計算量が $O(N^2)$）を、SNNと親和性の高いRNN形式（計算量が線形の $O(N)$）に置き換えたモデルである。  
  * *性能:* SpikeGPTは、非スパイクモデル（ANN版RWKV）に匹敵する言語モデリング性能を達成しつつ、ニューロモルフィック・ハードウェア上での実行を想定した場合、**32.2倍少ない演算**（＝エネルギー効率）を実現できると報告されている 89。  
  * *意義:* SNNが大規模な言語生成タスクを（単なる識別ではなく）実行可能であることを実証した。  
* Spiking Diffusion Models (SDMs) (画像生成):  
  2024年8月に提案された「Spiking Diffusion Models (SDMs)」12 は、SNNベースの画像生成拡散モデルである。  
  * *アーキテクチャ:* 拡散モデルの中核であるノイズ除去ネットワーク（通常U-Net）を、**完全にSNNで構築**した。  
  * *イノベーション:* 生物学的なシナプス可塑性（入力の変動性を捉える）にヒントを得た「Temporal-wise Spiking Mechanism (TSM)」を導入し、SNNがノイズ除去のダイナミックなプロセスを効果的に学習できるようにした 90。  
  * *性能:* SDMsは、従来のSNNベースの生成モデルの性能を劇的に改善し（FIDスコアで最大12倍）、対応するANNベースの拡散モデルと比較して**約60%のエネルギー削減**を達成したと報告されている 90。  
  * *意義:* SNNが、AIで現在最も計算集約的なタスクの一つである拡散モデルを実行し、かつANNに対する明確なエネルギー優位性を示した。

ロードマップへの示唆:  
「SNNがANNを超える」というロードマップの最終目標は、もはや「ResNetを超える」ことではない。それは、Transformer、GPT、Diffusion ModelといったANNの最重要フロンティアを、SNNのパラダイム（スパース、イベント駆動、時間処理）で置き換えることを意味する。SpikeGPT 88 とSDMs 91 という2024-2025年のブレークスルーは、この壮大な「工程」が非現実的な夢ではなく、具体的な工学的ターゲットとなったことを示している。

### **4.5. 提案テーブル(3)：SNN SOTA性能 vs. ANN：大規模タスク・ベンチマーク（2024-2025）**

以下のテーブルは、SNNがANNの牙城である大規模タスクにおいて、精度で匹敵し、効率で凌駕し始めたことを示す。これは、ロードマップの目標設定の妥当性を裏付ける強力な証拠となる。

| タスク領域 | ANNモデル / SOTAベンチマーク | ANN 精度 / 指標 | SNNモデル / SOTA (2024-2025) | SNN 精度 / 指標 | SNN側の主要イノベーション |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **画像分類** (ImageNet-1k) | DeiT-III / ViT-H \[22\] | 85.7% \- 86.7% | **SGLFormer** (2024) 13 | **83.73%** | SNNネイティブなTransformerとConvolutionの融合アーキテクチャ。 |
| **言語生成** (LLM) | RWKV (ANN) 88 | 基準性能 | **SpikeGPT** (2024) 88 | ANNに匹敵 (演算量: **32.2倍削減**) | SNN互換の線形Attention（RWKVベース）アーキテクチャの採用。 |
| **画像生成** (Diffusion) | DDPM (ANN) 91 | 基準 (FIDスコア) (高エネルギー消費) | **Spiking Diffusion Models (SDMs)** (2024) 90 | ANNに匹敵 (FID: SNN旧SOTAの12倍改善) (エネルギー: **\~60%削減**) | SNNベースU-Net \+ 生物学的可塑性（TSM）の導入。 |

---

## **V. 計算の要諦：SNNの優越性を担保するハードウェア・ソフトウェア協調設計**

本セクションは、ロードマップ全体の成否を分ける「リンチピン（要石）」である。セクションIで提示した「効率性の誤謬」の議論を、具体的な「工程」に落とし込む。

### **5.1. ニューロモルフィック・ハードウェアという至上命題**

本レポートで繰り返し論証してきた通り、SNNの「基本設計思想」（スパース、イベント駆動、非同期）は、GPU/CPUの「von Neumannアーキテクチャ」（高密度、同期的）と**根本的にミスマッチ**である 19。

SNNの「基本設計思想」を真に活かす（すなわち、ANNを超えるエネルギー効率と低レイテンシを実現する）ためには、SNNの計算モデルに合わせて設計された専用の「**ニューロモルフィック・ハードウェア**」（Non-von Neumann型アーキテクチャ）が**絶対的に不可欠**である 8。

ロードマップへの示唆:  
したがって、検証対象のロードマップ（roadmap.md）に、「ニューロモルフィック・ハードウェア戦略」が明確な「工程」として含まれていない場合、あるいはSNNの開発をGPU上のみで行うことを前提としている場合、そのロードマップは「基本設計思想」から逸脱しており、失敗が運命づけられていると結論付けざるを得ない。

### **5.2. ニューロモルフィック・ランドスケープ（2024-2025）**

ロードマップがターゲットとすべき、あるいは協調設計の対象とすべき主要なハードウェア・プラットフォームは、以下のように分類される。

* **大規模デジタルチップ（研究・サーバ用）:**  
  * Intel Loihi 2 93: 現在、SNN研究のSOTAプラットフォームとして広く利用されている。オンチップ学習（学習則のハードウェア実装）をサポートする 96。2025年の報告 97 では、Loihi 2が自律システムにおいてCPU比で100倍以上のエネルギー効率を持つとされている。  
  * SpiNNaker 2 93: 多数のARMコアをメッシュ状に接続し、大規模な神経シミュレーションとSNN実行に特化したマンチェスター大学のシステム。  
  * IBM NorthPole 93: TrueNorth 98 の後継機であり、メモリ内計算（In-memory computing）を特徴とするIBMの最新アーキテクチャ。  
* **アナログ/ミックスドシグナル・チップ:**  
  * BrainScaleS-2 93: ハイデルベルク大学のシステム。ニューロンのダイナミクスを物理的なアナログ回路で「エミュレート」し、生物学的な時間スケールよりも高速に動作させることが可能。  
* **エッジAIチップ（商用・組込み用）:**  
  * SynSense Speck 99: イベント駆動型IoTデバイス向けに設計され、mW（ミリワット）オーダーの超低消費電力を実現。  
  * BrainChip Akida 99: イベントベースの畳み込み処理などに特化した商用チップ。  
  * DYNAP-CNN 18: 100万ニューロンを搭載し、1mW未満の電力で動作可能な畳み込みSNNチップセット。  
* 未来の基盤技術:  
  ロードマップの長期的な視点としては、CMOS技術の先にあるデバイスも視野に入れる必要がある。これには、シナプス（重み）をアナログ的に記憶する「メムリスタ（抵抗変化型メモリ）」100 や、光で計算を行う「ニューロモルフィック・フォトニクス」100 が含まれる。

### **5.3. 協調設計（Co-Design）という最重要「工程」**

SNNがANNを超えるためのロードマップにおける最大の教訓は、SNNアルゴリズムとニューロモルフィック・ハードウェアは、**個別に開発されてはならない**ということである。

SNNアルゴリズム（ソフトウェア）とニューロモルフィック・ハードウェア（ハードウェア）は、「**ハードウェア・ソフトウェア協調設計（Co-Design）**」3 という単一の「工程」として、開発の最初期段階から統合的に扱われなければならない。

この協調設計は、双方向的である。

1. ハードウェアを意識したアルゴリズム（Hardware-Aware Algorithm）:  
   アルゴリズム開発者は、ターゲット・ハードウェアの物理的制約（例：利用可能なニューロンモデルの種類、サポートされる接続のスパース性、重みの量子化ビット数、メモリ帯域）を前提として学習アルゴリズムを設計する 103。  
   * *実証例:* 107 は、ハードウェア（SRAMベースのProcessing-In-Memory, PIM）の特性を考慮した「量子化認識トレーニング（Quantization-Aware Training, QAT）」35 を含むハイブリッド訓練アルゴリズムを協調設計した。その結果、標準的なデジタル実装と比較して、**700倍以上のEDP（Energy-Delay Product）改善**という驚異的な効率向上を達成した。  
2. アルゴリズムを意識したハードウェア（Algorithm-Aware Hardware）:  
   ハードウェア設計者は、実行したい特定のSNNアルゴリズム（例：特定の学習則STDP 108、特定のAttention機構 109）が最も効率的に動作するような専用の回路やデータフローを設計する。

ロードマップへの示唆:  
したがって、健全なロードマップは、「アルゴリズム開発チーム」と「ハードウェア・アーキテクチャチーム」が別々にタスクを進めるウォーターフォール型であってはならない。  
ロードマップは、開発の初期段階から「Sparsity-Aware Co-Design（スパース性を意識した協調設計）」103 や「Quantization-Aware Co-Design（量子化を意識した協調設計）」107 に共同で取り組む、アジャイルかつ統合的なプロセスを明確に定義しなければならない。

---

## **VI. 総括的検証とロードマップへの戦略的提言**

SNNがANNを超えるためのロードマップ（roadmap.md）の技術的妥当性は、それが「基本設計思想」に沿っているか、そして2024-2025年の最新の技術的ブレークスルーと根本的課題を反映しているかによって検証される。

いかなるロードマップも、以下の\*\*5つの戦略的移行（Strategic Transitions）\*\*を中核的な「工程」として明確に定義しているか否かによって、その有効性が判断されるべきである。

### **1.【実行基盤の移行】（"GPUの呪縛"からの脱却）**

* **検証ポイント:** ロードマップは、SNNの主要な実行基盤をGPU/CPU（von Neumann）からニューロモルフィック・ハードウェア（Non-von Neumann）へ移行することを、**開発の前提条件**として定義しているか？ 8  
* **戦略的提言:** ロードマップに「**ハードウェア・ソフトウェア協調設計（Co-Design）**」102 を、R\&Dの最終フェーズのタスクではなく、\*\*「フェーズ1」から始まる中核的な並行「工程」\*\*として位置づけなければならない。これがなければ、SNNの効率性の利点は決して実現されない。

### **2.【学習方法論の移行】（"模倣"からの脱却）**

* **検証ポイント:** ロードマップは、ANN-SNN変換（模倣）の限界（高レイテンシ問題 37）を認識し、代理勾配（Surrogate Gradient, SG）による**SNNネイティブな直接学習** 13 を、「ANNを超える」ための主流の「工程」として採用しているか？  
* **戦略的提言:** ANN-SNN変換（特に最新の超低レイテンシ技術 39）は、「短期的な製品化・橋渡し」トラックに限定すべきである。本質的なR\&Dトラック（ANNを超える）のリソースは、SGベースの直接学習に集中投下する必要がある。

### **3.【設計思想の移行】（"レートコーディングの罠"からの脱却）**

* **検証ポイント:** ロードマップは、開発するSNNが単なる「スパースなANN（＝レートコーディング）」32 になることを防ぎ、SNNの核心的価値である「**時間的情報処理**」能力を確実に引き出すための「工程」を含んでいるか？  
* **戦略的提言:** SG学習手法が**真の時間符号化（Temporal Coding）を学習している**こと 31 を実証するため、「時間符号化スキームの最適化（学習可能なエンコーダなど）」69 と、「TCA問題を真に問う時間的ベンチマーク（4 が指摘する罠を回避するもの）での性能評価」を、必須のマイルストーンとして設定すべきである。

### **4.【アーキテクチャの移行】（"ANNの亜種"からの脱却）**

* **検証ポイント:** ロードマップは、ANNの既存アーキテクチャ（例：ResNet）を単純にSNN化するだけでなく、SNNの「基本設計思想」（ダイナミクス、スパース性）に基づき、アーキテクチャを**SNNネイティブに再発明**する「工程」を含んでいるか？  
* **戦略的提言:** 以下の2つのR\&Dラインを最重要開発項目として設定すべきである。  
  1. **ニューロンダイナミクス:** ALIF（適応型ニューロン）72 やRSNN（学習可能な遅延を持つ再帰結合）77 など、TCA問題の解決と複雑な時間タスクの処理に不可欠な、よりリッチなニューロンモデルを導入する。  
  2. **SNNネイティブ・スケーラビリティ:** Spiking Self-Attention（80）、Spiking Diffusion Models 90、SpikeGPT 88 など、ANNの最前線（Transformer, Diffusion, LLM）に対抗可能な、SNNネイティブの大規模アーキテクチャを開発する。

### **5.【長期的戦略の移行】（"BPTTの限界"からの脱却）**

* **検証ポイント:** ロードマップは、現在の主流であるSG-BPTT手法のスケーラビリティと計算コストの限界 3 を見据え、その先にある「次世代」の学習則を探索する、より長期的かつハイリスクな「工程」を確保しているか？  
* **戦略的提言:** STDPは高性能化には不向きである 59。真にスケーラブルで、生物学的妥当性（＝ハードウェア親和性）を持ち、かつTCA問題を解く可能性を秘めた「**e-prop（Eligibility Propagation）**」63 のような、BPTTに代わる学習則の研究開発を、ハイリスク・ハイリターンのR\&Dトラックとして確保することを強く推奨する。

### **結論**

SNNがANNを超えるためのロードマップは、「精度」という単一の指標を追い求めるものであってはならない。それは、「**時間**」「**スパース性**」「**ハードウェア・アーキテクチャ**」というSNNの3つの「基本設計思想」を、「**協調設計（Co-Design）**」というプロセスによって統合する工学的試みである。

上記の5つの戦略的移行（工程）を欠いたロードマップは、2024-2025年の最新の技術的知見から見て不十分である。そのようなロードマップは、SNNの「基本設計思想」が持つ真のポテンシャルを活かせず、最終的にはANNの非効率な模倣（＝GPU上での高コストなシミュレーション）に終わり、目標を達成できない可能性が極めて高いと結論付ける。

#### **引用文献**

1. ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.726582/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.726582/full)  
2. Towards High-performance Spiking Transformers from ANN to SNN Conversion \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2502.21193v1](https://arxiv.org/html/2502.21193v1)  
3. Spiking Neural Network Architecture Search: A Survey \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.14235v1](https://arxiv.org/html/2510.14235v1)  
4. Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2502.09449v1](https://arxiv.org/html/2502.09449v1)  
5. Spiking Neural Network: a low power solution for physical layer authentication \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2505.12647v1](https://arxiv.org/html/2505.12647v1)  
6. Spiking Neural Networks \- SerpApi, 11月 1, 2025にアクセス、 [https://serpapi.com/blog/spiking-neural-networks/](https://serpapi.com/blog/spiking-neural-networks/)  
7. Spiking Neural Networks and Their Applications: A Review \- PMC \- PubMed Central \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/)  
8. Event-based Optical Flow on Neuromorphic Processor: ANN vs. SNN Comparison based on Activation Sparsification \- University of Twente Research Information, 11月 1, 2025にアクセス、 [https://research.utwente.nl/files/480124986/2407.20421v1.pdf](https://research.utwente.nl/files/480124986/2407.20421v1.pdf)  
9. Stochastic Spiking Neural Networks with First-to-Spike Coding \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2404.17719v2](https://arxiv.org/html/2404.17719v2)  
10. Benchmarking Spiking Neural Network Learning Methods with Varying Locality \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2402.01782v2](https://arxiv.org/html/2402.01782v2)  
11. The advantages and disadvantages of ANN and SNN are compared \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/figure/The-advantages-and-disadvantages-of-ANN-and-SNN-are-compared\_tbl1\_369092250](https://www.researchgate.net/figure/The-advantages-and-disadvantages-of-ANN-and-SNN-are-compared_tbl1_369092250)  
12. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
13. Direct training high-performance deep spiking neural networks: a review of theories and methods \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)  
14. Direct Training High-Performance Deep Spiking Neural Networks: A Review of Theories and Methods \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2405.04289v2](https://arxiv.org/html/2405.04289v2)  
15. ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2505.11455](https://arxiv.org/pdf/2505.11455)  
16. An Event-Based Time-Incremented SNN Architecture Supporting Energy-Efficient Device Classification \- MDPI, 11月 1, 2025にアクセス、 [https://www.mdpi.com/2079-9292/14/18/3712](https://www.mdpi.com/2079-9292/14/18/3712)  
17. Spiking Neural Networks for Multimodal Neuroimaging: A Comprehensive Review of Current Trends and the NeuCube Brain-Inspired Architecture \- PubMed Central, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/)  
18. The Promise of Spiking Neural Networks for Ubiquitous Computing: A Survey and New Perspectives \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2506.01737v1](https://arxiv.org/html/2506.01737v1)  
19. A Survey of Neuromorphic Computing and Neural Networks in Hardware \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/1705.06963](https://arxiv.org/pdf/1705.06963)  
20. Advancements in Image Classification: Comparing Spiking, Convolutional, and Artificial Neural Networks \- NHSJS, 11月 1, 2025にアクセス、 [https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/](https://nhsjs.com/2024/advancements-in-image-classification-comparing-spiking-convolutional-and-artificial-neural-networks/)  
21. Can neuromorphic computing help reduce AI's high energy cost? \- PNAS, 11月 1, 2025にアクセス、 [https://www.pnas.org/doi/10.1073/pnas.2528654122](https://www.pnas.org/doi/10.1073/pnas.2528654122)  
22. A comparative review of deep and spiking neural networks for edge AI neuromorphic circuits \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12528140/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12528140/)  
23. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.24461v1](https://arxiv.org/html/2510.24461v1)  
24. Canonic Signed Spike Coding for Efficient Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2408.17245v2](https://arxiv.org/html/2408.17245v2)  
25. Dopamine-driven synaptic credit assignment in neural networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.22178v1](https://arxiv.org/html/2510.22178v1)  
26. Attention-Based Deep Spiking Neural Networks for Temporal Credit Assignment Problems, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/368316095\_Attention-Based\_Deep\_Spiking\_Neural\_Networks\_for\_Temporal\_Credit\_Assignment\_Problems](https://www.researchgate.net/publication/368316095_Attention-Based_Deep_Spiking_Neural_Networks_for_Temporal_Credit_Assignment_Problems)  
27. On Temporal Credit Assignment and Data-Efficient Reinforcement Learning \- OpenReview, 11月 1, 2025にアクセス、 [https://openreview.net/forum?id=ek0ZniXLac](https://openreview.net/forum?id=ek0ZniXLac)  
28. Learning from delayed feedback: neural responses in temporal credit assignment \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC3208325/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3208325/)  
29. \[2312.01072\] A Survey of Temporal Credit Assignment in Deep Reinforcement Learning \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2312.01072](https://arxiv.org/abs/2312.01072)  
30. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2507.16043v1](https://arxiv.org/html/2507.16043v1)  
31. Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation \- NIPS papers, 11月 1, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/d1bdc488ec18f64177b2275a03984683-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/d1bdc488ec18f64177b2275a03984683-Paper-Conference.pdf)  
32. Benchmarking Artificial Neural Network Architectures for High-Performance Spiking Neural Networks \- MDPI, 11月 1, 2025にアクセス、 [https://www.mdpi.com/1424-8220/24/4/1329](https://www.mdpi.com/1424-8220/24/4/1329)  
33. Auto Deep Spiking Neural Network Design Based on an Evolutionary Membrane Algorithm \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12383992/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12383992/)  
34. High-accuracy deep ANN-to-SNN conversion using quantization-aware training framework and calcium-gated bipolar leaky integrate and fire neuron \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1141701/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1141701/full)  
35. Error-Free ANN-to-SNN Conversion for Extreme Edge Efficiency \- OpenReview, 11月 1, 2025にアクセス、 [https://openreview.net/pdf/258703fd81507ea27ba5013709048d04f57d6834.pdf](https://openreview.net/pdf/258703fd81507ea27ba5013709048d04f57d6834.pdf)  
36. Three-stage hybrid spiking neural networks fine-tuning for speech enhancement \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1567347/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1567347/full)  
37. Temporal Misinformation and Conversion through Probabilistic Spiking Neurons | OpenReview, 11月 1, 2025にアクセス、 [https://openreview.net/forum?id=sgke1JuVlc](https://openreview.net/forum?id=sgke1JuVlc)  
38. Efficient ANN-SNN Conversion with Error Compensation Learning \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2506.01968v1](https://arxiv.org/html/2506.01968v1)  
39. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.23383v1](https://arxiv.org/html/2510.23383v1)  
40. ICML Poster Differential Coding for Training-Free ANN-to-SNN Conversion, 11月 1, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/45408](https://icml.cc/virtual/2025/poster/45408)  
41. Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons \- ICML 2025, 11月 1, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/45627](https://icml.cc/virtual/2025/poster/45627)  
42. Towards High-performance Spiking Transformers from ANN to SNN Conversion \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2502.21193?](https://arxiv.org/pdf/2502.21193)  
43. Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications, 11月 1, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2025/papers/Bu\_Inference-Scale\_Complexity\_in\_ANN-SNN\_Conversion\_for\_High-Performance\_and\_Low-Power\_Applications\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.pdf)  
44. Surrogate Gradient Learning in Spiking Neural Networks | Request PDF \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/330726090\_Surrogate\_Gradient\_Learning\_in\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/330726090_Surrogate_Gradient_Learning_in_Spiking_Neural_Networks)  
45. The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks \- MIT Press Direct, 11月 1, 2025にアクセス、 [https://direct.mit.edu/neco/article/33/4/899/97482/The-Remarkable-Robustness-of-Surrogate-Gradient](https://direct.mit.edu/neco/article/33/4/899/97482/The-Remarkable-Robustness-of-Surrogate-Gradient)  
46. Surrogate gradient learning in spiking networks trained on event-based cytometry dataset \- Optica Publishing Group, 11月 1, 2025にアクセス、 [https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260)  
47. Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2406.19645v1](https://arxiv.org/html/2406.19645v1)  
48. Large Language Models Inference Engines based on Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.00133v2](https://arxiv.org/html/2510.00133v2)  
49. Enhancing Generalization of Spiking Neural Networks Through Temporal Regularization \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2506.19256v2](https://arxiv.org/html/2506.19256v2)  
50. \[2410.11488\] Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2410.11488](https://arxiv.org/abs/2410.11488)  
51. First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10577212/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10577212/)  
52. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2507.16043v2](https://arxiv.org/html/2507.16043v2)  
53. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2507.16043](https://arxiv.org/pdf/2507.16043)  
54. Delay learning based on temporal coding in Spiking Neural Networks \- PolyU Scholars Hub, 11月 1, 2025にアクセス、 [https://research.polyu.edu.hk/en/publications/delay-learning-based-on-temporal-coding-in-spiking-neural-network](https://research.polyu.edu.hk/en/publications/delay-learning-based-on-temporal-coding-in-spiking-neural-network)  
55. Time Is Of The Essence. Integrating Biologically-Inspired… | by Dean S Horak | Medium, 11月 1, 2025にアクセス、 [https://medium.com/@deanshorak/time-is-of-the-essense-27354ea835ba](https://medium.com/@deanshorak/time-is-of-the-essense-27354ea835ba)  
56. Neuronal Competition Groups with Supervised STDP for Spike-Based Classification \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2410.17066v1](https://arxiv.org/html/2410.17066v1)  
57. aidinattar/snn: Implementation of Spiking Neural Networks (SNNs) using SpykeTorch, featuring STDP and R-STDP training methods for efficient neural computation. \- GitHub, 11月 1, 2025にアクセス、 [https://github.com/aidinattar/snn](https://github.com/aidinattar/snn)  
58. Paired competing neurons improving STDP supervised local learning in spiking neural networks \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1401690/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1401690/full)  
59. Bi-sigmoid spike-timing dependent plasticity learning rule for magnetic tunnel junction-based SNN \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1387339/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1387339/full)  
60. FFGAF-SNN: The Forward-Forward Based Gradient Approximation Free Training Framework for Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2507.23643v1](https://arxiv.org/html/2507.23643v1)  
61. \[2307.04054\] Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2307.04054](https://arxiv.org/abs/2307.04054)  
62. Biologically Plausible Online Hebbian Meta‐Learning: Two‐Timescale Local Rules for Spiking Neural Brain Interfaces \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2509.14447v1](https://arxiv.org/html/2509.14447v1)  
63. EchoSpike Predictive Plasticity: An Online Local Learning Rule for Spiking Neural Networks, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2405.13976v1](https://arxiv.org/html/2405.13976v1)  
64. Brain-inspired learning in artificial neural networks: A review \- AIP Publishing, 11月 1, 2025にアクセス、 [https://pubs.aip.org/aip/aml/article/2/2/021501/3291446/Brain-inspired-learning-in-artificial-neural](https://pubs.aip.org/aip/aml/article/2/2/021501/3291446/Brain-inspired-learning-in-artificial-neural)  
65. \[2506.06904\] Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT for Neural Similarity? E-prop as an Example \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2506.06904](https://arxiv.org/abs/2506.06904)  
66. A Comprehensive Review of Spiking Neural Networks: Interpretation, Optimization, Efficiency, and Best Practices \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/369380134\_A\_Comprehensive\_Review\_of\_Spiking\_Neural\_Networks\_Interpretation\_Optimization\_Efficiency\_and\_Best\_Practices](https://www.researchgate.net/publication/369380134_A_Comprehensive_Review_of_Spiking_Neural_Networks_Interpretation_Optimization_Efficiency_and_Best_Practices)  
67. Rethinking skip connections in Spiking Neural Networks with Time-To-First-Spike coding \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1346805/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1346805/full)  
68. Enhanced representation learning with temporal coding in sparsely spiking neural networks \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10702559/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10702559/)  
69. Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure \- arXiv, 11月 1, 2025にアクセス、 [https://www.arxiv.org/pdf/2509.24411](https://www.arxiv.org/pdf/2509.24411)  
70. Sharing leaky-integrate-and-fire neurons for memory-efficient spiking neural networks \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1230002/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1230002/full)  
71. Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2408.07517v2](https://arxiv.org/html/2408.07517v2)  
72. Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2509.05356v2](https://arxiv.org/html/2509.05356v2)  
73. Application of Izhikevich-Based Spiking Neural Networks⋆ \- CEUR-WS.org, 11月 1, 2025にアクセス、 [https://ceur-ws.org/Vol-4048/paper16.pdf](https://ceur-ws.org/Vol-4048/paper16.pdf)  
74. Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2505.04034v1](https://arxiv.org/html/2505.04034v1)  
75. ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2505.11455v1](https://arxiv.org/html/2505.11455v1)  
76. DelRec: learning delays in recurrent spiking neural networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2509.24852v1](https://arxiv.org/html/2509.24852v1)  
77. Comparison of different transformer models on ImageNet-1K classification \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/figure/Comparison-of-different-transformer-models-on-ImageNet-1K-classification\_tbl1\_381784948](https://www.researchgate.net/figure/Comparison-of-different-transformer-models-on-ImageNet-1K-classification_tbl1_381784948)  
78. \[2409.02111\] Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2409.02111](https://arxiv.org/abs/2409.02111)  
79. Introducing Accurate Addition-Only Spiking Self-Attention for Transformer \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2503.00226v1](https://arxiv.org/html/2503.00226v1)  
80. Fourier or Wavelet bases as counterpart self-attention in spikformer for efficient visual classification \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1516868/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1516868/full)  
81. (PDF) Spikformer: When Spiking Neural Network Meets Transformer \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/364110937\_Spikformer\_When\_Spiking\_Neural\_Network\_Meets\_Transformer](https://www.researchgate.net/publication/364110937_Spikformer_When_Spiking_Neural_Network_Meets_Transformer)  
82. Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers \- CVF Open Access, 11月 1, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao\_Rethinking\_Spiking\_Self-Attention\_Mechanism\_Implementing\_a-XNOR\_Similarity\_Calculation\_in\_Spiking\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.pdf)  
83. SGSAFormer: Spike Gated Self-Attention Transformer and Temporal Attention \- MDPI, 11月 1, 2025にアクセス、 [https://www.mdpi.com/2079-9292/14/1/43](https://www.mdpi.com/2079-9292/14/1/43)  
84. Binary Event-Driven Spiking Transformer \- IJCAI, 11月 1, 2025にアクセス、 [https://www.ijcai.org/proceedings/2025/0458.pdf](https://www.ijcai.org/proceedings/2025/0458.pdf)  
85. Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformer \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2508.07710](https://arxiv.org/pdf/2508.07710)  
86. Attention mechanism and Transformer-based SNNs. This diagram illustrates how spiking versions of self-attention are integrated into SNN architectures, allowing for the efficient capture of global dependencies while preserving the energy efficiency of spike-based computation. \- OE Journals, 11月 1, 2025にアクセス、 [https://www.oejournal.org/ioe/supplement/c6cec146-3b5f-4224-9f03-35f66a235d31](https://www.oejournal.org/ioe/supplement/c6cec146-3b5f-4224-9f03-35f66a235d31)  
87. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2302.13939](https://arxiv.org/abs/2302.13939)  
88. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks, 11月 1, 2025にアクセス、 [https://openreview.net/forum?id=gcf1anBL9e](https://openreview.net/forum?id=gcf1anBL9e)  
89. Spiking Diffusion Models \- IEEE Computer Society, 11月 1, 2025にアクセス、 [https://www.computer.org/csdl/journal/ai/2025/01/10665907/1ZY7OwaZqAE](https://www.computer.org/csdl/journal/ai/2025/01/10665907/1ZY7OwaZqAE)  
90. Spiking Diffusion Models \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2408.16467](https://arxiv.org/html/2408.16467)  
91. AndyCao1125/SDM: \[IEEE Trans. AI 2024\] Spiking Diffusion Models \- GitHub, 11月 1, 2025にアクセス、 [https://github.com/andycao1125/sdm](https://github.com/andycao1125/sdm)  
92. What Is Neuromorphic Computing? \- IBM, 11月 1, 2025にアクセス、 [https://www.ibm.com/think/topics/neuromorphic-computing](https://www.ibm.com/think/topics/neuromorphic-computing)  
93. Digital neuromorphic technology: current and future prospects \- PMC, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10989295/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989295/)  
94. Spiking Neural Networks Hardware Implementations and Challenges: A Survey | Request PDF \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/332285563\_Spiking\_Neural\_Networks\_Hardware\_Implementations\_and\_Challenges\_A\_Survey](https://www.researchgate.net/publication/332285563_Spiking_Neural_Networks_Hardware_Implementations_and_Challenges_A_Survey)  
95. Spike-based dynamic computing with asynchronous sensing-computing neuromorphic chip \- PMC \- PubMed Central, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11127998/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11127998/)  
96. How Neuromorphic Computing Will Unlock Human-Centered Innovation \- Braden Kelley, 11月 1, 2025にアクセス、 [https://bradenkelley.com/2025/08/how-neuromorphic-computing-will-unlock-human-centered-innovation/](https://bradenkelley.com/2025/08/how-neuromorphic-computing-will-unlock-human-centered-innovation/)  
97. Neuromorphic Hardware Guide, 11月 1, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/hardware/](https://open-neuromorphic.org/neuromorphic-computing/hardware/)  
98. The Rise of Neuromorphic Computing: How Brain-Inspired AI is Shaping the Future in 2025, 11月 1, 2025にアクセス、 [https://www.ainewshub.org/post/the-rise-of-neuromorphic-computing-how-brain-inspired-ai-is-shaping-the-future-in-2025](https://www.ainewshub.org/post/the-rise-of-neuromorphic-computing-how-brain-inspired-ai-is-shaping-the-future-in-2025)  
99. Neuromorphic Computing 2025: Current SotA \- human / unsupervised, 11月 1, 2025にアクセス、 [https://humanunsupervised.com/papers/neuromorphic\_landscape.html](https://humanunsupervised.com/papers/neuromorphic_landscape.html)  
100. Neuromorphic Computing \- An Overview \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.06721v1](https://arxiv.org/html/2510.06721v1)  
101. Editorial: Hardware implementation of spike-based neuromorphic computing and its design methodologies \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9854259/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854259/)  
102. \[2408.14437\] Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2408.14437](https://arxiv.org/abs/2408.14437)  
103. Software-Hardware Co-Design of Neural Network Accelerators Using Emerging Technologies \- Curate ND, 11月 1, 2025にアクセス、 [https://curate.nd.edu/articles/dataset/Software-Hardware\_Co-Design\_of\_Neural\_Network\_Accelerators\_Using\_Emerging\_Technologies/25527742](https://curate.nd.edu/articles/dataset/Software-Hardware_Co-Design_of_Neural_Network_Accelerators_Using_Emerging_Technologies/25527742)  
104. (PDF) Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/383428834\_Sparsity-Aware\_Hardware-Software\_Co-Design\_of\_Spiking\_Neural\_Networks\_An\_Overview](https://www.researchgate.net/publication/383428834_Sparsity-Aware_Hardware-Software_Co-Design_of_Spiking_Neural_Networks_An_Overview)  
105. Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2408.14437v1](https://arxiv.org/html/2408.14437v1)  
106. ACE-SNN: Algorithm-Hardware Co-design of Energy-Efficient & Low-Latency Deep Spiking Neural Networks for 3D Image Recognition \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.815258/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.815258/full)  
107. A Survey on Neuromorphic Architectures for Running Artificial Intelligence Algorithms, 11月 1, 2025にアクセス、 [https://www.mdpi.com/2079-9292/13/15/2963](https://www.mdpi.com/2079-9292/13/15/2963)  
108. Hardware/software Co-design for Neuromorphic Systems \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/360705431\_Hardwaresoftware\_Co-design\_for\_Neuromorphic\_Systems](https://www.researchgate.net/publication/360705431_Hardwaresoftware_Co-design_for_Neuromorphic_Systems)  
109. The Role of Temporal Hierarchy in Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2407.18838v1](https://arxiv.org/html/2407.18838v1)