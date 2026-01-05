

# **ANNの先へ：SNNが次世代AIの覇権を握るための戦略的開発ロードマップ**

## **序章：パラダイムシフトの再定義 \- 模倣から超越へ**

本レポートは、スパイキングニューラルネットワーク（SNN）を単なる人工ニューラルネットワーク（ANN）の低電力な代替品としてではなく、計算の本質そのものを変革し、最終的にANNを超える潜在能力を持つ次世代AIパラダイムとして位置づける。その目的は、snn4プロジェクトが単にANNの性能に追いつく（性能パリティ）だけでなく、エネルギー効率、時間情報処理、継続学習、そして論理的推論といった、ANNが本質的に抱える限界をSNNがいかにして打ち破るか、そのための具体的な戦略とロードマップを提示することにある。

## **第1章 SNNの基本原理と現状の課題**

本章では、SNNの基本的な特性と、現在ANNに対して劣っているとされる点の技術的根本原因を分析する。これは、ANNを超えるための戦略を立てる上での必須の出発点となる。

### **1.1 SNNの二大優位性：エネルギー効率と時空間ダイナミクス**

SNNの計算単位は、生物学的なニューロンを模倣した離散的なバイナリの「スパイク」である 1。このイベント駆動型の性質が、SNNの卓越したエネルギー効率の源泉となっている。ニューロンはスパイクを発火させた時にのみ電力を消費するため、計算は本質的にスパース（疎）になる 3。これは、常に密な行列演算を行うANNとは対照的であり、特にニューロモルフィックハードウェア上では最大で2桁のエネルギー削減を可能にする 1。

さらに、SNNは内部状態（膜電位）を通じて時間を自然に処理するため、ビデオ、音声、イベントベースのセンサーデータといった時空間データの扱いに本質的に適している 1。

**表1：SNNとANNのパラダイム比較分析**

| 特性 | 人工ニューラルネットワーク (ANN) | スパイキングニューラルネットワーク (SNN) |
| :---- | :---- | :---- |
| **計算単位** | 連続値の活性化（例：シグモイド、ReLU） | 離散的なバイナリイベント（スパイク） |
| **情報伝達媒体** | 活性化の大きさ | スパイクのタイミング、レート、順序 |
| **データ処理** | 同期的、密なバッチ処理 | 非同期的、イベント駆動、スパース処理 |
| **時間ダイナミクス** | 主に静的。時間処理にはRNNなどの特殊な構造が必要。 | 内部状態（膜電位）を通じて本質的に時間を処理。 |
| **エネルギー消費** | 高い。密な行列演算による。 | 低い。スパースなイベント駆動計算による。 |
| **ハードウェア** | GPU/TPU（密行列演算に最適化） | ニューロモルフィックチップ（スパース、非同期処理に最適化） |
| **主要な学習機構** | 誤差逆伝播法（勾配ベース） | 代理勾配法、ANN-SNN変換、STDPなど多様。 |

### **1.2 SNNの三重苦：学習、性能、エコシステム**

SNNの普及を妨げている主要な障害は、以下の三点に集約される。

* **学習の難問**: スパイク生成が微分不可能な関数であるため、ANNで標準的な誤差逆伝播法を直接適用できない 1。  
* **性能のボトルネック**: 歴史的に、SNNは静的なベンチマークにおいてANNに精度で劣ってきた 2。また、精度を出すために長いシミュレーション時間（多数のタイムステップ）を要し、これが大きな推論遅延（レイテンシ）を引き起こすことがある 11。  
* **エコシステムの未成熟**: フレームワーク、ツール、そしてSNNの真価を引き出すニューロモルフィックハードウェアは、まだ発展途上である 3。

これらの課題は、「精度」「遅延」「生物学的妥当性」というトレードオフの三角形を形成しており、一つの利点を追求すると他の利点が犠牲になる関係にある。

## **第2章 ANN性能へのキャッチアップ戦略：既存技術の習熟と深化**

ANNを超えるためには、まずその性能に匹敵する能力を獲得する必要がある。本章では、そのための主要な学習アルゴリズムとアーキテクチャの進化について分析する。

### **2.1 学習アルゴリズムの三つの道筋**

SNNの学習問題（非微分可能性）を克服するため、主に三つのアプローチが開発されてきた。

* **代理勾配法 (SG)**: 逆伝播時に微分不可能なスパイク関数を、微分可能な「代理」関数に置き換えることで、勾配ベースの学習を可能にする直接学習法 1。低い推論遅延を達成できる可能性があるが、精度や学習コストに課題がある 7。  
* **ANN-SNN変換**: まず高性能なANNを学習させ、その重みや構造をSNNに変換する間接的な手法 8。高い精度を達成しやすいが、一般的に高い推論遅延を伴う 11。  
* **スパイクタイミング依存可塑性 (STDP)**: 生物学的な脳の学習に着想を得た、局所的な教師なし学習則 12。単独での性能は限定的だが、代理勾配法と組み合わせたハイブリッド学習（STDPで事前学習し、SGでファインチューニング）により、学習効率と堅牢性を大幅に向上させることができる 25。

**表2：SNN学習アルゴリズムの意思決定マトリクス**

| 評価指標 | 代理勾配法 (SG) | ANN-SNN変換 | STDPハイブリッド |
| :---- | :---- | :---- | :---- |
| **精度ポテンシャル** | 中～高 | 高～非常に高い | 中～高 |
|  | (近似勾配による限界) | (ソースANNの精度に依存) | (事前学習による改善) |
| **推論遅延** | 低 | 高 | 低～中 |
|  | (少ないタイムステップで動作) | (レートコーディングに長い時間が必要 14) | (SGファインチューニングに依存) |
| **学習コスト** | 高 | 中 | 中～高 |
|  | (BPTTによるメモリ/時間消費 7) | (ANN学習 \+ 変換コスト) | (事前学習 \+ ファインチューニング) |
| **実装の複雑さ** | 高 | 中 | 高 |
|  | (勾配の選択、安定化) | (誤差補正、正規化 13) | (2段階の学習パイプライン) |
| **時間的タスクへの適合性** | 高 | 低 | 高 |
|  | (時間ダイナミクスを直接学習) | (主に静的情報をマッピング) | (時間的相関を事前学習) |
| **ハードウェア親和性** | 中 | 低 | 高 |
|  | (BPTTはオンチップ学習に不向き) | (高いスパイクレートは非効率) | (STDPはオンチップ学習可能 8) |

### **2.2 アーキテクチャの進化：深層化と高性能化**

ANNの成功事例をSNNに適応させることで、性能は飛躍的に向上した。

* **スパイキングResNet**: ANNにおける残差接続（スキップ接続）をSNNに導入することで、勾配消失問題を緩和し、非常に深いSNNの学習を可能にした 11。  
* **スパイキングトランスフォーマー**: 自己注意（self-attention）機構をスパイクベースで再設計し、乗算を伴わないスパースな演算に置き換えることで、トランスフォーマーの高性能とSNNのエネルギー効率の両立を目指している 28。最新の研究では、空間情報と時間情報の両方を統合する時空間的自己注意（STAtten） 30 や、学習と推論を高速化するトークンスパース化 33 など、急速な進化を遂げている。

## **第3章 ANNを超えるための独自進化戦略**

ANNの性能に追いついた先で、SNNは独自の強みを活かしてANNを超える新しい価値を創造する。本章では、そのための三つの戦略的フロンティアを探る。

### **3.1 フロンティア1：イベント駆動型センサーとの完全融合**

SNNの真価は、静的な画像データセットではなく、データ自体がスパースで時間的な性質を持つ領域で発揮される。イベントベースビジョンセンサー（DVSなど）は、輝度変化があったピクセルのみが非同期にイベント（スパイク）を出力する 43。これはSNNの計算原理と完全に一致しており、両者を組み合わせることで、従来のフレームベースのカメラとANNの組み合わせでは達成不可能な、超低遅延・超低消費電力のリアルタイム認識システムが実現する 43。

* **応用分野**: この技術は、高速で移動する物体の追跡や、電力に厳しい制約のある自律型ドローン 45、次世代のロボット工学 47、高度運転支援システム（ADAS）52 など、ANNが苦手とするリアルタイム性が要求される動的な環境で圧倒的な優位性を示す。

### **3.2 フロンティア2：オンライン・継続学習と「破局的忘却」の克服**

ANNの重大な欠点の一つに「破局的忘却」がある。これは、新しいタスクを学習すると、過去に学習した内容を忘れてしまう現象である 53。SNN、特にSTDPのような生物学的に妥当な局所学習則は、この問題に対する有望な解決策を提示する。

* **オンチップ学習**: STDPは、大域的な誤差信号を必要としない局所的な学習ルールであるため、ニューロモルフィックチップ上でのオンライン学習（オンチップ学習）と非常に相性が良い 56。これにより、デバイスが実環境で動作しながら継続的に学習し、変化に適応していくことが可能になる。  
* **継続学習への応用**: カラム状の構造を持つSNNアーキテクチャと局所学習則を組み合わせることで、新しいタスクを学習する際に既存の知識への干渉を最小限に抑え、破局的忘却を大幅に緩和できることが示唆されている 58。これは、生涯にわたって学習し続ける真の自律システムの実現に向けた重要な一歩である。

### **3.3 フロンティア3：高信頼性論理とニューロシンボリックAI**

ANNはしばしば「ブラックボックス」と批判され、その意思決定プロセスを解釈することが難しい。SNNは、より決定論的で解釈可能な論理演算を構築するための道筋を提供する。

* **基礎論理の実装 (LogicSNN)**: 論理変数を2つのニューロン（一方が論理「0」、他方が論理「1」を表現）で符号化し、STDPを用いてAND、OR、XORといった基本的な論理ゲートを確実に学習させるLogicSNNパラダイムが提案されている 34。これにより、検証可能で信頼性の高い計算モジュールを構築できる。  
* **高次推論への道 (ニューロシンボリックAI)**: SNNの効率的なパターン認識能力と、記号AIの形式的推論能力を組み合わせるニューロシンボリックAI（NeSy）は、次世代AIの有望な方向性である 60。SNNを、イベント駆動データから記号的な知識を抽出する効率的な「システム1（直感的思考）」として用い、その出力を記号的推論エンジンである「システム2（論理的思考）」に供給することで、ANNの頑健性の欠如と記号AIの現実世界への接地能力の欠如という、双方の弱点を補い合うハイブリッドシステムを構築できる 62。

## **第4章 snn4プロジェクトのための戦略的ロードマップ**

以上の分析に基づき、snn4プロジェクトがANNを超えるための3段階の戦略的ロードマップを提案する。

### **フェーズ1：ANN性能パリティの達成（短期目標）**

* **目的**: 既存の静的ベンチマーク（画像分類など）において、最先端のANNと同等の精度を達成し、SNN技術の基礎体力を証明する。  
* **主要技術**:  
  * **学習**: 高精度を目指すなら**ANN-SNN変換**（特にRate Norm Layer 14 などの高度な手法）から着手。低遅延が求められる場合は、  
    **代理勾配法**（特にLSG 40 やSML 41 などの最新手法）を採用。  
  * **アーキテクチャ**: **スパイキングResNet** 11 を導入し、深層化による性能向上を図る。  
  * **ハイブリッド戦略**: **STDPによる事前学習** 25 を導入し、代理勾配法での学習を高速化・安定化させる。  
* **達成指標**: CIFAR-10/100やImageNetといった標準データセットで、同規模のANNモデルに匹敵する精度を、許容可能な遅延の範囲で達成する。

### **フェーズ2：SNN独自領域での優位性確立（中期目標）**

* **目的**: SNNが本質的に優位な領域に焦点を移し、ANNでは達成不可能な性能（特にエネルギー効率とリアルタイム性）を実証する。  
* **主要技術**:  
  * **アプリケーション**: **イベントベースビジョンセンサー（DVS）** 43 を用いたタスク（例：高速物体追跡、ドローン制御 45）に主軸を移す。  
  * **アーキテクチャ**: **スパイキングトランスフォーマー** 28 を導入し、複雑な時空間パターンの認識能力を獲得する。特に、時空間的自己注意（STAtten）31 の実装を検討する。  
  * **学習**: **STDPハイブリッド学習**を本格的に活用し、オンラインでの環境適応能力の基礎を構築する。  
* **達成指標**: イベントベースのデータセット（例：DVS-CIFAR10, N-Caltech101）において、ANNベースの手法と比較して、同等以上の精度を**1桁以上低いエネルギー消費と遅延**で達成する。

### **フェーズ3：次世代AIパラダイムへの飛躍（長期ビジョン）**

* **目的**: SNNの究極的なポテンシャルを解放し、自己進化し、人間と協調できる、信頼性の高いAIシステムを構築する。  
* **主要技術**:  
  * **ハードウェア**: **ニューロモルフィックハードウェア**への実装を本格的に検討し、ソフトウェアとハードウェアの協調設計を進める。  
  * **学習**: **継続学習**能力を実装し、**破局的忘却**を克服したモデルを実証する 58。  
  * **論理と推論**: **LogicSNN** 34 で構築した論理モジュールを基盤とし、外部の記号的推論エンジンと統合する  
    **ニューロシンボリックアーキテクチャ** 62 を探求する。  
* **達成指標**: 複数のタスクを逐次的に学習させても過去のタスクの性能がほとんど低下しないことを実証する。特定の推論タスクにおいて、その結論に至るまでの論理的な過程を説明できるシステムを構築する。

このロードマップを実行することで、snn4プロジェクトは単にANNの模倣に留まらず、その限界を超え、真に新しい価値を創造する次世代AIの開発をリードすることが可能となる。

#### **引用文献**

1. Advances in Artificial Neural Networks: Exploring Spiking Neural Models \- IEEE Computer Society, 10月 2, 2025にアクセス、 [https://www.computer.org/publications/tech-news/trends/spiking-neural-models](https://www.computer.org/publications/tech-news/trends/spiking-neural-models)  
2. Spiking neural network \- Wikipedia, 10月 2, 2025にアクセス、 [https://en.wikipedia.org/wiki/Spiking\_neural\_network](https://en.wikipedia.org/wiki/Spiking_neural_network)  
3. Spiking Neural Networks: The next “Big Thing” in AI? | by Dean S Horak | Medium, 10月 2, 2025にアクセス、 [https://medium.com/@deanshorak/spiking-neural-networks-the-next-big-thing-in-ai-efe3310709b0](https://medium.com/@deanshorak/spiking-neural-networks-the-next-big-thing-in-ai-efe3310709b0)  
4. Differentiable Spike: Rethinking Gradient-Descent for Training Spiking Neural Networks, 10月 2, 2025にアクセス、 [https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf)  
5. Spiking Neural Networks — SNN. Representing the human brain more… | by Goksselgunduz | Medium, 10月 2, 2025にアクセス、 [https://medium.com/@goksselgunduz/spiking-neural-networks-snn-40ef3fd369b4](https://medium.com/@goksselgunduz/spiking-neural-networks-snn-40ef3fd369b4)  
6. Surrogate Gradient Learning in Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/pdf/1901.09948](https://arxiv.org/pdf/1901.09948)  
7. 爆速・低消費電力で深層学習もグラフ探索も条件付き最適化も行える？！脳型計算機の定量的実力 | AI-SCHOLAR, 10月 2, 2025にアクセス、 [https://ai-scholar.tech/articles/survey/neuromorphics\_loihi](https://ai-scholar.tech/articles/survey/neuromorphics_loihi)  
8. Spiking Neural Network チュートリアル \- Speaker Deck, 10月 2, 2025にアクセス、 [https://speakerdeck.com/spatial\_ai\_network/spiking-neural-network-tutorial](https://speakerdeck.com/spatial_ai_network/spiking-neural-network-tutorial)  
9. Tutorial 6 \- Surrogate Gradient Descent in a Convolutional SNN ..., 10月 2, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_6.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)  
10. Training SNNs With Surrogate Gradients | INCF TrainingSpace, 10月 2, 2025にアクセス、 [https://training.incf.org/lesson/training-snns-surrogate-gradients](https://training.incf.org/lesson/training-snns-surrogate-gradients)  
11. Learnable Surrogate Gradient for Direct Training Spiking Neural Networks \- IJCAI, 10月 2, 2025にアクセス、 [https://www.ijcai.org/proceedings/2023/0335.pdf](https://www.ijcai.org/proceedings/2023/0335.pdf)  
12. ベイジアンフュージョンによる スパイキングニューラルネットワークの低エネルギ推論 \- 情報学広場, 10月 2, 2025にアクセス、 [https://ipsj.ixsq.nii.ac.jp/record/212619/files/IPSJ-DAS2021004.pdf](https://ipsj.ixsq.nii.ac.jp/record/212619/files/IPSJ-DAS2021004.pdf)  
13. Lec 8 Converting ANNs to SNNs with BrainCog, 10月 2, 2025にアクセス、 [https://www.brain-cog.network/docs/tutorial/8\_conversion.html](https://www.brain-cog.network/docs/tutorial/8_conversion.html)  
14. Optimal ANN-SNN Conversion for Fast and Accurate ... \- IJCAI, 10月 2, 2025にアクセス、 [https://www.ijcai.org/proceedings/2021/0321.pdf](https://www.ijcai.org/proceedings/2021/0321.pdf)  
15. \[2307.04054\] Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2307.04054](https://arxiv.org/abs/2307.04054)  
16. ゼロから作る Spiking Neural Networks, 10月 2, 2025にアクセス、 [https://compneuro-julia.github.io/\_static/pdf/SNN\_from\_scratch\_with\_python\_ver2\_1.pdf](https://compneuro-julia.github.io/_static/pdf/SNN_from_scratch_with_python_ver2_1.pdf)  
17. Tag: surrogate gradients \- Zenke Lab, 10月 2, 2025にアクセス、 [https://zenkelab.org/tag/surrogate-gradients/](https://zenkelab.org/tag/surrogate-gradients/)  
18. \[1901.09948\] Surrogate Gradient Learning in Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/1901.09948](https://arxiv.org/abs/1901.09948)  
19. Differential Coding for Training-Free ANN-to-SNN Conversion \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/pdf/2503.00301](https://arxiv.org/pdf/2503.00301)  
20. Differential Coding for Training-Free ANN-to-SNN Conversion \- OpenReview, 10月 2, 2025にアクセス、 [https://openreview.net/pdf?id=OxBWTFSGcv](https://openreview.net/pdf?id=OxBWTFSGcv)  
21. Differential Coding for Training-Free ANN-to-SNN Conversion \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2503.00301v1](https://arxiv.org/html/2503.00301v1)  
22. STDP Learning for LIF Neurons, 10月 2, 2025にアクセス、 [https://web-ext.u-aizu.ac.jp/misc/neuro-eng/book/NeuromorphicComputing/tutorials/stdp.html](https://web-ext.u-aizu.ac.jp/misc/neuro-eng/book/NeuromorphicComputing/tutorials/stdp.html)  
23. Spike-timing Dependent Plasticity (STDP) — Lava documentation, 10月 2, 2025にアクセス、 [https://lava-nc.org/lava/notebooks/in\_depth/tutorial08\_stdp.html](https://lava-nc.org/lava/notebooks/in_depth/tutorial08_stdp.html)  
24. Spike Timing-Dependent Plasticity (STDP) | INCF TrainingSpace, 10月 2, 2025にアクセス、 [https://training.incf.org/lesson/spike-timing-dependent-plasticity-stdp](https://training.incf.org/lesson/spike-timing-dependent-plasticity-stdp)  
25. Training Deep Spiking Convolutional Neural Networks ... \- Frontiers, 10月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00435/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00435/full)  
26. ResNet: Revolutionizing Deep Learning in Image Recognition \- Viso Suite, 10月 2, 2025にアクセス、 [https://viso.ai/deep-learning/resnet-residual-neural-network/](https://viso.ai/deep-learning/resnet-residual-neural-network/)  
27. Residual neural network \- Wikipedia, 10月 2, 2025にアクセス、 [https://en.wikipedia.org/wiki/Residual\_neural\_network](https://en.wikipedia.org/wiki/Residual_neural_network)  
28. Spike-driven Transformer, 10月 2, 2025にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)  
29. An Application-Driven Survey on Event-Based Neuromorphic Computer Vision \- MDPI, 10月 2, 2025にアクセス、 [https://www.mdpi.com/2078-2489/15/8/472](https://www.mdpi.com/2078-2489/15/8/472)  
30. Spiking Transformer with Spatial-Temporal ... \- CVF Open Access, 10月 2, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2025/papers/Lee\_Spiking\_Transformer\_with\_Spatial-Temporal\_Attention\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.pdf)  
31. The network structure of SNN logical operation module. \- ResearchGate, 10月 2, 2025にアクセス、 [https://www.researchgate.net/figure/The-network-structure-of-SNN-logical-operation-module\_fig2\_354260353](https://www.researchgate.net/figure/The-network-structure-of-SNN-logical-operation-module_fig2_354260353)  
32. Real-Time Neuromorphic Navigation: Integrating Event-Based Vision and Physics-Driven Planning on a Parrot Bebop2 Quadrotor \- ResearchGate, 10月 2, 2025にアクセス、 [https://www.researchgate.net/publication/381882855\_Real-Time\_Neuromorphic\_Navigation\_Integrating\_Event-Based\_Vision\_and\_Physics-Driven\_Planning\_on\_a\_Parrot\_Bebop2\_Quadrotor](https://www.researchgate.net/publication/381882855_Real-Time_Neuromorphic_Navigation_Integrating_Event-Based_Vision_and_Physics-Driven_Planning_on_a_Parrot_Bebop2_Quadrotor)  
33. \[2405.19687\] Autonomous Driving with Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2405.19687](https://arxiv.org/abs/2405.19687)  
34. LogicSNN: A Unified Spiking Neural Networks Logical Operation ..., 10月 2, 2025にアクセス、 [https://www.mdpi.com/2079-9292/10/17/2123](https://www.mdpi.com/2079-9292/10/17/2123)  
35. LogicSNN: A Unified Spiking Neural Networks Logical Operation Paradigm \- University of California Los Angeles, 10月 2, 2025にアクセス、 [https://search.library.ucla.edu/discovery/fulldisplay/cdi\_proquest\_journals\_2570775457/01UCS\_LAL:UCLA](https://search.library.ucla.edu/discovery/fulldisplay/cdi_proquest_journals_2570775457/01UCS_LAL:UCLA)  
36. Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2505.04034v1](https://arxiv.org/html/2505.04034v1)  
37. \[2503.00226\] Spiking Transformer:Introducing Accurate Addition-Only Spiking Self-Attention for Transformer \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2503.00226](https://arxiv.org/abs/2503.00226)  
38. Spiking Neural Networks and Their Applications: A Review \- PMC \- PubMed Central, 10月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/)  
39. \[2502.11269\] Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures: Benefits and Limitations \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2502.11269](https://arxiv.org/abs/2502.11269)  
40. A Novel Digital Realization of AdEx Neuron Model | Request PDF \- ResearchGate, 10月 2, 2025にアクセス、 [https://www.researchgate.net/publication/335499531\_A\_Novel\_Digital\_Realization\_of\_AdEx\_Neuron\_Model](https://www.researchgate.net/publication/335499531_A_Novel_Digital_Realization_of_AdEx_Neuron_Model)  
41. What else does attention need: Neurosymbolic approaches to general logical reasoning in LLMs? \- OpenReview, 10月 2, 2025にアクセス、 [https://openreview.net/pdf/b9cf1005d2770c37ff97daf5154df9dd4deb833c.pdf](https://openreview.net/pdf/b9cf1005d2770c37ff97daf5154df9dd4deb833c.pdf)  
42. Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2503.19643v1](https://arxiv.org/html/2503.19643v1)  
43. Event-based Cameras and Spiking Neural Networks (SNNs) for depth and optic flow processing under degraded visual conditions | IPAL, 10月 2, 2025にアクセス、 [https://ipal.cnrs.fr/event-based-cameras-and-spiking-neural-networks-snns-for-depth-and-optic-flow-processing-under-degraded-visual-conditions/](https://ipal.cnrs.fr/event-based-cameras-and-spiking-neural-networks-snns-for-depth-and-optic-flow-processing-under-degraded-visual-conditions/)  
44. Event-Based Trajectory Prediction Using Spiking Neural Networks \- PMC, 10月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC8180888/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8180888/)  
45. Real-Time Neuromorphic Navigation: Integrating Event-Based Vision and Physics-Driven Planning on a Parrot Bebop2 Quadrotor \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2407.00931v1](https://arxiv.org/html/2407.00931v1)  
46. Event-driven Vision and Control for UAVs on a Neuromorphic Chip \- Yulia Sandamirskaya, 10月 2, 2025にアクセス、 [https://sandamirskaya.eu/resources/SNN\_Control\_RAL.pdf](https://sandamirskaya.eu/resources/SNN_Control_RAL.pdf)  
47. Real-Time Neuromorphic Navigation: Guiding Physical Robots with Event-Based Sensing and Task-Specific Reconfigurable Autonomy Stack \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2503.09636v1](https://arxiv.org/html/2503.09636v1)  
48. Spiking neural network-based multi-task autonomous learning for mobile robots | Request PDF \- ResearchGate, 10月 2, 2025にアクセス、 [https://www.researchgate.net/publication/353272062\_Spiking\_neural\_network-based\_multi-task\_autonomous\_learning\_for\_mobile\_robots](https://www.researchgate.net/publication/353272062_Spiking_neural_network-based_multi-task_autonomous_learning_for_mobile_robots)  
49. Fully Spiking Neural Network for Legged Robots \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2310.05022v2](https://arxiv.org/html/2310.05022v2)  
50. A Survey of Robotics Control Based on Learning-Inspired Spiking Neural Networks, 10月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00035/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00035/full)  
51. Bio-Inspired Autonomous Learning Algorithm With Application to Mobile Robot Obstacle Avoidance \- Frontiers, 10月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.905596/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.905596/full)  
52. Autonomous Driving using Spiking Neural Networks on Dynamic Vision Sensor Data \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/pdf/2311.09225](https://arxiv.org/pdf/2311.09225)  
53. Slowing Down Forgetting in Continual Learning \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2411.06916v2](https://arxiv.org/html/2411.06916v2)  
54. Continual Learning and Catastrophic Forgetting \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2403.05175v1](https://arxiv.org/html/2403.05175v1)  
55. \[2403.05175\] Continual Learning and Catastrophic Forgetting \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2403.05175](https://arxiv.org/abs/2403.05175)  
56. Paired Competing Neurons Improving STDP Supervised Local Learning In Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2308.02194v2](https://arxiv.org/html/2308.02194v2)  
57. Event-Based, Timescale Invariant Unsupervised Online Deep Learning With STDP, 10月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2018.00046/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2018.00046/full)  
58. Continual Learning with Columnar Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2506.17169](https://arxiv.org/html/2506.17169)  
59. \[2506.17169\] Continual Learning with Columnar Spiking Neural Networks \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2506.17169](https://arxiv.org/abs/2506.17169)  
60. \[2501.05435\] Neuro-Symbolic AI in 2024: A Systematic Review \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/abs/2501.05435](https://arxiv.org/abs/2501.05435)  
61. Neuro-Symbolic AI in 2024: A Systematic Review \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2501.05435v1](https://arxiv.org/html/2501.05435v1)  
62. Towards Cognitive AI Systems: a Survey and Prospective on ... \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/pdf/2401.01040](https://arxiv.org/pdf/2401.01040)  
63. STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking \- arXiv, 10月 2, 2025にアクセス、 [https://arxiv.org/html/2505.11151v1](https://arxiv.org/html/2505.11151v1)