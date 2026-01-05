

# **SNS5: 超低消費電力と適応的知能の実現に向けた次世代AIアーキテクチャに関する包括的研究**

## **1\. 序論：ANNパラダイムの限界とポストANNへの転換**

現在、人工知能（AI）研究は、トランスフォーマー（Transformer）アーキテクチャと大規模言語モデル（LLM）の成功により、かつてない隆盛を極めている。しかし、この成功の裏には、計算資源と消費電力の指数関数的な増大という、持続可能性を脅かす深刻な課題が潜んでいる。GitHubプロジェクト「SNS5」が掲げる「ANN（人工ニューラルネットワーク）系のAIを超える省エネと知能」という目標は、現在のディープラーニングが直面している物理的・理論的限界に対する鋭い洞察に基づいている。本報告書は、提供されたロードマップと研究資料に基づき、SNS5が目指すべき技術的道筋を、**超低ビット量子化（BitNet）**、**線形注意機構（RWKV）**、**ニューロモーフィック・スパイキングニューラルネットワーク（SNN）**、**リキッドニューラルネットワーク（LNN）**、**能動的推論（Active Inference）**、そして**修正可能なニューロシンボリック・アーキテクチャ**という多角的な視点から包括的に調査・分析したものである。

従来のANN、特に現在主流のトランスフォーマーモデルは、浮動小数点演算（FP16やFP32）に依存した密な行列乗算を基盤としており、その計算量はシーケンス長の二乗$O(T^2)$で増大する。これは「メモリの壁（Memory Wall）」問題を悪化させ、推論時のエネルギー効率を著しく低下させる要因となっている1。さらに、静的な重みパラメータに知識を埋め込む学習方式は、新しい情報の追加や誤情報の修正において「破滅的忘却（Catastrophic Forgetting）」を引き起こしやすく、継続的な学習と適応を困難にしている3。

本研究では、これらの課題を克服するために、以下の3つのフロンティアを統合するアプローチを提案する。

1. **エネルギー効率のフロンティア**: 1.58ビット量子化技術とニューロモーフィック計算原理の導入により、計算コストを数桁オーダーで削減する。  
2. **知能のフロンティア**: 静的なパターンマッチングを超え、自由エネルギー原理に基づく能動的推論エージェントとして、環境との相互作用を通じて学習するシステムを構築する。  
3. **修正可能性のフロンティア**: 知識グラフとニューロシンボリックAIを組み合わせることで、パラメータの再学習なしに知識の更新・修正が可能な外部記憶構造を確立する。

## **2\. エネルギー効率のフロンティア：1.58ビットコンピューティングとポスト・トランスフォーマー**

SNS5が目指す「省エネ」を実現するためには、現代AIの最大のボトルネックであるメモリアクセスと浮動小数点演算を根本から見直す必要がある。調査結果は、極端な量子化技術と、トランスフォーマーの二次関数的な計算量を線形化する新しいアーキテクチャの融合が、その解であることを示唆している。

### **2.1 BitNet b1.58：1ビットLLMの衝撃と実装戦略**

2024年から2025年にかけての最も重要な技術的ブレイクスルーの一つが、**BitNet b1.58**の登場である。これは、LLMのすべての重みパラメータを${-1, 0, 1}$の3値（ternary）のみで表現するという急進的なアプローチでありながら、FP16（16ビット浮動小数点）で訓練されたモデルと同等の予測性能（Perplexity）を維持することが実証されている5。

#### **2.1.1 計算メカニズムと省エネルギー原理**

BitNet b1.58の核心は、従来のnn.Linear層を置き換える**BitLinear**層にある。通常のニューラルネットワークでは、入力ベクトルと重み行列の間で高価な積和演算（Multiply-Accumulate Operations, MACs）が行われる。しかし、重みが${-1, 0, 1}$に限定されるBitNetでは、この乗算プロセスが単純な\*\*加算（Addition）と減算（Subtraction）\*\*に置き換わる7。

| 特性 | 従来のFP16 LLM | BitNet b1.58 | 改善要因 |
| :---- | :---- | :---- | :---- |
| **重み表現** | 16ビット浮動小数点 | 1.58ビット（3値: \-1, 0, 1） | メモリ帯域幅の劇的な削減 |
| **主要演算** | 浮動小数点 行列乗算 | 整数 加算・減算 | 演算器（ALU）の簡素化と低電力化 |
| **メモリ消費** | 大（モデルサイズに比例） | 極小（FP16比で約1/10） | 重み圧縮によるDRAMアクセス削減 |
| **推論速度** | メモリ帯域律速 | 高スループット | メモリ転送時間の短縮 |

特筆すべきは、単なる2値（-1, 1）ではなく「0」を含めた3値である点である。この「0」の存在により、モデルは入力特徴量の中で不要な情報を明示的にフィルタリング（遮断）することが可能となり、これがスパース性（疎性）を生み出し、高いモデル性能を維持する鍵となっている9。この特性は、生物学的ニューロンにおける「発火しない」状態を模倣しているとも解釈でき、SNS5が目指す生物的な知能の実現と整合する。

#### **2.1.2 ハードウェア実装とエネルギー削減効果**

BitNet b1.58のエネルギー効率への貢献は理論上だけのものではない。bitnet.cppなどの推論フレームワークを用いた実験では、ARM CPU上で1.37倍から5.07倍の高速化、x86 CPU上で2.37倍から6.17倍の高速化が確認されており、エネルギー消費量は55.4%から82.2%削減されることが報告されている10。  
さらに重要な洞察として、1.58ビットモデルは1000億パラメータ（100B）クラスの巨大モデルであっても、単一のCPUで実用的な速度（毎秒5-7トークン）で動作可能であることが示されている。これは、高価で電力消費の激しいGPUクラスタを必要とせず、エッジデバイスやローカル環境での高度な知能実現を可能にする技術であり、SNS5の省エネ目標達成において中核的な役割を果たすべきである10。

### **2.2 RWKVアーキテクチャ：線形注意機構による効率化**

BitNetが「重みの精度」に関する革命であるならば、\*\*RWKV（Receptance Weighted Key Value）\*\*は「計算の複雑度」に関する革命である。トランスフォーマーの自己注意機構（Self-Attention）は、文脈長が長くなるにつれて計算量が二次関数的に増大するため、長い文脈を扱う際のエネルギー効率が悪い。

#### **2.2.1 RNNとトランスフォーマーの融合**

RWKVは、トランスフォーマーのような並列学習が可能でありながら、推論時にはRNN（リカレントニューラルネットワーク）として動作するハイブリッドアーキテクチャである。これにより、推論時のメモリ使用量は文脈長に関わらず一定（$O(1)$）となり、計算量は線形（$O(T)$）に抑えられる12。

このアーキテクチャは「チャネル指向の注意機構」とも呼べる線形注意機構を採用しており、従来のドット積注意機構（Dot-Product Attention）を置き換えることで、メモリボトルネックを解消している。具体的には、RWKVは過去の情報を「Receptance（受容）」「Weight（減衰）」「Key」「Value」というベクトルを用いて圧縮・保持し、時間ステップごとに状態を更新する。これにより、KVキャッシュが肥大化するトランスフォーマーとは異なり、極めて軽量な推論が可能となる14。

#### **2.2.2 BitRWKV：究極の効率化に向けた統合**

SNS5への提言として、BitNetの量子化技術とRWKVのアーキテクチャを統合した\*\*BitRWKV（1.58-bit RWKV）\*\*の実装を推奨する。この統合により、以下の相乗効果が期待される。

1. **メモリ転送の最小化**: 1.58ビット重みにより、RNN状態更新に必要な重み読み出し時間が最小化される。  
2. **演算の簡素化**: RWKVの行列ベクトル積が、BitLinearによる加算処理に置き換わり、推論時の消費電力が極限まで低下する。  
3. **スケーラビリティ**: 70Bクラスのモデルであっても、コンシューマー向けハードウェアでの動作が視野に入る。

実際に、Spiking RWKVのような派生研究において、RWKV構造がスパイクベースの処理と親和性が高いことが示されており、SNNへの移行の前段階、あるいは実用的な代替案として極めて有望である16。

### **2.3 リキッドニューラルネットワーク（LNN）と適応性**

「知能」の観点からSNS5が注目すべきもう一つのアーキテクチャが、\*\*リキッドニューラルネットワーク（LNN）\*\*である。LNNは、線虫の神経系に触発されたモデルであり、ニューロンの動作を微分方程式で記述する。

#### **2.3.1 時間連続的な適応能力**

LNNの最大の特徴は、学習後も推論時に環境に応じてパラメータ（液状の時定数など）が動的に変化する点にある。これにより、静的なANNでは対応が難しい、未知の環境変化や時系列データの揺らぎに対して、高い適応能力と堅牢性を示す18。  
MITの研究チームによって開発された「Liquid Foundation Models (LFMs)」は、トランスフォーマーと比較して大幅に少ないパラメータ数で同等の性能を発揮し、特にドローンやロボティクスなどのエッジAI領域での有効性が実証されている。30億パラメータのモデルがラップトップ上で動作し、トランスフォーマーよりも低いメモリフットプリントで推論可能であるという事実は、SNS5の省エネ目標と合致する19。  
LNNは「因果関係の学習」や「解釈可能性」においても優位性を持つ。ニューロンの活性化が微分方程式として明示的に記述されているため、出力に至るプロセスを遡って解析することが従来のブラックボックス型モデルよりも容易である19。これは、後述する「修正可能性」の要件に対してもプラスに作用する。

### **2.4 Kolmogorov-Arnold Networks (KAN)**

さらなるアーキテクチャの選択肢として、\*\*Kolmogorov-Arnold Networks (KAN)\*\*の導入も検討に値する。従来のMLP（多層パーセプトロン）がノード（ニューロン）に固定の活性化関数を持つのに対し、KANはエッジ（重み）自体に学習可能な活性化関数（Bスプライン曲線など）を持たせる20。

#### **2.4.1 少ないパラメータでの高精度表現**

KANは「Kolmogorov-Arnold表現定理」に基づいており、複雑な多変数関数をより少ないパラメータ数で近似できることが示されている。研究によれば、特定の科学的・工学的タスクにおいて、KANはMLPよりも高い精度と効率性を発揮し、さらに解釈可能性（Explainability）が高い。具体的には、学習された活性化関数を視覚化することで、モデルがどのような法則性を獲得したかを人間が直感的に理解できる場合がある22。

SNS5において、特に物理法則の理解や数理的な推論が求められるサブモジュールとしてKANを採用することで、モデル全体のパラメータ数を削減しつつ、知能密度を向上させることが可能となる。

## **3\. ニューロモーフィックコンピューティングとスパイキングニューラルネットワーク（SNN）**

「ANN系のAIを超える」という目標を達成するためには、従来のフォン・ノイマン型アーキテクチャ（メモリとプロセッサの分離）からの脱却が不可欠である。人間の脳を模倣した**ニューロモーフィック・コンピューティング**と\*\*スパイキングニューラルネットワーク（SNN）\*\*は、エネルギー効率の理論的上限に挑む技術である。

### **3.1 イベント駆動型処理による究極の省エネ**

SNNは、情報を通時的な「スパイク（パルス信号）」として表現し、入力があった時のみニューロンが発火・動作する\*\*イベント駆動型（Event-Driven）\*\*の処理を行う。常に全ニューロンが活性化値を計算するANNとは異なり、SNNはスパース（疎）な活動を行うため、理論上、消費電力を数桁オーダーで削減可能である15。

#### **3.1.1 ニューロモーフィック・ハードウェアの優位性**

SNNの真価は、専用のニューロモーフィック・チップ上で発揮される。

* **IBM NorthPole**: 完全にデジタルなニューロモーフィックチップであり、メモリと演算回路を融合させることで、従来のGPUと比較して**25倍のエネルギー効率**と**22倍の面積効率**、**5倍の高速化**を実現している。外部メモリへのアクセスを排除することで「メモリの壁」を物理的に解決している25。  
* **Intel Loihi 2**: 非同期回路を採用し、ミリ秒単位の細かい時間分解能でのスパイク処理が可能。最大100万ニューロン級の集積度を持ちながら、極めて低い消費電力で動作する25。

### **3.2 SpikeLLM：大規模言語モデルのSNN化**

これまでSNNは学習の難しさから、大規模モデルへの適用が困難とされてきた。しかし、近年の研究により、700億（70B）パラメータ規模のLLMをSNN化する**SpikeLLM**のような手法が確立されつつある26。

#### **3.2.1 実現技術：GIFニューロンとOBSフレームワーク**

SpikeLLMは、従来のANN量子化の代替として、生物学的妥当性の高い**一般化積分発火（Generalized Integrate-and-Fire, GIF）ニューロン**を採用する。

* **Optimal Brain Spiking (OBS)**: 事前学習済みのLLMの重みを、SNNに変換する際に生じる誤差（変換損失）を最小化するためのフレームワーク。チャネルごとのサリエンシー（重要度）に基づき、異常値を持つチャネルと通常のチャネルを分離して処理することで、低ビットレートのスパイク通信でも情報を劣化させずに伝達することを可能にした26。  
* **性能**: SpikeLLMは、WikiText2などのベンチマークにおいて、従来の量子化モデル（W4A4など）と比較しても低いPerplexity（予測誤差）を達成しており、一部の推論タスクでは精度向上も見られる。これは、SNNが単なる省エネ版ANNではなく、独自の計算特性によってLLMの能力を維持・拡張できることを示している28。

SNS5への統合戦略:  
現時点では専用チップの入手性が課題となるが、SNNのアルゴリズム（イベント駆動、スパース性）をソフトウェアレベルでシミュレートするだけでも、ハードウェアアクセラレーション（FPGAや専用ASIC）と組み合わせることで大幅な効率化が見込める。特に、前述のBitNetやRWKVと組み合わせた「Spiking RWKV」は、SNNのスパース性とRWKVの線形性を併せ持つ有望なハイブリッド領域である17。

## **4\. 知能のフロンティア：能動的推論とエージェンシー**

省エネルギー化が「身体」の効率化であるならば、知能の向上は「脳」の動作原理の革新である。SNS5は、単にテキストを生成する受動的なモデルではなく、自ら環境を探索し、知識を更新する自律的なエージェントを目指すべきである。このための理論的支柱となるのが、\*\*自由エネルギー原理（Free Energy Principle, FEP）**に基づく**能動的推論（Active Inference）\*\*である。

### **4.1 受動的学習から能動的推論へ**

現在のLLMは、与えられたプロンプトに対して確率的に尤もらしい続きを予測するだけであり、自己の不確実性を能動的に解消しようとする動機を持たない。対して能動的推論エージェントは、\*\*「期待自由エネルギー（Expected Free Energy, G）」\*\*を最小化するように行動を選択する29。

#### **4.1.1 期待自由エネルギーの二重性**

期待自由エネルギー$G$の最小化は、以下の二つの要素のバランスを取るプロセスとして定式化される。

1. **実用的価値（Pragmatic Value）**: 自身が望む選好（Preferences、C行列）を満たす状態へ到達しようとする行動。強化学習における「報酬最大化」に相当する。  
2. **認識的価値（Epistemic Value）**: 環境に対する不確実性や驚き（Surprise）を減らそうとする行動。「情報の獲得」や「好奇心」に相当し、エージェントは自分の知識の欠落を埋めるために質問したり、探索したりする動機を持つ29。

この理論を導入することで、SNS5は「答えられない質問」に対して適当なハルシネーションを生成するのではなく、「情報を検索する」「ユーザーに問い返す」といった行動を、数理的な必然性として選択できるようになる。

### **4.2 実装アーキテクチャ：LLMによる生成モデル（POMDP）**

能動的推論をLLMに実装するためには、LLMをPOMDP（部分観測マルコフ決定過程）の\*\*生成モデル（Generative Model）\*\*として位置づける必要がある32。

| 行列 | 機能 | LLMにおける役割 |
| :---- | :---- | :---- |
| **A行列 (Likelihood)** | 隠れ状態$s$から観測$o$への写像 $P(o\\|s$ | 「現在の文脈や真実（状態）」から「生成されるテキスト（観測）」の確率分布。 |
| **B行列 (Transitions)** | 状態$s\_t$から次の状態$s\_{t+1}$への遷移 $P(s\_{t+1}\\|s\_t, u$ | 行動$u$（検索、回答など）によって文脈がどう変化するかの予測モデル。 |
| **C行列 (Preferences)** | 選好分布 $P(o)$ | エージェントの目標（例：正確な回答、エネルギー節約、倫理的安全性）。 |
| **D行列 (Priors)** | 初期状態の信念 $P(s\_0)$ | 会話開始時点での前提知識や文脈理解。 |

実装ツールとしてのpymdpとRxInfer.jl:  
Pythonライブラリであるpymdpや、Julia言語のRxInfer.jlは、この離散状態空間における能動的推論ループを実装するために設計されている。

* **pymdp**: エージェントの知覚、推論、学習、行動選択の標準的なパイプラインを提供する。LLMの出力を離散的な「観測」として扱い、エージェント内部の「信念（Belief）」を更新するメタコントローラーとして機能させることができる32。  
* **RxInfer.jl**: ファクターグラフ上でのメッセージパッシング（Belief Propagation）を用いた変分推論を高速に行うライブラリ。大規模なモデルや複雑な確率的依存関係を持つエージェントの推論エンジンとして、リアルタイム性の高い処理が可能である35。

### **4.3 マルチエージェント・オーケストレーション**

さらに高度な知能を実現するために、「オーケストレーター（Orchestrator）」と呼ばれる構成が提案されている。これは、タスクを実行する複数のLLMエージェントの上位に、能動的推論に基づくメタ認知層（Meta-Cognition Layer）を配置するものである。オーケストレーターは、各エージェントの状態を監視し、全体の不確実性を最小化するようにタスクの割り振りやプロンプトの動的な調整を行う30。これにより、単一のLLMでは解決できない複雑な問題を、専門化されたエージェント群の協調によって解決する「社会的な知能」が実現する。

## **5\. 修正可能性のフロンティア：事後的修正が可能なLLMの実現**

ユーザーの要件にある「後から修正できるLLM」は、現在のAI開発における最大の難関の一つである。ファインチューニングによる知識の更新は、コストが高いだけでなく、既存の知識を破壊するリスクが高い。SNS5では、**重み更新に頼らない知識管理**のアプローチを採用すべきである。

### **5.1 モデル編集（Model Editing）の限界と「破滅的忘却」**

ROME（Rank-One Model Editing）やMEMITといった手法は、特定の事実（例：「エッフェル塔はローマにある」という反事実的修正など）に対応する重みを直接書き換える技術として注目された。しかし、最新の研究（2024-2025年）は、これらの手法が\*\*連続的な編集（Sequential Editing）\*\*に弱いことを明らかにしている38。  
複数の事実を次々と修正していくと、モデルの重みが過度に歪み、編集していない無関係な知識まで想起できなくなる「知識の歪曲（Knowledge Distortion）」や、推論能力全体の低下（Model Collapse）が発生する40。したがって、SNS5の基本戦略として、長期的な知識の保存・更新手段としてモデルの重み編集を採用することは推奨されない。

### **5.2 解決策：ニューロシンボリックRAGと知識グラフ**

「後から修正可能」かつ「忘却しない」システムを実現する唯一の堅実な方法は、知識をニューラルネットワークの外部に切り出し、\*\*Retrieval-Augmented Generation (RAG)**と**知識グラフ（Knowledge Graph）\*\*を組み合わせることである。

#### **5.2.1 GraphRAG：構造化された外部記憶**

従来のベクトル検索ベースのRAGは、意味的な類似性で情報を検索するが、事実関係の構造（誰が、何を、どうした）を理解していないため、複雑な推論に弱い。これに対し、**GraphRAG**は、エンティティ間の関係性をグラフ構造（ノードとエッジ）として明示的に保持する42。

* **修正の容易性**: 知識の修正が必要な場合、再学習は一切不要である。知識グラフ上の該当するノードやエッジを更新（Update）または削除（Delete）するだけでよい。これにより、100%確実な「修正」が即座に反映される。  
* **ニューロシンボリックな統合**: LLM（ニューラル）は自然言語のインターフェースおよび推論エンジンとして機能し、知識グラフ（シンボリック）は信頼できる事実の保管庫として機能する。この分離により、AIの「推論能力」と「知識量」を独立して管理・スケーリングできる44。

#### **5.2.2 ハイブリッド記憶システム（Memori）**

さらに進んだ実装として、ベクトルデータベース（短期・中期記憶）と知識グラフ/リレーショナルデータベース（長期・不変記憶）を組み合わせたハイブリッド記憶システムの構築が有効である。「Memori」のようなシステムは、SQLのような堅牢なデータベース技術をAIの記憶として再評価しており、重要な事実やルールを構造化データとして保存することで、長期にわたる一貫性を保証している45。

### **5.3 制約付きデコーディング（Constrained Decoding）による制御**

知識だけでなく、AIの「振る舞い」や「論理」を修正・制御したい場合、\*\*制約付きデコーディング（Constrained Decoding）\*\*が強力なツールとなる。  
これは、LLMがトークンを生成する際、文法（Grammar）や論理制約（Logic）に違反するトークンの生成確率を強制的にゼロにする技術である47。

* **再学習不要の制御**: 例えば「出力は必ずJSON形式でなければならない」や「特定の差別用語を使ってはならない」といったルールを、モデルの重みを変更することなく、推論時のフィルタリングのみで厳密に適用できる。  
* **ニューロシンボリック・ラッパー**: 「SymCode+」や「Guardrails AI」のようなフレームワークを用いることで、LLMの出力が論理的に正しいかをシンボリックなソルバーで検証し、誤りがあれば即座に自己修正ループを回すことが可能になる49。これにより、ハルシネーションを抑制し、高い信頼性を担保できる。

## **6\. SNS5に向けた統合ロードマップ**

以上の調査に基づき、SNS5が「ANN超え」を実現するための具体的な開発ロードマップを提案する。

### **フェーズ1：高効率バックボーンの構築（BitRWKV）**

* **目標**: 既存のトランスフォーマー比で10倍以上のエネルギー効率を持つ基盤モデルの構築。  
* **アクション**:  
  * **アーキテクチャ**: RWKV-6またはRWKV-7を採用し、線形注意機構による推論メモリの定数化（$O(1)$）を実現する。  
  * **量子化**: BitNet b1.58の学習レシピ（AbsMean量子化）を適用し、重みを1.58ビット（$\\{-1, 0, 1\\}$）で学習する**BitRWKV**を開発する。  
  * **実装**: bitnet.cppをベースに、CPUおよびエッジデバイス（ARM/x86）向けの最適化カーネルを実装し、GPUレスでの高速推論を確立する。

### **フェーズ2：能動的知能の実装（Active Inference Agent）**

* **目標**: 受動的なテキスト生成から、自律的な探索と目標達成を行うエージェントへの進化。  
* **アクション**:  
  * **制御ループ**: pymdpまたはRxInfer.jlを用いて、BitRWKVを生成モデルとして組み込んだPOMDPエージェントループを構築する。  
  * **状態空間設計**: ユーザーの意図、対話の文脈、外部知識の状態を隠れ状態として定義し、期待自由エネルギー最小化に基づく行動選択（回答、検索、質問、沈黙など）を実装する。

### **フェーズ3：永続的かつ修正可能な記憶システム（Neuro-symbolic GraphRAG）**

* **目標**: 再学習なしで知識を更新・修正でき、長期間運用可能な記憶の実現。  
* **アクション**:  
  * **知識グラフ統合**: Neo4jや軽量なグラフDBをエージェントの長期記憶として統合する。  
  * **修正ツールの実装**: エージェントに「記憶修正ツール」を持たせ、ユーザーからの指摘や新しい情報に基づいて、グラフ内のノードを自律的に更新させる機能を実装する。  
  * **制約付き生成**: グラフクエリ言語（Cypher/SPARQL）の生成において、文法制約デコーディングを適用し、構文エラーのない確実なデータベース操作を保証する。

## **7\. 結論**

SNS5プロジェクトが目指す「ANN系のAIを超える省エネと知能」は、既存の技術の延長線上には存在しない。本調査は、\*\*1.58ビット量子化とRWKVアーキテクチャの融合（BitRWKV）\*\*が、ハードウェアの制約を打破する物理的な鍵であることを示した。そして、**能動的推論**が、受動的な機械学習モデルを自律的な知的エージェントへと昇華させる理論的な鍵である。さらに、「後から修正できる」という要件に対しては、重み更新という不確実な手法を捨て、**知識グラフと制約付きデコーディングによるニューロシンボリックなアプローチ**を採用することが、エンジニアリングとして最も確実かつ持続可能な解である。

これらの技術要素――BitRWKV、Active Inference、Neuro-symbolic GraphRAG――を統合することで、SNS5は、現在の巨大で非効率なLLMパラダイムとは一線を画す、真に効率的で、賢く、そして信頼できる次世代のAIシステムとなり得るだろう。

#### **引用文献**

1. South Korea’s Semiconductor Supercycle: AI Demand Ignites Price Surge, Threatening Global Electronics, 11月 19, 2025にアクセス、 [https://markets.financialcontent.com/wral/article/tokenring-2025-11-18-south-koreas-semiconductor-supercycle-ai-demand-ignites-price-surge-threatening-global-electronics](https://markets.financialcontent.com/wral/article/tokenring-2025-11-18-south-koreas-semiconductor-supercycle-ai-demand-ignites-price-surge-threatening-global-electronics)  
2. d-Matrix Secures $275 Million, Claims 10x Faster AI Than Nvidia with Revolutionary In-Memory Compute, 11月 19, 2025にアクセス、 [https://markets.financialcontent.com/wral/article/tokenring-2025-11-18-d-matrix-secures-275-million-claims-10x-faster-ai-than-nvidia-with-revolutionary-in-memory-compute](https://markets.financialcontent.com/wral/article/tokenring-2025-11-18-d-matrix-secures-275-million-claims-10x-faster-ai-than-nvidia-with-revolutionary-in-memory-compute)  
3. Spurious Forgetting in Continual Learning of Language Models \- OpenReview, 11月 19, 2025にアクセス、 [https://openreview.net/forum?id=ScI7IlKGdI](https://openreview.net/forum?id=ScI7IlKGdI)  
4. Catastrophic Forgetting or the Challenge of Continuous Learning | by Thomas Zilliox, 11月 19, 2025にアクセス、 [https://medium.com/@thomas.zilliox/catastrophic-forgetting-or-the-challenge-of-continuous-learning-1278a1179811](https://medium.com/@thomas.zilliox/catastrophic-forgetting-or-the-challenge-of-continuous-learning-1278a1179811)  
5. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2402.17764v1](https://arxiv.org/html/2402.17764v1)  
6. \[2402.17764\] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)  
7. The Future of AI Efficiency with BitNet b1.58 and 1-Bit LLMs \- CloudThat, 11月 19, 2025にアクセス、 [https://www.cloudthat.com/resources/blog/the-future-of-ai-efficiency-with-bitnet-b1-58-and-1-bit-llms](https://www.cloudthat.com/resources/blog/the-future-of-ai-efficiency-with-bitnet-b1-58-and-1-bit-llms)  
8. So what happened to the 1.58bit models "revolution" ? : r/LocalLLaMA \- Reddit, 11月 19, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1hsa0tm/so\_what\_happened\_to\_the\_158bit\_models\_revolution/](https://www.reddit.com/r/LocalLLaMA/comments/1hsa0tm/so_what_happened_to_the_158bit_models_revolution/)  
9. \[BitNet b1.58\] Achieved accuracy better than Llama by expressing model parameters in three values\! | AI-SCHOLAR, 11月 19, 2025にアクセス、 [https://ai-scholar.tech/en/articles/large-language-models/BitNet1-58b](https://ai-scholar.tech/en/articles/large-language-models/BitNet1-58b)  
10. microsoft/BitNet: Official inference framework for 1-bit LLMs \- GitHub, 11月 19, 2025にアクセス、 [https://github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)  
11. Reimagining AI Efficiency: A Practical Guide to Using BitNet's 1-Bit LLM on CPUs Without Sacrificing Performance | by Kondwani Nyirenda | Medium, 11月 19, 2025にアクセス、 [https://medium.com/@kondwani0099/reimagining-ai-efficiency-a-practical-guide-to-using-bitnets-1-bit-llm-on-cpus-without-ef804d3fb875](https://medium.com/@kondwani0099/reimagining-ai-efficiency-a-practical-guide-to-using-bitnets-1-bit-llm-on-cpus-without-ef804d3fb875)  
12. Introduction to RWKV: An Evolution in AI Sequence Modeling | by Eugenii Shevchenko, 11月 19, 2025にアクセス、 [https://medium.com/@eugenesh4work/introduction-to-rwkv-an-evolution-in-ai-sequence-modeling-9659c2a2de35](https://medium.com/@eugenesh4work/introduction-to-rwkv-an-evolution-in-ai-sequence-modeling-9659c2a2de35)  
13. How RWKV creates more efficient LLMs | by Devansh \- Medium, 11月 19, 2025にアクセス、 [https://machine-learning-made-simple.medium.com/how-rwkv-creates-more-efficient-llms-04ddf197b219](https://machine-learning-made-simple.medium.com/how-rwkv-creates-more-efficient-llms-04ddf197b219)  
14. A Survey of RWKV \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2412.14847v1](https://arxiv.org/html/2412.14847v1)  
15. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2302.13939v5](https://arxiv.org/html/2302.13939v5)  
16. SDiT: Spiking Diffusion Model with Transformer \- alphaXiv, 11月 19, 2025にアクセス、 [https://www.alphaxiv.org/overview/2402.11588v2](https://www.alphaxiv.org/overview/2402.11588v2)  
17. SpikeRWKV:Energy-efficient Large Language Model with Spiking Neural Network, 11月 19, 2025にアクセス、 [http://poster-openaccess.com/files/ICIC2025/4202.pdf](http://poster-openaccess.com/files/ICIC2025/4202.pdf)  
18. Understanding Liquid Neural Networks: The Future of Adaptable AI \- Medium, 11月 19, 2025にアクセス、 [https://medium.com/@pg2196577/understanding-liquid-neural-networks-the-future-of-adaptable-ai-49ea78017f67](https://medium.com/@pg2196577/understanding-liquid-neural-networks-the-future-of-adaptable-ai-49ea78017f67)  
19. The Big Picture: Has Liquid AI hit a gusher with its first non-transformer models?, 11月 19, 2025にアクセス、 [https://www.coloradoai.news/the-big-picture-has-liquid-ai-hit-a-gusher-with-its-first-non-transformer-models/](https://www.coloradoai.news/the-big-picture-has-liquid-ai-hit-a-gusher-with-its-first-non-transformer-models/)  
20. SineKAN: Kolmogorov-Arnold Networks using sinusoidal activation functions \- Frontiers, 11月 19, 2025にアクセス、 [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1462952/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1462952/full)  
21. Efficiency Analysis of Kolmogorov-Arnold Networks for Visual Data Processing \- MDPI, 11月 19, 2025にアクセス、 [https://www.mdpi.com/2673-4591/79/1/68](https://www.mdpi.com/2673-4591/79/1/68)  
22. Optimizing Building Energy Efficiency with Kolmogorov-Arnold Network (KAN) backed Regression \- IEEE Xplore, 11月 19, 2025にアクセス、 [https://ieeexplore.ieee.org/iel8/11063690/11063701/11064661.pdf](https://ieeexplore.ieee.org/iel8/11063690/11063701/11064661.pdf)  
23. Opening the Black-Box: Symbolic Regression with Kolmogorov-Arnold Networks for Energy Applications \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2504.03913v1](https://arxiv.org/html/2504.03913v1)  
24. Synergies and Divergences Between Spiking Neural Networks and Large Language Models \- Preprints.org, 11月 19, 2025にアクセス、 [https://www.preprints.org/manuscript/202507.0525](https://www.preprints.org/manuscript/202507.0525)  
25. Can neuromorphic computing help reduce AI's high energy cost? \- PNAS, 11月 19, 2025にアクセス、 [https://www.pnas.org/doi/10.1073/pnas.2528654122](https://www.pnas.org/doi/10.1073/pnas.2528654122)  
26. \[2407.04752\] SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/abs/2407.04752](https://arxiv.org/abs/2407.04752)  
27. SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking | alphaXiv, 11月 19, 2025にアクセス、 [https://www.alphaxiv.org/overview/2407.04752v1](https://www.alphaxiv.org/overview/2407.04752v1)  
28. SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2407.04752v3](https://arxiv.org/html/2407.04752v3)  
29. From pixels to planning: scale-free active inference \- PMC \- PubMed Central, 11月 19, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12217590/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12217590/)  
30. Active Inference for Self-Organizing Multi-LLM Systems: A Bayesian Thermodynamic Approach to Adaptation \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2412.10425v1](https://arxiv.org/html/2412.10425v1)  
31. Active Inference for Self-Organizing Multi-LLM Systems: A Bayesian Thermodynamic Approach to Adaptation \- ResearchGate, 11月 19, 2025にアクセス、 [https://www.researchgate.net/publication/387104300\_Active\_Inference\_for\_Self-Organizing\_Multi-LLM\_Systems\_A\_Bayesian\_Thermodynamic\_Approach\_to\_Adaptation](https://www.researchgate.net/publication/387104300_Active_Inference_for_Self-Organizing_Multi-LLM_Systems_A_Bayesian_Thermodynamic_Approach_to_Adaptation)  
32. Active Inference with pymdp: Tutorial 2 \- Colab, 11月 19, 2025にアクセス、 [https://colab.research.google.com/github/infer-actively/pymdp/blob/master/docs/notebooks/using\_the\_agent\_class.ipynb](https://colab.research.google.com/github/infer-actively/pymdp/blob/master/docs/notebooks/using_the_agent_class.ipynb)  
33. Tutorial 1: Active inference from scratch \- pymdp's documentation\! \- Read the Docs, 11月 19, 2025にアクセス、 [https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active\_inference\_from\_scratch.html](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html)  
34. Tutorial 2: the Agent API \- pymdp's documentation\! \- Read the Docs, 11月 19, 2025にアクセス、 [https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using\_the\_agent\_class.html](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html)  
35. RxInfer.jl \- Fast and Flexible Bayesian Inference, 11月 19, 2025にアクセス、 [https://rxinfer.com/](https://rxinfer.com/)  
36. Large Language Models \- RxInfer.jl Examples, 11月 19, 2025にアクセス、 [https://examples.rxinfer.com/categories/experimental\_examples/large\_language\_models/](https://examples.rxinfer.com/categories/experimental_examples/large_language_models/)  
37. Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks \- ResearchGate, 11月 19, 2025にアクセス、 [https://www.researchgate.net/publication/395354569\_Orchestrator\_Active\_Inference\_for\_Multi-Agent\_Systems\_in\_Long-Horizon\_Tasks](https://www.researchgate.net/publication/395354569_Orchestrator_Active_Inference_for_Multi-Agent_Systems_in_Long-Horizon_Tasks)  
38. Model Editing at Scale leads to Gradual and Catastrophic Forgetting \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2401.07453v2](https://arxiv.org/html/2401.07453v2)  
39. Model Editing at Scale leads to Gradual and Catastrophic Forgetting \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2401.07453v4](https://arxiv.org/html/2401.07453v4)  
40. Exploring Pitfalls of Knowledge Editing in Large Language Models \- Medium, 11月 19, 2025にアクセス、 [https://medium.com/@techsachin/exploring-pitfalls-of-knowledge-editing-in-large-language-models-a0ab043909d0](https://medium.com/@techsachin/exploring-pitfalls-of-knowledge-editing-in-large-language-models-a0ab043909d0)  
41. Understanding Robustness of Model Editing in Code LLMs: An Empirical Study \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2511.03182v1](https://arxiv.org/html/2511.03182v1)  
42. How to Improve Multi-Hop Reasoning With Knowledge Graphs and LLMs \- Neo4j, 11月 19, 2025にアクセス、 [https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/](https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/)  
43. Knowledge graph vs vector database: Which one to choose? \- FalkorDB, 11月 19, 2025にアクセス、 [https://www.falkordb.com/blog/knowledge-graph-vs-vector-database/](https://www.falkordb.com/blog/knowledge-graph-vs-vector-database/)  
44. Combining retrieval augmented generation with knowledge graphs for more reliable AI analytics \- Outshift | Cisco, 11月 19, 2025にアクセス、 [https://outshift.cisco.com/blog/combining-retrieval-augmented-generation-knowledge-graphs](https://outshift.cisco.com/blog/combining-retrieval-augmented-generation-knowledge-graphs)  
45. A-MEM: Agentic Memory for LLM Agents \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/pdf/2502.12110](https://arxiv.org/pdf/2502.12110)  
46. Everyone's trying vectors and graphs for AI memory. We went back to SQL. : r/AI\_Agents \- Reddit, 11月 19, 2025にアクセス、 [https://www.reddit.com/r/AI\_Agents/comments/1nkx0bz/everyones\_trying\_vectors\_and\_graphs\_for\_ai\_memory/](https://www.reddit.com/r/AI_Agents/comments/1nkx0bz/everyones_trying_vectors_and_graphs_for_ai_memory/)  
47. Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2403.06988v1](https://arxiv.org/html/2403.06988v1)  
48. Controlling your LLM: Deep dive into Constrained Generation | by Andrew Docherty, 11月 19, 2025にアクセス、 [https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20](https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)  
49. SymCode: A Neurosymbolic Approach to Mathematical Reasoning via Verifiable Code Generation \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2510.25975v1](https://arxiv.org/html/2510.25975v1)  
50. OneShield \- the Next Generation of LLM Guardrails \- arXiv, 11月 19, 2025にアクセス、 [https://arxiv.org/html/2507.21170v1](https://arxiv.org/html/2507.21170v1)