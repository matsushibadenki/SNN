# **高次元情報空間におけるマルチプレクシング：AIの学習・推論効率を最大化するための技術的・理論的枠組みに関する包括的調査報告書**

## **1\. エグゼクティブサマリー**

人工知能（AI）研究の最前線において、計算資源とパラメータ効率の限界を突破するための鍵として「マルチプレクシング（Multiplexing、多重化）」の概念が急速に重要性を増している。従来のAIモデル開発は、性能向上をパラメータ数とデータ量の増大（スケーリング則）に依存してきたが、物理的なメモリ制約や推論レイテンシ、電力消費の問題が顕在化するにつれ、限られたリソース内でいかに多くの情報や機能を同時に表現・処理するかという「情報密度」の向上が求められるようになった。マルチプレクシングとは、通信工学における多重化技術と同様に、単一のニューラルネットワークやベクトル表現の中に、複数のタスク、概念、あるいは文脈を干渉なく埋め込む技術を指す。

本報告書は、ユーザーの要求に基づき、AIの学習および推論フェーズにおいてマルチプレクシングを採用しやすくするための技術、アルゴリズム、および理論的基盤を網羅的に調査・分析したものである。調査の結果、マルチプレクシングの実用化を阻む主要な課題は「情報の干渉（Interference）」、「破滅的忘却（Catastrophic Forgetting）」、「解釈性の欠如（Black Box）」の3点に集約されることが判明した。これらの課題を克服し、マルチプレクシングを容易にするための技術群は、以下の5つの主要なカテゴリに分類される。

1. **ニューロンの「重ね合わせ」と幾何学的制御**: Anthropic社の研究に代表される「スーパーポジション（Superposition）」のメカニズムを解明し、ニューロン数を超える特徴量を埋め込む際の干渉を制御する理論。  
2. **パラメータ空間の直交化（Orthogonalization）**: O-LoRAやGOLA、スペクトル正則化など、学習プロセスにおいて重み行列や更新ベクトルの直交性を強制することで、異なるタスクや機能が互いに「不可視」な状態で共存することを可能にする技術。  
3. **表現の分離（Disentanglement）**: $\\beta$-VAEやTopDisのように、潜在空間において情報の生成因子を独立した軸として分離・整列させ、多重化の際の「素材」を純粋化する技術。  
4. **アーキテクチャによる動的ルーティング**: Mixture of Experts (MoE) や動的アダプタ（Dynamic LoRA）を用い、入力に応じて計算経路を空間的・時間的に分割・多重化するシステム設計。  
5. **ハイパーディメンショナルコンピューティング（HDC）**: ベクトル記号アーキテクチャ（VSA）に基づき、高次元ベクトルの代数演算（Binding, Bundling）によって明示的かつ可逆的に情報を多重化する数学的枠組み。

本報告書では、これらの技術がいかにしてAIモデルの「可塑性（学習しやすさ）」と「安定性（記憶の保持）」のトレードオフを解消し、次世代の効率的なAIシステムの基盤となり得るかを、詳細な理論的背景と実装例を交えて論じる。

## ---

**2\. スーパーポジション仮説と情報容量の物理学**

AIにおけるマルチプレクシングの最も根源的な形態は、ニューラルネットワークの内部表現において自然発生的に生じる「スーパーポジション（Superposition）」である。これを理解し制御することは、意図的なマルチプレクシングを実現するための第一歩となる。

### **2.1 ニューラルネットワークにおける「有効次元」の拡張**

従来の直感では、ニューラルネットワークが表現できる独立した概念（特徴量）の数は、物理的なニューロンの数 $N$ に制限されると考えられてきた。しかし、近年の大規模言語モデル（LLM）の解析、特にAnthropic社の研究チームによる「トイモデル（Toy Models）」を用いた実験は、この直感を覆す結果を示している 1。

#### **多義性（Polysemanticity）の再評価**

解析の結果、単一のニューロンが「学術論文の引用」と「韓国語のテキスト」といった、全く無関係な複数の概念に反応する現象が頻繁に観察された。これを「多義性（Polysemanticity）」と呼ぶ。従来、これはモデルの解釈性を阻害する「バグ」と見なされていたが、スーパーポジション仮説に基づけば、これはモデルが限られたニューロン数 $N$ を超える数 $M$ の特徴量（$M \\gg N$）を表現するために編み出した、高度な「圧縮戦略」であると解釈できる 1。

#### **ジョンソン・リンデンシュトラウスの補題と「ほぼ直交」**

この圧縮が可能になる数学的根拠は、高次元空間の幾何学的特性にある。ジョンソン・リンデンシュトラウスの補題が示唆するように、高次元空間（例えば $10^4$ 次元）においては、ランダムに選択された2つのベクトルが「ほぼ直交（Almost Orthogonal）」する確率が極めて高い。  
AIモデルはこの性質を利用し、各特徴量を完全に直交する基底（One-hot表現）ではなく、わずかな内積（コサイン類似度）を持つ「ほぼ直交」する方向ベクトルとして配置する。これにより、許容可能なノイズ（干渉）の範囲内で、物理的な次元数を遥かに超える数の特徴量をパッキング（多重化）することが可能になる 4。

### **2.2 干渉（Interference）とスパース性のトレードオフ**

スーパーポジションによるマルチプレクシングは、「無料のランチ」ではない。特徴量Ａと特徴量Ｂが同じニューロン群を共有し、かつ非直交である場合、一方の活性化は他方の読み出しに対してノイズ（干渉）として作用する。

$$Interference \= \\sum\_{j \\neq i} (W\_i \\cdot W\_j) x\_j$$

ここで $W$ は特徴量の埋め込みベクトル、$x$ は活性化値である。  
この干渉を抑制し、マルチプレクシングを成立させるための重要な条件が「スパース性（Sparsity）」である 1。現実世界のデータにおいては、ある瞬間に活性化している概念は全体のごく一部である。特徴量の生起確率が十分に低ければ（スパースであれば）、特徴量Ａが活性化している瞬間に、干渉相手である特徴量Ｂも同時に活性化している確率は無視できるほど小さくなる。  
したがって、AIの学習においてマルチプレクシングを採用しやすくするためには、データのスパース性を最大化するような表現学習（Sparse Representation Learning）や、活性化関数（ReLUやTop-K）の選択が重要となる。

### **2.3 幾何学的構造化：Uniform Polytopes**

さらに興味深いことに、学習が進んだネットワークでは、特徴量ベクトルはランダムに配置されるのではなく、干渉を最小化するための最適な幾何学的構造を形成することが明らかになっている。これを「Uniform Polytopes（均一多面体）」と呼ぶ 4。  
例えば、2次元平面に3つの特徴量を埋め込む場合、モデルはそれらを互いに120度の角度を持つ「メルセデス・ベンツのロゴ」のような形状に配置する。これにより、どの2つの特徴量間の干渉（内積の絶対値）も均等かつ最小になるように調整される。  
この知見は、マルチプレクシングのための初期化や正則化において、特徴量ベクトルがこのような最適配置をとるように誘導すること（幾何学的正則化）が有効であることを示唆している。

## ---

**3\. 学習を安定化させるパラメータ空間の直交化技術**

スーパーポジションは自然発生的な現象であるが、実用的なAIシステム開発、特に継続学習（Continual Learning）やマルチタスク学習においては、より制御された明示的なマルチプレクシング技術が必要となる。その核心となるのが「直交性（Orthogonality）」の強制である。

### **3.1 Orthogonal Low-Rank Adaptation (O-LoRA)**

Low-Rank Adaptation (LoRA) は、事前学習済みモデルの重み行列 $W\_0$ を固定し、低ランク行列 $A, B$ の積 $\\Delta W \= BA$ のみを学習する手法として広く普及している。しかし、単純なLoRAを複数のタスクに順次適用すると、パラメータ更新領域が重なり合い、過去のタスク知識を破壊する「破滅的忘却」が発生する。これを解決し、単一モデル内でのタスク多重化を容易にするのが **O-LoRA (Orthogonal LoRA)** である 5。

#### **サブスペースの分離による干渉回避**

O-LoRAの基本原理は、新しいタスクのための更新行列が、過去のタスクの更新行列が張る部分空間（Subspace）に対して直交するように制約をかけることである。  
具体的には、タスク $t$ の学習において、損失関数に以下の直交化ペナルティ項を追加する。

$$L\_{orth} \= \\lambda \\sum\_{i \< t} \\| \\Delta W\_t^T \\Delta W\_i \\|\_F^2$$

あるいは、勾配更新時にグラム・シュミットの正規直交化プロセスに類似した射影操作を行い、過去のタスク空間への成分を強制的に除去する。これにより、タスクAのパラメータ調整がタスクBの機能に影響を与えること（干渉）を物理的に防ぐことができる。

#### **O-LoRAの実装効果**

O-LoRAを採用することで、開発者は以下のメリットを享受できる。

1. **Replay Bufferの不要化**: 過去のデータを保存・再学習する必要がなくなり、プライバシー保護やストレージ効率が向上する。  
2. **容量の最大化**: モデルのパラメータ空間を「互いに干渉しない個室」に区切って使用するため、単一モデルに詰め込めるタスク数（多重化率）が大幅に向上する。  
3. **線形加法性の向上**: 各タスクのアダプタが直交しているため、推論時に複数のアダプタを単純加算してマージしても、性能劣化（干渉ノイズ）が起きにくくなる。

### **3.2 GOLA: グループ直交性によるマルチモーダル多重化**

RGB画像と熱赤外線（Thermal）画像を用いたトラッキングのようなマルチモーダル学習においては、異なるモダリティ間の情報重複がリソースの無駄遣いとなる。**GOLA (Group Orthogonal Low-Rank Adaptation)** は、この問題に対処するために開発された技術である 7。

#### **ランクの機能分解とグループ化**

GOLAでは、学習可能なランク（特異値分解における成分）を、事前学習済みの知識を保持する「重要ランク（Crucial Ranks）」と、新しいタスクに適応させるための「冗長ランク（Redundant Ranks）」に分解する。さらに、冗長ランクを複数のグループに分割し、グループ間での直交性制約（Inter-group Orthogonal Constraint）を課す。

$$L\_{GOLA} \= L\_{task} \+ \\lambda \\| G\_i^T G\_j \\|\_F^2 \\quad (i \\neq j)$$

これにより、あるグループは「RGB固有の特徴（テクスチャなど）」を、別のグループは「熱画像固有の特徴（温度分布など）」を学習するように誘導される。結果として、限られたパラメータ数の中で異なる種類の情報を効率的に多重化し、冗長性を排除した高密度な表現学習が可能となる。

### **3.3 Spectral Regularization（スペクトル正則化）と可塑性**

マルチプレクシングを長期間維持するためには、学習プロセス自体がモデルの「可塑性（Plasticity）」を損なわないようにする必要がある。学習が進むと重み行列の特異値分布が偏り、新しい情報を書き込むための有効な次元（直交する余地）が失われる現象が知られている 8。

スペクトル正則化は、各層の重み行列 $W$ の特異値 $\\sigma\_i$ が学習中も適切な分布（例えば最大特異値が1付近）を保つように制御する。

$$L\_{spectral} \= \\beta \\| \\sigma\_{max}(W) \- 1 \\|^2$$

この正則化により、ネットワークは「初期化直後のような学習しやすさ」を維持し続けることができる。これは、次々と新しいタスクを追加していく継続的なマルチプレクシングにおいて、後半のタスク学習が停滞するのを防ぐために不可欠な理論的基盤である。また、直交初期化（Orthogonal Initialization） 8 を用いて学習を開始することも、多重化能力を最大化するための前提条件となる。

## ---

**4\. アーキテクチャによる動的マルチプレクシングとルーティング**

パラメータの直交化が「静的」な多重化支援であるのに対し、アーキテクチャレベルで計算リソースを動的に割り当てる技術は、時間的・空間的な多重化を実現する。

### **4.1 Mixture of Experts (MoE) における直交ルーティング**

**Mixture of Experts (MoE)** は、入力に応じて巨大なモデルの一部（エキスパート）のみを活性化させる技術であり、本質的に「空間分割多重」のアーキテクチャである。近年の研究では、このMoEに直交性の概念を導入することで、タスク間の分離能力をさらに高める手法が提案されている 11。

#### **Orthogonal Routing（直交ルーティング）**

継続学習において新しいタスクを追加する際、MoEのルーター（Gating Network）が新しい入力に対して、既存タスクで使用されているエキスパートとは異なるエキスパート群を選択するように学習させる。具体的には、ルーターの更新勾配を、既存タスクの入力部分空間と直交する方向に射影する。  
これにより、モデル内には物理的に分離された「タスク専用の計算パス」が動的に形成される。タスクAとタスクBは同じモデル（筐体）の中に同居しているが、ルーターによる交通整理のおかげで互いに遭遇することはない。これは、クラウドコンピューティングにおける「マルチテナント分離」をニューラルネットワーク内で実現していると言える。

#### **Expert Choice Routingによる負荷分散**

従来の「トークンがエキスパートを選ぶ（Token Choice）」方式では、特定の有能なエキスパートに負荷が集中し、多重化の効率（リソース利用率）が低下する問題があった。これに対し、\*\*「Expert Choice（エキスパートがトークンを選ぶ）」\*\*方式 13 は、各エキスパートが処理能力の上限（バッファサイズ）まで、自分に適したトークンを上位 $k$ 個選択する。  
これにより、すべてのエキスパートが常にフル稼働する状態が保たれ、モデル全体の情報処理容量（スループット）が最大化される。これは、マルチプレクシングにおける「帯域幅の最適化」に相当する技術である。

### **4.2 C-LoRAとDynamic Routing**

C-LoRA (Continual LoRA) 12 は、LoRAとMoEの概念を融合させた技術である。学習可能なルーティング行列 $\\mathcal{R}$ を導入し、入力データに応じてパラメータ更新の適用先を動的に切り替える。  
ここでも直交性が鍵となる。ルーティング行列は、過去のタスクに関連するパラメータ成分と、新規タスクの成分が直交するように設計されており、干渉を最小化しつつ、過去の知識（共有可能な低ランク基底）の再利用を促進する。これにより、タスクごとに独立したアダプタを用意する従来の非効率な方法（パラメータ数がタスク数に比例して増大）を回避し、単一の統合モデル内でのスケーラブルな多重化を実現している。

## ---

**5\. 表現の分離（Disentanglement）による多重化の純化**

マルチプレクシングを成功させるための前処理として、情報を「混ぜ合わせても大丈夫な純粋な成分」に分解しておくことが極めて有効である。これが\*\*Disentangled Representation Learning（もつれのない表現学習）\*\*の役割である。

### **5.1 $\\beta$\-VAEと独立因子分解**

$\\beta$-VAE 14 は、教師なし学習において、データの生成因子（例：色、形、位置）を互いに独立した潜在変数にマッピングする。  
通常のVAEの損失関数（ELBO）に対し、KLダイバージェンス項に係数 $\\beta \> 1$ を掛けることで、潜在変数 $z$ の各次元間の相互情報量を最小化し、統計的な独立性を強制する。  
$$ L\_{\\beta-VAE} \= E\_{q(z|x)}\[\\log p(x|z)\] \- \\beta D\_{KL}(q(z|x) |  
| p(z)) $$  
このようにして得られた「もつれのない表現」は、各次元が単一の意味を持つため、それらを足し合わせたり（重ね合わせ）、一部を差し替えたりしても、予期せぬ副作用（Entanglementによる干渉）が生じにくい。これは、マルチプレクシングの素材としての品質を高めるプロセスである。

### **5.2 TopDis: 位相的構造の保存**

統計的な独立性だけでなく、データの幾何学的・位相的な構造（トポロジー）を保存することも、多重化のしやすさに寄与する。TopDis (Topological Disentanglement) 18 は、データの多様体上での近傍関係や穴（ホモロジー群）といった構造を、潜在空間においても維持するように正則化を加える。  
トポロジーが保存された空間では、表現の連続性が保証されるため、Task Arithmeticのようなベクトル演算（後述）を行った際にも、表現が意味の通る領域から逸脱しにくくなる。

## ---

**6\. ハイパーディメンショナルコンピューティング（HDC）：理論的極致**

これまでの技術は、従来のディープラーニングの枠組みの中でマルチプレクシングを模索するものであったが、**ハイパーディメンショナルコンピューティング（HDC）**、または\*\*ベクトル記号アーキテクチャ（VSA）\*\*は、計算の公理そのものを「情報の多重化」に置くパラダイムである 20。

### **6.1 明示的な多重化演算：BindingとBundling**

HDCでは、情報を $10,000$ 次元以上の超高次元ベクトル（ハイパーベクトル）で表現する。この空間では、以下の代数演算によって情報の結合・分離が定義される。

#### **Binding（結合演算, $\\otimes$）**

2つのベクトル（例：変数ラベル $x$ と値 $a$）を結びつけ、新しいベクトルを生成する。

$$V\_{bound} \= V\_x \\otimes V\_a$$

通常、要素ごとの排他的論理和（XOR）や巡回畳み込みが用いられる。重要な性質として、生成された $V\_{bound}$ は元の $V\_x$ や $V\_a$ と直交する（似ていない）。これにより、結合された情報は元の成分とは異なる新しい概念として扱われる。

#### **Bundling（束ね演算/重ね合わせ, $\\oplus$）**

複数のベクトルを足し合わせ、1つのベクトルに多重化する。

$$V\_{set} \= V\_1 \\oplus V\_2 \\oplus \\dots \\oplus V\_n$$

通常、要素ごとの加算（と正規化）が用いられる。重要な性質として、生成された $V\_{set}$ は構成要素 $V\_i$ との類似性（高い内積）を保つ。これにより、多重化されたベクトルの中から、特定の要素が含まれているかを検索（クエリ）することが可能になる。

#### **記憶の多重化と検索**

これらを組み合わせることで、キー・バリュー形式の構造化データを1本のベクトルに多重化できる。

$$V\_{memory} \= (K\_1 \\otimes V\_1) \\oplus (K\_2 \\otimes V\_2) \\oplus (K\_3 \\otimes V\_3)$$

この $V\_{memory}$ から $V\_2$ を取り出すには、逆演算（$K\_2$ とのBinding）を行う。

$$V\_{out} \= V\_{memory} \\otimes K\_2^{-1} \\approx V\_2 \+ \\text{Noise}$$

高次元空間の準直交性により、他項からの干渉（クロスクトークノイズ）は分散され、高い確率で正しい $V\_2$ が復元される。

### **6.2 エッジAIとニューロモルフィックへの実装**

HDCの演算は、複雑な浮動小数点計算ではなく、単純なビット演算や並列加算で行えるため、ハードウェア実装時のエネルギー効率が極めて高い。特に、インメモリコンピューティング（In-Memory Computing）やニューロモルフィックチップ上での実装に適している 26。  
これにより、ドローンやウェアラブルデバイスなどの電力制約が厳しいエッジ環境において、複数のセンサー情報（視覚、聴覚、加速度など）を単一のハイパーベクトルに多重化し、低遅延で学習・推論を行うシステムが構築可能となる。HDCは、AIモデルの「推論への採用しやすさ」をハードウェアレベルで加速する技術である。

## ---

**7\. Task Arithmeticとモデルマージング**

学習後のモデルパラメータを直接操作して機能を多重化する\*\*Task Arithmetic（タスク算術）\*\*は、再学習コストゼロでマルチプレクシングを実現する実用的な技術である 28。

### **7.1 タスクベクトルによる機能の加法混色**

タスクベクトル $\\tau\_t$ は、タスク $t$ で微調整されたモデルの重み $\\theta\_t$ と、ベースモデルの重み $\\theta\_{pre}$ の差分として定義される。

$$\\tau\_t \= \\theta\_t \- \\theta\_{pre}$$

驚くべきことに、これらのタスクベクトルを線形結合してベースモデルに加算するだけで、複数のタスク能力を同時に持つモデルを合成できる。

$$\\theta\_{multi} \= \\theta\_{pre} \+ \\lambda\_A \\tau\_A \+ \\lambda\_B \\tau\_B$$

これにより、「フランス語翻訳」ベクトルと「Pythonコーディング」ベクトルを足し合わせて「フランス語でコード解説ができるモデル」を作るといった運用が可能になる。

### **7.2 線形モード接続性と接空間微調整**

この単純な加算が成立するためには、重み空間において各タスクの解が「線形に接続されている（Linear Mode Connectivity）」必要がある。  
研究 31 によると、モデルを\*\*接空間（Tangent Space）\*\*で微調整する、すなわちモデルの出力を重みに対する線形近似（Neural Tangent Kernel領域）として扱うことで、この線形加法性が大幅に向上する。  
また、Task Vector Bases 30 のような技術は、多数のタスクベクトルを少数の直交基底に圧縮・分解し、その組み合わせとして管理することで、大規模なタスクライブラリの効率的な運用（デマルチプレクシング）を支援する。

## ---

**8\. 解釈性と安全性のためのデマルチプレクシング：Sparse Autoencoders**

マルチプレクシング技術の最大の懸念点は、モデル内部が「混ぜ合わせ」によってブラックボックス化することである。これを解消し、採用の心理的・実務的ハードルを下げるのが**Sparse Autoencoder (SAE)** による解読技術である 1。

### **8.1 仮想ニューロンによる特徴量の抽出**

SAEは、学習済みモデルの中間層の活性化ベクトルを入力とし、それをさらに高次元かつ極めてスパースな特徴空間に写像（エンコード）し、再構成（デコード）するように学習される。

$$f \= \\text{ReLU}(W\_{enc} x \+ b\_{enc})$$

$$\\hat{x} \= W\_{dec} f \+ b\_{dec}$$

ここで抽出された特徴 $f$ の各次元は、モデルがスーパーポジションによって圧縮していた個々の概念（単義的特徴）に対応する。これらは「仮想ニューロン」とも呼ばれ、物理的なニューロンには現れない詳細な概念（例：「DNAの二重螺旋構造」や「特定のプログラミング構文」）をピンポイントで特定できる。

### **8.2 安全なマルチプレクシングのための監視**

SAEを用いることで、開発者は「モデルが今、どの概念を多重化して処理しているか」をリアルタイムで監視できる。もし「差別的なバイアス」や「欺瞞的な意図」といった危険な特徴量が活性化していれば、その成分だけを特定して抑制（Clamp）することが可能になる。  
この「デマルチプレクシング」技術の存在は、企業がコンプライアンスを遵守しつつ、高効率なマルチプレクシングモデルを採用するための強力な担保となる。

## ---

**9\. 結論と実装ガイドライン**

本調査により、AIにおけるマルチプレクシングは、単なる理論的可能性を超え、O-LoRAやMoE、HDCといった具体的な技術によって実装可能なフェーズに入っていることが確認された。これらの技術は、互いに排他的ではなく補完的である。

### **実装のための推奨戦略**

1. **学習の安定化**: まず**O-LoRA**や**GOLA**、**直交初期化**を導入し、パラメータ空間での干渉を物理的に阻止する。これにより、タスク追加時の破滅的忘却を防ぐ。  
2. **表現の純化**: データの前処理や事前学習段階で\*\*$\\beta$-VAE**や**TopDis\*\*の損失関数を採用し、特徴量の直交性を高めておく。  
3. **動的・事後的な多重化**: 推論時には**MoE**の動的ルーティングや**Task Arithmetic**によるアダプタ合成を活用し、コンテキストに応じた最適な計算パスと機能セットを提供する。  
4. **透明性の確保**: **SAE**をデバッグツールとして常備し、多重化された内部表現を定期的に監査することで、ブラックボックス化のリスクを管理する。

これらの技術スタックを統合することで、次世代のAIシステムは、人間の脳のように限られたリソースで無限に近いタスクと概念を操る「高密度な知能」へと進化するだろう。

### ---

**データ・技術対照表**

| カテゴリ | 技術名 | 主な機能・メカニズム | マルチプレクシングへの貢献 | 参照ID |
| :---- | :---- | :---- | :---- | :---- |
| **学習制御** | **O-LoRA** | パラメータ更新の直交化 | タスク間の干渉排除、継続学習の実現 | 5 |
|  | **GOLA** | ランクのグループ化と直交制約 | マルチモーダル情報の分離と効率化 | 7 |
|  | **Spectral Reg.** | 特異値分布の制御 | 学習可塑性の維持、ヌル空間の確保 | 8 |
| **表現学習** | **$\\beta$-VAE** | 潜在変数の独立化 | 特徴量のもつれ解消（素材の純化） | 14 |
|  | **TopDis** | トポロジー保存正則化 | 幾何学的構造の維持、演算耐性の向上 | 18 |
| **アーキテクチャ** | **MoE** | Orthogonal Routing / Expert Choice | 空間分割多重、リソース利用率の最大化 | 11 |
|  | **Dynamic LoRA** | アダプタの動的選択・合成 | 時分割多重、コンテキスト適応 | 35 |
| **演算理論** | **Task Arithmetic** | タスクベクトルの加算・減算 | 機能の事後的な合成（マージ） | 28 |
|  | **HDC/VSA** | Binding / Bundling演算 | 明示的・可逆的な情報の多重化 | 20 |
| **解釈性** | **SAE** | スパース特徴抽出 | 内部表現の解読（デマルチプレクシング） | 1 |

#### **引用文献**

1. Superposition as Lossy Compression – Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability \- OpenReview, 1月 2, 2026にアクセス、 [https://openreview.net/pdf/52771bf2fbb9d8e574634fdc3d8d42a338e055bc.pdf](https://openreview.net/pdf/52771bf2fbb9d8e574634fdc3d8d42a338e055bc.pdf)  
2. Polysemanticity and Capacity in Neural Networks \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2210.01892v4](https://arxiv.org/html/2210.01892v4)  
3. Mechanistic Interpretability: A Survey | by Shav Vimalendiran \- Medium, 1月 2, 2026にアクセス、 [https://medium.com/@shavtge/mechanistic-interpretability-a-survey-c7b8c5411767](https://medium.com/@shavtge/mechanistic-interpretability-a-survey-c7b8c5411767)  
4. The Geometry of Intelligence: Unpacking Superposition, Polysemanticity, and the Architecture of Sparse Autoencoders in Large Language Models | Uplatz Blog, 1月 2, 2026にアクセス、 [https://uplatz.com/blog/the-geometry-of-intelligence-unpacking-superposition-polysemanticity-and-the-architecture-of-sparse-autoencoders-in-large-language-models/](https://uplatz.com/blog/the-geometry-of-intelligence-unpacking-superposition-polysemanticity-and-the-architecture-of-sparse-autoencoders-in-large-language-models/)  
5. Orthogonal Subspace Learning for Language Model Continual Learning \- OpenReview, 1月 2, 2026にアクセス、 [https://openreview.net/forum?id=L7ZBpZZ8Va](https://openreview.net/forum?id=L7ZBpZZ8Va)  
6. \[2509.21433\] DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/abs/2509.21433](https://arxiv.org/abs/2509.21433)  
7. Group Orthogonal Low-Rank Adaptation for RGB-T Tracking \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2512.05359v1](https://arxiv.org/html/2512.05359v1)  
8. LEARNING CONTINUALLY BY SPECTRAL REGULARIZATION \- ICLR Proceedings, 1月 2, 2026にアクセス、 [https://proceedings.iclr.cc/paper\_files/paper/2025/file/5565ab682d6c7f8d9da34ba0919974b0-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/5565ab682d6c7f8d9da34ba0919974b0-Paper-Conference.pdf)  
9. Learning Continually by Spectral Regularization \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2406.06811v2](https://arxiv.org/html/2406.06811v2)  
10. Dynamics of learning when learning dynamics using neural networks \- OpenReview, 1月 2, 2026にアクセス、 [https://openreview.net/forum?id=T65jHpSX7i](https://openreview.net/forum?id=T65jHpSX7i)  
11. Training Consistent Mixture-of-Experts-Based Prompt Generator for Continual Learning, 1月 2, 2026にアクセス、 [https://ojs.aaai.org/index.php/AAAI/article/view/34108/36263](https://ojs.aaai.org/index.php/AAAI/article/view/34108/36263)  
12. C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2502.17920v1](https://arxiv.org/html/2502.17920v1)  
13. Mixture-of-Experts with Expert Choice Routing \- NeurIPS, 1月 2, 2026にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf)  
14. Learning and Evaluating Deep Generative Models for Disentanglement （Disentanglement のための深層生成モデルの学習, 1月 2, 2026にアクセス、 [https://repository.dl.itc.u-tokyo.ac.jp/record/2011071/files/A39897.pdf](https://repository.dl.itc.u-tokyo.ac.jp/record/2011071/files/A39897.pdf)  
15. Beta-VAE \- Yukun Chen, 1月 2, 2026にアクセス、 [https://yukunchen113.github.io/project/2019/06/09/Beta-Variational-Autoencoder.html](https://yukunchen113.github.io/project/2019/06/09/Beta-Variational-Autoencoder.html)  
16. β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK, 1月 2, 2026にアクセス、 [https://www.cs.toronto.edu/\~bonner/courses/2022s/csc2547/papers/generative/disentangled-representations/beta-vae,-higgins,-iclr2017.pdf](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/generative/disentangled-representations/beta-vae,-higgins,-iclr2017.pdf)  
17. \[Beta-VAE\] Learning Basic Visual Concepts with a Constrained Variational Framework, 1月 2, 2026にアクセス、 [https://letter-night.tistory.com/180](https://letter-night.tistory.com/180)  
18. Disentanglement Learning via Topology \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2308.12696v4](https://arxiv.org/html/2308.12696v4)  
19. Disentanglement Learning via Topology \- OpenReview, 1月 2, 2026にアクセス、 [https://openreview.net/forum?id=23OEmHVkpq](https://openreview.net/forum?id=23OEmHVkpq)  
20. (PDF) A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations \- ResearchGate, 1月 2, 2026にアクセス、 [https://www.researchgate.net/publication/360721967\_A\_Survey\_on\_Hyperdimensional\_Computing\_aka\_Vector\_Symbolic\_Architectures\_Part\_I\_Models\_and\_Data\_Transformations](https://www.researchgate.net/publication/360721967_A_Survey_on_Hyperdimensional_Computing_aka_Vector_Symbolic_Architectures_Part_I_Models_and_Data_Transformations)  
21. Hyperdimensional Computing Provides a Programming Paradigm for Oscillatory Systems This work was supported by DOE ASCR and BES Microelectronics Threadwork. This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357. \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2312.11783v1](https://arxiv.org/html/2312.11783v1)  
22. On separating long- and short-term memories in hyperdimensional computing \- Frontiers, 1月 2, 2026にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.867568/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.867568/full)  
23. Hyperdimensional computing: A fast, robust, and interpretable paradigm for biological data, 1月 2, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11421772/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11421772/)  
24. Hyperdimensional computing in biomedical sciences: a brief review \- PMC, 1月 2, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12192801/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192801/)  
25. Hyperdimensional Computing as a Framework for Systematic Aggregation of Image Descriptors, 1月 2, 2026にアクセス、 [https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert\_Hyperdimensional\_Computing\_as\_a\_Framework\_for\_Systematic\_Aggregation\_of\_Image\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.pdf)  
26. Near-Memory Computing Architectures and Circuits for Ultra-Low Power Near-Sensor Processors, 1月 2, 2026にアクセス、 [https://www.research-collection.ethz.ch/bitstreams/6cbbb5d2-f91b-4262-be83-ddcc6c5e64d9/download](https://www.research-collection.ethz.ch/bitstreams/6cbbb5d2-f91b-4262-be83-ddcc6c5e64d9/download)  
27. Generic architectures for efficient Hyper-Dimensional Computing \- Berkeley EECS, 1月 2, 2026にアクセス、 [https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-54.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-54.pdf)  
28. Task Arithmetic: Model Editing Paradigm \- Emergent Mind, 1月 2, 2026にアクセス、 [https://www.emergentmind.com/topics/task-arithmetic-ta](https://www.emergentmind.com/topics/task-arithmetic-ta)  
29. Cross-Model Transfer of Task Vectors via Few-Shot Orthogonal Alignment \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2505.12021v1](https://arxiv.org/html/2505.12021v1)  
30. Task Vector Bases: A Unified and Scalable Framework for Compressed Task Arithmetic, 1月 2, 2026にアクセス、 [https://openreview.net/forum?id=tcuaVzKm3e](https://openreview.net/forum?id=tcuaVzKm3e)  
31. Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models \- NeurIPS, 1月 2, 2026にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2023/file/d28077e5ff52034cd35b4aa15320caea-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2023/file/d28077e5ff52034cd35b4aa15320caea-Paper-Conference.pdf)  
32. Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging \- arXiv, 1月 2, 2026にアクセス、 [https://arxiv.org/html/2505.22934v1](https://arxiv.org/html/2505.22934v1)  
33. Transformers Use Causal World Models in Maze-Solving Tasks \- ResearchGate, 1月 2, 2026にアクセス、 [https://www.researchgate.net/publication/387104793\_Transformers\_Use\_Causal\_World\_Models\_in\_Maze-Solving\_Tasks](https://www.researchgate.net/publication/387104793_Transformers_Use_Causal_World_Models_in_Maze-Solving_Tasks)  
34. Personalized Federated Fine-Tuning of Vision Foundation Models for Healthcare, 1月 2, 2026にアクセス、 [https://chatpaper.com/paper/199746](https://chatpaper.com/paper/199746)  
35. Dynamic LoRA Adapters for Efficient Adaptation \- Emergent Mind, 1月 2, 2026にアクセス、 [https://www.emergentmind.com/topics/dynamic-lora-adapters](https://www.emergentmind.com/topics/dynamic-lora-adapters)