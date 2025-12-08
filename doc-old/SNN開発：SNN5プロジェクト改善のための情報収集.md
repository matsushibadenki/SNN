

# **SNNベースAIの精度と効率の飛躍的向上のための戦略的技術レポート（2024-2025 SOTA分析）**

## **1\. エグゼクティブ・サマリーと戦略的インサイト**

### **1.1. 目的とスコープ**

本レポートは、スパイキングニューラルネットワーク（SNN）ベースのAIプロジェクト「SNN5」の精度と効率を向上させるという目的のもと、最新の学術論文および技術情報を包括的に分析し、戦略的な推奨事項を提供します。

要求されたプロジェクト（https://github.com/matsushibadenki/SNN5）のコードレビューは、対象リポジトリへのアクセスが不能であったため（Inaccessible）1、実行できませんでした。この制約に対応するため、本レポートは特定のコードベースの診断からピボット（方向転換）し、SNN5プロジェクトが直面しているであろう一般的な核心的課題（精度の頭打ち、高い推論レイテンシ、期待されるエネルギー効率の未達成）を解決するための、**2024年から2025年にかけての最先端（SOTA）技術の包括的な戦略的プレイブック**を提供することに焦点を当てます。

### **1.2. 2024-2025年のパラダイムシフト：模倣から超越へ**

SNNの研究開発は、従来型人工ニューラルネットワーク（ANN）の性能を単純に模倣する段階から、SNN固有の特性（時間的ダイナミクス、イベント駆動型のスパース性、生物学的メカニズム）を活用し、特定領域においてANNを**超越**する段階へと移行しています 3。

このシフトは、従来のLeaky Integrate-and-Fire (LIF) ニューロンモデルの表現力の限界に対する認識 5、ANN-SNN変換時に発生する誤差の体系的な克服手法の登場 6、そしてSpiking TransformersやSpiking State Space Models (SpikingSSMs) のようなSNNネイティブな先進的アーキテクチャの台頭によって強力に推進されています 7。

### **1.3. 中核的分析：エネルギー効率の神話とその厳格な実現条件**

SNNは、そのイベント駆動型の性質から「本質的にエネルギー効率が高い」と広く認識されています 10。しかし、2024年から2025年のハードウェア実装を考慮した詳細な分析によれば、この認識は**無条件の真実ではありません** 13。

SNNが、最適化されたANN（例：高密度の積和演算（MAC）ユニットを利用）に対して明確なエネルギー効率の優位性を確保するには、**2つの厳格な条件**を満たす必要があります。

1. 極めて高いスパース性 ($s$) の達成:  
   SNNのエネルギー消費は、実行されるシナプス操作（AC）の数に依存し、これはニューロンの活動（スパイク）のスパース性（疎性）に直結します 15。ある分析では、VGG16モデル（タイムステップ $T=6$）において、ニューロンのスパース性レート $s$ が\*\*93%\*\*を超えなければ、ほとんどのデータフローアーキテクチャにおいてANNに対するエネルギー効率の優位性が確保できないと試算されています 16。  
2. 極めて短いタイムステップ ($T$) での動作:  
   SNNのエネルギー消費と推論レイテンシは、シミュレーションのタイムステップ $T$ に比例して増加します。高精度を求めて $T$ を大きく設定する（例：$T=64$ 以上）と、SNNの総計算コストはANNのそれを容易に上回ってしまいます 14。

SNN5プロジェクトが「精度」のみを追求し、その結果としてモデルのスパース性が低下（スパイクが密になる）したり、タイムステップ $T$ が増加したりした場合、プロジェクトの核となる前提（エネルギー効率）が崩壊している可能性があります。精度と効率（高スパース性・低 $T$）は、独立した目標ではなく、**トレードオフの関係にある厳格な制約条件**として同時に最適化されなければなりません。

### **1.4. 主要な戦略的推奨事項（ハイレベル）**

上記の分析に基づき、SNN5プロジェクトの精度と効率を飛躍させるため、以下の4つの戦略的行動を推奨します。

1. **トレーニング戦略の岐路（セクション2）:** プロジェクトの優先順位（純粋な精度 vs. 効率 vs. 生物学的妥当性）に基づき、3つのSOTAトレーニングパラダイム（サロゲート勾配による直接訓練、ANN-SNN変換、バックプロパゲーション・フリー）から最適な経路を再選択します。  
2. **ANN-SNN変換の革新（セクション3）:** もしSNN5が「SOTAの精度」と「超低レイテンシ（例：$T=1$ または $T=2$）」の両立を目指す場合、2025年のICMLなどで発表された「エラー補償学習 (Error Compensation Learning)」6 や「Scale-and-Fire (SFN) ニューロン」18 の導入が最優先課題となります。  
3. **効率の徹底的追求（セクション4）:** 上記1.3の「93%の壁」を達成するため、「時空間（Spatio-Temporal）プルーニング」19 と「膜電位（Membrane Potential）量子化」20 というSNN固有の圧縮技術を導入します。  
4. **アーキテクチャの超越（セクション5）:** もしSNN5がオーディオ処理やロボティクス制御のような**長期時系列タスク**を扱っている場合、主流のSpiking Transformerから脱却し、2025年のAAAIでSOTAを達成した「**Spiking State Space Models (SpikingSSMs)**」8 へのアーキテクチャ移行を強く推奨します。

---

## **2\. SNNトレーニングパラダイムの戦略的選択**

SNNのトレーニングは、その非連続なスパイク生成メカニズムにより、依然として最も困難な課題の一つです 22。精度と効率は、ここで選択するパラダイムに根本的に依存します。

### **2.1. 経路1：直接トレーニング (Direct Training) とサロゲートグラディエント (SG) 法の最適化**

これは、SNNをゼロから直接訓練するアプローチです。

* 基本原理:  
  スパイク生成関数（ヘヴィサイド関数）は微分不可能（勾配が0または無限大）であるため、標準的なバックプロパゲーション（BP）が適用できません。サロゲートグラディエント（SG）法は、BPの逆伝播時のみ、この微分不可能な関数を滑らかな「サロゲート（代理）」関数（例：Sigmoid 24、Atan 25）で置き換え、勾配を近似計算する手法です 3。  
* 標準的手法と課題:  
  SNNは時間的な内部状態（膜電位）を持つため、Recurrent Neural Networks (RNNs) と同様に、Backpropagation Through Time (BPTT) 7 や、それを空間ドメインにも拡張したSpatio-Temporal Backpropagation (STBP) 27 が基本となります。これらの手法の重大な欠点は、タイムステップ $T$ に比例してメモリ消費量と計算時間が増大することです 7。  
* 2024-2025 SOTA (1): 適応型サロゲートグラディエント  
  最新の研究（2025年10月）30 は、これまで固定のハイパーパラメータと見なされてきたサロゲート関数の傾き ($slope$) が、学習ダイナミクスを支配する重要な変数であることを明らかにしました 30。  
  この傾きの設定には明確なトレードオフが存在します。  
  1. **浅い傾き ($Shallower slopes$):** 深い層における勾配の大きさ（$magnitude$）を増加させ、勾配消失問題を緩和します。しかし同時に、計算された勾配が「真の勾配」から乖離し、アライメント（一致度）が低下します 30。  
  2. **急な傾き ($Steeper slopes$):** 勾配のアライメントは高いですが、勾配消失を引き起こしやすくなります。

このトレードオフの最適解は、タスクに依存します。教師あり学習（Supervised Learning）では、これら2つの効果はバランスが取れています 31。しかし、**強化学習（RL）** の設定では、探索の促進が重要となるため、**「浅い傾き」が強く選好される**ことが発見されました 30。ある研究では、傾きを浅く、あるいは適応的にスケジュールするだけで、RLタスクの性能が2.1倍向上しました 33。SNN5がRL（例：ロボティクス制御 30）タスクである場合、これは即時適用可能な知見です。

* 2024-2025 SOTA (2): Jump-Start Reinforcement Learning (JSRL)  
  SNNをRLに適用する際のもう一つの課題は、SNNが内部状態（膜電位）を安定させるために必要な「ウォームアップ期間」です 30。学習初期はSNNのポリシーが未熟なため、十分な長さのシーケンスを経験できず、ウォームアップが完了しません。JSRLは、学習初期に特権情報（privileged information）を持つ非スパイク（ANN）の「指導ポリシー」を使用してSNNの学習をブートストラップし、このウォームアップ問題を解決します 30。

### **2.2. 経路2：ANN-SNN変換 (Conversion) によるSOTA精度の追求**

これは、SNNの直接訓練の難しさを回避し、ANNのSOTAアーキテクチャ（例：Transformer）の恩恵を受けるための最も一般的なアプローチです 7。

* 基本原理:  
  まず、ReLUベースの高性能なANNを訓練します。その後、ReLUの連続的な活性化値を、SNNニューロンの離散的な平均発火率にマッピング（変換）します 26。  
* 中核的課題 (The "Conversion Error"):  
  このアプローチの最大の欠点は、ANNの連続値とSNNの離散スパイクとの間の固有の不一致によって生じる「変換エラー（Conversion Error）」39 です。このエラーが、変換後のSNNの精度低下と、発火「率」を正確に表現するための長いタイムステップ $T$ の要求につながっていました。2024-2025年の研究により、このエラーは主に3つの要素に分解されています 6:  
  1. **クリッピングエラー ($Clipping Error$):** ANNの活性化がSNNの発火しきい値を超えることによる情報の損失 6。  
  2. **量子化エラー ($Quantization Error$):** 連続的な活性化を離散的なスパイク数（時間）で近似することによる誤差 6。  
  3. **不均一性エラー ($Unevenness Error$):** 特に $T$ が小さい場合に、膜電位の残差によって引き起こされる誤差 6。

SNN5が深いアーキテクチャ（ResNet, VGGなど）を採用している場合、層ごとのキャリブレーション 37 では、浅い層からのキャリブレーションエラーが深い層に**累積**するという重大な問題が発生します 44。このため、層ごと（layer-wise）の調整は不十分であり、エラーをニューロンモデル自体（neuron-level）で根本的に解決するアプローチ（セクション3で詳述）が不可欠となります。

### **2.3. 経路3：新地平 \- バックプロパゲーション (BP) フリー学習**

これは、SNNの生物学的妥当性とハードウェア効率を最大限に追求する、最も革新的なアプローチです。

* 基本原理 (Forward-Forward Algorithm):  
  Geoffrey Hintonによって提案された、BPの順伝播と逆伝播のパスを、2つの順伝播パス（「ポジティブ」データ（実データ）と「ネガティブ」データ（対照データ）を使用）に置き換える手法です 45。  
* **利点:**  
  1. **生物学的妥当性:** BPが持つ生物学的な「ありえなさ」（例：重みの対称性、フィードバック経路）を克服します 45。  
  2. **効率:** BPTTのように中間活性を全タイムステップにわたって保存する必要がなく、層ごとにローカライズされた学習が可能です。これにより、計算効率が向上し、ニューロモーフィックハードウェアへの実装に非常に適しています 46。  
* 2024-2025 SOTA (SNNへの適用):  
  FFアルゴリズムはSNNに適用され始めており、「BPフリーSNN」が実現しています 46。arXiv:2502.20411 45 の実験結果は、提案されたFFベースのSNNフレームワークが、MNISTやFashion-MNISTだけでなく、複雑なスパイクデータセットであるSHDにおいても、BPベースのSOTA SNNに匹敵する、あるいはそれを凌駕する精度を達成することを示しています 47。  
* 実装:  
  snnTorchはFFアルゴリズムの公式チュートリアルを公開しており 46、SpikingJellyを使用した実装も報告されています 56。

### **2.4. 表1: SNNトレーニングパラダイムの戦略的比較**

| パラダイム | 基本原理 | 主な利点 | 主な課題 | 2024-2025 SOTAアプローチ |
| :---- | :---- | :---- | :---- | :---- |
| **経路1: 直接トレーニング (SG)** | 代理勾配（SG）を用いたBPTT/STBPによる端から端までの学習。 | 時間ダイナミクスを直接学習可能。生物学的特徴（例：ALIF）を組み込みやすい。 | BPTT/STBPの計算・メモリコスト（$T$ に比例）7。勾配消失。 | 適応型サロゲートグラディエント（RLタスクで特に有効）30。JSRL \[34\]。 |
| **経路2: ANN-SNN変換** | 訓練済みのANN活性化（ReLU）をSNNの発火率にマッピング。 | SOTAのANNアーキテクチャ（例：Transformer）を利用可。高精度を達成しやすい \[35\]。 | 変換エラー（Clipping, Quantization, Unevenness）6。伝統的に高レイテンシ（高 $T$）が必要。 | エラー補償学習（$T=2$）6。Scale-and-Fireニューロン（$T=1$）18。 |
| **経路3: BPフリー (Forward-Forward)** | 2回の順伝播（Positive/Negativeデータ）による層ごとのローカル学習。 | BP不要。BPTTのメモリコストがゼロ \[46\]。生物学的妥当性が高い 45。 | 比較的新しい手法。ネガティブデータの生成方法が課題。 | FFベースSNN (arXiv:2502.20411)。snnTorchで実装可能 \[46, 55\]。 |

---

## **3\. \[高精度・超低レイテンシ\] ANN-SNN変換のSOTAソリューション**

ANN-SNN変換の最大の課題は、「変換エラー」39 であり、これが精度低下と高いタイムステップ $T$ の要求につながっていました。2024-2025年の最新技術は、このエラーをニューロンモデルレベルで根本的に解決し、「高精度」と「超低レイテンシ（$T=1$ または $T=2$）」の両立を可能にしました 6。

### **3.1. SOTAイノベーション 1：エラー補償学習 (Error Compensation Learning) (ICML 2025\)**

* **論文:** "Efficient ANN-SNN Conversion with Error Compensation Learning" (ICML 2025\) 6。  
* **目的:** セクション2.2で述べた3つの変換エラー（クリッピング、量子化、不均一性）を体系的に解決する統一フレームワークを提供します 6。  
* **主要コンポーネント** 6:  
  1. **学習可能なしきい値を持つクリッピング関数:** ANNの活性化分布にクリッピングしきい値を適応させ、情報の損失（クリッピングエラー）を最小化します 6。  
  2. **デュアルしきい値ニューロン ($Dual Threshold Neuron$):** 2つのしきい値を持つことで、量子化エラーを動的に削減します 6。  
  3. **最適化された膜電位の初期化:** $T$ が小さい場合に特に問題となる不均一性エラーを最小限に抑えます 6。  
* **パフォーマンス:** この手法により、ResNet-18アーキテクチャを用いてCIFAR-10データセットで、わずか\*\*$T=2$ タイムステップ**で**94.75%\*\*というSOTAの精度を達成しました 6。これは、SNNがANNとほぼ同等のレイテンシで、競争力のある精度を達成できることを示しています。  
* **実装:** ICML 2025の論文 60。arXivのメタデータ 61 によれば、コードは公開GitHubリポジトリにアップロードされたと記載されていますが、Awesome-SNNリスト 59 にはまだリンクが掲載されていません 36。

### **3.2. SOTAイノベーション 2：Scale-and-Fire (SFN) ニューロン (T=1推論)**

* **論文:** "One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons" (Preprint 2025\) 18。  
* 核となる理論：「時間-空間 等価理論」  
  このアプローチは、ANN-SNN変換に関する根本的なパラダイムシフトを提示します。  
  * **従来:** ANNの活性化は、SNNの**時間的**な平均発火率（$rate$）に等しいとされていました 38。この前提では、発火「率」を正確に表現するために多くのタイムステップ $T$ が必要でした。  
  * **SFNの理論:** 時間的なスパイク積分のプロセス（$T\>1$ で発生）は、**空間的**なマルチしきい値メカニズム（単一のタイムステップ $T=1$ で処理）を通じて**正確に再構築**できる、というものです 18。

これは、SNNがもはや「時間（$T$）を犠牲にして精度を得る」必要がなく、$T=1$（つまりANNと同等）のレイテンシで高精度を達成できる可能性を示しています 18。

* メカニズム:  
  SFNニューロンは、「スケーリングメカニズム」と「適応型発火関数」を組み合わせて、この時間-空間等価性を $T=1$ で実現します 18。  
* 応用:  
  このSFNをベースにした\*\*Spiking Transformer (SFormer)\*\*も開発されており、マルチしきい値設計のオーバーヘッドを軽減しながらTransformerアーキテクチャに適用されています 18。  
* **実装:** コードリポジトリは現時点では特定されていません 59。

### **3.3. SNN5への戦略的意味**

SNN5が現在、$T\>10$ のような長いタイムステップで運用されており、レイテンシとエネルギー消費（セクション1.3参照）に課題を抱えている場合、「エラー補償学習」または「SFN」の導入は、精度を維持または向上させつつ、レイテンシと計算コストを**1桁削減**できる可能性を秘めた、最もインパクトのあるアップデートとなります。

---

## **4\. \[高効率\] 徹底的なモデル圧縮とエネルギー最適化**

セクション1.3で示した通り、SNNの真の効率は「スパース性 ($s$)」と「タイムステップ ($T$)」に依存します 14。ANNの圧縮（重みプルーニング）と異なり、SNNのエネルギー効率（SynOps: シナプス操作数）は、動的な**スパイク活動**に依存するため 68、SNN固有の圧縮戦略が必要です 13。

### **4.1. テクニック 1：SNN固有のプルーニング（Pruning）**

* ANNプルーニングとの違い:  
  ANNのプルーニングは主に空間的な重み（パラメータ）を削減します 19。SNNはこれに加え、\*\*時間的な冗長性（不要なタイムステップ）\*\*も削減対象となります 19。  
* **SOTA (2025): 時空間プルーニング ($Spatio-Temporal Pruning$)**  
  * **論文:** "Dynamic spatio-temporal pruning for efficient spiking neural networks" (Frontiers 2025\) 19。  
  * **メカニズム:** 2つの要素を動的に削減します 19:  
    1. **適応的・時間プルーニング ($Temporal$):** ニューロン出力のKLダイバージェンス（情報量）を監視し、情報の蓄積が飽和した「冗長なタイムステップ」を適応的に削除します 19。  
    2. **LAMPSベース・空間プルーニング ($Spatial$):** レイヤー間の重み分布に基づき、重要な接続を保護しながら（例：既に疎な層は過度にプルーニングしない）、バランスの取れた層ごとのプルーニングを行います 19。  
  * **結果:** パラメータ空間を最大98%削減しつつ、精度低下を最小限に抑えます 19。  
  * **実装:** このアルゴリズムは、**SpikingJelly**フレームワーク上で実装されています 19。

### **4.2. テクニック 2：SNN固有の量子化（Quantization）**

* ANN量子化との違い:  
  SNNは活性化（スパイク）が既にバイナリ（1-bit）であるため、活性化の量子化が不要です 70。その代わり、SNNは「膜電位 ($Membrane Potential$)」というANNにない内部状態変数を持ち、これがメモリフットプリントの主要なボトルネックとなります 20。  
* **SOTA (2024): SpQuant-SNN**  
  * **論文:** "SpQuant-SNN: ultra-low precision membrane potential with sparse activations..." (Frontiers 2024\) 20。  
  * **メカニズム:** 膜電位をFP32から超低精度の整数（Integer-only）に量子化します。具体的には、\*\*3値 ($Ternary$) 表現（-1, 0, \+1）\*\*に圧縮します 20。  
* 量子化とスパース化の二重圧縮（Double Compression）:  
  SpQuant-SNNの著者らは、膜電位を3値（-1, 0, \+1）に量子化した後、さらに重要な発見をしました。  
  * **発見:** 負の膜電位（"-1"）をスパイクプロセスから「プルーニング」（式(19) $Umask \= Bool(Q(ut) \< 0)$ によりマスク）しても、**精度低下は最小限**であることを見出しました 20。  
  * **相乗効果:** これは単なる量子化（メモリ削減）ではなく、量子化が**スパース性を（追加コストなしで）向上させる**（計算削減）という、強力な相乗効果（二重圧縮 20）を示しています。SNN5は、膜電位の量子化を導入する際、単にビット幅を減らすだけでなく、この「負の値のプルーニング」を同時にテストすべきです。  
  * **実装:** 論文 81 は https://github.com/Ganpei576/BD-SNN というリポジトリを指していますが、現時点ではアクセス不能です 82。

### **4.3. 圧縮戦略に関する重要な注意点**

プルーニング（スパース化）と量子化は、しばしば直交する（互いに影響しない）技術として個別に適用されがちです。しかし、ICLR 2025で発表された論文 83 は、「スパース性と量子化が**非直交**であることの**初の数学的証明**」を提示しました。

* **危険性:** 量子化を**先**に適用すると、重みの相対的な重要性が破壊され、その後のプルーニングステップで「（量子化前は）重要であった」要素が誤って削除される可能性があります 83。  
* **結論:** SNN5が最高の圧縮効率（例：QP-SNN 59）を求める場合、圧縮の順序は\*\*「1. プルーニング → 2\. 量子化」\*\*でなければなりません。順序を逆にすると、両方の手法の誤差が「複合」し、精度が著しく損なわれるリスクがあります 83。

---

## **5\. \[高精度・複雑なタスク\] 先進的ニューロンモデルとSNNアーキテクチャ**

標準的なLIFニューロンは、計算効率は高いものの、生物学的なニューロンの持つ豊かなダイナミクスを欠いており、特に複雑な時間パターンの学習能力に限界があります 5。SNNの精度を（特に時間的タスクにおいて）ANNに匹敵させる、あるいは凌駕するため、より高度なニューロンモデルへの移行が2024-2025年のトレンドとなっています。

### **5.1. SOTAニューロンモデル（LIFの超越）**

* **1\. Adaptive LIF (ALIF):**  
  * **メカニズム:** 単純なLIFに「動的な（適応型）発火しきい値」を導入します 85。  
  * **利点:** 生物学的な「スパイク周波数適応 (SFA)」85 を模倣でき、ニューロンが最近の活動履歴に基づいて感度を調整できます。これにより、より複雑な時間パターンを捕捉し、計算効率を向上させることができます 85。  
* **2\. Gated LIF (GLIF) (NeurIPS 2022):**  
  * **メカニズム:** 複数の生物学的特徴（異なる時定数やリセットメカニズムなど）を、**学習可能な「ゲーティング係数」** を用いて動的に融合させます 5。  
  * **利点:** ネットワークがタスクに応じてニューロンのダイナミクス自体を学習できるため、ニューロンの不均一性（heterogeneity）と適応性が向上します。CIFAR-100において、既存のSNNのニューロンをGLIFに置き換えるだけで、SOTAの精度 (77.35%) を達成しました 5。  
  * **実装:** https://github.com/Ikarosy/Gated-LIF 5。  
* **3\. Two-Compartment LIF (TC-LIF) (AAAI 2024):**  
  * **メカニズム:** ニューロンを生物学的に精緻な「体細胞 (Somatic)」と「樹状突起 (Dendritic)」の2つの区画（コンパートメント）に分離します 92。  
  * **利点:** この構造は、特に「**長期時系列依存性 ($long-term temporal dependency$)**」の学習を促進するように設計されています 92。これは、標準的なLIFやRNNが苦手とする課題です。理論的にも、誤差勾配を長期間にわたって伝播させる能力が優れていることが示されています 94。  
  * **実装:** https://github.com/ZhangShimin1/TC-LIF 94。

### **5.2. 表2: 先進的ニューロンモデルの機能比較**

| ニューロンモデル | 主なメカニズム | 計算オーバーヘッド | 主な利点（解決する課題） | 論文/リポジトリ |
| :---- | :---- | :---- | :---- | :---- |
| **LIF** (標準) | 膜電位の漏洩と固定しきい値による発火。 | 低 | ベースライン。単純な時間積分。 | (N/A) |
| **ALIF** | 動的な（適応型）発火しきい値 \[88\]。 | 中 | スパイク周波数適応 (SFA) 85。時間パターンの捕捉能力強化 85。 | 85 |
| **GLIF** | **学習可能**なゲートで、複数の生物学的特徴（時定数等）を動的に融合 \[91\]。 | 中〜高 | 高い適応性と不均一性。汎用的な精度向上 5。 | (NeurIPS 2022\) 5 / Gated-LIF |
| **TC-LIF** | 2区画（体細胞・樹状突起）モデル 93。 | 中〜高 | **長期時系列依存性の克服**に特化 92。 | (AAAI 2024\) 94 / TC-LIF |

### **5.3. SOTAアーキテクチャ（シーケンスモデリング）**

* **1\. Spiking Transformers (ViTベース):**  
  * TransformerアーキテクチャをSNNに導入する試みが活発化しています 7。  
  * **課題:** 標準的な自己注意（Self-Attention）メカニズムは、SNNのバイナリかつ時空間的なスパイク列と**ミスマッチ**を起こし、空間的関連性や時間的相互作用の把握を妨げます 96。  
  * **SOTA (2025): Saccadic Spike Self-Attention (SSSA)**  
    * **論文:** "Spiking Vision Transformer with Saccadic Attention" (ICLR 2025\) 96。  
    * **着想:** 生物学的な「**サッカード（Saccadic）**」（急速な眼球運動）の注意メカニズム 96。  
    * **メカニズム:** 空間的にはスパイク分布に基づきQuery-Keyの関連性を評価し、時間的には「サッカード相互作用モジュール」が各タイムステップで特定の視覚領域に動的に焦点を当てます 96。  
    * **利点:** SNNの時空間特性に適合し、線形の計算複雑性 ($linear computational complexity$) でSOTAの性能を達成します 96。  
* **2\. Spiking State Space Models (SpikingSSMs):**  
  * 長期シーケンスでは、SpikingSSMsがSpiking Transformersを凌駕する:  
    ANNの世界では、Transformer（特に自己注意）の二乗計算量が課題となり、より効率的なState Space Models (SSMs) (例: Mamba) が台頭しています。このトレンドはSNNにも波及しています。  
    * **発見:** 2025年のAAAI論文 8 は、「**SSMベースのSNN**が、確立された**長期シーケンスモデリングベンチマーク（Long-Range Arena）の全てのタスクでTransformerを凌駕**できる」ことを初めて体系的に示しました 8。  
    * **背景:** SSMは、SNNの基本素子であるLIFニューロンと同様の再帰的な計算プリミティブを持っており、親和性が非常に高いことが示されています 8。  
    * **結論:** SNN5のタスクがNLP、オーディオ処理、長期の制御タスクなど、**長い依存関係**を必要とする場合、Spiking Transformerへの投資は最適解ではない可能性があります。次世代アーキテクチャとして、SpikingSSMs 21 への移行を直接検討すべきです。  
    * **実装 (SpikingSSMs):** https://github.com/shenshuaijie/SDN 59。

---

## **6\. ベンチマーク、フレームワーク、および実装ロードマップ**

### **6.1. SOTAパフォーマンス分析 (ニューロモーフィックデータセット)**

SNNの性能評価は、静的データセット（CIFAR 7）と、時間的情報が豊富なニューロモーフィックデータセット（N-MNIST, DVS Gesture, SHD）で行われます 7。

* ベンチマークの「時間的難易度」の罠:  
  SNNは時間処理が得意とされますが、驚くべきことに、一般的なベンチマークがその能力を正しく測れていない可能性が指摘されています。  
  * **発見:** 「N-MNIST」や「DvsGesture」データセットにおいて、時間的モデリングを行わない（NoTD）手法が、時間的モデリングを行うSTBPなどと同等かそれ以上の性能を達成したと報告されています 86。  
  * **結論:** 86は、「DvsGestureデータセットは、モデルが時間処理能力を持つことを**要求しない**」と結論付けています。

もしSNN5がDVS Gesture 108 でモデルを評価している場合、そこで得られた精度の向上は、真の時間処理能力の向上ではなく、単なる空間的特徴抽出の改善である可能性があります。SNNの真の時間処理能力を測るには、**SHD (Spiking Heidelberg Digits)** 109 やSSC (Spiking Speech Commands) 114 といった、より複雑な時間的依存性を持つオーディオデータセット、あるいはLong-Range Arena 8 へのベンチマーク切り替えを推奨します。

* **DVS Gesture & SHD SOTA (2024-2025): TSkips**  
  * **メカニズム:** SNNアーキテクチャに、明示的な「時間的遅延」を持つスキップ接続（順方向・逆方向）を導入する手法（TSkips）が2025年に提案されました 115。  
  * **結果 (SHD):** SHDデータセットにおいて、ベースライン（84.32%）からTSkips（BTSkips \- 1）導入で\*\*93.64%\*\*へと飛躍的に精度が向上しています 117。  
  * **結果 (DVS Gesture):** DVS128 Gestureにおいても8%の精度向上を達成し 114、SOTAのSpiking Transformerよりも少ないパラメータでこれを達成したと報告されています 114。

### **6.2. 表3: SOTAパフォーマンスベンチマーク (ニューロモーフィックデータセット 2024-2025)**

| データセット | モデル/手法 | パラメータ(M) | 精度 (%) | 主な特徴（論文） |
| :---- | :---- | :---- | :---- | :---- |
| **SHD** | **TSkips (BTSkips \- 1\)** | (N/A) | **93.64** | 時間的遅延を持つスキップ接続 \[115, 117\] |
| SHD | TSkips (FTSkips \- 1\) | (N/A) | 92.32 | 時間的遅延を持つスキップ接続 \[115, 117\] |
| SHD | Baseline (for TSkips) | (N/A) | 84.32 | 117 |
| SHD | Adaptive Skip Recurrent | (N/A) | 81.93 | 適応型スキップ接続 (Xu et al., 2025\) 111 |
| SHD | SE-adLIF | (N/A) | 80.4±0.3 | 適応型LIFニューロン (Baronig et al., 2025\) 111 |
| **DVS Gesture** | **TSkips** | (N/A) | 8%向上 (Baseline比) | 時間的遅延を持つスキップ接続 114。Spiking Transformerより少ないパラメータでSOTA 114。 |

### **6.3. フレームワークの選定：snnTorch vs. SpikingJelly**

どちらもPyTorchベースの主要なSNNフレームワークです 7。

* **SpikingJelly** 7:  
  * **利点（速度）:** **カスタムCUDAカーネル**（CuPyバックエンド）を利用しており、非常に高速なシミュレーションが可能です 121。トレーニングを最大11倍高速化できると報告されています 118。  
  * **欠点（柔軟性）:** カスタムCUDAカーネルは特定のニューロンモデル（例：LIF）用に最適化されており、新しいカスタムニューロン（例：TC-LIF）の定義・高速化が困難です 121。  
  * **適性:** SOTA論文（例：時空間プルーニング 19）の**再現**や、標準LIFベースの**高速な本番環境**に適しています。  
* **snnTorch** 7:  
  * **利点（柔軟性）:** PyTorchの機能を純粋に拡張しており、低レベルコードを追加していません 121。これにより、PyTorchネイティブな方法で**カスタムニューロンを非常に柔軟に定義**できます 121。  
  * **利点（速度のキャッチアップ）:** PyTorch 2.0のtorch.compile（JITコンパイラ）を使用することで、カスタムCUDAカーネルに匹敵する速度向上が見込めるようになりました 121。  
  * **適性:** 新しいニューロンモデル（GLIF, TC-LIF）や新しい学習ルール（FFアルゴリズム 53）の**研究開発**に最適です。  
* JITコンパイラ（torch.compile）と適応型モデルの危険な相互作用:  
  snnTorchのtorch.compileによる速度向上は魅力的ですが、124は重大な実装上の落とし穴を指摘しています。  
  * **問題:** 「適応型サロゲートグラディエント」のように、メンバー変数（例：self.k）として適応的なパラメータを持つPython**クラス**として実装されたニューロンは、JITコンパイル時にその変数が**定数値（例：$k=2$）としてフリーズ**してしまう可能性があります 124。  
  * **結論:** snnTorchで適応型モデル（GLIF, TC-LIF, 適応型SG）を実装し、torch.compileで高速化を図る場合、適応型パラメータがコンパイル時に「固まって」しまい、適応性が失われる（＝学習が失敗する）という、検出が困難なバグに細心の注意を払う必要があります。

### **6.4. 表4: SOTA実装のための主要GitHubリポジトリ (2024-2025)**

| SOTA概念 | 主要論文（会議/年） | GitHubリポジトリ | 関連スニペット |
| :---- | :---- | :---- | :---- |
| **SpikingSSMs** (長期シーケンス) | SpikingSSMs: Learning Long Sequences... (AAAI 2025\) | https://github.com/shenshuaijie/SDN | 59 |
| **TC-LIF** (長期依存性ニューロン) | TC-LIF: A Two-Compartment Spiking... (AAAI 2024\) | https://github.com/ZhangShimin1/TC-LIF | 94 |
| **GLIF** (適応型ニューロン) | GLIF, a unified spiking neuron... (NeurIPS 2022\) | https://github.com/Ikarosy/Gated-LIF | 5 |
| **SGLFormer** (Spiking Transformer) | SGLFormer: Spiking Global-Local-Fusion... (Preprint) | https://github.com/ZhangHanN1/SGLFormer | \[95\] |
| **FF in SNN** (BPフリー学習) | Backpropagation-free Spiking Neural... (arXiv 2025\) | snnTorchチュートリアル 55 を参照 | \[45, 46, 55\] |
| **SpQuant-SNN** (膜電位量子化) | SpQuant-SNN: ultra-low precision... (Frontiers 2024\) | https://github.com/Ganpei576/BD-SNN (※アクセス注意) | 81 |
| **Error Comp. Learning** ($T=2$ 変換) | Efficient ANN-SNN Conversion with... (ICML 2025\) | \[61\] | \[59, 61\] |
| **Spatio-Temporal Pruning** | Dynamic spatio-temporal pruning... (Frontiers 2025\) | SpikingJelly 19 ベース。要検索。 | 19 |
| **S4NN** (ランクオーダー学習) | Temporal Backpropagation for SNNs... (ResearchGate 2019\) | https://github.com/SRKH/S4NN | \[126\] |

---

## **7\. プロジェクト「SNN5」への戦略的推奨事項**

収集されたSOTA情報に基づき、SNN5プロジェクトの「精度」と「効率」を最大化するための、具体的かつ優先順位付けされたアクションプランを提案します。

### **7.1. フェーズ1：診断と即時的改善（Quick Wins）**

1. **\[効率\] エネルギー効率の現実の直視（1.3に基づく）:**  
   * **アクション:** SNN5の現行モデルの「平均スパース性 ($s$)」と「タイムステップ ($T$)」を計測します。  
   * **判定:** スパース性が93%を（大幅に）下回るか、$T$ が（例：$T\>16$）と長い場合、SNN5は最適化されたANNよりも多くのエネルギーを消費している危険性があります 14。  
   * **対策:** 損失関数にスパイク活動を抑制する正則化項を追加し、スパース性を強制的に高めます 16。  
2. **\[精度\] ベンチマークの再評価（6.1に基づく）:**  
   * **アクション:** 現在の主要ベンチマークが「DVS Gesture」や「N-MNIST」であるか確認します。  
   * **判定:** もしそうであれば、SNN5は「時間処理」能力を過大評価している可能性があります。これらのデータセットは時間処理能力を必須としないことが示唆されています 86。  
   * **対策:** プロジェクトの目標が真の時間処理能力（オーディオ、制御、長期予測）である場合、ベンチマークを「**SHD**」111、「SSC」114、または「Long-Range Arena」8 に移行します。  
3. **\[効率\] 圧縮順序の確認（4.3に基づく）:**  
   * **アクション:** プルーニングと量子化の両方を使用している場合、その適用順序を確認します。  
   * **対策:** 精度低下を防ぐため、順序は必ず\*\*「1. プルーニング → 2\. 量子化」\*\*とします 83。

### **7.2. フェーズ2：アーキテクチャの抜本的刷新（Mid-Term）**

4. **\[戦略\] トレーニングパラダイムの選択（セクション2に基づく）:**  
   * **選択A（精度・レイテンシ最優先）:** **ANN-SNN変換**を採用します。セクション3で詳述した「**エラー補償学習**」6 または「**SFN**」18 を実装し、$T=1$ または $T=2$ でのSOTA精度を目指します。  
   * **選択B（時間処理・妥当性優先）:** **直接トレーニング**を継続または採用します。  
5. **\[精度\] ニューロンモデルのアップグレード（選択Bの場合）（5.1に基づく）:**  
   * **タスクが長期時系列（オーディオ、制御等）の場合:** 標準LIFを「**TC-LIF**」94 に置き換えます。これは長期依存性 93 のために設計されています。  
   * **タスクが上記以外（画像分類等）の場合:** 標準LIFを「**GLIF**」5 に置き換えます。これは学習可能なダイナミクスにより、汎用的な精度向上（CIFAR-100 SOTA）が報告されています 5。  
6. **\[効率\] 「二重圧縮」の実装（4.2に基づく）:**  
   * **アクション1:** SpikingJellyベースの「**時空間プルーニング**」を導入し、空間的（重み）および時間的（タイムステップ）な冗長性を削減します 19。  
   * **アクション2:** 「**SpQuant-SNN**」を参考に、膜電位を3値（-1, 0, \+1）に量子化し、さらに「-1」の値をプルーニングして、メモリと計算の両方を削減します 20。

### **7.3. フェーズ3：次世代アーキテクチャへの移行（Long-Term）**

7. **\[アーキテクチャ\] 長期シーケンス課題への対応（5.3に基づく）:**  
   * **アクション:** SNN5のタスクが長いシーケンス（数千ステップ）を扱う場合、Spiking Transformerの構築は**推奨しません**。  
   * **対策:** ANN領域でのSSMの台頭に倣い、Transformerを飛び越して「**SpikingSSMs**」（AAAI 2025 SOTA）8 のアーキテクチャ（リポジトリ: 59）を直接採用することを強く推奨します。  
8. **\[アーキテクチャ\] Vision Transformer課題への対応:**  
   * **アクション:** SNN5がVision Transformerベースである場合、標準の自己注意メカニズムを、SNNの時空間特性に適合させた「**Saccadic Spike Self-Attention (SSSA)**」97 に置き換えます。  
9. **\[戦略\] BPフリーへの移行（2.3に基づく）:**  
   * **アクション:** ハードウェアの制約が最も厳しい（例：オンチップ学習、極低メモリ）場合、snnTorch 55 を使用して「**Forward-Forward (FF) アルゴリズム**」の導入を検討します。これはBPTTのメモリコストを完全に排除します 45。

#### **引用文献**

1. 1月 1, 1970にアクセス、 [https://github.com/matsushibadenki/SNN5](https://github.com/matsushibadenki/SNN5)  
2. 1月 1, 1970にアクセス、 [https://github.com/matsushibadenki/SNN5/issues](https://github.com/matsushibadenki/SNN5/issues)  
3. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2507.16043v2](https://arxiv.org/html/2507.16043v2)  
4. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2507.16043v1](https://arxiv.org/html/2507.16043v1)  
5. GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks, 11月 2, 2025にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2022/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf)  
6. \[2506.01968\] Efficient ANN-SNN Conversion with Error Compensation Learning \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/abs/2506.01968](https://arxiv.org/abs/2506.01968)  
7. Direct training high-performance deep spiking neural networks: a review of theories and methods \- PMC \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)  
8. Learning Long Sequences in Spiking Neural Networks with Matei Stan, 11月 2, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/student-talks/learning-long-sequences-in-snns/](https://open-neuromorphic.org/neuromorphic-computing/student-talks/learning-long-sequences-in-snns/)  
9. SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2408.14909v1](https://arxiv.org/html/2408.14909v1)  
10. Reducing the spike rate of deep spiking neural networks based on time-encoding \- DOAJ, 11月 2, 2025にアクセス、 [https://doaj.org/article/1d3dd57b75cd4e649208dee63e065fb3](https://doaj.org/article/1d3dd57b75cd4e649208dee63e065fb3)  
11. Linear leaky-integrate-and-fire neuron model based spiking neural networks and its mapping relationship to deep neural networks \- PMC \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/)  
12. Deep Learning With Spiking Neurons: Opportunities and Challenges \- Frontiers, 11月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00774/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00774/full)  
13. Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework | OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/forum?id=eoSeaK4QJo](https://openreview.net/forum?id=eoSeaK4QJo)  
14. arXiv:2409.08290v1 \[cs.NE\] 29 Aug 2024, 11月 2, 2025にアクセス、 [https://arxiv.org/pdf/2409.08290?](https://arxiv.org/pdf/2409.08290)  
15. TOWARDS ENERGY EFFICIENT SPIKING NEURAL NET- WORKS: AN UNSTRUCTURED PRUNING FRAMEWORK \- ICLR Proceedings, 11月 2, 2025にアクセス、 [https://proceedings.iclr.cc/paper\_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf)  
16. Reconsidering the energy efficiency of spiking neural networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2409.08290v1](https://arxiv.org/html/2409.08290v1)  
17. Reconsidering the energy efficiency of spiking neural networks \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/384057650\_Reconsidering\_the\_energy\_efficiency\_of\_spiking\_neural\_networks](https://www.researchgate.net/publication/384057650_Reconsidering_the_energy_efficiency_of_spiking_neural_networks)  
18. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2510.23383v1](https://arxiv.org/html/2510.23383v1)  
19. Dynamic spatio-temporal pruning for efficient spiking neural networks \- Frontiers, 11月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full)  
20. SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks applications \- Frontiers, 11月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full)  
21. Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling \- OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/pdf/c9cf2a54177f6c93e5a6537cd87e028e699ff4aa.pdf](https://openreview.net/pdf/c9cf2a54177f6c93e5a6537cd87e028e699ff4aa.pdf)  
22. Spiking Neural Networks: The Next Evolution in AI | by AI\_Pioneer | Nov, 2025 \- Medium, 11月 2, 2025にアクセス、 [https://medium.com/@tejasdalvi927/spiking-neural-networks-the-next-evolution-in-ai-71c77d9a261d](https://medium.com/@tejasdalvi927/spiking-neural-networks-the-next-evolution-in-ai-71c77d9a261d)  
23. To Spike or Not to Spike, that is the Question \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2407.19566v3](https://arxiv.org/html/2407.19566v3)  
24. Spike Function and Surrogate Gradient Function. | Download Scientific Diagram \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/figure/Spike-Function-and-Surrogate-Gradient-Function\_fig3\_379702386](https://www.researchgate.net/figure/Spike-Function-and-Surrogate-Gradient-Function_fig3_379702386)  
25. Dynamic spatio-temporal pruning for efficient spiking neural networks \- PMC, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11975901/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11975901/)  
26. Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure \- arXiv, 11月 2, 2025にアクセス、 [https://www.arxiv.org/pdf/2509.24411](https://www.arxiv.org/pdf/2509.24411)  
27. Enhancing Generalization of Spiking Neural Networks Through Temporal Regularization \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2506.19256v3](https://arxiv.org/html/2506.19256v3)  
28. Surrogate gradient learning in spiking networks trained on event-based cytometry dataset, 11月 2, 2025にアクセス、 [https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260)  
29. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 11月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
30. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2510.24461v1](https://arxiv.org/html/2510.24461v1)  
31. \[Literature Review\] Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- Moonlight | AI Colleague for Research Papers, 11月 2, 2025にアクセス、 [https://www.themoonlight.io/en/review/adaptive-surrogate-gradients-for-sequential-reinforcement-learning-in-spiking-neural-networks](https://www.themoonlight.io/en/review/adaptive-surrogate-gradients-for-sequential-reinforcement-learning-in-spiking-neural-networks)  
32. Computer Science \- arXiv, 11月 2, 2025にアクセス、 [https://www.arxiv.org/list/cs/recent?skip=50\&show=2000](https://www.arxiv.org/list/cs/recent?skip=50&show=2000)  
33. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks | OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/forum?id=oGmROC4e4W](https://openreview.net/forum?id=oGmROC4e4W)  
34. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- ChatPaper, 11月 2, 2025にアクセス、 [https://chatpaper.com/chatpaper/paper/204363](https://chatpaper.com/chatpaper/paper/204363)  
35. Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformer \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2508.07710v1](https://arxiv.org/html/2508.07710v1)  
36. Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/353833695\_Optimal\_ANN-SNN\_Conversion\_for\_Fast\_and\_Accurate\_Inference\_in\_Deep\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/353833695_Optimal_ANN-SNN_Conversion_for_Fast_and_Accurate_Inference_in_Deep_Spiking_Neural_Networks)  
37. Artificial to Spiking Neural Networks Conversion with Calibration in Scientific Machine Learning \- SIAM Publications Library, 11月 2, 2025にアクセス、 [https://epubs.siam.org/doi/10.1137/24M1643232](https://epubs.siam.org/doi/10.1137/24M1643232)  
38. Optimising Event-Driven Spiking Neural Network with Regularisation and Cutoff \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2301.09522v4](https://arxiv.org/html/2301.09522v4)  
39. Efficient ANN-SNN Conversion with Error Compensation Learning \- OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/attachment?id=9lw5HYPT4Y\&name=pdf](https://openreview.net/attachment?id=9lw5HYPT4Y&name=pdf)  
40. Efficient ANN-SNN Conversion with Error Compensation Learning \- ICML 2025, 11月 2, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/46208](https://icml.cc/virtual/2025/poster/46208)  
41. Reducing ANN-SNN Conversion Error through Residual Membrane Potential, 11月 2, 2025にアクセス、 [https://ojs.aaai.org/index.php/AAAI/article/view/25071/24843](https://ojs.aaai.org/index.php/AAAI/article/view/25071/24843)  
42. \[2305.19868\] Fast-SNN: Fast Spiking Neural Network by Converting Quantized ANN \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/abs/2305.19868](https://arxiv.org/abs/2305.19868)  
43. Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks | Proceedings of the AAAI Conference on Artificial Intelligence, 11月 2, 2025にアクセス、 [https://ojs.aaai.org/index.php/AAAI/article/view/32150](https://ojs.aaai.org/index.php/AAAI/article/view/32150)  
44. A Fast and Accurate ANN-SNN Conversion Algorithm with Negative Spikes \- IJCAI, 11月 2, 2025にアクセス、 [https://www.ijcai.org/proceedings/2025/0719.pdf](https://www.ijcai.org/proceedings/2025/0719.pdf)  
45. Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2502.20411v1](https://arxiv.org/html/2502.20411v1)  
46. Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2502.20411v2](https://arxiv.org/html/2502.20411v2)  
47. Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/pdf/2502.20411](https://arxiv.org/pdf/2502.20411)  
48. FFGAF-SNN: The Forward-Forward Based Gradient Approximation Free Training Framework for Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2507.23643v1](https://arxiv.org/html/2507.23643v1)  
49. \[PDF\] Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm, 11月 2, 2025にアクセス、 [https://www.semanticscholar.org/paper/Backpropagation-free-Spiking-Neural-Networks-with-Ghader-Kheradpisheh/3d38e4457edfc971ce0e6bf0981b462c4695f064](https://www.semanticscholar.org/paper/Backpropagation-free-Spiking-Neural-Networks-with-Ghader-Kheradpisheh/3d38e4457edfc971ce0e6bf0981b462c4695f064)  
50. (PDF) Backpropagation-free Spiking Neural Networks with the, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/389510097\_Backpropagation-free\_Spiking\_Neural\_Networks\_with\_the\_Forward-Forward\_Algorithm](https://www.researchgate.net/publication/389510097_Backpropagation-free_Spiking_Neural_Networks_with_the_Forward-Forward_Algorithm)  
51. SpikingChen/SNN-Daily-Arxiv: Update arXiv papers about Spiking Neural Networks daily., 11月 2, 2025にアクセス、 [https://github.com/SpikingChen/SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv)  
52. Tutorials — snntorch 0.9.4 documentation, 11月 2, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/index.html](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)  
53. The Forward-Forward Algorithm with a Spiking Neural Network \- snnTorch \- Read the Docs, 11月 2, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_forward\_forward.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_forward_forward.html)  
54. Tutorial 5 \- Training Spiking Neural Networks with snntorch \- Read the Docs, 11月 2, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_5.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)  
55. Code | Neuromorphic Computing Group \- UC Santa Cruz, 11月 2, 2025にアクセス、 [https://ncg.ucsc.edu/category/code/](https://ncg.ucsc.edu/category/code/)  
56. Sign Gradient Descent-based Neuronal Dynamics: ANN-to-SNN Conversion Beyond ReLU Network \- GitHub, 11月 2, 2025にアクセス、 [https://raw.githubusercontent.com/mlresearch/v235/main/assets/oh24b/oh24b.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/oh24b/oh24b.pdf)  
57. Sign Gradient Descent-based Neuronal Dynamics: ANN-to-SNN Conversion Beyond ReLU Network \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2407.01645v1](https://arxiv.org/html/2407.01645v1)  
58. Efficient ANN-SNN Conversion with Error Compensation Learning \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2506.01968v1](https://arxiv.org/html/2506.01968v1)  
59. zhouchenlin2096/Awesome-Spiking-Neural-Networks: A ... \- GitHub, 11月 2, 2025にアクセス、 [https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks](https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks)  
60. ICML 2025 Schedule, 11月 2, 2025にアクセス、 [https://icml.cc/virtual/2025/calendar](https://icml.cc/virtual/2025/calendar)  
61. Machine Learning Jun 2025 \- arXiv, 11月 2, 2025にアクセス、 [https://www.arxiv.org/list/cs.LG/2025-06?skip=100\&show=2000](https://www.arxiv.org/list/cs.LG/2025-06?skip=100&show=2000)  
62. Artificial Intelligence Jun 2025 \- arXiv, 11月 2, 2025にアクセス、 [https://www.arxiv.org/list/cs.AI/2025-06?skip=875\&show=2000](https://www.arxiv.org/list/cs.AI/2025-06?skip=875&show=2000)  
63. RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/343461203\_RMP-SNN\_Residual\_Membrane\_Potential\_Neuron\_for\_Enabling\_Deeper\_High-Accuracy\_and\_Low-Latency\_Spiking\_Neural\_Network](https://www.researchgate.net/publication/343461203_RMP-SNN_Residual_Membrane_Potential_Neuron_for_Enabling_Deeper_High-Accuracy_and_Low-Latency_Spiking_Neural_Network)  
64. Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 2, 2025にアクセス、 [https://www.arxiv.org/pdf/2510.23383](https://www.arxiv.org/pdf/2510.23383)  
65. Quantization Framework for Fast Spiking Neural Networks \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/362134992\_Quantization\_Framework\_for\_Fast\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/362134992_Quantization_Framework_for_Fast_Spiking_Neural_Networks)  
66. 神经与进化计算/符号计算2025\_10\_28 \- ArXiv Daily, 11月 2, 2025にアクセス、 [http://arxivdaily.com/thread/73187](http://arxivdaily.com/thread/73187)  
67. \[Papierüberprüfung\] Optimal ANN-SNN Conversion with Group, 11月 2, 2025にアクセス、 [https://www.themoonlight.io/de/review/optimal-ann-snn-conversion-with-group-neurons](https://www.themoonlight.io/de/review/optimal-ann-snn-conversion-with-group-neurons)  
68. SPEAR: Structured Pruning for Spiking Neural Networks via Synaptic Operation Estimation and Reinforcement Learning \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/pdf/2507.02945](https://arxiv.org/pdf/2507.02945)  
69. Pruning Everything, Everywhere, All at Once \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/pdf/2506.04513](https://arxiv.org/pdf/2506.04513)  
70. Edge Intelligence with Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2507.14069v1](https://arxiv.org/html/2507.14069v1)  
71. Dynamic spatio-temporal pruning for efficient spiking neural networks \- eScholarship, 11月 2, 2025にアクセス、 [https://escholarship.org/content/qt3xr67678/qt3xr67678.pdf](https://escholarship.org/content/qt3xr67678/qt3xr67678.pdf)  
72. Dynamic spatio-temporal pruning for efficient spiking neural networks \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/390191451\_Dynamic\_spatio-temporal\_pruning\_for\_efficient\_spiking\_neural\_networks](https://www.researchgate.net/publication/390191451_Dynamic_spatio-temporal_pruning_for_efficient_spiking_neural_networks)  
73. SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/384237731\_SpikingResformer\_Bridging\_ResNet\_and\_Vision\_Transformer\_in\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/384237731_SpikingResformer_Bridging_ResNet_and_Vision_Transformer_in_Spiking_Neural_Networks)  
74. Adaptation and learning of spatio-temporal thresholds in spiking neural networks | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/391773687\_Adaptation\_and\_learning\_of\_spatio-temporal\_thresholds\_in\_spiking\_neural\_networks](https://www.researchgate.net/publication/391773687_Adaptation_and_learning_of_spatio-temporal_thresholds_in_spiking_neural_networks)  
75. Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/358846417\_Temporal\_Efficient\_Training\_of\_Spiking\_Neural\_Network\_via\_Gradient\_Re-weighting](https://www.researchgate.net/publication/358846417_Temporal_Efficient_Training_of_Spiking_Neural_Network_via_Gradient_Re-weighting)  
76. Sharing leaky-integrate-and-fire neurons for memory-efficient spiking neural networks \- PMC, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10423932/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10423932/)  
77. Q-SpiNN: A Framework for Quantizing Spiking Neural Networks, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/354712995\_Q-SpiNN\_A\_Framework\_for\_Quantizing\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/354712995_Q-SpiNN_A_Framework_for_Quantizing_Spiking_Neural_Networks)  
78. SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks applications \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11408473/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11408473/)  
79. SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks applications \- Frontiers, 11月 2, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/pdf](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/pdf)  
80. MINT: Multiplier-less INTeger Quantization for Energy Efficient Spiking Neural Networks | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/379281403\_MINT\_Multiplier-less\_INTeger\_Quantization\_for\_Energy\_Efficient\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/379281403_MINT_Multiplier-less_INTeger_Quantization_for_Energy_Efficient_Spiking_Neural_Networks)  
81. Memory cost and FLOPs reduction of proposed SpQuant-SNN with YOLO-V2 on Prophesse Gen 1\. \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/figure/Memory-cost-and-FLOPs-reduction-of-proposed-SpQuant-SNN-with-YOLO-V2-on-Prophesse-Gen-1\_fig1\_384087196](https://www.researchgate.net/figure/Memory-cost-and-FLOPs-reduction-of-proposed-SpQuant-SNN-with-YOLO-V2-on-Prophesse-Gen-1_fig1_384087196)  
82. 1月 1, 1970にアクセス、 [https://github.com/Ganpei576/BD-SNN](https://github.com/Ganpei576/BD-SNN)  
83. Effective Interplay between Sparsity and Quantization: From Theory to Practice, 11月 2, 2025にアクセス、 [https://openreview.net/forum?id=wJv4AIt4sK](https://openreview.net/forum?id=wJv4AIt4sK)  
84. QP-SNNs: Quantized and Pruned Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2502.05905v2](https://arxiv.org/html/2502.05905v2)  
85. Spike frequency adaptation: bridging neural models and neuromorphic applications, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/377914868\_Spike\_frequency\_adaptation\_bridging\_neural\_models\_and\_neuromorphic\_applications](https://www.researchgate.net/publication/377914868_Spike_frequency_adaptation_bridging_neural_models_and_neuromorphic_applications)  
86. Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2502.09449v1](https://arxiv.org/html/2502.09449v1)  
87. Efficient and Accurate Spiking Neural Networks \- TUE Research portal \- Eindhoven University of Technology, 11月 2, 2025にアクセス、 [https://research.tue.nl/files/234498286/20221214\_Yin\_B.\_hf.pdf](https://research.tue.nl/files/234498286/20221214_Yin_B._hf.pdf)  
88. ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network \- OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/pdf?id=KsIhAcB84m](https://openreview.net/pdf?id=KsIhAcB84m)  
89. ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2505.11455v1](https://arxiv.org/html/2505.11455v1)  
90. Spike frequency adaptation: bridging neural models and neuromorphic applications \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11053160/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053160/)  
91. GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/abs/2210.13768](https://arxiv.org/abs/2210.13768)  
92. Exploring temporal information dynamics in Spiking Neural Networks: Fast Temporal Efficient Training | Semantic Scholar, 11月 2, 2025にアクセス、 [https://www.semanticscholar.org/paper/b7a30e3d67cbbe7d32421e0c7df572059b9ed0ed](https://www.semanticscholar.org/paper/b7a30e3d67cbbe7d32421e0c7df572059b9ed0ed)  
93. (PDF) TC-LIF: A Two-Compartment Spiking Neuron Model for Long-term Sequential Modelling \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/373437962\_TC-LIF\_A\_Two-Compartment\_Spiking\_Neuron\_Model\_for\_Long-term\_Sequential\_Modelling](https://www.researchgate.net/publication/373437962_TC-LIF_A_Two-Compartment_Spiking_Neuron_Model_for_Long-term_Sequential_Modelling)  
94. TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling, 11月 2, 2025にアクセス、 [https://ira.lib.polyu.edu.hk/bitstream/10397/109062/1/29625-Article%20Text-33679-1-2-20240324.pdf](https://ira.lib.polyu.edu.hk/bitstream/10397/109062/1/29625-Article%20Text-33679-1-2-20240324.pdf)  
95. Efficient Speech Command Recognition Leveraging Spiking Neural Networks and Progressive Time-scaled Curriculum Distillation \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/396898473\_Efficient\_Speech\_Command\_Recognition\_Leveraging\_Spiking\_Neural\_Networks\_and\_Progressive\_Time-scaled\_Curriculum\_Distillation](https://www.researchgate.net/publication/396898473_Efficient_Speech_Command_Recognition_Leveraging_Spiking_Neural_Networks_and_Progressive_Time-scaled_Curriculum_Distillation)  
96. \[2502.12677\] Spiking Vision Transformer with Saccadic Attention \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/abs/2502.12677](https://arxiv.org/abs/2502.12677)  
97. Spiking Vision Transformer with Saccadic Attention \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2502.12677v1](https://arxiv.org/html/2502.12677v1)  
98. Revision History for Spiking Vision Transformer with... \- OpenReview, 11月 2, 2025にアクセス、 [https://openreview.net/revisions?id=Ykq8a5imXV](https://openreview.net/revisions?id=Ykq8a5imXV)  
99. SPIKING VISION TRANSFORMER WITH SACCADIC ATTENTION, 11月 2, 2025にアクセス、 [https://researchportal.northumbria.ac.uk/files/198173592/7196\_Spiking\_Vision\_Transforme.pdf](https://researchportal.northumbria.ac.uk/files/198173592/7196_Spiking_Vision_Transforme.pdf)  
100. \[Quick Review\] Spiking Vision Transformer with Saccadic Attention, 11月 2, 2025にアクセス、 [https://liner.com/review/spiking-vision-transformer-with-saccadic-attention](https://liner.com/review/spiking-vision-transformer-with-saccadic-attention)  
101. TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/379730019\_TCJA-SNN\_Temporal-Channel\_Joint\_Attention\_for\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/379730019_TCJA-SNN_Temporal-Channel_Joint_Attention_for_Spiking_Neural_Networks)  
102. Luziwei Leng's research works | Huawei Technologies and other places \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/scientific-contributions/Luziwei-Leng-2231869595](https://www.researchgate.net/scientific-contributions/Luziwei-Leng-2231869595)  
103. A LIF-based Legendre Memory Unit as neuromorphic State Space Model benchmarked on a second-long spatio-temporal task | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/393517121\_A\_LIF-based\_Legendre\_Memory\_Unit\_as\_neuromorphic\_State\_Space\_Model\_benchmarked\_on\_a\_second-long\_spatio-temporal\_task](https://www.researchgate.net/publication/393517121_A_LIF-based_Legendre_Memory_Unit_as_neuromorphic_State_Space_Model_benchmarked_on_a_second-long_spatio-temporal_task)  
104. Learning long sequences in spiking neural networks \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/384199502\_Learning\_long\_sequences\_in\_spiking\_neural\_networks](https://www.researchgate.net/publication/384199502_Learning_long_sequences_in_spiking_neural_networks)  
105. Efficient Deep Spiking Multilayer Perceptrons With Multiplication-Free Inference | Request PDF \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/publication/380757492\_Efficient\_Deep\_Spiking\_Multilayer\_Perceptrons\_With\_Multiplication-Free\_Inference](https://www.researchgate.net/publication/380757492_Efficient_Deep_Spiking_Multilayer_Perceptrons_With_Multiplication-Free_Inference)  
106. Gesture-SNN: Co-optimizing accuracy, latency and energy of SNNs for neuromorphic vision sensors \- NSF PAR, 11月 2, 2025にアクセス、 [https://par.nsf.gov/servlets/purl/10326920](https://par.nsf.gov/servlets/purl/10326920)  
107. Biologically inspired heterogeneous learning for accurate, efficient and low-latency neural network | National Science Review | Oxford Academic, 11月 2, 2025にアクセス、 [https://academic.oup.com/nsr/article/12/1/nwae301/7746334](https://academic.oup.com/nsr/article/12/1/nwae301/7746334)  
108. SNN accuracy (%) is benchmarked on the DVS Gesture dataset, with... | Download Scientific Diagram \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/figure/SNN-accuracy-is-benchmarked-on-the-DVS-Gesture-dataset-with-results-averaged-across\_tbl2\_393081789](https://www.researchgate.net/figure/SNN-accuracy-is-benchmarked-on-the-DVS-Gesture-dataset-with-results-averaged-across_tbl2_393081789)  
109. Exploring the Limitations of Layer Synchronization in Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2408.05098v2](https://arxiv.org/html/2408.05098v2)  
110. BrainScale, Enabling Scalable Online Learning in Spiking Neural ..., 11月 2, 2025にアクセス、 [https://www.biorxiv.org/content/10.1101/2024.09.24.614728v3.full-text](https://www.biorxiv.org/content/10.1101/2024.09.24.614728v3.full-text)  
111. Spiking Heidelberg Digits and Spiking Speech Commands \- Zenke Lab, 11月 2, 2025にアクセス、 [https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)  
112. Spiking Heidelberg Digits: Speech Recognition Benchmark \- YouTube, 11月 2, 2025にアクセス、 [https://www.youtube.com/shorts/NKLrjhXdgzc](https://www.youtube.com/shorts/NKLrjhXdgzc)  
113. Learning-efficient spiking neural networks with multi-compartment spatio-temporal backpropagation \- PMC \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12284053/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284053/)  
114. TSkips: Efficiency Through Explicit Temporal Delay Connections in Spiking Neural Networks, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2411.16711v2](https://arxiv.org/html/2411.16711v2)  
115. Weekly TMLR digest for Jul 20, 2025 \- Google Groups, 11月 2, 2025にアクセス、 [https://groups.google.com/g/tmlr-announce-weekly/c/3-YtGJ2hbbo](https://groups.google.com/g/tmlr-announce-weekly/c/3-YtGJ2hbbo)  
116. Illustration of the structure of spiking neurons, comparing the... \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/figure/Illustration-of-the-structure-of-spiking-neurons-comparing-the-traditional-LIF-model-a\_fig1\_384268823](https://www.researchgate.net/figure/Illustration-of-the-structure-of-spiking-neurons-comparing-the-traditional-LIF-model-a_fig1_384268823)  
117. TSkips: Efficiency Through Explicit Temporal Delay Connections in Spiking Neural Networks, 11月 2, 2025にアクセス、 [https://openreview.net/forum?id=hwz32S06G4](https://openreview.net/forum?id=hwz32S06G4)  
118. SNNtrainer3D: Training Spiking Neural Networks Using a User-Friendly Application with 3D Architecture Visualization Capabilities \- MDPI, 11月 2, 2025にアクセス、 [https://www.mdpi.com/2076-3417/14/13/5752](https://www.mdpi.com/2076-3417/14/13/5752)  
119. Comparison between spiking neural network simulation libraries. \- ResearchGate, 11月 2, 2025にアクセス、 [https://www.researchgate.net/figure/Comparison-between-spiking-neural-network-simulation-libraries\_tbl1\_329601153](https://www.researchgate.net/figure/Comparison-between-spiking-neural-network-simulation-libraries_tbl1_329601153)  
120. SNNAX \- Spiking Neural Networks in JAX \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2409.02842v2](https://arxiv.org/html/2409.02842v2)  
121. Spiking Neural Network (SNN) Library Benchmarks \- Open Neuromorphic, 11月 2, 2025にアクセス、 [https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)  
122. SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. \- GitHub, 11月 2, 2025にアクセス、 [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)  
123. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence \- PMC \- NIH, 11月 2, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)  
124. Spyx: A Library for Just-In-Time Compiled Optimization of Spiking Neural Networks \- arXiv, 11月 2, 2025にアクセス、 [https://arxiv.org/html/2402.18994v1](https://arxiv.org/html/2402.18994v1)  
125. Hands-On Session with snnTorch \- Jason Eshraghian, University of California Santa Cruz, 11月 2, 2025にアクセス、 [https://www.youtube.com/watch?v=aUjWRpisRRg](https://www.youtube.com/watch?v=aUjWRpisRRg)