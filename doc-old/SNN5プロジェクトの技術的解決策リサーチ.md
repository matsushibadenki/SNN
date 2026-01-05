

# **ニューロモルフィック・フロンティアの踏破：SNNプロジェクト「SNN5」のための最先端理論、SOTAアーキテクチャ、および展開戦略（2025-2026）**

## **現代SNN設計における中心的戦略パラダイムの分析**

スパイキングニューラルネットワーク（SNN）プロジェクトが直面する最初の、そして最も重要な戦略的岐路は、「訓練パラダイムの選択」です。2025年から2026年の技術的ランドスケープは、主に3つの異なるアプローチによって定義されます。それぞれが特定のタスク領域（時間的処理、静的推論、オンライン適応）において明確な優位性を持っており、プロジェクトの目標に応じて選択されるべきです。

### **ダイレクト・トレーニング：サロゲート勾配（SG）法による最適化の道**

SNNの直接訓練における主流かつ最も成功しているアプローチは、サロゲート勾配（SG）法です 1。

理論的基盤とSOTA（最先端）の動向  
中核となる課題は、ニューロンの発火イベントが数学的に微分不可能なヘヴィサイド関数（Heaviside step function）である点にあります 2。サロゲート勾配法は、この問題を解決するため、順伝播（forward pass）では発火を維持しつつ、逆伝播（backward pass）の際にのみ、この微分不可能な関数を滑らかで微分可能な「サロゲート」関数（例えばArctan 2 や高速シグモイド 3）に置き換えます 6。  
この技術的な「トリック」により、SNNは標準的な誤差逆伝播法（Backpropagation）の全能力を利用できるようになります 6。これは、SNNがPyTorchのような既存のディープラーニング・エコシステム 6、バッチ正規化（Batch Normalization）1、残差接続（Residual Connections）8 といった強力なツール群を直接活用できることを意味します。

性能面では、SG法で訓練されたSNNは、従来のANN（人工ニューラルネットワーク）の精度に極めて肉薄し（1-2%以内）、20エポック程度で高速に収束し、10ミリ秒という低レイテンシーを達成することが示されています 9。

「レートコーディングを超えて」— SGによる真の時間的学習の実現  
SG法は、単にANNの精度（レートコーディング、すなわち発火率の模倣）を達成するための手段ではありません。2025年の最新研究（arXiv:2507.16043）は、SG法がレートコーディングを超えた情報、すなわち正確なスパイクタイミングを学習できることを理論的・実証的に示しています 10。  
このメカニズムは、SG法によって可能になったバックプロパゲーションが、ニューロン間のスパイク間隔（Inter-Spike Intervals, ISI）やクロスニューロン同期（Coincidence）といった、きめ細かな時間構造（fine-grained temporal structure）から情報を抽出できるというものです 10。この決定的な証拠として、訓練済みのネットワークにおいて、入力スパイクシーケンスの時間を反転させると性能が劇的に低下することが挙げられます 11。もしネットワークが単純な発火率（レート）のみを学習していた場合、時間反転は性能に影響を与えないはずです。

これは、プロジェクト「SNN5」にとって極めて重要な示唆を与えます。もしプロジェクトの目標が、音声認識 15、DVS（イベントベース）ビジョン 16、その他 17 のような本質的に時間的なデータを扱うことであるならば、SG法はSNN独自の**時間的処理能力** 15 を引き出すための、理論的に最適なアプローチとなります。

「適応的サロゲート勾配」— 強化学習（RL）タスクにおけるSOTA戦略  
SG法の最適化はさらに進んでいます。従来、SG関数の「傾き（slope）」は固定のハイパーパラメータでした。しかし、NeurIPS 2025で発表された論文 18 が、この傾き設定の重要性を明らかにしました。  
分析によれば、傾きを浅く（shallower）設定すると、深い層での勾配消失を防ぐ効果（勾配の大きさが増加する）がある一方で 19、勾配の「真の値」との整合性（alignment）を低下させるというトレードオフが存在します 19。

教師あり学習ではこのトレードオフは均衡しますが、強化学習（RL）の文脈（例：ドローンの位置制御タスク 18）においては、浅い傾きが引き起こす「勾配ノイズ」が、探索（exploration）を促進するという予期せぬ利益をもたらします。「SNN5」がロボティクスや自律システム 21 のようなRLタスクを目指す場合、固定されたSG法では最適解に到達できません。**適応的な勾配傾斜スケジュール（adaptive slope schedule）** 19 の実装が必須となります。この手法は、学習速度と最終性能の両方で従来技術（TD3BCなど）を**2.1倍**上回るという劇的な改善をもたらす、SOTAの技術的アイデアです 19。

### **ANN-SNN変換：精度と超低レイテンシーの最短経路**

SNNを構築するもう一つの主流な経路は、SOTAのANN（特にReLUベース）を訓練し、その重みをSNNに「変換」するアプローチです 1。理論的には、ReLUの活性値はIF（Integrate-and-Fire）ニューロンの発火率に比例します 22。

歴史的課題  
しかし、この変換アプローチは長らく2つの大きな問題に悩まされてきました。

1. **精度低下:** 変換プロセスで生じる誤差（量子化誤差、クリッピング誤差、閾値設定ミス）により、元のANNの精度が低下する 23。  
2. **高レイテンシー:** ANNの連続値をSNNの発火率で正確に近似するためには、非常に長いシミュレーション時間ステップ（$T$）が必要でした 22。これにより、SNNの利点であるはずの「低レイテンシー」が失われていました 9。

「One-Timestep is Enough」— $T=1$推論によるパラダイムシフト  
2025年後半に発表された論文「One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons」 26 が、上記の歴史的課題を根本的に解決しました。これは、ANN-SNN変換の分野における真のブレークスルーです。  
この研究は「**Temporal-to-Spatial Equivalence Theory（時間的-空間的等価性理論）**」を確立しました 26。これは、「*複数の時間ステップ（Temporal）にわたるスパイクの積分プロセス*は、*単一時間ステップ（$T=1$）内の空間的な（Spatial）マルチ閾値メカニズム*によって、正確に再構築できる」ことを理論的に証明したものです 26。

この理論を具現化するのが「**Scale-and-Fire Neuron (SFN)**」と名付けられた新しいニューロンモデルです 26。SFNは、膜電位の適応的スケーリング（Scale）と適応的発火関数（Fire）を組み合わせることで、従来の「時間積分」を不要にし、単一時間ステップでの推論を可能にします 26。

このSFNを用いたSNNは、**ImageNetにおいて$T=1$（単一ステップ）で88.8%** という驚異的なTop-1精度を達成しました 26。これは、SGベースのSOTA（約81% 31）を遥かに凌駕する結果です。

SGとANN-SNN変換の戦略的棲み分けの確立  
SFNの登場 26 は、SNNの戦略的ランドスケープを完全に塗り替えました。これまでは「精度と時間的学習能力のSG」対「（潜在的に）高速だが精度の低い変換」というトレードオフでした 9。しかし、SFNの登場により、「変換」は「最高の精度」と「最小のレイテンシー（$T=1$）」を同時に達成する手法へと変貌しました。  
これにより、「SNN5」プロジェクトが取るべき戦略は明確に分岐します。

1. **静的データ（画像分類、物体検出）が主目的の場合:** SG法でSNNを直接訓練するアプローチは、もはや最善ではありません。SOTAのANN（例：Transformer）を訓練し、SFN 26 を用いて$T=1$のSNNに変換するアプローチが、精度・レイテンシー・エネルギー効率の全てにおいて最適解となります。  
2. **動的・時間的データ（DVS、音声、制御）が主目的の場合:** $T=1$の推論は意味を持たないため、\*\*SG法（1.1節）\*\*が引き続き必須のアプローチとなります。

### **生物学的可塑性：STDPと3因子学習ルールによるオンチップ適応**

第3の戦略的パラダイムは、生物学的な学習ルールに基づいています。

理論的基盤とSOTA（最先端）の動向  
スパイクタイミング依存可塑性（STDP）は、最も生物学的に妥当性の高い学習ルールです 32。これは「プリシナプスのスパイク」と「ポストシナプスのスパイク」という2つの要素の相対的タイミングに基づいてシナプス強度を更新する、ローカルな（局所的な）ルールです 34。この局所性（locality）は、ニューロモルフィック・ハードウェア上でのオンチップ学習（ハードウェア上での実時間学習）に不可欠です 35。  
しかし、STDP（ヘブ則）だけでは、「どの学習がタスク全体にとって有益だったか」を判断できません 36。この問題を解決するのが「**3因子学習ルール（Three-Factor Learning Rule）**」です 32。これは、STDP（2因子）に、ドーパミン 32 のような「**グローバルな変調信号（第3因子）**」を追加します。この第3因子は、「報酬シグナル」や「環境からのフィードバック」として機能し、STDPによる可塑性の適用を「ゲート」または「変調」します 32。

3因子学習による強化学習（RL）への橋渡し  
3因子学習ルールは、STDPを単なる教師なし学習から、本格的な\*\*強化学習（RL）\*\*へと昇華させるための理論的メカニズムです。2因子（STDP）は「いつスパイクが来たか」しか知りませんが、3因子目（報酬）が「そのスパイクの結果、良いことが起きたか」を教えます。これにより、SNNはローカルなスパイクイベントとグローバルなタスク報酬を関連付ける「信用割り当て（Credit Assignment）」38 が可能になります。  
「SNN5」が適応的なエッジデバイス（例：継続的に環境から学習するロボット）を目指す場合、SG法（1.1節）のようなGPUベースのオフライン訓練は非現実的です。Intel Loihi 2のようなハードウェアは、この**3因子学習ルールをオンチップでサポート**しています 39。したがって、STDP/3因子学習は、リアルタイムの**オンライン適応学習**を実現するための、SG法や変換に代わる第3の戦略的パラダイムとなります。

---

## **プロジェクトSNN5のための最先端アーキテクチャとモデルの選定**

セクション1で概説した理論を具現化する、具体的なSOTAモデルアーキテクチャを分析します。

### **Spiking Transformers (SFormer / Spikformer V2): 複雑な時空間タスクへの回答**

ANNにおけるTransformer（ViT）の圧倒的成功を受け、SNNへの導入が活発化しています 2。

SOTAモデルの分岐と性能  
現在、Spiking TransformerのSOTAは、訓練パラダイムの違いによって2つの系譜に分かれています。

1. **SFormer (SFNベース, ANN-SNN変換):** 論文「One-Timestep is Enough」26 で提案された\*\*Spiking Transformer (SFormer)\*\*は、1.2節で述べたSFNニューロンの応用例です 27。SFormerは、Transformerアーキテクチャ（特にSoftMax後）の非常に偏った（skewed）活性化分布に対応するため、SFNの適応的発火関数をカスタマイズしています 45。これにより、\*\*ImageNet $T=1$で88.8%\*\*というSNNの歴史上最高の精度を達成しています 26。  
2. **Spikformer V2 (SGベース, 直接訓練):** 2024年初頭にSNNの精度を初めて80%の大台に乗せた、SGベースのSpiking Transformerです 31。その性能は**ImageNet $T=1$で81.10%**、T=4で80.38%です 31。

SNNアーキテクチャにおける「変換」の「直接訓練」に対する勝利  
2024年（Spikformer V2）から2025年（SFormer/SFN）にかけてのSOTAの変遷（81.1% vs 88.8%）は、決定的な傾向を示しています。SGベースの直接訓練（Spikformer V2）は強力ですが、超大規模なViTをSNNとしてゼロから訓練することは、依然として困難（例：勾配消失、ハイパーパラメータ調整）を伴います。一方、SFNベースの変換（SFormer）は、(1) まずANNとして最適に訓練し、(2) ほぼロスレスで$T=1$のSNNに変換する、という2段階のアプローチを取ります。  
この結果は、少なくとも**静的画像分類タスクにおいては、SFNを用いたANN-SNN変換アプローチが、SG法による直接訓練アプローチに対して明確に勝利した**ことを示しています。「SNN5」が静的タスクでSOTAを目指す場合、SFormer (SFN) 27 のアーキテクチャと手法を採用すべきです。

重要なコンポーネント: CPG-PE (NeurIPS 2024\)  
Spiking TransformerのようなシーケンシャルSNNには、ハードウェアフレンドリーなスパイク形式の位置エンコーディング（PE）が必要でした 2。NeurIPS 2024で発表されたCPG-PE 2 は、人間の脳内にあるリズミカルなパターンを生成する中枢パターン生成器（CPG）に着想を得た、新しいPE技術です。「SNN5」がTransformerを実装する場合、従来のSinusoidal PEよりも生物学的で効率的なCPG-PE 2 の採用を検討すべきです。

### **FEEL-SNN (NeurIPS 2024): 堅牢性（ロバストネス）のブレークスルー**

従来のSNNは「生物学的にインスパイアされているため、本質的に堅牢である」と信じられてきましたが、これは理論的裏付けに欠けていました 47。FEEL-SNN 47 は、この問題に正面から取り組んだNeurIPS 2024の論文です。

技術的メカニズムの詳細 47  
FEEL-SNNの中核は、生物学的な2つのメカニズムを模倣するモジュールを導入し、SNNの堅牢性を工学的に強化する点にあります 47。

1. **Frequency Encoding (FE):**  
   * **生物学的着想:** 選択的視覚注意メカニズム 47。  
   * **動作:** 従来のSNNエンコーディングとは異なり、FEは時間ステップごとに異なる周波数帯域に焦点を当てます 47。これにより、特定の周波数帯に集中するノイズ（例：ガウシアンノイズ）を効果的にフィルタリングします 47。  
2. **Evolutionary Leak factor (EL):**  
   * **生物学的着想:** 非固定の膜電位リーク 47。生物のニューロンのリーク率は、イオン濃度などによって動的に変化します。  
   * **動作:** 従来のSNNでは固定値であったLIFニューロンのリーク係数（beta）を、ニューロンごと、かつ時間ステップごとに**学習可能**かつ**動的に進化**させます 47。これにより、ネットワークはデータやノイズの特性に応じて、情報の「保持」と「忘却」のバランスを最適化できます 47。

堅牢性を「設計可能なコンポーネント」として扱う  
FEEL-SNNの最大の貢献は、SNNの堅牢性を「期待される副産物」から「設計可能なコンポーネント」へと変えたことです。論文は、SNNの堅牢性に関する統一された理論的フレームワークを提示し 47、エンコーディングとリークファクターが堅牢性に直結することを示しました。  
「SNN5」が、ノイズの多い実世界（例：DVSセンサーからの入力 16 や、敵対的攻撃 47）で確実に動作する必要がある場合、FEEL-SNNのアーキテクチャ 47 を、セクション2.1で選定したSOTAモデル（SFormerなど）に**統合**することが、技術的に極めて重要なアイデアとなります。

---

## **実装と展開：SNN5をGPUからニューロモルフィック・ハードウェアへ**

SNN5プロジェクトを構築し、最終目標であるニューロモルフィック・ハードウェア上で実行するための実践的な「アイデア」とワークフローを詳述します。

### **ソフトウェア・フレームワークの戦略的選定**

現代のSNN研究は、PyTorchベースのフレームワークが主流です 54。

* snnTorch 3:  
  * **特徴:** PyTorchの哲学に忠実で 57、非常にユーザーフレンドリーです。ドキュメントとチュートリアル 4 が豊富で、学習、プロトタイピング、教育に最適です 59。LIF、Synaptic、SLSTMなど多様なニューロンモデルをサポートし 61、多様なサロゲート勾配（ATan、Sigmoidなど）を簡単に切り替え可能です 3。  
  * **弱点:** パフォーマンス。純粋なPyTorch実装であるため、カスタムCUDAカーネルを持つSpikingJellyと比較して遅いことがベンチマークで示されています 64。また、特定のニューロモルフィック・ハードウェア展開への明確なパスが不足しています 65。  
* SpikingJelly 66:  
  * **特徴:** 同じくPyTorchベース 55 ですが、パフォーマンスを最優先に設計されています。  
  * **強み:** **速度**。カスタムCUDAカーネルとCuPyバックエンドを活用し、snnTorchのような純粋なPyTorch実装に比べて桁違いの高速シミュレーションを実現します（16kニューロンのネットワークで0.26秒 vs 2.5秒以上）64。さらに重要な点として、**Intel Loihi 2への展開パス** 1 を明示的に提供する「フルスタック・ツールキット」であると述べられています 1。

「使いやすさ」と「展開性能」の間のトレードオフ  
この2つのフレームワークは、「ニューロモルフィック・コンピューティングの断片化された風景」68 の典型例です。snnTorchは「学習の容易さ」59 を、SpikingJellyは「シミュレーション速度と展開性」1 を提供します。  
「SNN5」プロジェクトは、**ハイブリッド・フレームワーク戦略**を採用することが最善のアイデアです。

1. **フェーズ1（研究・プロトタイプ）:** **snnTorch** 57 を使用し、新しいアイデア（例：カスタムSG、FE/ELモジュール）を迅速にテストし、チームのSNN習熟度を高めます。  
2. **フェーズ2（SOTA実装・展開）:** **SpikingJelly** 66 を使用し、SOTAモデル（SFormerなど）を再現し、パフォーマンスを最大化し、Loihi 2への展開（3.2節）に備えます。

### **Intel Loihi 2 展開ワークフロー：理論から実践へ**

**Loihi 2とLavaフレームワークの概要**

* **Loihi 2:** Intelの第2世代ニューロモルフィック研究チップであり 69、1チップあたり最大100万ニューロンをサポートし 69、オンチップ学習（3因子ルール）が可能です 40。  
* **Lava:** Loihi 2のための公式オープンソース・ソフトウェアフレームワークです 69。Loihi 2実機だけでなく、CPU/GPUシミュレーションもサポートするプラットフォーム非依存設計が特徴です 71。  
* **Lava-DL:** Lavaのディープラーニング・ライブラリであり 72、SG法による直接訓練（SLAYER 72）とANN-SNN変換（Bootstrap 72）の両方をサポートします。  
* **Lava-nativeワークフロー:** Lava-DL (SLAYERなど) 74 を使用してGPUで訓練し、Lavaのnetxライブラリ 72 を介してLoihi 2に展開します 74。

SpikingJellyからLavaへの「ショートカット」ワークフロー  
Lava-DL (SLAYER) 72 をゼロから学ぶのは、「簡単ではない」74 と報告されています。しかし、3.1節の提言に従いSpikingJellyを採用した場合、Lavaエコシステムにロックインされる必要はありません。  
SpikingJellyは、spikingjelly.activation\_based.lava\_exchangeという専用モジュールを提供しています 67。このワークフローは以下の通りです 67：

1. **SpikingJelly (PyTorch):** SNNモデルを通常通りSpikingJellyで構築・訓練します。  
2. **lava\_exchange:** このモジュールが、SpikingJellyのニューロン（LIFNode）をLavaのニューロンに変換し（to\_lava\_neuron 75）、データ形式（T,N,... → N,...,T）を変換します（TNX\_to\_NXT 75）。  
3. **Lava-DL:** 変換されたモデルはLava-DLフォーマットになります。  
4. **Lava HDF5:** モデルをHDF5ファイルにエクスポートします 67。  
5. **Loihi 2:** LavaフレームワークがこのHDF5ファイルを読み込み、Loihi 2実機またはシミュレータで実行します 67。

これは、「SNN5」プロジェクトにとって**最も実践的かつ強力な展開アイデア**です。チームは、使い慣れた高性能なSpikingJelly (PyTorch) 環境で開発の95%を完結させ、lava\_exchange 75 を「コンパイラ」として使用するだけで、Loihi 2への展開が可能です。

### **ニューロモルフィック・ネイティブ学習：オンチップでの適応**

Loihi 2は、単なる推論アクセラレータではなく、**オンチップでの学習**が可能です 40。具体的には、ニューロコア（Neuro-Cores）が**3因子学習ルール**（1.3節）をハードウェアレベルでサポートしています 40。Lavaフレームワークは、この機能を活用するためのAPIとチュートリアル（例：tutorial01\_Reward\_Modulated\_STDP）を完備しています 39。

「メタ学習」としてのオンチップ学習ワークフロー  
41は、オンチップ学習の非常に高度な利用法、すなわちハイブリッド・メタラーニングを提案しています。  
そのワークフローは以下の通りです 41：

1. **ステージ1 (GPU / メタ訓練):** GPU上で、Lavaのシミュレータと*微分可能な*可塑性ルール（SG）を使用し、タスク（例：ワンショット学習）に最適な「**可塑性のハイパーパラメータ**」自体を*訓練*します。  
2. **ステージ2 (Loihi 2 / 展開):** 最適化された可塑性ルール（例：R-STDP）とそのハイパーパラメータをLoihi 2に展開します。

この結果、Loihi 2は、GPUでの再訓練なしに、新しいクラスのデータを**リアルタイムで**、**ワンショット（1回の提示）で**学習できるようになります 41。「SNN5」が目指すべき究極的な「アイデア」の一つは、Loihi 2を「学習するシステム」として展開し、エッジでリアルタイムに適応・パーソナライズする 41 ことです。

---

## **プロジェクトSNN5のための戦略的提言と技術的ロードマップ**

提供されたリサーチ78からはプロジェクト「SNN5」の具体的な目標を特定できませんでしたが、その名称が示唆するSNN（スパイキングニューラルネットワーク）分野のSOTA（最先端）を達成するための技術的戦略は、以下のように明確に提言できます。

### **提言：タスク駆動型ハイブリッド・アプローチの採用**

2025-2026年のSNNランドスケープにおいて、「全てのタスクに最適な単一の技術」は存在しません。SG法（1.1節）、SFN変換（1.2節）、3因子STDP（1.3節）は、それぞれが異なる問題領域（時間処理、静的推論、オンライン適応）のチャンピオンです。

「SNN5」プロジェクトは、以下の**タスク駆動型ハイブリッド戦略**を採用すべきです。

* **目標1: 静的高精度・超低遅延タスク（例：ImageNet分類、物体検出）**  
  * **理論:** ANN-SNN変換（1.2節）。  
  * **アイデア:** **Scale-and-Fire Neuron (SFN)** 26 と**SFormer** 27 のアーキテクチャを採用し、SOTAのANN-ViTを $T=1$ のSNNに変換する。ImageNet 88.8% 26 を目標ベンチマークとする。  
* **目標2: 時間的・シーケンシャル・タスク（例：DVSジェスチャー、時系列予測）**  
  * **理論:** サロゲート勾配（SG）法（1.1節）。  
  * **アイデア:** **Spiking Transformer** 31 アーキテクチャをSG法で直接訓練する。SG法がレートコーディングを超えた時間的学習 11 を可能にすることを活用する。  
* **目標3: 実世界での堅牢性（上記2つに共通）**  
  * **理論:** 堅牢性の工学的設計（2.2節）。  
  * **アイデア:** **FEEL-SNN** 47 の\*\*Frequency Encoding (FE)**と**Evolutionary Leak factor (EL)\*\*モジュール 47 を、目標1と目標2のアーキテクチャに統合する。  
* **目標4: エッジでの適応・オンライン学習（例：ロボティクス、RL）**  
  * **理論:** 3因子学習ルール（1.3節）。  
  * **アイデア:** **Intel Loihi 2**のオンチップ学習機能を活用し、Lavaフレームワーク 39 を用いて\*\*報酬変調型STDP (R-STDP)\*\*を実装する。41に示されるメタ学習アプローチを検討する。

### **技術的ロードマップのアイデア**

* **フェーズ1: ベースライン構築とSOTA再現（GPU）**  
  * **使用フレームワーク:** **SpikingJelly** 64。  
  * **タスクA (静的):** SFormer (SFN) 27 の論文を追試し、ImageNet $T=1$でのSOTA精度（88.8%）を再現する 26。  
  * **タスクB (動的):** Spikformer V2 31 とCPG-PE 2 を実装し、SG法によるImageNet SOTA（81.1%）を再現する 31。  
* **フェーズ2: 堅牢性の組み込みと時間的学習の検証（GPU）**  
  * **使用フレームワーク:** SpikingJelly \+ snnTorch（プロトタイピング用）57。  
  * **タスクA (堅牢性):** FEEL-SNN 47 のFE/ELモジュールをフェーズ1のモデルに統合し、ノイズありデータセットで堅牢性をベンチマークする 47。  
  * **タスクB (時間):** フェーズ1のSGモデルをDVS-Gesture 16 で訓練し、「時間反転」11 実験を行い、レートコーディングを超えた学習 11 が行われていることを確認する。  
* **フェーズ3: ニューロモルフィック展開と検証（Loihi 2）**  
  * **使用ワークフロー:** **SpikingJelly-to-Lava**（3.2節）のlava\_exchange 67。  
  * **タスクA (変換):** フェーズ1, 2で訓練したSNNモデルをLava HDF5フォーマットに変換する。  
  * **タスクB (検証):** Loihi 2シミュレータおよび実機上で推論を実行し、GPUシミュレーションとの精度一致 82 を確認する。  
  * **タスクC (計測):** Loihi 2上での推論あたりのエネルギー（ミリジュール）9 とレイテンシー（ミリ秒）9 を計測し、GPU（NVIDIA Jetson 82）との比較で桁違いの効率向上（例：250倍 82）を実証する。  
* **フェーズ4: オンチップ適応（Loihi 2 ネイティブ）**  
  * **使用フレームワーク:** **Intel Lava** 39。  
  * **タスクA (PoC):** R-STDPチュートリアル 39 に基づき、Loihi 2上で動作する小規模なオンライン学習タスクのプロトタイプを構築する。  
  * **タスクB (拡張):** 41のメタ学習アプローチを研究し、オンチップ・ワンショット学習の実装を試みる。

---

### **付録：主要戦略およびフレームワークの比較**

**表1: SNN訓練パラダイムの戦略的比較（2025-2026年）**

| パラダイム | 理論的中核 | SOTA精度 (ImageNet) | 最小レイテンシー (T) | 生物学的妥当性 | Loihi 2 オンチップ学習 | 最適タスク |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **サロゲート勾配 (SG) 法** | 微分不可能な発火を滑らかな関数で近似し、BPを可能にする 2 | $\\approx$ 81.1% (Spikformer V2) 31 | 低 (例: 4-10 ms) 9 | 低 (BPは非生物的) 83 | 不可 (オフライン訓練) | **時間的タスク** (DVS, 音声, 制御) 11 |
| **ANN-SNN変換 (SFN)** | 時間的-空間的等価性理論。時間積分を空間的マルチ閾値で代替 26 | **$\\approx$ 88.8% (SFormer)** 26 | **超低 ($T=1$)** 26 | 低 (ANNベース) | 不可 (オフライン訓練) | **静的タスク** (画像分類, 物体検出) |
| **STDP / 3因子学習** | 局所的なスパイクタイミング（2因子）＋グローバルな報酬信号（3因子） 32 | N/A (タスクによる) | N/A (リアルタイム) | **高** 33 | **可** 39 | **オンライン適応** (エッジRL, 継続学習) |

**表2: SOTA SNNモデルの性能ベンチマーク (2024-2025年)**

| モデル名 | アーキテクチャ | 訓練手法 | ベンチマーク | Top-1 精度 (%) | 時間ステップ (T) | 発表 (年) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **SFormer** 27 | Transformer | ANN-SNN変換 (SFN) | ImageNet | **88.8** | **1** | 2025 (arXiv) 26 |
| **Spikformer V2** 31 | Transformer | SG (直接訓練) | ImageNet | 81.1 | 1 | 2024 (arXiv) 31 |
| **QKFormer** 46 | Transformer | SG (直接訓練) | ImageNet | \> 85.0 (※) | 4 | 2024 (記事) 46 |
| **FEEL-SNN** 47 | N/A | SG (直接訓練) | CIFAR-100 (Noise) | N/A (堅牢性向上) | N/A | 2024 (NeurIPS) 47 |
| **HAS-8-ResNet** 84 | ResNet-18 | SG (直接訓練) | CIFAR-100 | 48.23 | N/A | 2025 (arXiv) 84 |
| 26 |  |  |  |  |  |  |

**表3: SNN実装フレームワークのエコシステム分析**

| フレームワーク | ベース | 主要機能 | GPUパフォーマンス | Loihi 2 展開サポート | Loihi 2 への具体的パス | オンチップ学習サポート |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **snnTorch** 57 | PyTorch | 豊富なチュートリアル 58, 使いやすさ 59, 多様なニューロン 61 | 標準 (Pure PyTorch) 64 | 限定的 65 | N/A | 不可 |
| **SpikingJelly** 66 | PyTorch | **カスタムCUDAカーネル** 66, SOTA実装 67, フルスタック 1 | **非常に高速** 64 | **あり** 1 | **lava\_exchange** 75 | 不可 |
| **Intel Lava** 70 | (独自) | Loihi 2 ネイティブサポート 69, プラットフォーム非依存 71 | 可 (Lava-DLシミュレーション) 72 | **ネイティブ** | (HDF5経由) 72 | **あり (R-STDP)** 39 |

#### **引用文献**

1. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
2. Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators \- NIPS papers, 11月 8, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf)  
3. snntorch.surrogate \- Read the Docs, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html)  
4. Tutorial 5 \- Training Spiking Neural Networks with snntorch \- Read the Docs, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_5.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)  
5. Tutorial 6 \- Surrogate Gradient Descent in a Convolutional SNN \- snnTorch \- Read the Docs, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_6.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)  
6. Adaptive Smoothing Gradient Learning for Spiking Neural Networks, 11月 8, 2025にアクセス、 [https://proceedings.mlr.press/v202/wang23j/wang23j.pdf](https://proceedings.mlr.press/v202/wang23j/wang23j.pdf)  
7. Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2406.19645v1](https://arxiv.org/html/2406.19645v1)  
8. TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2206.10177v3](https://arxiv.org/html/2206.10177v3)  
9. \[2510.27379\] Spiking Neural Networks: The Future of Brain-Inspired ..., 11月 8, 2025にアクセス、 [https://www.arxiv.org/abs/2510.27379](https://www.arxiv.org/abs/2510.27379)  
10. Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2507.16043v1](https://arxiv.org/html/2507.16043v1)  
11. Beyond Rate Coding: Surrogate Gradients Enable Spike ... \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2507.16043](https://arxiv.org/abs/2507.16043)  
12. Spiking Neural Networks \- Aussie AI, 11月 8, 2025にアクセス、 [https://www.aussieai.com/research/spiking-neural-networks](https://www.aussieai.com/research/spiking-neural-networks)  
13. Bluesky Embed, 11月 8, 2025にアクセス、 [https://embed.bsky.app/embed/did:plc:niqde7rkzo7ua3scet2rzyt7/app.bsky.feed.post/3lupzbfiakk2m?id=5943970666266978\&ref\_url=https%253A%252F%252Fneural-reckoning.org%252Fpub\_beyond\_rate\_coding.html](https://embed.bsky.app/embed/did:plc:niqde7rkzo7ua3scet2rzyt7/app.bsky.feed.post/3lupzbfiakk2m?id=5943970666266978&ref_url=https%253A%252F%252Fneural-reckoning.org%252Fpub_beyond_rate_coding.html)  
14. Exploiting heterogeneous delays for efficient computation in low-bit neural networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2510.27434v1](https://arxiv.org/html/2510.27434v1)  
15. Spiking Neural Network Architecture Search: A Survey \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2510.14235v1](https://arxiv.org/html/2510.14235v1)  
16. \[2409.12691\] A dynamic vision sensor object recognition model based on trainable event-driven convolution and spiking attention mechanism \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2409.12691](https://arxiv.org/abs/2409.12691)  
17. Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2502.09449v1](https://arxiv.org/html/2502.09449v1)  
18. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/397006451\_Adaptive\_Surrogate\_Gradients\_for\_Sequential\_Reinforcement\_Learning\_in\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/397006451_Adaptive_Surrogate_Gradients_for_Sequential_Reinforcement_Learning_in_Spiking_Neural_Networks)  
19. \[Literature Review\] Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- Moonlight | AI Colleague for Research Papers, 11月 8, 2025にアクセス、 [https://www.themoonlight.io/en/review/adaptive-surrogate-gradients-for-sequential-reinforcement-learning-in-spiking-neural-networks](https://www.themoonlight.io/en/review/adaptive-surrogate-gradients-for-sequential-reinforcement-learning-in-spiking-neural-networks)  
20. NeurIPS Poster Adaptive Surrogate Gradients for Sequential ..., 11月 8, 2025にアクセス、 [https://neurips.cc/virtual/2025/poster/116054](https://neurips.cc/virtual/2025/poster/116054)  
21. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2510.24461v1](https://arxiv.org/html/2510.24461v1)  
22. Quantization Framework for Fast Spiking Neural Networks \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.918793/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.918793/full)  
23. High-accuracy deep ANN-to-SNN conversion using quantization-aware training framework and calcium-gated bipolar leaky integrate and fire neuron \- PMC \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10030499/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030499/)  
24. FAS: Fast ANN-SNN Conversion for Spiking Large Language Models \- arXiv, 11月 8, 2025にアクセス、 [https://www.arxiv.org/pdf/2502.04405](https://www.arxiv.org/pdf/2502.04405)  
25. Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2508.20392v1](https://arxiv.org/html/2508.20392v1)  
26. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2510.23383v1](https://arxiv.org/html/2510.23383v1)  
27. Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 8, 2025にアクセス、 [https://www.arxiv.org/pdf/2510.23383](https://www.arxiv.org/pdf/2510.23383)  
28. \[2510.23383\] One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 8, 2025にアクセス、 [https://www.arxiv.org/abs/2510.23383](https://www.arxiv.org/abs/2510.23383)  
29. Achieving High-performance ANN-to-SNN Conversion via ... \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2510.23383](https://arxiv.org/abs/2510.23383)  
30. Object Detection in 20 Years: A Survey | Request PDF \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/367483279\_Object\_Detection\_in\_20\_Years\_A\_Survey](https://www.researchgate.net/publication/367483279_Object_Detection_in_20_Years_A_Survey)  
31. Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2401.02020v1](https://arxiv.org/html/2401.02020v1)  
32. Three-Factor Learning in Spiking Neural Networks: An Overview of Methods and Trends from a Machine Learning Perspective \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2504.05341v2](https://arxiv.org/html/2504.05341v2)  
33. Developmental Spike Timing-Dependent Long-Term Depression Requires Astrocyte d-Serine at L2/3-L2/3 Synapses of the Mouse Somatosensory Cortex | Journal of Neuroscience, 11月 8, 2025にアクセス、 [https://www.jneurosci.org/content/44/48/e0805242024](https://www.jneurosci.org/content/44/48/e0805242024)  
34. Causal Spike Timing Dependent Plasticity Prevents Assembly Fusion in Recurrent Networks \- bioRxiv, 11月 8, 2025にアクセス、 [https://www.biorxiv.org/content/10.1101/2025.01.14.633085v1.full.pdf](https://www.biorxiv.org/content/10.1101/2025.01.14.633085v1.full.pdf)  
35. Stimulating STDP to Exploit Locality for Lifelong Learning without Catastrophic Forgetting \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/vc/arxiv/papers/1902/1902.03187v1.pdf](https://arxiv.org/vc/arxiv/papers/1902/1902.03187v1.pdf)  
36. Three-Factor Learning in Spiking Neural Networks: An Overview of Methods and Trends from a Machine Learning Perspective \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2504.05341v1](https://arxiv.org/html/2504.05341v1)  
37. Three-Factor Learning in Spiking Neural Networks: An Overview of Methods and Trends from a Machine Learning Perspective \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/390601647\_Three-Factor\_Learning\_in\_Spiking\_Neural\_Networks\_An\_Overview\_of\_Methods\_and\_Trends\_from\_a\_Machine\_Learning\_Perspective](https://www.researchgate.net/publication/390601647_Three-Factor_Learning_in_Spiking_Neural_Networks_An_Overview_of_Methods_and_Trends_from_a_Machine_Learning_Perspective)  
38. Three-Factor Learning in Spiking Neural Networks: An ... \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2504.05341](https://arxiv.org/abs/2504.05341)  
39. Three Factor Learning with Lava, 11月 8, 2025にアクセス、 [https://lava-nc.org/lava/notebooks/in\_depth/three\_factor\_learning/tutorial01\_Reward\_Modulated\_STDP.html](https://lava-nc.org/lava/notebooks/in_depth/three_factor_learning/tutorial01_Reward_Modulated_STDP.html)  
40. Legendre-SNN on Loihi-2: Evaluation and Insights \- OpenReview, 11月 8, 2025にアクセス、 [https://openreview.net/pdf?id=wUUvWjdE0K](https://openreview.net/pdf?id=wUUvWjdE0K)  
41. Emulating Brain-like Rapid Learning in Neuromorphic Edge Computing \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2408.15800v1](https://arxiv.org/html/2408.15800v1)  
42. Spiking neural networks for object detection and semantic segmentation across event-driven and frame-based modalities \- OE Journals, 11月 8, 2025にアクセス、 [https://www.oejournal.org/ioe/en/article/pdf/preview/10.29026/ioe.2025.250007.pdf](https://www.oejournal.org/ioe/en/article/pdf/preview/10.29026/ioe.2025.250007.pdf)  
43. Spiking neural networks for object detection and semantic segmentation across event-driven and frame-based modalities: A review \- OE Journals, 11月 8, 2025にアクセス、 [https://www.oejournal.org/ioe/article/doi/10.29026/ioe.2025.250007](https://www.oejournal.org/ioe/article/doi/10.29026/ioe.2025.250007)  
44. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/396967604\_One-Timestep\_is\_Enough\_Achieving\_High-performance\_ANN-to-SNN\_Conversion\_via\_Scale-and-Fire\_Neurons](https://www.researchgate.net/publication/396967604_One-Timestep_is_Enough_Achieving_High-performance_ANN-to-SNN_Conversion_via_Scale-and-Fire_Neurons)  
45. \[Literature Review\] One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons, 11月 8, 2025にアクセス、 [https://www.themoonlight.io/en/review/one-timestep-is-enough-achieving-high-performance-ann-to-snn-conversion-via-scale-and-fire-neurons](https://www.themoonlight.io/en/review/one-timestep-is-enough-achieving-high-performance-ann-to-snn-conversion-via-scale-and-fire-neurons)  
46. Direct training high-performance deep spiking neural networks: a review of theories and methods \- PMC \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)  
47. FEEL-SNN: Robust Spiking Neural Networks with ... \- NIPS papers, 11月 8, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/a73474c359ed523e6cd3174ed29a4d56-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/a73474c359ed523e6cd3174ed29a4d56-Paper-Conference.pdf)  
48. NeurIPS 2024 Papers, 11月 8, 2025にアクセス、 [https://nips.cc/virtual/2024/papers.html](https://nips.cc/virtual/2024/papers.html)  
49. FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and Evolutionary Leak Factor \- proceedings.com, 11月 8, 2025にアクセス、 [https://www.proceedings.com/079017-2917.html](https://www.proceedings.com/079017-2917.html)  
50. (PDF) Synergy Between the Strong and the Weak: Spiking Neural Networks are Inherently Self-Distillers \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/396373930\_Synergy\_Between\_the\_Strong\_and\_the\_Weak\_Spiking\_Neural\_Networks\_are\_Inherently\_Self-Distillers](https://www.researchgate.net/publication/396373930_Synergy_Between_the_Strong_and_the_Weak_Spiking_Neural_Networks_are_Inherently_Self-Distillers)  
51. Edge Intelligence with Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2507.14069v1](https://arxiv.org/html/2507.14069v1)  
52. FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and Evolutionary Leak Factor | OpenReview, 11月 8, 2025にアクセス、 [https://openreview.net/forum?id=TuCQdBo4NC\&referrer=%5Bthe%20profile%20of%20Huajin%20Tang%5D(%2Fprofile%3Fid%3D\~Huajin\_Tang1)](https://openreview.net/forum?id=TuCQdBo4NC&referrer=%5Bthe+profile+of+Huajin+Tang%5D\(/profile?id%3D~Huajin_Tang1\))  
53. FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and Evolutionary Leak Factor, 11月 8, 2025にアクセス、 [https://neurips.cc/media/neurips-2024/Slides/95008.pdf](https://neurips.cc/media/neurips-2024/Slides/95008.pdf)  
54. 1\. Getting started — Norse, 11月 8, 2025にアクセス、 [https://norse.github.io/norse/](https://norse.github.io/norse/)  
55. DEGREE PROJECT \- DiVA portal, 11月 8, 2025にアクセス、 [https://ltu.diva-portal.org/smash/get/diva2:1801633/FULLTEXT02.pdf](https://ltu.diva-portal.org/smash/get/diva2:1801633/FULLTEXT02.pdf)  
56. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence \- PMC \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)  
57. snnTorch Documentation — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/](https://snntorch.readthedocs.io/)  
58. Tutorials — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/index.html](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)  
59. Spiking Neural Networks : r/compmathneuro \- Reddit, 11月 8, 2025にアクセス、 [https://www.reddit.com/r/compmathneuro/comments/1hjnl9q/spiking\_neural\_networks/](https://www.reddit.com/r/compmathneuro/comments/1hjnl9q/spiking_neural_networks/)  
60. Spiking Neural Networks : r/neuro \- Reddit, 11月 8, 2025にアクセス、 [https://www.reddit.com/r/neuro/comments/1hjnljz/spiking\_neural\_networks/](https://www.reddit.com/r/neuro/comments/1hjnljz/spiking_neural_networks/)  
61. is designed to be intuitively used with PyTorch, as though each spiking neuron were simply another activation in a sequence of layers. \- snnTorch, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/snntorch.html](https://snntorch.readthedocs.io/en/latest/snntorch.html)  
62. Tutorial 4 \- 2nd Order Spiking Neuron Models — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_4.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html)  
63. Regression with SNNs: Part I — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_regression\_1.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html)  
64. Spiking Neural Network (SNN) Library Benchmarks \- Open Neuromorphic, 11月 8, 2025にアクセス、 [https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)  
65. Spiking Neural Networks for Multimodal Neuroimaging: A Comprehensive Review of Current Trends and the NeuCube Brain-Inspired Architecture \- PubMed Central, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/)  
66. SNNAX \- Spiking Neural Networks in JAX \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2409.02842v1](https://arxiv.org/html/2409.02842v1)  
67. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence arXiv:2310.16620v1 \[cs.NE\], 11月 8, 2025にアクセス、 [https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf](https://ai-data-base.com/wp-content/uploads/2023/10/2310.16620v1.pdf)  
68. From Idea to Implementation: Reimagining the Neuromorphic Workflow, 11月 8, 2025にアクセス、 [https://open-neuromorphic.org/blog/workflow-vision-neuromorphic/](https://open-neuromorphic.org/blog/workflow-vision-neuromorphic/)  
69. A Look at Loihi 2 \- Intel \- Open Neuromorphic, 11月 8, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/)  
70. lava-nc/lava: A Software Framework for Neuromorphic Computing \- GitHub, 11月 8, 2025にアクセス、 [https://github.com/lava-nc/lava](https://github.com/lava-nc/lava)  
71. Lava Software Framework — Lava documentation, 11月 8, 2025にアクセス、 [https://lava-nc.org/](https://lava-nc.org/)  
72. Deep Learning — Lava documentation, 11月 8, 2025にアクセス、 [https://lava-nc.org/dl.html](https://lava-nc.org/dl.html)  
73. N-DriverMotion: Driver motion learning and prediction using an event-based camera and directly trained spiking neural networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2408.13379v2](https://arxiv.org/html/2408.13379v2)  
74. Lava Tutorial: Mnist Training On Gpu And Evaluation On Loihi2 | R Gaurav's Blog, 11月 8, 2025にアクセス、 [https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html](https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html)  
75. Convert to Lava for Loihi Deployment — spikingjelly alpha 文档, 11月 8, 2025にアクセス、 [https://spikingjelly.readthedocs.io/zh-cn/latest/activation\_based\_en/lava\_exchange.html](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/lava_exchange.html)  
76. spikingjelly.activation\_based.neuron package \- Read the Docs, 11月 8, 2025にアクセス、 [https://spikingjelly.readthedocs.io/zh-cn/latest/sub\_module/spikingjelly.activation\_based.neuron.html](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html)  
77. Taking Neuromorphic Computing with Loihi 2 to the Next Level Technology Brief \- Intel, 11月 8, 2025にアクセス、 [https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf)  
78. 1月 1, 1970にアクセス、 [https://github.com/matsushibadenki/SNN5/blob/main/Objective.md](https://github.com/matsushibadenki/SNN5/blob/main/Objective.md)  
79. 1月 1, 1970にアクセス、 [https://github.com/matsushibadenki/SNN5/blob/main/ROADMAP.md](https://github.com/matsushibadenki/SNN5/blob/main/ROADMAP.md)  
80. github.com, 11月 8, 2025にアクセス、 [https://github.com/matsushibadenki/SNN5](https://github.com/matsushibadenki/SNN5)  
81. \[2209.14915\] Spiking Neural Networks for event-based action recognition: A new task to understand their advantage \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2209.14915](https://arxiv.org/abs/2209.14915)  
82. A Complete Pipeline for deploying SNNs with Synaptic Delays on Loihi 2 \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2510.13757v1](https://arxiv.org/html/2510.13757v1)  
83. Benchmarking Spiking Neural Network Learning Methods with Varying Locality \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2402.01782v2](https://arxiv.org/html/2402.01782v2)  
84. Hybrid ANN-SNN With Layer-Wise Surrogate Spike Encoding-Decoding Structure \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2509.24411v1](https://arxiv.org/html/2509.24411v1)