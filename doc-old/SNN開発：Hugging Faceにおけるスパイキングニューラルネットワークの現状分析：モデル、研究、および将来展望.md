

# **Hugging Faceにおけるスパイキングニューラルネットワークの現状分析：モデル、研究、および将来展望**

## **序論：ニューロモーフィックの最前線—大規模AI時代におけるスパイキングニューラルネットワーク**

### **ANNからSNNへのパラダイムシフト**

従来型の人工ニューラルネットワーク（ANN）は、同期的で連続値に基づき、膨大なエネルギーを消費する計算モデルです 1。これに対し、スパイキングニューラルネットワーク（SNN）は、非同期かつイベント駆動型であり、スパース（疎）な計算モデルを特徴としています 3。SNNの根幹をなすのは、ニューロンの発火（スパイク）、膜電位、そして時間的情報エンコーディングといった概念です。これらの生物学的な脳の仕組みに着想を得た原理は、特に専用のニューロモーフィックハードウェア上で実行された際に、飛躍的なエネルギー効率の向上をもたらす可能性を秘めています 3。ANNが層全体のニューロンを同時に活性化させ、高密度の行列演算を必要とするのに対し、SNNは情報の変化があった場合にのみスパイクを生成・伝達するため、計算リソースを大幅に削減できるのです。

### **SNN開発のハブとしてのHugging Face**

Hugging Faceプラットフォームは、主にTransformerベースの事前学習済みANNモデルのリポジトリとして広く認知されています。しかし、その役割は単なるモデルの保管庫にとどまりません。近年では、SNNのような次世代アーキテクチャの普及と議論を促進するエコシステムとしての側面を強めています。本レポートの目的は、Hugging Face Hubおよび関連するPapersリポジトリを通じてアクセス可能な主要なSNNモデルと研究イニシアチブを体系的に調査・分析し、この分野における「現状」を包括的に明らかにすることです。

Hugging FaceがTransformerモデルの標準化とアクセシビリティを通じて成功を収めたことは周知の事実です。SNNのような新しいアーキテクチャがこのプラットフォーム上に登場するという事実は、それ自体が重要な意味を持ちます。これは、SNNが一定レベルのユーザビリティ、性能、そしてより広範なAIコミュニティへの関連性を達成したことを示唆しています。後述するBrainTransformersやSpikeGPTのようなモデルは、もはや単なる理論上の概念ではありません。それらは実装され、学習され、場合によっては使い慣れたtransformersライブラリ風のインターフェースを通じて直接利用可能になっています 5。したがって、これらのモデルがHugging Faceに登録されているという事実は、SNNという分野が純粋な学術研究から、より広い開発者層を惹きつけようとする応用工学へと移行しつつあることを示す先行指標と言えるでしょう。

## **第1部：Hugging Face Hubで公開されているSNNベース大規模言語モデル**

このセクションでは、Hugging Face Hub上でモデルの重みとコードが直接利用可能なプロジェクトに焦点を当てます。これらは、言語タスク向けSNNの実用化に向けた最も成熟した事例と言えます。

### **1\. BrainTransformers-3B-Chat：Transformer LLMのスパイキング実装**

#### **モデルプロファイル**

* **リンク:**([https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat)) 5  
* **開発元:** LumenscopeAI 6  
* **パラメータ数:** 30.9億 6  
* **ライセンス:** Apache 2.0 7

#### **アーキテクチャの詳細**

* **基本概念:** このモデルは、Transformerアーキテクチャからの抜本的な脱却ではなく、その主要コンポーネントをSNNの原理を用いて丹念に再実装したものです。「BrainTransformers」と呼ばれるフレームワークに基づいており、具体的な実装名はBrainGPTForCausalLMです 5。  
* **SNNネイティブコンポーネント:** 本モデルの核心的なイノベーションは、SNNMatmul、SNNSoftmax、SNNSiLUといった、SNNと互換性のあるTransformerの構成要素を設計した点にあります 8。このアプローチは、Transformerアーキテクチャが持つ強力な帰納バイアスを維持しつつ、スパイキング計算による潜在的な効率性の利点を享受することを目的としています。  
* **ハイブリッドな性質:** 現時点では、このモデルはハイブリッドシステムです。標準的なハードウェア（GPU）上での計算効率を維持するため、一部の浮動小数点演算が残されています。将来的には、さらなる最適化と専用のSNNハードウェアへの適応が計画されています 5。

#### **学習手法と系譜**

* **基盤モデル:** モデルの学習は、事前学習済みのANN、具体的にはANN-Base-Qwen2からブートストラップされています 5。これは、実用的な「ANNからSNNへ」の知識転移戦略を採用していることを示す重要な詳細です。  
* **3段階の学習:** 学習プロセスは3つの異なる段階を経ており、その中には「SNN固有のニューロンシナプス可塑性トレーニング」という専門的なフェーズが含まれています 5。これは、SNNが標準的なバックプロパゲーションを超えた独自の最適化課題を抱えていることを浮き彫りにしています。

#### **性能と評価**

このモデルは、広範な標準的LLMベンチマークで評価され、競争力のある性能を示しています。自己報告されている主要なスコアには、MMLU (63.2)、BBH (54.1)、ARC-C (54.3)、GSM8K (76.3)、HumanEval (40.5) などがあります 6。これらの結果は、SNNモデルが同規模の従来型LLMに対して、まだ初期段階ではあるものの、実行可能な競合相手として位置づけられることを示す上で極めて重要です。

#### **利用可能性と使用法**

* モデルはHugging FaceおよびWiseModelで公開されています 5。関連するGitHubリポジトリは、完全なtransformersパッケージの代替を提供しており、既存の開発環境へ比較的シームレスに統合することが可能です 5。  
* 明確に述べられている重要な制約として、このモデルは標準的なANNのファインチューニング技術をサポートしていません。開発チームはSNN向けの専用ファインチューニングツールを開発中であり、これは現在のエコシステムにおけるツール不足を示唆しています 5。

#### **分析と示唆**

BrainTransformersは、SNNを普及させるための重要な戦略、すなわち「再発明ではなく翻訳」というアプローチを体現しています。全く新しいアーキテクチャを設計する代わりに、現在主流であるTransformerパラダイムをSNNの領域に適応させているのです。AI業界は、Transformerアーキテクチャの最適化に莫大な計算時間と知的資本を投じてきました。この知識ベースを放棄してゼロから新しいアーキテクチャを構築するのは、非常にリスクが高く非効率的です。BrainTransformersがTransformerのコンポーネント（SNNMatmulなど）に対応するSNN版を作成したアプローチは、Transformerが持つ実証済みのアーキテクチャ上の強みを活用することを可能にしました 8。さらに、学習済みのANN（Qwen2）から学習を開始することで、良好な初期重みを見つけるための膨大でコストのかかる探索プロセスをショートカットしています。これは知識転移として知られる手法です 5。この実用的な「中間的」アプローチは、プロジェクトのリスクを大幅に軽減し、その性能主張をより信頼性が高く、既存のベンチマークと直接比較可能なものにしています。

一方で、ファインチューニングツールが存在しないという明確な言及は 5、SNN開発者エコシステムの未熟さを示す重要な指標です。現代のLLMの価値は、事前学習された状態だけでなく、ファインチューニングによる適応能力にもあります。主要なSNNモデルが標準的なファインチューニング手法を利用できないという事実は、多くの実用的なアプリケーションへの導入における大きな障壁となります。これは、SNNの課題がモデル設計だけでなく、ツール、ライブラリ、確立されたベストプラクティス（例えば、SNN版のPEFTやLoRA）といった周辺エコシステム全体の構築にあることを明らかにしています。したがって、BrainTransformersのようなモデルの将来的な成功は、中核となるモデルアーキテクチャの改善と同じくらい、これらのツールの開発にかかっていると言えるでしょう。

### **2\. SpikeGPT：エネルギー効率に優れたRWKV準拠のスパイキング言語モデル**

#### **モデルプロファイル**

* **リンク:**([https://huggingface.co/ridger/SpikeGPT-BookCorpus](https://huggingface.co/ridger/SpikeGPT-BookCorpus)) 11,([https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M)) 12  
* **関連研究:** *SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks* (arXiv:2302.13939) 3  
* **パラメータ数:** 4500万および2億1600万のバリアントが存在します 3。公開されているモデルはL18-D768（18層、埋め込み次元768）と特定されています 11。

#### **アーキテクチャの詳細**

* **基本概念:** SpikeGPTは、「バイナリでイベント駆動型のスパイキング活性化ユニット」を用いて構築された生成言語モデルです 3。その主な目的は、高いエネルギー効率を達成し、計算の複雑さを削減することです。  
* **RWKVからの着想:** このアーキテクチャは、Receptance Weighted Key Value (RWKV) 言語モデルから明確に着想を得ています 3。これは極めて重要な設計上の選択です。RWKVは、Transformerレベルの性能を持つ再帰型ニューラルネットワーク（RNN）でありながら、標準的なアテンション機構の二次的な計算量（）とは対照的に、系列長に対して線形の計算量（）を実現しています。  
* **線形計算量:** RWKVのような逐次処理アプローチを採用することで、SpikeGPTは自己アテンションの二次的な計算量を回避し、理論上、非常に長い系列に対してより効率的になります 3。これにより、BrainTransformersが採用したTransformer適応アプローチとは異なる選択肢を提示しています。

#### **学習と利用可能性**

* **学習データ:** 公開されているモデルは、BookCorpus 11 や OpenWebText 12 といった著名なコーパスで学習されています。GitHubリポジトリには、WikiText-103などのデータセットを用いた事前学習やファインチューニングに関する詳細な手順が記載されています 15。  
* **モデルの利用可能性:** 2億1600万パラメータモデルの事前学習済み重みはHugging Face Hubで利用可能です 12。プロジェクトのGitHubが、コードと学習スクリプトの主要な情報源となっています 3。

#### **性能に関する主張**

* SpikeGPTの中心的な主張は、適切なニューロモーフィックハードウェア上で処理された場合、非スパイキングモデルと同等のベンチマーク性能を維持しつつ、演算回数を大幅に（最大で20分の1に）削減できるという点です 3。これは、SNNソフトウェアと専用ハードウェアが相互に依存しあって初めて、その効率性の利点を最大限に引き出せることを示唆しています。

#### **分析と示唆**

SpikeGPTとBrainTransformersは、SNNベースのLLMを構築するための2つの競合する哲学を代表しています。BrainTransformersは主流であるアテンションベースのパラダイムを「適応」させるのに対し、SpikeGPTはSNNの逐次的で状態を持つ性質と本質的により整合性の高い「代替的」な再帰型パラダイム（RWKV）を採用しています。Transformerは基本的に並列処理であり、自己アテンションを介して系列内の全トークンを同時に処理します。これは、SNNが持つ時間的でステップバイステップの処理様式とは相容れない側面があります。対照的に、RNNは本質的に逐次的であり、一度に1つのトークンを処理し、状態を維持します。この特性は、スパイキングニューロンの時間的ダイナミクスと完全に一致します。RWKVアーキテクチャは、RNN設計でもTransformerレベルの性能を達成できることを示しました。したがって、SpikeGPTがRWKVを基盤として選択したのは偶然ではなく、戦略的な賭けです。つまり、「本質的に並列」なTransformerアーキテクチャをスパイキングフレームワークに無理やり押し込むよりも、「本質的に逐次的」なアーキテクチャの方がSNNの実装に適しており、より効率的でエレガントであるという考え方です。これは、SNN LLMの分野における根本的な設計上の分岐点を示しています。

さらに、SpikeGPTの性能に関する主張は、「ニューロモーフィックハードウェア」の使用に決定的に依存しています 3。論文では演算回数が「20分の1」に削減されると主張されていますが、これは標準的なGPUでは達成不可能です。GPUはANNの主要な演算である高密度の行列積に最適化されています。スパースでイベント駆動型の加算を行うSNNは、このハードウェアには不向きであり、特別に設計されていない限り、実際にはGPU上でより低速に動作することさえあります（BrainTransformersのハイブリッドアプローチがこの問題への対処を示唆しています）。SNNの真の利点は、スパース性や非同期イベントを活用するように設計されたハードウェア（例：IntelのLoihi）でのみ発揮されます。これは、SpikeGPTのようなモデルがもたらす広範な実用的利益は、この種の専用ハードウェアが将来的に普及し、利用可能になるかどうかにかかっていることを意味します。これは、理想的なハードウェアの登場を待つソフトウェアソリューションと言えるでしょう。

## **第2部：Hugging Face Papersを通じて発見可能な著名なSNN研究プロジェクト**

このセクションでは、モデル自体はHugging Face Hubで直接ホストされていないものの、技術論文を通じてHugging Faceエコシステム内で高い可視性を持つ、影響力のあるSNNプロジェクトを探ります。これらはSNN研究の最先端を代表し、将来の方向性を示唆するものです。

### **3\. SpikingBrain：効率的な長文脈SNNのためのフレームワーク**

#### **プロジェクトプロファイル**

* **関連研究:** *SpikingBrain Technical Report: Spiking Brain-inspired Large Models* (arXiv:2509.05276) 17  
* **開発元:** 中国科学院 脳型コンピューティング研究所 (BICLab) 19  
* **モデル:** SpikingBrain-7B（線形LLM）およびSpikingBrain-76B（ハイブリッド線形Mixture-of-Experts LLM）を含むモデルファミリー 17

#### **主要な目的とイノベーション**

* **長文脈における効率性:** 主な目標は、長い系列を処理する際のTransformerの効率性のボトルネックを解決することです 17。二次的な計算量から脱却するため、線形およびハイブリッド線形の注意機構アーキテクチャを採用しています 20。  
* **ハードウェアの独立性（非NVIDIA）:** 際立った特徴は、NVIDIA製ではないGPUクラスタ（MetaX）上で開発および学習が行われた点です 17。これは、支配的なハードウェア供給元から独立した大規模AIシステムを構築するための重要な戦略的取り組みです。  
* **適応的スパイキング:** モデルは「適応的スパイキングニューロン」と専用のスパイクコーディングフレームワークを使用し、活性化をスパースなスパイク列に変換することで、効率的な加算ベースの計算を可能にしています 18。

#### **性能と利用可能性**

* **性能に関する主張:** このプロジェクトは、長い系列に対する劇的な高速化を報告しており、SpikingBrain-7Bは400万トークンの系列に対して最初のトークン生成時間（TTFT）で100倍以上の高速化を達成したとされています 17。また、69.15%という高いスパース性を実現しており、これは低電力動作の鍵となります 18。  
* **利用可能性:** コードはGitHubで公開されています（([https://github.com/BICLab/SpikingBrain-7B](https://github.com/BICLab/SpikingBrain-7B))） 20。モデルの重みは、Hugging Face Hub自体ではなく、それに類似したプラットフォームであるModelScopeでホストされています 21。プロジェクトは、Hugging Face互換フォーマット、vLLM推論版、量子化版など、様々なデプロイメントバージョンを提供しています 21。

#### **分析と示唆**

SpikingBrainがNVIDIA製ではない（MetaX）ハードウェア上で明示的に開発されたことは、単なる技術的な選択ではなく、戦略的な選択です。世界のAI産業は、大規模モデルの学習においてNVIDIA製GPUに決定的に依存しています。この依存はサプライチェーンのリスクを生み出し、半導体の輸出規制といった地政学的な圧力の影響を受けやすくなります 19。中国の主要な国立研究機関（中国科学院）が国内のハードウェア上で最先端のAIモデルを開発することは、主権を持つ垂直統合されたAIエコシステムを構築しようとする明確な努力です。したがって、SpikingBrainは単なる研究論文ではなく、AIにおける技術的独立性のための概念実証であり、SNNがこの代替ハードウェアに最適化されたアーキテクチャの道筋を提供する可能性を示唆しています。

また、SpikingBrainが「ハイブリッド線形」アテンション（線形、局所、標準アテンションの組み合わせ）を使用していることは、純粋なSNNアプローチだけではまだ最先端の性能を達成するには不十分かもしれないことを示しています 20。純粋な線形アテンションモデルは、すべてのタスクで完全な二次アテンションの性能に匹敵するのに苦労することがよくあります。SpikingBrainのアーキテクチャは、おそらく重要な局所的文脈のために計算コストの高い標準アテンションを使用し、長距離の依存関係のためにより効率的な線形アテンションを使用するなど、異なるアテンションタイプを混合しています。このハイブリッドアプローチは、実用的なエンジニアリング上の妥協点です。つまり、性能と効率のより良いバランスを達成するために、アーキテクチャの純粋性を犠牲にしているのです。これは、効率的なLLMへの道が単一の「魔法の弾丸」的アーキテクチャではなく、異なる特化コンポーネントの巧妙な組み合わせにある可能性を示唆しています。

### **4\. SpikeLLM：SNNを700億パラメータ以上にスケールアップさせるための研究イニシアチブ**

#### **プロジェクトプロファイル**

* **関連研究:** *SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking* (arXiv:2407.04752) 22  
* **開発元:** 清華大学などの研究者による共同研究（29などの論文の著者所属から推測）

#### **主要な目的とイノベーション**

* **大規模スケーリング:** プロジェクトの明確な目標は、SNNを現代のLLM（70億から700億パラメータ）のレベル、さらには人間の脳の規模（860億ニューロン）にまでスケールアップさせることです 22。  
* **一般化積分発火（GIF）ニューロン:** 主要な理論的貢献は、GIFニューロンの提案です。この技術は、スパイク長（時間ステップ数）を圧縮することで、スパイクベースのエンコーディングをより効率的にすることを目的としており、従来のSNNにおける主要なボトルネックに対処するものです 22。  
* **顕著性ベースのスパイキング（Optimal Brain Spiking）:** このフレームワークは、ネットワーク内の「外れ値」または「顕著な」チャネルを特定し、それらにより多くの計算リソース（つまり、より多くのスパイキング時間ステップ）を割り当てる一方、重要度の低いチャネルにはより少ないリソースを割り当てる手法を導入しています 22。これは、時間領域における混合精度量子化に類似した考え方です。

#### **性能と利用可能性**

* **性能に関する主張:** 論文では、SpikeLLMがLLAMA-7Bのようなモデルにおいて、OmniQuantやGPTQといった標準的な量子化手法を上回る性能向上を達成できると主張しており、その生物学的に妥当なスパイキング機構が単純な低ビット量子化よりも効果的であることを示しています 22。  
* **利用可能性:** このプロジェクトは主に研究実装です。コードはGitHubで公開されていますが（([https://github.com/Xingrun-Xing2/SpikeLLM](https://github.com/Xingrun-Xing2/SpikeLLM))）、現時点ではHugging Faceや他のプラットフォームでダウンロード可能な事前学習済みモデルの重みはありません 24。

#### **分析と示唆**

SpikeLLMは、SNNに関する議論を再構築しています。それは、スパイキング機構を単なる生物学に着想を得た興味深い技術としてではなく、本質的により知的で効率的な量子化の一形態として位置づけています。量子化は、メモリを節約し計算を高速化するために、重みや活性化の精度を（例えば32ビット浮動小数点数から4ビット整数へ）低下させます。スパイキングは、連続的な活性化値をバイナリイベント（スパイクの有無）に変換するため、究極の量子化と見なすことができます。しかし、単純な二値化はLLMでは失敗することが多いです。SpikeLLMの論文は、その手法（GIFニューロン、顕著性ベースのスパイキング）が、より洗練された動的な量子化形式を提供し、より頑健であると主張しています。同じベースモデル（LLAMA-7B）上で、GPTQのような確立された量子化手法と比較して優れた性能を実証することで 22、SpikeLLMはSNNの原理がモデル圧縮のための直接的でドロップイン可能な改善策となり得るという強力な事例を提示しています。これは非常に価値があり実用的な応用です。

この研究は、BrainTransformersと共に、ANNとSNNの間の境界線が曖昧になっていることを示しています。将来は、一方が他方を完全に置き換えるのではなく、概念の融合が進む可能性があります。SpikeLLMは、新しいSNNをゼロから構築するのではなく、既存のLLM（LLaMAなど）をスパイキング機構で「再設計」します 22。これは、量子化やプルーニングが今日適用されているように、「スパイキング層」や「スパイキングモード」が既存のANNアーキテクチャに統合される未来を示唆しています。LLMが、あるタスクには標準的なANN層を使用し、他のタスクには高効率なスパイキングモードに切り替える、あるいはハイブリッドな構成を持つようになるかもしれません。この融合アプローチは、業界が既存のモデルやインフラへの莫大な投資を活用しつつ、ニューロモーフィックコンピューティングの利点を段階的に取り入れることを可能にするため、採用に向けてより実用的です。

## **第3部：テーマ別分析と将来展望**

### **1\. SNN言語モデルアーキテクチャの比較分析**

このセクションでは、第1部と第2部で得られた知見を統合し、異なるアーキテクチャ哲学を比較します。

* **Transformer適応型 (BrainTransformers):**  
  * **長所:** 強力でよく理解されているTransformerアーキテクチャを活用しており、性能ベンチマークが直接比較可能。  
  * **短所:** 並列アーキテクチャを時間的計算パラダイムにマッピングする非効率な方法である可能性がある。  
* **再帰型/逐次型 (SpikeGPT):**  
  * **長所:** アーキテクチャ的にSNNとより自然に適合し、線形計算量を約束する。  
  * **短所:** 主流のTransformerパラダイムから離れるため、直接的な比較や統合がより複雑になる。  
* **ハイブリッド/コンポーネントベース (SpikingBrain, SpikeLLM):**  
  * **長所:** 実用的で、両者の長所を組み合わせたアプローチ。既存のモデルに適用可能。  
  * **短所:** アーキテクチャの複雑性が増す。

以下に、主要なSNN言語モデルの比較概要を示します。この表は、研究者やエンジニアがこれらの複雑なプロジェクト間の主な違いを迅速に把握するためのものです。選択された列は、誰が作ったのか、どのくらいの大きさか、何に基づいているのか、何が新しいのか、今すぐ使えるのか、どのハードウェアを対象としているのか、といった最も重要な問いに直接答えることを目的としています。

| モデル名 | 主要開発元 | パラメータサイズ | ベースアーキテクチャ | 主要なイノベーション | Hugging Face Hubでの利用可能性 | ハードウェア焦点 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **BrainTransformers-3B-Chat** | LumenscopeAI | 30.9億 | Transformer (Qwen2) | SNN互換のTransformerコンポーネント | あり 6 | GPU (現在), SNNハードウェア (将来) |
| **SpikeGPT** | Rui-Jie Zhu, et al. | 4500万, 2億1600万 | RWKV (RNN) | 線形計算量を持つSNN-ネイティブなRNN | あり 11 | ニューロモーフィックハードウェア |
| **SpikingBrain** | BICLab (中国科学院) | 70億, 760億 | ハイブリッド線形アテンション | 長文脈効率、非NVIDIAハードウェアでの学習 | なし (コードはGitHub、重みはModelScope) 21 | MetaX GPU (非NVIDIA) |
| **SpikeLLM** | Xingrun Xing, et al. | 70億～700億 (研究対象) | Transformer (LLaMA) | SNNを高度な量子化として利用、GIFニューロン | なし (コードはGitHub) 27 | 汎用 (GPU), ニューロモーフィックハードウェア |

### **2\. SNNの広範なトレンドと応用**

本レポートはLLMに焦点を当てていますが、SNNがAIの様々な分野で応用されていることに言及することは重要です。収集された情報からは、以下の分野での研究が示唆されています。

* **コンピュータビジョン:** 3Dシーン再構成のためのSpikingNeRF 28、EMS-YOLOによる物体検出 1、ロボット工学のための視覚的場所認識 30。  
* **ロボット工学:** ロボットの行動軌道生成のための脳型行動生成 31。  
* **信号処理:** 脳波（EEG）データからのてんかん発作検出 32。

これらの事例は、特に時間的データを含むタスクや、エッジデバイスでの低電力・低遅延推論が求められるタスクにおいて、SNNパラダイムが持つ多様性を示しています 1。

### **3\. 根強い課題と今後の道のり**

* **学習の複雑性:** SNNの学習の難しさは繰り返し指摘されるテーマです。SNNは微分不可能であるため、代理勾配法 29 や複雑な多段階の学習パイプライン 5 が必要となります。これは、ANNで確立されたバックプロパゲーション法と比較して、依然として大きな障壁です。  
* **ツールとエコシステムのギャップ:** BrainTransformersが明らかにしたように、ファインチューニングのようなタスクのための成熟したツールが存在しないことは、実用的な導入における大きな障壁です 5。SNNコミュニティは、ライブラリやベストプラクティスからなる堅牢なエコシステムを構築する必要があります。  
* **ハードウェアへの依存:** SNNの潜在能力を最大限に引き出すことは、専用のニューロモーフィックハードウェアの利用可能性と本質的に結びついています 3。SpikingBrainのようなプロジェクトは代替的な従来型ハードウェアを模索していますが、このパラダイムの最終的な成功は、ソフトウェアとハードウェアの共進化にかかっています。

## **結論：Hugging FaceにおけるSNNの現状—転換点に立つ分野**

Hugging FaceにおけるSNNの現状は、いくつかの先駆的で利用可能なモデルと、その背後にある活発な最先端研究の流れによって特徴づけられます。主要なアーキテクチャに関する議論、すなわちTransformerを適応させるか、新しい再帰型設計を採用するかが続いています。

また、SNNが既存のANNに対する高度で生物学的に妥当なモデル圧縮および効率向上技術の一形態として台頭している傾向が明らかになりました。

最終的な評価として、学習、ツール、ハードウェアの面で依然として大きな課題は残るものの、Hugging Faceのような主流のプラットフォーム上でこれらのプロジェクトが存在し、その質が高いことは、SNNが理論的な概念から、より効率的で高性能な人工知能を追求する上で、実用的かつますます重要となる最前線へと移行しつつあることを示しています。

#### **引用文献**

1. Daily Papers \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers?q=Spiking%20U-Net](https://huggingface.co/papers?q=Spiking+U-Net)  
2. Daily Papers \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers?q=Spiking%20neural%20networks](https://huggingface.co/papers?q=Spiking+neural+networks)  
3. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2302.13939](https://huggingface.co/papers/2302.13939)  
4. SpikeGPT: researcher releases code for largest-ever spiking neural network for language generation \- UC Santa Cruz \- News, 10月 9, 2025にアクセス、 [https://news.ucsc.edu/2023/03/eshraghian-spikegpt/](https://news.ucsc.edu/2023/03/eshraghian-spikegpt/)  
5. BrainTransformers-SNN-LLM \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM](https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM)  
6. LumenscopeAI/BrainTransformers-3B-Chat · Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat)  
7. 4.52 kB \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat/resolve/1abe35d6bffbb9405ce72cea59f68b86bb115369/README.md?download=true](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat/resolve/1abe35d6bffbb9405ce72cea59f68b86bb115369/README.md?download=true)  
8. Paper page \- BrainTransformers: SNN-LLM \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2410.14687](https://huggingface.co/papers/2410.14687)  
9. Update README.md · LumenscopeAI/BrainTransformers-3B-Chat at e52ee89, 10月 9, 2025にアクセス、 [https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat/commit/e52ee8987dd7321e802021f24ca3eb152da78c88](https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat/commit/e52ee8987dd7321e802021f24ca3eb152da78c88)  
10. BrainTransformers 3B Chat · Models \- Dataloop, 10月 9, 2025にアクセス、 [https://dataloop.ai/library/model/lumenscopeai\_braintransformers-3b-chat/](https://dataloop.ai/library/model/lumenscopeai_braintransformers-3b-chat/)  
11. ridger/SpikeGPT-BookCorpus · Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/ridger/SpikeGPT-BookCorpus](https://huggingface.co/ridger/SpikeGPT-BookCorpus)  
12. ridger/SpikeGPT-OpenWebText-216M · Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M)  
13. SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2302.13939](https://arxiv.org/abs/2302.13939)  
14. README.md · ridger/SpikeGPT-BookCorpus at main \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/ridger/SpikeGPT-BookCorpus/blob/main/README.md](https://huggingface.co/ridger/SpikeGPT-BookCorpus/blob/main/README.md)  
15. Implementation of "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks" \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/ridgerchu/SpikeGPT](https://github.com/ridgerchu/SpikeGPT)  
16. ridger/SpikeGPT-OpenWebText-216M at ca3ff6b \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M/commit/ca3ff6b10738970914883d6f0400dde0b9c07175](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M/commit/ca3ff6b10738970914883d6f0400dde0b9c07175)  
17. SpikingBrain Technical Report: Spiking Brain-inspired Large Models \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2509.05276](https://huggingface.co/papers/2509.05276)  
18. \[2509.05276\] SpikingBrain Technical Report: Spiking Brain-inspired Large Models \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2509.05276](https://arxiv.org/abs/2509.05276)  
19. China's SpikingBrain-7B: 100x Faster Brain-Inspired AI Model \- WebProNews, 10月 9, 2025にアクセス、 [https://www.webpronews.com/chinas-spikingbrain-7b-100x-faster-brain-inspired-ai-model/](https://www.webpronews.com/chinas-spikingbrain-7b-100x-faster-brain-inspired-ai-model/)  
20. SpikingBrain Technical Report: Spiking Brain-inspired Large Models \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)  
21. BICLab/SpikingBrain-7B \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/BICLab/SpikingBrain-7B](https://github.com/BICLab/SpikingBrain-7B)  
22. Paper page \- SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2407.04752](https://huggingface.co/papers/2407.04752)  
23. \[2407.04752\] SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2407.04752](https://arxiv.org/abs/2407.04752)  
24. SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/html/2407.04752v3](https://arxiv.org/html/2407.04752v3)  
25. \[Quick Review\] SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- Liner, 10月 9, 2025にアクセス、 [https://liner.com/review/spikellm-scaling-up-spiking-neural-network-to-large-language-models](https://liner.com/review/spikellm-scaling-up-spiking-neural-network-to-large-language-models)  
26. Daily Papers \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers?q=PB-LLM](https://huggingface.co/papers?q=PB-LLM)  
27. Xingrun-Xing2/SpikeLLM: This is the implentation of our paper "SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking" in ICLR 2025\. \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/Xingrun-Xing2/SpikeLLM](https://github.com/Xingrun-Xing2/SpikeLLM)  
28. SpikingNeRF: Making Bio-inspired Neural Networks See through the Real World \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/blog/mikelabs/spikingnerf-making-bio-inspired-neural-networks-se](https://huggingface.co/blog/mikelabs/spikingnerf-making-bio-inspired-neural-networks-se)  
29. Deep Directly-Trained Spiking Neural Networks for Object Detection \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2307.11411](https://huggingface.co/papers/2307.11411)  
30. Daily Papers \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers?q=neuromorphic%20spiking%20neural%20networks](https://huggingface.co/papers?q=neuromorphic+spiking+neural+networks)  
31. Brain-inspired Action Generation with Spiking Transformer Diffusion Policy Model \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers/2411.09953](https://huggingface.co/papers/2411.09953)  
32. Daily Papers \- Hugging Face, 10月 9, 2025にアクセス、 [https://huggingface.co/papers?q=spiking%20neuron%20layer](https://huggingface.co/papers?q=spiking+neuron+layer)