

# **スパイキングニューラルネットワークの総合ガイド：フレームワーク、モデル、実践的考察**

## **序論**

スパイキングニューラルネットワーク（SNN）は、生物の脳が情報を処理する方法に着想を得た、次世代のニューラルネットワークとして注目を集めています。従来の人工ニューラルネットワーク（ANN）が連続的な値で情報を扱うのに対し、SNNは「スパイク」と呼ばれる離散的なイベントを用いて通信します。このイベント駆動型の計算モデルは、特にニューロモルフィックハードウェア上で実行された際に、驚異的なエネルギー効率を実現する可能性を秘めています。

本ドキュメントは、GitHub上で公開されている高性能なSNNプロジェクトに関する包括的な調査結果をまとめたものです。SNN開発を支える主要なソフトウェアフレームワークの分析から、学習済みモデルが公開されている最先端の研究プロジェクトのカタログ、さらには各ライブラリの実践的な利点と欠点に至るまで、SNNの現状を多角的に掘り下げます。このドキュメントが、研究者やエンジニアがSNNの世界を探求し、自身のプロジェクトに応用するための確かな指針となることを目指します。

## **第I部：SNNフレームワークのエコシステム**

SNNの研究開発を支えるソフトウェアライブラリは、それぞれ異なる設計思想と目標を持っています。本セクションでは、主要なフレームワークを詳細に分析し、その特徴と技術的背景を明らかにします。

### **1.1 PyTorchベースフレームワークの隆盛**

現代の高性能SNNフレームワークの多くは、深層学習で広く利用されているPyTorchを基盤としています 1。これは、PyTorchが提供する柔軟な自動微分エンジンautogradが、SNNの学習における最大の課題であった「スパイク発火の非微分可能性」を、代理勾配法（Surrogate Gradient Method）という技術で克服するための理想的な基盤を提供したためです 2。

* **SpikingJelly**: SNN研究のためのフルスタックなツールキットであり、データセットの前処理からモデル構築、最適化、ニューロモルフィックチップへのデプロイまでを一貫してサポートします 3。特にCUDAで強化されたニューロンモジュールは、従来のシミュレーションを大幅に高速化します 3。そのAPIはPyTorchと酷似しており、開発者は低い学習コストで高性能なSNNを構築できます 3。  
* **snnTorch**: アクセシビリティと使いやすさを最重視して設計されており、SNN初学者にとって最適な入門ツールです 5。代理勾配法に特化し、豊富なチュートリアルと優れたドキュメントを通じて、SNNの基本原理を迅速に理解し実践に移すことができます 6。  
* **Norse**: 深層学習の実践的なアプローチと、生物学的な着想に基づく神経コンポーネントの融合を目指しています 7。PyTorch Lightningとの統合により、小規模な実験からHPCクラスターでの大規模研究まで、容易にスケールアップが可能です 7。  
* **BindsNET**: 生物学的妥当性を強く志向し、スパイクタイミング依存可塑性（STDP）を主要な学習メカニズムとして採用しています 8。計算論的神経科学の文脈で、学習プロセスそのものを探求する研究に強力なツールとなります 8。  
* **PySNN**: GPUアクセラレーションを活用した効率的なシミュレーションを目指し、特に相関ベースの学習則での使用を意図しています 9。ネットワークモジュールをNeuronとConnectionに明確に分離した設計が特徴です 9。

### **1.2 広範なエコシステム：特化型ツール**

PyTorchベースのフレームワーク以外にも、多様なニーズに応える特化したツールが存在します。

* **神経科学シミュレータ (NEST, Brian2)**: 機械学習の性能よりも、生物学的な詳細度の高さを優先して設計されています。脳の情報処理メカニズムを探求するための標準ツールです。  
* **JAXエコシステム (Spyx)**: JAXのJust-In-Time (JIT) コンパイラを活用し、SNNのトレーニングと推論の効率を極限まで高めることを目指しています。  
* **ハードウェア中心フレームワーク (Lava)**: IntelのLoihiのようなニューロモルフィックハードウェア上での効率的な実行を最終目標として設計されています。ソフトウェアとハードウェアの協調設計を体現しています。

### **表1：主要SNNフレームワークの比較**

| フレームワーク名 | 主要バックエンド | 主な学習パラダイム | 主要な特徴 |
| :---- | :---- | :---- | :---- |
| **SpikingJelly** | PyTorch | 代理勾配法, ANN-SNN変換 | CUDAによる高速化、フルスタックツールキット |
| **snnTorch** | PyTorch | 代理勾配法 | 優れたドキュメントとチュートリアル、使いやすさ |
| **Norse** | PyTorch | 代理勾配法 | 生物学的着想に基づくプリミティブ、HPC対応 |
| **BindsNET** | PyTorch | STDP (ヘブ学習) | 生物学的妥当性、強化学習への応用 |
| **PySNN** | PyTorch | 相関ベース学習 | NeuronとConnectionのモジュール分離 |
| **Spyx** | JAX | 代理勾配法 | JITコンパイルによる高性能化 |
| **NEST / Brian2** | C++ / Python | \- | 大規模、高忠実度な神経科学シミュレーション |
| **Lava** | \- | \- | Intel Loihiハードウェアへのデプロイ |

## **第II部：高性能SNNモデルのカタログ**

本セクションでは、学習済みモデルを公開している影響力の大きい研究プロジェクトを厳選して紹介します。これにより、読者は最先端のSNNモデルを即座に自身の環境で試すことができます。

### **2.1 基礎モデル**

* **SpikingVGGおよびSpikingResNet (SpikingJelly経由)**  
  * **概要**: VGGやResNetといった古典的なANNアーキテクチャのスパイク版を提供します 10。pretrained=Trueと指定するだけで、ImageNetで事前学習されたANNの重みをSNNに変換して利用でき、高性能なバックボーンを容易に構築できます 11。  
  * **リポジトリ**: [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) 3

### **2.2 先進的モデルと学習手法**

* **OTTT-SNN: オンライン時間的学習**  
  * **概要**: メモリ効率に優れたオンライン学習アルゴリズム「OTTT」を提案 12。従来のBPTT（Backpropagation Through Time）が抱えていた、タイムステップ数に比例してメモリ消費量が増大する問題を解決し、非常に長い時系列データでの学習を可能にしました 12。  
  * **リポジトリ**:([https://github.com/pkuxmq/OTTT-SNN](https://github.com/pkuxmq/OTTT-SNN)) 12  
  * **モデルリンク**:([https://drive.google.com/drive/folders/1eDn3mVgfBHTLBfb--WawgA5Qms4oFbZ4?usp=sharing),(https://pan.baidu.com/s/1q0ljZiCVIUW41Hh-aol2Zg](https://drive.google.com/drive/folders/1eDn3mVgfBHTLBfb--WawgA5Qms4oFbZ4?usp=sharing),([https://pan.baidu.com/s/1q0ljZiCVIUW41Hh-aol2Zg](https://pan.baidu.com/s/1q0ljZiCVIUW41Hh-aol2Zg))) (コード: gppq) 12  
* **ProxyLearning-SNN: ANNプロキシを介した学習**  
  * **概要**: SNNとANNが重みを共有し、学習時の誤差を微分可能なANNを通じて逆伝播させる独創的な手法 13。SNNの非微分可能性の問題を回避しつつ、短いシミュレーション時間（低遅延）で高い性能を達成します 13。  
  * **リポジトリ**:([https://github.com/SRKH/ProxyLearning](https://github.com/SRKH/ProxyLearning)) 13  
  * **モデルリンク**:([https://www.dropbox.com/s/xhg0e2ndesqu5tj/Pretrained.zip?dl=0](https://www.dropbox.com/s/xhg0e2ndesqu5tj/Pretrained.zip?dl=0)) 13  
* **SNASNet: SNNのためのニューラルアーキテクチャ探索**  
  * **概要**: SNNに特化したニューラルアーキテクチャ探索（NAS）を初めて成功させた研究の一つ 15。特に、時間的なフィードバック結合（後方結合）を持つアーキテクチャが性能を向上させることを発見し、SNNにおけるリカレントな情報処理の重要性を示しました 15。  
  * **リポジトリ**:([https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks](https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks)) 15  
  * **モデルリンク**:([https://drive.google.com/file/d/1irW6V4MNt0BOkNWAP55X\_yjrcIt-ulVK/view?usp=sharing](https://drive.google.com/file/d/1irW6V4MNt0BOkNWAP55X_yjrcIt-ulVK/view?usp=sharing)) 15

### **表2：公開学習済みモデルを持つ高性能SNN**

| プロジェクト名 | 主な貢献 | アーキテクチャ | タスク/データセット | 報告性能（精度 & タイムステップ） |
| :---- | :---- | :---- | :---- | :---- |
| **SpikingVGG/ResNet** | ANNからの転移学習 | VGG, ResNet | 画像分類 / ImageNet | ANNに準ずる |
| **OTTT-SNN** | メモリ効率に優れたオンライン学習 | VGG-like | CIFAR-10/100, ImageNet | 92.55% (CIFAR-10, T=4) |
| **ProxyLearning-SNN** | ANNを代理としたSNN学習 | Deep ConvSNN | Fashion-MNIST, CIFAR-10 | 94.56% (F-MNIST, T=50) |
| **SNASNet** | SNN特化のNAS | 探索された新規アーキテクチャ | CIFAR-10, CIFAR-100 | 93.73% (CIFAR-10, T=5) |

## **第III部：ライブラリの欠点と実践的考察**

各ライブラリは強力な機能を持つ一方で、その設計思想に起因するトレードオフや欠点も存在します。ここでは、ライブラリ選定の際に考慮すべき実践的な注意点を解説します。

### **3.1 主要ライブラリの欠点**

* **SpikingJelly**:  
  * **複雑性**: フルスタックな多機能性ゆえに、初学者には学習コストが高い可能性があります。  
  * **カスタムCUDAへの依存**: 高速化は独自のCUDAカーネルに依存しており、これが柔軟性の低下や特定のGPU環境への依存につながることがあります。CPUのみでの実行がサポートされないバックエンドも存在します 3。  
* **snnTorch**:  
  * **パフォーマンスの限界**: 使いやすさを重視し、純粋なPyTorchの拡張として設計されているため、カスタムカーネルを持つSpikingJellyと比較して、大規模ネットワークにおける計算速度で劣る可能性があります。  
  * **機能の焦点**: 主に代理勾配法に特化しており、ANN-SNN変換などの機能は限定的です 2。  
* **Spyx (JAXベース)**:  
  * **学習コスト**: JAXはPyTorchとは異なる関数型プログラミングのパラダイムを採用しており、習得が難しい場合があります。エコシステムもPyTorchほど成熟していません。  
  * **アーキテクチャの制約**: JITコンパイルの設計上、任意のリカレント結合（層をまたぐフィードバック接続など）のサポートが困難という大きな制約があります。  
* **Lava (Intel製)**:  
  * **ハードウェアへの強い依存**: IntelのLoihiチップに最適化されているため、専用ハードウェアへのアクセスがないユーザーはその真価を発揮できません。  
  * **独自性の高いプログラミングモデル**: 習得に時間が必要な独自のプロセスベースのプログラミングモデルを採用しています。  
* **Norse**:  
  * **ニッチな立ち位置**: 初心者向けのsnnTorchと、高性能なSpikingJellyの中間に位置し、特定の研究目的以外では他のライブラリが選ばれがちです。学習済みモデルも提供されていません 7。  
* **BindsNET**:  
  * **性能の限界**: 生物学的妥当性を重視するSTDP学習は、複雑なタスクにおける精度で代理勾配法に劣る傾向があります。深層ネットワークへの適用も困難です 19。

### **表3：主要SNNライブラリの欠点比較**

| ライブラリ | 主な欠点・トレードオフ |
| :---- | :---- |
| **SpikingJelly** | 複雑性が高く、学習コストがかかる。カスタムCUDAへの依存。 |
| **snnTorch** | 純粋なPyTorch実装のため、性能面で劣る可能性がある。 |
| **Spyx (JAX)** | JAX特有の学習コスト。アーキテクチャに制約がある。 |
| **Lava (Intel)** | 専用ハードウェアへのアクセスがなければ利点が限定的。 |
| **Norse** | 機能がニッチで、学習済みモデルが不足。 |
| **BindsNET** | 代理勾配法に比べ性能が低い。深層ネットワークへの適用が困難。 |

## **第IV部：将来展望と結論**

### **4.1 方法論的トリレンマと今後の課題**

現在のSNN開発は、**(1) 高い精度、(2) 低い遅延／高い効率、(3) 生物学的妥当性**という3つの目標の間でトレードオフを迫られる「トリレンマ」に直面しています。

* **ANN-SNN変換**: 高い精度を達成できるが、遅延が大きい 2。  
* **代理勾配法**: 高い精度と低い遅延を両立するが、生物学的妥当性はない 2。  
* **生物学的妥当学習 (STDP)**: 生物学的に妥当だが、現状では精度が低い 2。

このトリレンマを解決し、3つの特性を同時に満たす学習アルゴリズムの開発が、SNN研究における究極の目標の一つと言えるでしょう。

### **4.2 主流AIとの収束**

かつてニッチな分野と見なされていたSNNは、今や主流のAI技術と急速に融合しつつあります。SpikingBERT 20、SpikeLLM 21、Spiking Transformer 22 といったプロジェクトは、SNNの原理を現代AIの最先端アーキテクチャに適用する試みです。SNNの未来は、ANNを「置き換える」のではなく、LLMのような巨大モデルのエネルギー効率を改善する要素技術として「融合する」ことにあるのかもしれません。

### **結論**

本調査を通じて、SNNの分野が基礎研究の段階を越え、実用的な性能と工学的な再現性を追求する新たなフェーズに突入していることが明らかになりました。PyTorchベースのフレームワークがエコシステムの中心となり、代理勾配法によって性能が飛躍的に向上しています。また、トップレベルの研究において学習済みモデルの公開が標準化しつつあり、これは分野の成熟を示す重要な兆候です。

一方で、各ライブラリは依然として性能、使いやすさ、生物学的妥当性といった要素の間でトレードオフを抱えており、開発者はプロジェクトの目的に応じて慎重な選択を迫られます。この「方法論的トリレンマ」の解決と、Transformerなどの主流AIアーキテクチャとのさらなる融合が、SNNの未来を形作る鍵となるでしょう。本ドキュメントが、このエキサイティングな分野における研究開発の一助となれば幸いです。

#### **引用文献**

1. Spiking Neural Network (SNN) Frameworks \- Open Neuromorphic, 10月 9, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/)  
2. Has anyone used Spiking Neural Networks (SNNs) for image processing? \- Reddit, 10月 9, 2025にアクセス、 [https://www.reddit.com/r/computervision/comments/tu15ev/has\_anyone\_used\_spiking\_neural\_networks\_snns\_for/](https://www.reddit.com/r/computervision/comments/tu15ev/has_anyone_used_spiking_neural_networks_snns_for/)  
3. SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)  
4. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence \- ResearchGate, 10月 9, 2025にアクセス、 [https://www.researchgate.net/publication/374526125\_SpikingJelly\_An\_open-source\_machine\_learning\_infrastructure\_platform\_for\_spike-based\_intelligence](https://www.researchgate.net/publication/374526125_SpikingJelly_An_open-source_machine_learning_infrastructure_platform_for_spike-based_intelligence)  
5. Best Python Library for Spiking Neural Networks? \- Data Science Stack Exchange, 10月 9, 2025にアクセス、 [https://datascience.stackexchange.com/questions/111886/best-python-library-for-spiking-neural-networks](https://datascience.stackexchange.com/questions/111886/best-python-library-for-spiking-neural-networks)  
6. jeshraghian/snntorch: Deep and online learning with ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/jeshraghian/snntorch](https://github.com/jeshraghian/snntorch)  
7. norse/norse: Deep learning with spiking neural networks ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/norse/norse](https://github.com/norse/norse)  
8. BindsNET/bindsnet: Simulation of spiking neural networks ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/BindsNET/bindsnet](https://github.com/BindsNET/bindsnet)  
9. BasBuller/PySNN: Efficient Spiking Neural Network ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/BasBuller/PySNN](https://github.com/BasBuller/PySNN)  
10. spikingjelly.activation\_based.model.spiking\_vgg 源代码 \- Read the Docs, 10月 9, 2025にアクセス、 [https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/\_modules/spikingjelly/activation\_based/model/spiking\_vgg.html](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/spikingjelly/activation_based/model/spiking_vgg.html)  
11. SyOPs counter for spiking neural networks \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/iCGY96/syops-counter](https://github.com/iCGY96/syops-counter)  
12. pkuxmq/OTTT-SNN: \[NeurIPS 2022\] Online Training ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/pkuxmq/OTTT-SNN](https://github.com/pkuxmq/OTTT-SNN)  
13. SRKH/ProxyLearning: Spiking neural networks trained via ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/SRKH/ProxyLearning](https://github.com/SRKH/ProxyLearning)  
14. Spiking neural networks trained via proxy arXiv:2109.13208v3 \[cs ..., 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2109.13208](https://arxiv.org/abs/2109.13208)  
15. Intelligent-Computing-Lab-Panda/Neural-Architecture-Search-for-Spiking-Neural-Networks \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks](https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks)  
16. \[2201.10355\] Neural Architecture Search for Spiking Neural Networks \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2201.10355](https://arxiv.org/abs/2201.10355)  
17. Neural Architecture Search for Spiking Neural Networks \- European ..., 10月 9, 2025にアクセス、 [https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/papers/136840036.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840036.pdf)  
18. norse/notebooks: Notebooks illustrating the use of Norse, a ... \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/norse/notebooks](https://github.com/norse/notebooks)  
19. Spiking Neural Networks and Their Applications: A Review \- PMC \- PubMed Central, 10月 9, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/)  
20. yfguo91/Awesome-Spiking-Neural-Networks \- GitHub, 10月 9, 2025にアクセス、 [https://github.com/yfguo91/Awesome-Spiking-Neural-Networks](https://github.com/yfguo91/Awesome-Spiking-Neural-Networks)  
21. SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/html/2407.04752v1](https://arxiv.org/html/2407.04752v1)  
22. \[2505.11151\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking \- arXiv, 10月 9, 2025にアクセス、 [https://arxiv.org/abs/2505.11151](https://arxiv.org/abs/2505.11151)