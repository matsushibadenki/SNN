# **Artificial Brain Architecture v14.2 (SNN-Native)**

## **概要**

人間の脳構造（大脳皮質、海馬、大脳基底核、小脳、扁桃体など）を模倣し、SNN（スパイキングニューラルネットワーク）で実装した認知アーキテクチャ。  
「予測符号化（Predictive Coding）」と「能動的推論（Active Inference）」を動作原理の核とし、省エネかつ適応的な知的振る舞いを目指す。  
graph TD  
    %% Style definitions  
    classDef entry fill:\#f9f,stroke:\#333,stroke-width:2px  
    classDef app fill:\#e1f5fe,stroke:\#0277bd,stroke-width:2px  
    classDef brain fill:\#e8f5e9,stroke:\#2e7d32,stroke-width:2px  
    classDef core fill:\#fff3e0,stroke:\#ef6c00,stroke-width:2px  
    classDef green fill:\#e0f2f1,stroke:\#00695c,stroke-width:2px  
    classDef train fill:\#f3e5f5,stroke:\#7b1fa2,stroke-width:2px  
    classDef hw fill:\#eceff1,stroke:\#455a64,stroke-width:2px

    %% Entry Points & Scripts  
    CLI(\[snn-cli.py\]):::entry  
    RunScripts\[\[scripts/runners\]\]:::entry  
    GradioApp(\[app/main.py\]):::entry

    %% Application Layer  
    DI\_Container\[containers.py\<br/\>Dependency Injector\]:::app  
    ChatSvc\[ChatService\]:::app  
    ImgSvc\[ImageClassificationService\]:::app  
    WebCrawl\[WebCrawler\]:::app  
    LangChain\[SNNLangChainAdapter\]:::app  
    InferenceEngine\[deployment.py\<br/\>SNNInferenceEngine\]:::app

    %% Cognitive Architecture  
    LifeForm\[DigitalLifeForm\]:::brain  
    BrainKernel\[ArtificialBrain\]:::brain  
    GW\[GlobalWorkspace\<br/\>Consciousness\]:::brain  
    Astrocyte\[AstrocyteNetwork\<br/\>Resource Manager\]:::brain  
    Scheduler\[NeuromorphicScheduler\]:::brain  
    Hippo\[Hippocampus\<br/\>Episodic Memory\]:::brain  
    Cortex\[Cortex\<br/\>Semantic Memory\]:::brain  
    RAG\[RAGSystem\<br/\>Long-term Knowledge\]:::brain  
    Visual\[VisualPerception\<br/\>Vision Encoder\]:::brain  
    PercCortex\[PerceptionCortex\<br/\>Multimodal\]:::brain  
    Amyg\[Amygdala\<br/\>Emotion/Value\]:::brain  
    Motor\[MotorCortex\<br/\>Action Generation\]:::brain  
    Basal\[BasalGanglia\<br/\>Action Selection\]:::brain  
      
    %% Agents & Planner  
    Agents\[AutonomousAgent\]:::brain  
    Planner\[PlannerSNN\<br/\>System 2 Reasoning\]:::brain  
    CausalEng\[CausalInferenceEngine\]:::brain  
    ModelRegistry\[ModelRegistry\]:::brain

    %% Green AI & Alternative Computing (New)  
    HDC\[HDC Engine\<br/\>Vector Symbolic Arch\]:::green  
    Tsetlin\[Tsetlin Machine\<br/\>Logic Automata\]:::green  
    ONN\[Oscillatory NN\<br/\>Sync Computing\]:::green  
    FF\_Trainer\[Forward-Forward\<br/\>BP-Free Learning\]:::green

    %% Core SNN Layer  
    SNNCore\[SNNCore\<br/\>Spike Orchestration\]:::core  
    ArchReg\[ArchitectureRegistry\]:::core  
      
    %% Models  
    SFormer\[SpikingTransformer\]:::core  
    BioPC\[BioPCNetwork\]:::core  
    SpikeTrans\[SpikingTransformer\]:::core  
    SpikeCNN\[SpikingCNN\]:::core  
    SEMM\[SpikingEMM\]:::core  
    BitNet\[BitNet 1.58b\]:::core

    %% Building Blocks  
    Neurons\[LIF / Izhikevich / Hodgkin-Huxley\]:::core  
    Layers\[Synaptic Layers\]:::core  
    Rules\[STDP / R-STDP / ETD\]:::core

    %% Training & Conversion  
    Trainers\[Training Manager\]:::train  
    Converter\[ANN-SNN Converter\]:::train  
    Calibrator\[Bio-Calibrator\]:::train  
    HSEO\[HSEO Optimizer\]:::train

    %% Hardware Interface  
    Compiler\[Neuromorphic Compiler\]:::hw  
    Simulator\[Event-Driven Simulator\]:::hw  
    IO\_Dev\[IO Devices\<br/\>Camera/Motor\]:::hw

    %% Connections  
    CLI \--\> DI\_Container  
    RunScripts \--\> DI\_Container  
    GradioApp \--\> DI\_Container

    DI\_Container \--\> BrainKernel  
    DI\_Container \--\> Agents  
    DI\_Container \--\> ChatSvc  
    DI\_Container \--\> ImgSvc  
    DI\_Container \--\> WebCrawl  
    DI\_Container \--\> LangChain

    ChatSvc \--\> InferenceEngine  
    ImgSvc \--\> InferenceEngine  
    LangChain \--\> InferenceEngine

    %% Cognitive Internal  
    LifeForm \--\> BrainKernel  
    LifeForm \--\> Agents  
    BrainKernel \--\> GW  
    BrainKernel \--\> Astrocyte  
    BrainKernel \--\> Scheduler  
    BrainKernel \--\> Hippo  
    BrainKernel \--\> Cortex  
    BrainKernel \--\> RAG  
    BrainKernel \--\> Visual  
    BrainKernel \--\> PercCortex  
    BrainKernel \--\> Amyg  
    BrainKernel \--\> Motor  
    BrainKernel \--\> Basal  
      
    %% Green AI Connections  
    Cortex \--\> HDC  
    Cortex \--\> Tsetlin  
    Hippo \--\> ONN  
    Trainers \--\> FF\_Trainer

    Agents \--\> Planner  
    Agents \--\> WebCrawl  
    Planner \--\> ModelRegistry  
    Planner \--\> CausalEng

    %% Cognitive to Core  
    InferenceEngine \--\> SNNCore  
    BrainKernel \--\> SNNCore  
    Visual \--\> SNNCore  
    PercCortex \--\> SNNCore

    %% Training to Core  
    Trainers \--\> SNNCore  
    Converter \--\> SNNCore  
    Calibrator \--\> SNNCore  
    Calibrator \--\> HSEO

    %% Core Internal  
    SNNCore \--\> ArchReg  
    ArchReg \--\> SFormer  
    ArchReg \--\> BioPC  
    ArchReg \--\> SpikeTrans  
    ArchReg \--\> SpikeCNN  
    ArchReg \--\> SEMM  
    ArchReg \--\> BitNet  
    SFormer \--\> Neurons  
    BioPC \--\> Neurons  
    SpikeTrans \--\> Neurons  
    Neurons \--\> Layers  
    Layers \--\> Rules

    %% Hardware  
    Compiler \--\> SNNCore  
    Simulator \--\> SNNCore  
    BrainKernel \--\> IO\_Dev

    %% Data Flow  
    RAG \-- Context \--\> Planner  
    Hippo \-- Replay \--\> Cortex  
    Basal \-- Gating \--\> Thalamus

## **1\. 認知アーキテクチャ (Cognitive Layer)**

脳の高次機能を模倣する最上位レイヤー。情報の統合、意思決定、記憶管理を行う。

* **Global Workspace (意識の座):** 複数のモジュールからの情報を統合し、最も重要な情報を放送（Broadcast）する。  
* **Prefrontal Cortex (前頭前野):** 長期的な計画、推論、抑制制御を担当。System 2思考の司令塔。  
* **Hippocampus (海馬):** エピソード記憶の形成と想起、短期記憶から長期記憶への固定化（Consolidation）を管理。  
* **Basal Ganglia (大脳基底核):** 行動選択（Action Selection）と強化学習による報酬予測。  
* **Amygdala (扁桃体):** 情動価（Valence）の評価、危険検知、記憶への感情タグ付け。  
* **Astrocyte Network (アストロサイト):** ニューロンへのエネルギー供給、神経活動の変調、睡眠サイクルの制御。

## **2\. コアSNNエンジン (Core Layer)**

生物学的妥当性と計算効率を両立させるためのスパイク処理エンジン。

* **SNNCore:** スパイクの発生、伝播、可塑性を管理するカーネル。  
* **Bio-Microcircuit (PD14):** 大脳皮質の6層構造（L2/3, L4, L5, L6）と層間結合を忠実に再現した回路モデル。  
* **BitNet 1.58b:** 重みを {-1, 0, 1} の3値に量子化し、乗算を排除した超省エネLLMモデル。

## **3\. 学習・適応メカニズム (Learning & Plasticity)**

* **STDP (Spike-Timing Dependent Plasticity):** スパイクタイミングに基づく局所的なシナプス可塑性。  
* **Predictive Coding (予測符号化):** 脳は常に次の入力を予測し、予測誤差のみを伝播させて学習する。  
* **GRPO (Group Relative Policy Optimization):** 複数の思考パスを生成し、相対的な良さで方策を更新する強化学習手法。

## **4\. 記憶システム (Memory Systems)**

* **Short-term / Working Memory:** 海馬および前頭前野の反響回路による一時的な保持。  
* **Long-term / Semantic Memory:** 知識グラフ (GraphRAG) と大脳皮質モデルによる構造化された知識。  
* **Procedural Memory:** 小脳および大脳基底核による技能・習慣の記憶。

## **5\. 知覚・行動モジュール (Perception & Action)**

* **Visual Cortex (視覚野):** DVS (Dynamic Vision Sensor) データや画像をスパイク列に変換し、特徴抽出を行う。  
* **Motor Cortex (運動野):** 決定された行動を具体的な運動指令（アクチュエータ制御信号）に変換する。

## **6\. Neuromorphic OS (System Software)**

脳型ハードウェアのリソース（ニューロン、シナプス、エネルギー）を効率的に管理するオペレーティングシステム。

* **Scheduler:** タスクの優先度と脳の状態（覚醒度、エネルギー）に基づいてプロセスを割り当てる。  
* **Sleep Consolidation:** アイドル時や睡眠モード時に、記憶の整理とシナプス重みの再スケーリングを行う。

## **7\. 新しい計算パラダイムの統合 (Green AI Extensions)**

従来のSNN（スパイク発火）に加え、生物学的妥当性とエネルギー効率を追求した新しい計算モデルを統合しています。これらはGPUに依存しない「真の脳型計算」を目指すものです。

### **7.1 シンボル接地と超次元計算 (HDC)**

従来の「シンボル＝埋め込みベクトル（密ベクトル）」というアプローチに加え、**Hyperdimensional Computing (HDC)** を導入しました。

* **役割:** 概念操作（結合、分離、類推）を、浮動小数点演算なしのビット演算で行います。  
* **利点:** 瞬時の学習（ワンショット）、高いノイズ耐性、ハードウェアへの親和性。  
* **統合:** SymbolGrounding モジュールの一部として、知覚情報（スパイク）をHDCのハイパーベクトルへ変換し、高速な推論を行います。

### **7.2 振動と同期 (Oscillatory Dynamics)**

脳内の情報の結びつき（Binding Problem）を解決するために、**Oscillatory Neural Network (ONN)** を導入しました。

* **役割:** 異なる領域にあるニューロン群が、同一の対象を処理しているときに「位相を同期」させます。  
* **メカニズム:** Kuramotoモデルに基づき、互いに引き込み合いながらエネルギー最小状態（安定した記憶）へ収束します。

### **7.3 推論のみによる学習 (Forward-Forward)**

脳には誤差逆伝播（Backpropagation）の回路が存在しないという説に基づき、**Forward-Forward Algorithm** を採用しています。

* **プロセス:** 「正解データ（Positive）」と「偽データ（Negative）」をネットワークに流し、それぞれの「良さ（Goodness）」を局所的に最大化/最小化します。  
* **意義:** 全体的な同期（Global Error）を必要とせず、局所的な可塑性のみで学習が進行するため、超省エネかつ並列化が容易です。

### **7.4 論理オートマトン学習 (Tsetlin Machine)**

ニューラルネットワークのブラックボックス性を解消し、超高速な推論を実現するために導入しました。

* **役割:** 入力パターンに対する論理的規則（Clause）を学習します。  
* **メカニズム:** 多数のオートマトンが協調して、命題論理式（AND/OR/NOT）を構成します。行列演算を一切使用しません。
