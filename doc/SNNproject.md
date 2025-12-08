```mermaid
graph TD
    %% --- スタイル設定 ---
    classDef entry fill:#f9f,stroke:#333,stroke-width:2px;
    classDef app fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef train fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px;

    %% --- エントリーポイント & スクリプト ---
    subgraph EntryPoints ["🚀 Entry Points & Scripts"]
        direction TB
        CLI([snn-cli.py]):::entry
        RunScripts[[scripts/runners/*]]:::entry
        GradioApp([app/main.py]):::entry
    end

    %% --- アプリケーション層 ---
    subgraph AppLayer ["📱 Application Layer (app/)"]
        DI_Container[containers.py<br/>(Dependency Injector)]:::app
        
        subgraph Services [Services]
            ChatSvc[ChatService]:::app
            ImgSvc[ImageClassificationService]:::app
            WebCrawl[WebCrawler]:::app
            LangChain[SNNLangChainAdapter]:::app
        end
        
        InferenceEngine[deployment.py<br/>SNNInferenceEngine]:::app
    end

    %% --- 認知アーキテクチャ層 ---
    subgraph CognitiveLayer ["🧠 Cognitive Architecture (snn_research/)"]
        LifeForm[DigitalLifeForm]:::brain
        BrainKernel[ArtificialBrain]:::brain
        
        subgraph OS_Kernel [Neuromorphic OS]
            GW[GlobalWorkspace<br/>(Consciousness)]:::brain
            Astrocyte[AstrocyteNetwork<br/>(Resource Manager)]:::brain
            Scheduler[NeuromorphicScheduler]:::brain
        end

        subgraph MemorySystem [Memory & Reasoning]
            Hippo[Hippocampus<br/>(Short-Term Memory)]:::brain
            Cortex[Cortex<br/>(Long-Term Memory)]:::brain
            RAG[RAGSystem<br/>(GraphRAG)]:::brain
            Planner[HierarchicalPlanner]:::brain
            CausalEng[CausalInferenceEngine]:::brain
        end

        subgraph Perception_Action [Perception & Action]
            Visual[VisualCortex]:::brain
            PercCortex[HybridPerceptionCortex]:::brain
            Amyg[Amygdala<br/>(Emotion)]:::brain
            Motor[MotorCortex]:::brain
            Basal[BasalGanglia<br/>(Decision Making)]:::brain
        end
        
        Agents[AutonomousAgent<br/>SelfEvolvingAgent]:::brain
    end

    %% --- 学習・変換・最適化 ---
    subgraph TrainOps ["⚙️ Training, Conversion & Optimization"]
        Trainers[Trainers<br/>(Breakthrough, Distillation, BioRL)]:::train
        Losses[Loss Functions<br/>(Combined, CausalTrace)]:::train
        Converter[AnnToSnnConverter]:::train
        Calibrator[DeepBioCalibrator]:::train
        HSEO[HSEO Optimization]:::train
        ModelRegistry[ModelRegistry]:::train
    end

    %% --- SNNコア & モデル層 ---
    subgraph CoreLayer ["⚡ SNN Core & Models"]
        SNNCore[SNNCore<br/>(Unified Wrapper)]:::core
        ArchReg[ArchitectureRegistry]:::core
        
        subgraph Architectures [Model Architectures]
            SFormer[SFormer (T=1)]:::core
            BioPC[BioPCNetwork]:::core
            SpikeTrans[SpikingTransformer]:::core
            SpikeCNN[SpikingCNN]:::core
            SEMM[SEMM (MoE)]:::core
            BitNet[BitSpikingRWKV]:::core
        end
        
        subgraph NeuralComponents [Core Components]
            Neurons[Neurons<br/>(LIF, SFN, EL-LIF)]:::core
            Layers[Layers<br/>(SDSA, PC-Layer, FEEL)]:::core
            Rules[Learning Rules<br/>(STDP, CausalTrace)]:::core
        end
    end

    %% --- ハードウェア & IO ---
    subgraph HW_IO ["🔌 Hardware & I/O"]
        Compiler[NeuromorphicCompiler]:::hw
        Simulator[EventDrivenSimulator]:::hw
        IO_Dev[SensoryReceptor / Actuator]:::hw
    end

    %% --- 依存関係の定義 ---
    
    %% Entry -> App
    CLI --> DI_Container
    RunScripts --> DI_Container
    GradioApp --> DI_Container
    
    %% App -> Cognitive
    DI_Container --> LifeForm
    DI_Container --> BrainKernel
    DI_Container --> Agents
    DI_Container --> Services
    
    Services --> InferenceEngine
    LangChain --> InferenceEngine
    
    %% Cognitive Internal
    LifeForm --> BrainKernel
    LifeForm --> Agents
    BrainKernel --> OS_Kernel
    BrainKernel --> MemorySystem
    BrainKernel --> Perception_Action
    
    Agents --> Planner
    Agents --> WebCrawl
    Planner --> ModelRegistry
    
    %% Cognitive -> Core
    InferenceEngine --> SNNCore
    BrainKernel --> SNNCore
    Visual --> SNNCore
    PercCortex --> SNNCore
    
    %% Training -> Core
    Trainers --> SNNCore
    Converter --> SNNCore
    Calibrator --> SNNCore
    Calibrator -.-> HSEO
    
    %% Core Internal
    SNNCore --> ArchReg
    ArchReg --> Architectures
    Architectures --> NeuralComponents
    
    %% Hardware
    Compiler --> SNNCore
    Simulator --> SNNCore
    BrainKernel --> IO_Dev

    %% Data Flow
    RAG <==> Cortex
    RAG <==> Planner
```