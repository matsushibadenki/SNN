```mermaid
graph TD
    %% Style definitions
    classDef entry fill:#f9f,stroke:#333,stroke-width:2px
    classDef app fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef train fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px

    %% Entry Points & Scripts
    CLI([snn-cli.py]):::entry
    RunScripts[[scripts/runners]]:::entry
    GradioApp([app/main.py]):::entry

    %% Application Layer
    DI_Container[containers.py<br/>Dependency Injector]:::app
    ChatSvc[ChatService]:::app
    ImgSvc[ImageClassificationService]:::app
    WebCrawl[WebCrawler]:::app
    LangChain[SNNLangChainAdapter]:::app
    InferenceEngine[deployment.py<br/>SNNInferenceEngine]:::app

    %% Cognitive Architecture
    LifeForm[DigitalLifeForm]:::brain
    BrainKernel[ArtificialBrain]:::brain
    GW[GlobalWorkspace<br/>Consciousness]:::brain
    Astrocyte[AstrocyteNetwork<br/>Resource Manager]:::brain
    Scheduler[NeuromorphicScheduler]:::brain
    Hippo[Hippocampus<br/>Short-Term Memory]:::brain
    Cortex[Cortex<br/>Long-Term Memory]:::brain
    RAG[RAGSystem<br/>GraphRAG]:::brain
    Planner[HierarchicalPlanner]:::brain
    CausalEng[CausalInferenceEngine]:::brain
    Visual[VisualCortex]:::brain
    PercCortex[HybridPerceptionCortex]:::brain
    Amyg[Amygdala<br/>Emotion]:::brain
    Motor[MotorCortex]:::brain
    Basal[BasalGanglia<br/>Decision Making]:::brain
    Agents[AutonomousAgent<br/>SelfEvolvingAgent]:::brain

    %% Training & Optimization
    Trainers[Trainers<br/>Breakthrough Distillation BioRL]:::train
    Losses[Loss Functions<br/>Combined CausalTrace]:::train
    Converter[AnnToSnnConverter]:::train
    Calibrator[DeepBioCalibrator]:::train
    HSEO[HSEO Optimization]:::train
    ModelRegistry[ModelRegistry]:::train

    %% SNN Core & Models
    SNNCore[SNNCore<br/>Unified Wrapper]:::core
    ArchReg[ArchitectureRegistry]:::core
    SFormer[SFormer T=1]:::core
    BioPC[BioPCNetwork]:::core
    SpikeTrans[SpikingTransformer]:::core
    SpikeCNN[SpikingCNN]:::core
    SEMM[SEMM MoE]:::core
    BitNet[BitSpikingRWKV]:::core
    Neurons[Neurons<br/>LIF SFN EL-LIF]:::core
    Layers[Layers<br/>SDSA PC-Layer FEEL]:::core
    Rules[Learning Rules<br/>STDP CausalTrace]:::core

    %% Hardware & I/O
    Compiler[NeuromorphicCompiler]:::hw
    Simulator[EventDrivenSimulator]:::hw
    IO_Dev[SensoryReceptor / Actuator]:::hw

    %% Dependencies - Entry to App
    CLI --> DI_Container
    RunScripts --> DI_Container
    GradioApp --> DI_Container

    %% App to Services
    DI_Container --> LifeForm
    DI_Container --> BrainKernel
    DI_Container --> Agents
    DI_Container --> ChatSvc
    DI_Container --> ImgSvc
    DI_Container --> WebCrawl
    DI_Container --> LangChain

    ChatSvc --> InferenceEngine
    ImgSvc --> InferenceEngine
    LangChain --> InferenceEngine

    %% Cognitive Internal
    LifeForm --> BrainKernel
    LifeForm --> Agents
    BrainKernel --> GW
    BrainKernel --> Astrocyte
    BrainKernel --> Scheduler
    BrainKernel --> Hippo
    BrainKernel --> Cortex
    BrainKernel --> RAG
    BrainKernel --> Visual
    BrainKernel --> PercCortex
    BrainKernel --> Amyg
    BrainKernel --> Motor
    BrainKernel --> Basal

    Agents --> Planner
    Agents --> WebCrawl
    Planner --> ModelRegistry
    Planner --> CausalEng

    %% Cognitive to Core
    InferenceEngine --> SNNCore
    BrainKernel --> SNNCore
    Visual --> SNNCore
    PercCortex --> SNNCore

    %% Training to Core
    Trainers --> SNNCore
    Converter --> SNNCore
    Calibrator --> SNNCore
    Calibrator --> HSEO

    %% Core Internal
    SNNCore --> ArchReg
    ArchReg --> SFormer
    ArchReg --> BioPC
    ArchReg --> SpikeTrans
    ArchReg --> SpikeCNN
    ArchReg --> SEMM
    ArchReg --> BitNet
    SFormer --> Neurons
    BioPC --> Neurons
    SpikeTrans --> Neurons
    Neurons --> Layers
    Layers --> Rules

    %% Hardware
    Compiler --> SNNCore
    Simulator --> SNNCore
    BrainKernel --> IO_Dev

    %% Data Flow
    RAG --> Cortex
    Cortex --> RAG
    RAG --> Planner
    Planner --> RAG
    ```