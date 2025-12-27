```mermaid
graph TD
  %% Style definitions
  classDef entry fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef container fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5,color:#000
  classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
  classDef module fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef core fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
  classDef memory fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
  classDef io fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#000

  %% Entry Points
  subgraph EntryPoints["Interface Layer"]
    CLI["snn-cli.py CLI"]:::entry
    SimScript["run_brain_simulation.py"]:::entry
    API["app/main.py FastAPI"]:::entry
  end

  %% DI Containers
  subgraph DI_Containers["DI Container app/containers.py"]
    BrainCont[BrainContainer]:::container
    AgentCont[AgentContainer]:::container
    TrainCont[TrainingContainer]:::container
    
    BrainCont --> AgentCont
    BrainCont --> TrainCont
  end

  %% Artificial Brain Architecture
  subgraph Artificial_Brain["Artificial Brain - Cognitive Architecture"]
    direction TB
    AB_Class[ArtificialBrain Class]:::brain
    
    subgraph Consciousness["Consciousness & Control"]
      GW["Global Workspace<br/>Attention Control"]:::module
      Astrocyte["Astrocyte Network<br/>Energy Homeostasis"]:::module
      Motivation["Intrinsic Motivation<br/>Curiosity Drive"]:::module
    end

    subgraph Perception["Perception Modules"]
      VisCortex["Visual Cortex<br/>SNN-DSA / CNN"]:::module
      PercCortex["Perception Cortex<br/>Multi-modal"]:::module
      Receptor[Sensory Receptor]:::io
    end

    subgraph Cognition["Cognition Memory Emotion"]
      PFC["Prefrontal Cortex<br/>Planner / Executive"]:::module
      Hippo["Hippocampus<br/>Short-term / Episode"]:::module
      Amygdala["Amygdala<br/>Emotion / Value"]:::module
      CausalEng["Causal Inference Engine"]:::module
    end

    subgraph Action["Action & Motor"]
      Basal["Basal Ganglia<br/>Action Selection / RL"]:::module
      Motor[Motor Cortex]:::module
      Actuator["Actuator<br/>Voice / Robot"]:::io
      Reflex["Reflex Module<br/>System 1 Response"]:::module
    end

    %% Brain connections
    Receptor --> VisCortex
    Receptor --> PercCortex
    VisCortex --> GW
    PercCortex --> GW
    GW <--> PFC
    GW <--> Hippo
    GW <--> Basal
    PFC --> Basal
    Amygdala --> GW
    Amygdala --> Basal
    Basal --> Motor
    Motor --> Actuator
    Reflex --> Motor
    Astrocyte -.-> GW
    Astrocyte -.-> Basal
  end

  %% Agent Layer
  subgraph Agent_Layer["Agent Layer"]
    DLF["Digital Life Form<br/>Embodiment Wrapper"]:::brain
    AutoAgent[Autonomous Agent]:::brain
    SelfEvol[Self Evolving Agent]:::brain
    
    DLF --> AB_Class
  end

  %% Memory System
  subgraph Memory_System["External Memory System"]
    RAG[RAG System]:::memory
    VecStore["Vector Store"]:::memory
    LTM["Cortex / Long-Term Memory"]:::memory
    
    Hippo --> LTM
    LTM <--> RAG
    RAG <--> VecStore
  end

  %% SNN Core
  subgraph SNN_Core_Tech["SNN Core & Models"]
    SNNCore[SNN Core Engine]:::core
    
    subgraph Models["Model Group"]
      SFormer["SFormer T=1"]:::core
      LGSNN["Logic Gated SNN<br/>1.58-bit Weights"]:::core
      BitNet[BitNet / RWKV]:::core
    end
    
    subgraph Trainers["Training System"]
      Distill["Distillation Trainer<br/>System2 to System1"]:::core
      BioRL[Bio-RL Trainer]:::core
      HSEO[HSEO Optimization]:::core
    end
  end

  %% Connections
  
  %% Entry to Container
  CLI --> BrainCont
  SimScript --> BrainCont
  API --> BrainCont

  %% Container instantiation
  BrainCont -- create --> AB_Class
  BrainCont -- create --> GW
  BrainCont -- create --> VisCortex
  
  AgentCont -- create --> DLF
  AgentCont -- create --> AutoAgent
  AgentCont -- create --> RAG

  TrainCont -- create --> SNNCore
  TrainCont -- create --> Trainers

  %% Component dependencies
  AB_Class --> GW
  AB_Class --> Astrocyte
  AB_Class --> Motivation
  AB_Class --> VisCortex
  AB_Class --> Perception
  AB_Class --> Receptor
  AB_Class --> PFC
  AB_Class --> Hippo
  AB_Class --> Amygdala
  AB_Class --> CausalEng
  AB_Class --> Basal
  AB_Class --> Motor
  AB_Class --> Actuator
  
  %% Agent and Brain collaboration
  AutoAgent --> PFC
  SelfEvol --> Distill

  %% Core dependencies
  PFC --> SNNCore
  VisCortex --> SNNCore
  PercCortex --> SNNCore
  Distill --> SNNCore
  SNNCore --> Models

  %% External data
  AutoAgent -- Web Crawl --> RAG
  ```