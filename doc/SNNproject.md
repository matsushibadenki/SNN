```mermaid
graph TD
  %% --- スタイル定義 ---
  classDef entry fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef container fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5,color:#000
  classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
  classDef module fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef core fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
  classDef memory fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
  classDef io fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#000

  %% --- 1. エントリーポイント ---
  subgraph EntryPoints [インターフェース層]
    CLI(["snn-cli.py (CLI)"]):::entry
    SimScript(["run_brain_simulation.py"]):::entry
    API(["app/main.py (FastAPI)"]):::entry
  end

  %% --- 2. DIコンテナ (Wiring) ---
  subgraph DI_Containers [DIコンテナ (app/containers.py)]
    BrainCont[BrainContainer]:::container
    AgentCont[AgentContainer]:::container
    TrainCont[TrainingContainer]:::container
    
    BrainCont --> AgentCont
    BrainCont --> TrainCont
  end

  %% --- 3. 人工脳アーキテクチャ ---
  subgraph Artificial_Brain [Artificial Brain (認知アーキテクチャ)]
    direction TB
    AB_Class[ArtificialBrain Class]:::brain
    
    subgraph Consciousness [意識・制御]
      GW[Global Workspace<br/>(意識の座・注意制御)]:::module
      Astrocyte[Astrocyte Network<br/>(エネルギー恒常性・代謝)]:::module
      Motivation[Intrinsic Motivation<br/>(好奇心・動機づけ)]:::module
    end

    subgraph Perception [知覚モジュール]
      VisCortex[Visual Cortex<br/>(SNN-DSA / CNN)]:::module
      PercCortex[Perception Cortex<br/>(Multi-modal)]:::module
      Receptor[Sensory Receptor]:::io
    end

    subgraph Cognition [認知・記憶・感情]
      PFC[Prefrontal Cortex<br/>(Planner / 実行機能)]:::module
      Hippo[Hippocampus<br/>(短期記憶 / エピソード)]:::module
      Amygdala[Amygdala<br/>(感情 / 価値評価)]:::module
      CausalEng[Causal Inference Engine<br/>(因果推論)]:::module
    end

    subgraph Action [行動・運動]
      Basal[Basal Ganglia<br/>(行動選択 / 強化学習)]:::module
      Motor[Motor Cortex]:::module
      Actuator[Actuator<br/>(Voice / Robot)]:::io
      Reflex[Reflex Module<br/>(System 1 反射)]:::module
    end

    %% 脳内接続
    Receptor --> VisCortex & PercCortex
    VisCortex & PercCortex --> GW
    GW <--> PFC & Hippo & Basal
    PFC --> Basal
    Amygdala --> GW & Basal
    Basal --> Motor
    Motor --> Actuator
    Reflex --> Motor
    Astrocyte -.-> GW & Basal
  end

  %% --- 4. エージェント & 身体性 ---
  subgraph Agent_Layer [エージェント層]
    DLF[Digital Life Form<br/>(身体性ラッパー)]:::brain
    AutoAgent[Autonomous Agent]:::brain
    SelfEvol[Self Evolving Agent]:::brain
    
    DLF --> AB_Class
  end

  %% --- 5. メモリ & RAG ---
  subgraph Memory_System [外部記憶システム]
    RAG[RAG System]:::memory
    VecStore[(Vector Store)]:::memory
    LTM[Cortex / Long-Term Memory]:::memory
    
    Hippo --> LTM
    LTM <--> RAG
    RAG <--> VecStore
  end

  %% --- 6. SNNコア & モデル ---
  subgraph SNN_Core_Tech [SNN Core & Models]
    SNNCore[SNN Core Engine]:::core
    
    subgraph Models [モデル群]
      SFormer[SFormer (T=1)]:::core
      LGSNN[Logic Gated SNN<br/>(1.58-bit Weights)]:::core
      BitNet[BitNet / RWKV]:::core
    end
    
    subgraph Trainers [学習システム]
      Distill[Distillation Trainer<br/>(System2 -> System1)]:::core
      BioRL[Bio-RL Trainer]:::core
      HSEO[HSEO Optimization]:::core
    end
  end

  %% --- 接続関係 ---
  
  %% エントリーからコンテナへ
  CLI --> BrainCont
  SimScript --> BrainCont
  API --> BrainCont

  %% コンテナによるインスタンス化
  BrainCont -- 生成 --> AB_Class
  BrainCont -- 生成 --> GW
  BrainCont -- 生成 --> VisCortex
  
  AgentCont -- 生成 --> DLF
  AgentCont -- 生成 --> AutoAgent
  AgentCont -- 生成 --> RAG

  TrainCont -- 生成 --> SNNCore
  TrainCont -- 生成 --> Trainers

  %% コンポーネント間の依存
  AB_Class --> GW & Astrocyte & Motivation
  AB_Class --> VisCortex & Perception & Receptor
  AB_Class --> PFC & Hippo & Amygdala & CausalEng
  AB_Class --> Basal & Motor & Actuator
  
  %% エージェントと脳の連携
  AutoAgent --> PFC
  SelfEvol --> Distill

  %% コアへの依存
  PFC & VisCortex & PercCortex --> SNNCore
  Distill --> SNNCore
  SNNCore --> Models

  %% 外部データ
  AutoAgent -- Web Crawl --> RAG
  ```
