```mermaid
graph TD
  %% Styles
  classDef entry fill:#f9f,stroke:#333,stroke-width:2px
  classDef app fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
  classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
  classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
  classDef train fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
  classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px
  classDef infra fill:#fff8e1,stroke:#6d4c41,stroke-width:2px

  %% Entry / UI
  CLI(["snn-cli.py"]):::entry
  RunScripts[["scripts/runners"]]:::entry
  GradioApp(["app/main.py"]):::entry

  %% App layer
  DI_Container["DependencyInjector containers.py"]:::app
  ChatSvc["ChatService"]:::app
  ImgSvc["ImageClassificationService"]:::app
  WebCrawl["WebCrawler"]:::app
  LangChain["SNNLangChainAdapter"]:::app
  InferenceEngine["Deployment Adapter SNNInferenceEngine"]:::app

  %% Life & Body
  LifeForm["DigitalLifeForm Body Interface"]:::brain
  IO_Dev["SensoryReceptor / Actuator"]:::hw

  %% Brain kernel
  BrainKernel["ArtificialBrainKernel"]:::brain
  GW["GlobalWorkspace Consciousness/Attention"]:::brain
  Astrocyte["AstrocyteNetwork Homeostasis"]:::brain
  Scheduler["NeuromorphicScheduler Resource Manager"]:::brain

  %% Memory & Symbol
  Hippo["Hippocampus Short-Term Memory / Indexing"]:::brain
  Cortex["Cortex Long-Term Memory"]:::brain
  RAG["RAG System External KB Adapter"]:::brain
  GraphRAG["GraphRAG Symbolic Interface"]:::brain

  %% Perception & Valence
  Visual["Visual Cortex DVS & Image"]:::brain
  PercCortex["HybridPerception Cortex Audio/Text/DVS"]:::brain
  Amyg["Amygdala Valence / Safety / Empathy"]:::brain

  %% Decision, Planning, Actuation
  Planner["Hierarchical Planner"]:::brain
  Basal["Basal Ganglia Decision Policy"]:::brain
  Motor["Motor Cortex / Actuator Planner"]:::brain
  Agents["AutonomousAgent SelfEvolvingAgent"]:::brain

  %% Core SNN Models
  SNNCore["SNNCore Unified Compute"]:::core
  ArchReg["ArchitectureRegistry"]:::core
  ExpertMgr["Expert Manager Energy-aware Router"]:::core
  SFormer["SFormer T=1"]:::core
  SEMM["SEMM MoE Spiking Experts"]:::core
  SpikeTrans["SpikingTransformer"]:::core
  BitNet["BitSpikingRWKV"]:::core
  Neurons["Neuron Library LIF/SFN/EL-LIF"]:::core
  Layers["Layer Primitives SDSA/PC/FEEL"]:::core
  Rules["Learning Rules STDP/CausalTrace"]:::core

  %% Training & Distillation
  Trainers["Trainers Distill & BioRL"]:::train
  Distill["Distillation Pipeline"]:::train
  Calibrator["DeepBioCalibrator"]:::train
  HSEO["HSEO Optimization"]:::train
  ModelRegistry["ModelRegistry"]:::train

  %% Hardware infra
  Compiler["NeuromorphicCompiler Codegen"]:::hw
  Simulator["EventDrivenSimulator"]:::hw
  TritonK["Triton/CUDA Kernels"]:::hw

  %% Entry -> App
  CLI --> DI_Container
  RunScripts --> DI_Container
  GradioApp --> DI_Container

  %% App -> Life & Brain
  DI_Container --> LifeForm
  DI_Container --> BrainKernel
  DI_Container --> Agents
  DI_Container --> ChatSvc
  DI_Container --> ImgSvc
  DI_Container --> WebCrawl
  DI_Container --> LangChain
  DI_Container --> InferenceEngine

  %% LifeForm -> IO
  LifeForm --> IO_Dev
  IO_Dev --> Visual
  IO_Dev --> PercCortex

  %% Brain core coordination
  BrainKernel --> GW
  BrainKernel --> Astrocyte
  BrainKernel --> Scheduler
  BrainKernel --> Hippo
  BrainKernel --> Cortex

  %% Perception -> Core compute
  Visual --> SNNCore
  PercCortex --> SNNCore
  Hippo --> SNNCore
  Planner --> SNNCore
  RAG --> SNNCore

  %% SNN Core internal wiring
  SNNCore --> ArchReg
  ArchReg --> ExpertMgr
  ExpertMgr --> SEMM
  ExpertMgr --> SFormer
  ArchReg --> SpikeTrans
  ArchReg --> BitNet
  SFormer --> Neurons
  SEMM --> Neurons
  SpikeTrans --> Neurons
  Neurons --> Layers
  Layers --> Rules

  %% Training & Distillation -> Core
  Trainers --> Distill
  Distill --> SFormer
  Distill --> SEMM
  Calibrator --> SNNCore
  HSEO --> Calibrator
  ModelRegistry --> Distill

  %% Planner/Decision loops
  GW --> Planner
  Planner --> Basal
  Planner --> GraphRAG
  Basal --> Motor
  Motor --> IO_Dev

  %% RAG / Cortex interactions
  Cortex --> RAG
  RAG --> Cortex
  RAG --> Planner

  %% Amygdala connections
  Amyg --> Planner
  Amyg --> Basal
  Amyg --> Motor
  Amyg --> GW
  Amyg --> Scheduler

  %% Agents autonomy
  Agents --> Planner
  Agents --> WebCrawl
  Agents --> ModelRegistry
  Agents --> GW

  %% Inference / Deployment
  InferenceEngine --> SNNCore
  Compiler --> SNNCore
  Simulator --> SNNCore
  TritonK --> SNNCore

  %% Monitoring & Safety
  BrainKernel --> ModelRegistry
  Astrocyte --> Scheduler
  Scheduler --> ExpertMgr
  Scheduler --> Distill
  ```