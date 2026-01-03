```mermaid
graph TD
  %% Styles
  classDef region fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
  classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
  classDef memory fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
  classDef emotion fill:#ffebee,stroke:#c62828,stroke-width:2px
  classDef exec fill:#ede7f6,stroke:#5e35b1,stroke-width:2px
  classDef sensory fill:#f1f8e9,stroke:#7cb342,stroke-width:2px

  %% Core coordinating structures
  Thalamus["Thalamus Sensory Relay & Attention Switching"]:::core
  Brainstem["Brainstem Autonomic Control / Arousal"]:::core
  Cerebellum["Cerebellum Motor Coordination / Prediction"]:::core

  %% Global Workspace & Executive
  PFC["Prefrontal Cortex Planning / Reasoning / Executive"]:::exec
  GW["Global Neuronal Workspace Conscious Broadcasting"]:::exec
  ACC["Anterior Cingulate Cortex Conflict Monitoring / Error Detection"]:::exec

  %% Sensory areas
  V1["Primary Visual Cortex V1"]:::sensory
  V2["V2/V4 Shape/Color"]:::sensory
  A1["Primary Auditory Cortex A1"]:::sensory
  S1["Somatosensory Cortex"]:::sensory
  Olf["Olfactory Cortex"]:::sensory
  Insula["Insular Cortex Interoception / Body State"]:::sensory

  %% Memory circuits
  Hippocampus["Hippocampus Episodic Memory / Indexing"]:::memory
  EC["Entorhinal Cortex Memory Encoding Gateway"]:::memory
  Cortex["Cerebral Cortex Semantic / Long-Term Memory"]:::memory

  %% Emotion & Valence
  Amygdala["Amygdala Valence / Fear / Social Signals"]:::emotion
  Striatum["Striatum Basal Ganglia input Reward / Action Selection"]:::emotion
  Hypo["Hypothalamus Homeostasis / Drive"]:::emotion

  %% Basal ganglia loops
  GPi["Globus Pallidus internus"]:::core
  STN["Subthalamic Nucleus"]:::core
  BG["Basal Ganglia Loop Action Gating"]:::core

  %% Motor pathways
  PMC["Premotor Cortex"]:::exec
  M1["Primary Motor Cortex"]:::exec
  Spinal["Spinal Cord"]:::core

  %% Connectivity
  %% Sensory input routing
  V1 --> V2
  V2 --> Thalamus
  A1 --> Thalamus
  S1 --> Thalamus
  Olf --> Insula

  %% Thalamus integration
  Thalamus --> PFC
  Thalamus --> GW
  Thalamus --> Hippocampus
  Thalamus --> Insula

  %% Memory formation
  Hippocampus --> EC
  EC --> Cortex
  Cortex --> EC
  Hippocampus --> Cortex

  %% Emotion & valuation
  Amygdala --> PFC
  Amygdala --> GW
  Amygdala --> Striatum
  Amygdala --> Hypo

  %% Basal ganglia loops
  PFC --> Striatum
  Striatum --> GPi
  GPi --> STN
  STN --> PFC
  Striatum --> BG
  BG --> PMC

  %% Executive and conscious workspace
  PFC --> GW
  GW --> PFC
  ACC --> PFC
  ACC --> GW

  %% Motor output
  PMC --> M1
  M1 --> Cerebellum
  Cerebellum --> M1
  M1 --> Spinal

  %% Interoception and autonomic control
  Insula --> PFC
  Hypo --> Brainstem
  Brainstem --> Hypo

  %% Final routing
  Spinal --> Brainstem
  Brainstem --> Cortex
  ```