```mermaid
flowchart TB

  %% --- 感覚入力層 ---
  subgraph SensoryLayer [感覚入力層]
    SENSORY["感覚入力\n(光・音・化学・機械)"]
    RECEPTOR["受容器応答\n(アナログ変換)"]
    ENCODING["スパイク符号化\nRate / Temporal / Latency"]
    SENSORY --> RECEPTOR --> ENCODING
  end

  %% --- 単一ニューロン層（LIFモデル） ---
  subgraph Neuron [単一ニューロン（LIFモデル基準）]
    direction TB
    DENDRITES["樹状突起\n多数のシナプス入力\n(加重和・時定数 τ)"]
    SOMA["細胞体\n統合点"]
    MEMBRANE["膜電位 Vm(t)\n静止: -70 mV\n漏れ統合"]
    THRESH["閾値判定\nVth ≈ -55 mV"]
    SPIKE["活動電位\n(スパイク発火)"]
    REFRAC["不応期\n(2-5 ms)\nVm ← Vreset (-65 mV)"]
    AXON["軸索伝導\n(有髄化 / 伝導速度)"]
    TERMINAL["軸索末端 (Presynaptic)"]

    DENDRITES --> SOMA
    SOMA --> MEMBRANE
    MEMBRANE -->|Vm ≥ Vth| THRESH
    THRESH --> SPIKE
    SPIKE --> REFRAC
    REFRAC -->|Vm ← Vreset| MEMBRANE
    SPIKE --> AXON
    AXON --> TERMINAL
  end

  %% --- シナプス層（興奮性・抑制性） ---
  subgraph Synapse [シナプス（興奮性 / 抑制性）]
    direction LR
    DELAY["シナプス遅延\n(0.5–2 ms)"]
    NT_E["興奮性伝達物質\nGlutamate\n(AMPA / NMDA)"]
    NT_I["抑制性伝達物質\nGABA\n(Cl⁻流入)"]
    POSTSYN["シナプス後電位\n(EPSP / IPSP)\n→ ΔVm(t)"]

    TERMINAL --> DELAY
    DELAY --> NT_E
    DELAY --> NT_I
    NT_E --> POSTSYN
    NT_I --> POSTSYN
    POSTSYN -->|ΔVm| MEMBRANE
  end

  %% --- 可塑性・学習機構 ---
  subgraph Plasticity [可塑性・学習機構]
    STDP["STDP\n(Spike-Timing-Dependent Plasticity)"]
    LTP["LTP\n(長期増強)"]
    LTD["LTD\n(長期抑圧)"]
    WEIGHT["シナプス重み w\n(動的更新)"]
    NEUROMOD["神経修飾\n(Dopamine / 報酬信号)"]

    SPIKE --> STDP
    STDP -->|Δt > 0| LTP
    STDP -->|Δt < 0| LTD
    LTP --> WEIGHT
    LTD --> WEIGHT
    NEUROMOD --> STDP
    WEIGHT -->|伝達効率変化| NT_E
  end

  %% --- ネットワーク構造 ---
  subgraph Network [ネットワーク構造]
    direction TB
    FF["フィードフォワード結合\n(層間伝播)"]
    REC["リカレント結合\n(再帰・短期記憶)"]
    EI["E/Iバランス\n(興奮 ≒ 80% / 抑制 ≒ 20%)"]
    OSC["同期振動\nGamma (30–100 Hz)\nBeta (12–30 Hz)\nAlpha (8–12 Hz)\nTheta (4–8 Hz)"]
    LONG["長距離結合\n(皮質間・遅延含む)"]
    HIERARCHY["階層構造\n(皮質6層構造 / 視床-皮質ループ)"]

    %% 提案5を反映
    ENCODING --> FF
    FF --> REC
    REC --> FF
    POSTSYN --> EI
    EI --> OSC
    FF --> LONG
    REC --> LONG

    %% 階層構造へのリンク（コメントは独立行）
    FF --> HIERARCHY
    HIERARCHY --> FF
  end

  %% --- 情報表現層 ---
  subgraph Info [情報表現・符号化]
    RATE["Rate Code\n(発火頻度符号化)"]
    TEMP["Temporal Code\n(スパイクタイミング / 位相)"]
    POPULATION["Population Code\n(集団符号化)"]
    SPARSE["Sparse Coding\n(省エネ・高識別性)"]

    SPIKE --> RATE
    SPIKE --> TEMP
    RATE --> POPULATION
    TEMP --> POPULATION
    POPULATION --> SPARSE
  end

  %% --- 出力・復号化層 ---
  subgraph Output [出力・復号化層]
    DECODE["スパイク復号化\n(積分 / 投票 / フィルタ)"]
    PERCEPT["知覚・判断"]
    MOTOR["運動出力"]
    FEEDBACK["フィードバック\n(誤差信号 / 感覚再入力)"]

    LONG --> DECODE
    DECODE --> PERCEPT
    DECODE --> MOTOR
    PERCEPT --> FEEDBACK
    FEEDBACK --> NEUROMOD
  end

  %% --- 生物学的制約 ---
  subgraph Meta [生物学的制約（SNNでは通常省略）]
    NOISE["ノイズ\n(熱雑音・確率的放出)"]
    ENERGY["代謝コスト\n(ATP消費)"]
    GLIA["グリア細胞\n(恒常性維持・ATP供給・イオン調整)"]

    NOISE -.->|膜電位変動| MEMBRANE
    ENERGY -.->|発火制約| SPIKE
    GLIA -.->|サポート| POSTSYN
    GLIA -.->|ATP供給| ENERGY
  end

  %% --- フィードバックループ ---
  FEEDBACK -.->|学習信号| STDP
  MOTOR -.->|感覚フィードバック| RECEPTOR

  %% --- スタイル定義 ---
  classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
  classDef learning fill:#fff3e0,stroke:#f57c00,stroke-width:2px
  classDef optional fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5

  class MEMBRANE,SPIKE,POSTSYN,WEIGHT core
  class STDP,LTP,LTD,NEUROMOD learning
  class NOISE,ENERGY,GLIA optional

  ```

  
SNN構造への導入アイデア（改訂版）このドキュメントでは、Mermaid図の各構造要素をどのようにSpiking Neural Network (SNN) に採用できるかを、実装レベルで整理します。1. 生物構造とSNNモデルの対応表図の構成ブロックSNNモデルでの実装対応実装ライブラリ例 / 備考 (提案1, 2反映)感覚入力層 (SENSORY / ENCODING)スパイク符号化器（RateCodeEncoder, TemporalEncoder）snnTorch, BindsNET, Nengo など。詳細はPhase 1参照。単一ニューロン層 (MEMBRANE / THRESH / REFRACT)LIFニューロン or AdExモデルsnn.LIFCell()。膜時定数 τm (10-20 ms)。不応期 (2-5 ms) は絶対（発火不可）または相対（閾値上昇）で実装。シナプス (NT_E / NT_I / POSTSYN)重み付きスパイク伝達（E/Iバランス付き）シナプス時定数 τs (5-10 ms)。E/I比率（ニューロン数比 80:20 が一般的）。Dale's principle（ニューロンはE/Iどちらか一方）を実装。可塑性 (STDP / LTP / LTD / WEIGHT)Spike-Timing-Dependent Plasticity (STDP) ルール( \Delta w = A^+ e^{-\Delta t/\tau^+} - A^- e^{\Delta t/\tau^-} )。STDP時定数 τ+/τ- (例: 20 ms / 20 ms)。ネットワーク構造 (FF / REC / EI / OSC / HIERARCHY)フィードフォワード + リカレント + 階層構造CNN + Recurrent SNN / Liquid State Machine。皮質6層構造や視床-皮質ループのモデル化（高度化）。情報表現層 (RATE / TEMP / POPULATION)出力スパイク統計解析PSTH, ISI統計, 同期解析出力層 (DECODE / MOTOR)スパイク積分または投票復号化SpikeIntegrator, RateDecoderフィードバック / 報酬信号 (NEUROMOD)報酬STDP (Dopamine-like RPE)強化学習型SNNMeta層 (NOISE / ENERGY / GLIA)ノイズ注入・エネルギー正則化生物的安定性の模倣2. 構築手順（フェーズ別）🔹 Phase 1：符号化層（Encoding） (提案3反映)感覚入力をスパイク列に変換。Rate Coding (レート符号化):概要: 一定時間内のスパイク数（発火頻度）で情報の強度を表現。使い分け: 静的な情報（例: 画像認識のピクセル強度）や、時間解像度が重要でないタスク。実装例 (Poisson-like):# 入力x (0-1) に比例した発火率のポアソンスパイクを生成
# time_steps: シミュレーションステップ数
rate = x / x.max()
spike_train = torch.rand(time_steps, *rate.shape) < rate
Temporal / Latency Coding (時間符号化 / 潜時符号化):概要: 最初のスパイクが発火するまでの時間（潜時）で情報を表現。入力が強いほど早く発火。使い分け: 高速な応答が求められるタスク（例: 音声認識、触覚処理）。時間的パターン自体が重要な場合。実装例 (Latency encoding):# 入力x (0-1) が大きいほど潜時が短くなる (t_maxから減少)
t_max = 20 # 最大潜時ステップ
spike_times = t_max * (1.0 - x)
# 実際には各ステップで spike_times < current_step かを判定して発火
🔹 Phase 2：ニューロンモデル（Neuron） (提案1反映)LIFモデルで膜電位動態を実装。$$\tau_m \frac{dV_m}{dt} = - (V_m - V_{rest}) + R_m I_{syn}
\]  * **τm (膜時定数, 10-20 ms)**: 膜電位が変化に応答する速さ。これが短いと入力変化に敏感に、長いと入力を時間的に積分しやすくなる。
* **発火条件**: ( V\_m \ge V\_{th} \Rightarrow V\_m \leftarrow V\_{reset} )
* **不応期 (Refractory Period, 2-5 ms)**:
* **絶対不応期**: 発火直後の一定期間、一切発火しない（`snnTorch` の `reset_mechanism`）。
* **相対不応期**: 発火直後、閾値 `Vth` を一時的に上昇させ、より強い入力がないと発火しにくくする（より生物学的）。

<!-- end list -->

```python
import snntorch as snn
# betaは時定数τmから計算 (例: beta = exp(-dt/τm))
# dt=1ms, τm=20ms の場合
beta = torch.exp(-1.0/20.0) # 約 0.95
lif = snn.Leaky(beta=beta, threshold=1.0, reset_mechanism="subtract")
```

-----

### 🔹 Phase 3：シナプスと学習（Synapse & Plasticity） (提案1, 4反映)

STDPに基づき、スパイク時刻差で重み更新。

\[ \tau\_s \frac{dI\_{syn}}{dt} = -I\_{syn} + \sum\_j w\_{ij} \delta(t - t\_j) \]

* **τs (シナプス時定数, 5-10 ms)**: シナプス後電流（または電位）が減衰する速さ。

STDP則:
\[ \Delta w = \text{if } \Delta t \> 0: A^+ e^{-\Delta t/\tau^+} \quad (\text{LTP}) \]
\[ \Delta w = \text{if } \Delta t \< 0: -A^- e^{\Delta t/\tau^-} \quad (\text{LTD}) \]

* **τ+/τ- (STDP時定数, 例: 20 ms / 20 ms)**: LTP/LTDが誘導される時間窓。
* **A\_plus / A\_minus (学習率, 提案4反映)**:
* 重み変化の最大量。
* **生物学的妥当性**: 固定値ではなく、ニューロンの状態や神経修飾物質によって動的に変化します。
* **実装上の値**: スケールに依存しますが、`w` の範囲を [0, 1] と正規化した場合、`A_plus = 0.01`, `A_minus = 0.005` など、LTPとLTDで非対称かつ小さな値に設定されることが多いです。安定性を保つため、LTDをLTPよりわずかに大きく設定する（`A_minus > A_plus`）場合もあります。

<!-- end list -->

```python
def stdp_update(pre_t, post_t, w,
A_plus=0.01, A_minus=0.005, # (提案4)
tau_plus=20.0, tau_minus=20.0): # (提案1)
dt = post_t - pre_t
dw = torch.where(dt > 0,
A_plus * torch.exp(-dt / tau_plus),
-A_minus * torch.exp(dt / tau_minus))
w += dw
w.clamp_(0, 1) # 重みの範囲を[0, 1]に制限
```

NEUROMODノードはドーパミン報酬信号 `r(t)` としてSTDPに乗算可能（報酬STDP）。

\[ \Delta w = r(t) \cdot STDP(\Delta t) \]

-----

### 🔹 Phase 4：ネットワーク構造（Network） (提案2, 5反映)

| 要素 | SNN構成例 |
|:---|:---|
| FF | 畳み込み型SNN (`snn.Conv2d`) |
| REC | リカレントスパイクネット (`snn.RLeaky`) |
| EI | (提案2) **E/Iバランス層**。ニューロン集団を80%の興奮性(E)と20%の抑制性(I)ニューロンに分割。**Dale's principle** に従い、Eニューロンは正の重みのみ、Iニューロンは負の重みのみを持つように制約。ネットワークの活動を安定化させる。 |
| OSC | 同期振動を解析対象に（FFT, コヒーレンス） |
| HIERARCHY | (提案5) **階層構造**。`4. 高度化アイデア` 参照。 |

-----

### 🔹 Phase 5：出力復号（Decoding）

スパイク列を最終的な出力（クラスラベルなど）に変換。

* **Rate Decoding (レート復号)**: 一定時間（または全時間）のスパイク総数を積分し、最も発火したニューロンを選択。
```python
# spike_rec: (time_steps, batch_size, num_outputs)
out_activity = spike_rec.sum(dim=0) # (batch_size, num_outputs)
pred = out_activity.argmax(dim=1)
```
* **Temporal Decoding (時間復号)**: 最初に発火したニューロンを選択（Latency符号化の逆）。

> `DECODE → PERCEPT / MOTOR` に対応。

-----

### 🔹 Phase 6：フィードバック学習

誤差信号を報酬STDPに反映し、自己教師的適応を実現。

\[ \Delta w = r(t) \cdot STDP(\Delta t) \]

-----

## 3\. 推奨ディレクトリ構成

```
SNN_Model/
│
├── encoders/
│   ├── rate_encoder.py
│   └── temporal_encoder.py
│
├── neurons/
│   ├── lif_cell.py
│   └── adex_cell.py
│
├── synapses/
│   ├── stdp.py
│   ├── dopamine_mod.py
│   └── ei_balance.py (Dale's principle実装)
│
├── network/
│   ├── feedforward.py
│   ├── recurrent.py
│   ├── cortical_column.py (皮質6層構造など)
│   └── oscillation_monitor.py
│
├── decoders/
│   └── rate_decoder.py
│
└── meta/
├── noise_injector.py
├── energy_constraint.py
└── glia_support.py
```

> 各ファイルがMermaid図のノードに1対1で対応。

-----

## 4\. 評価指標とベンチマーク (提案6反映)

SNNの性能は、従来の精度（Accuracy）だけでなく、その生物学的特性も評価対象となります。

| 指標 | 計算方法 | 生物学的意義 / SNNにおける重要性 |
|:---|:---|:---|
| **Classification Accuracy** | 標準精度測定（正解数 / 全データ数） | タスクの遂行能力（基本指標） |
| **Spike Efficiency (Sparseness)** | 総スパイク数 / 正解数、または全ニューロンの平均発火率 | **エネルギー効率**。スパイクが疎（スパース）であるほど低消費電力。 |
| **Temporal Precision** | ISI（スパイク間隔）変動係数 (CV) | **情報伝達精度**。CVが低いほど時間的に正確な情報を持つ。 |
| **Synchrony Index** | スパイク時間の相互相関、またはコヒーレンス | **ネットワーク協調性**。ガンマ波などの振動と同期した発火が情報処理に重要とされる。 |
| **Latency to First Spike** | 入力から最初の出力スパイクまでの時間 | **応答速度**。特にTemporal/Latency符号化で重要。 |

-----

## 5\. 高度化アイデア (提案5反映)

1.  **(提案5) 階層的ネットワーク構造**:
* **皮質6層構造（Cortical Column）**: 入力（L4）、処理（L2/3）、出力（L5/6）など、層ごとに異なるニューロンモデルや結合パターンを実装し、より複雑な局所処理を実現する。
* **視床-皮質ループ**: 視床（リレー核）と皮質間の再帰的ループをモデル化し、注意（Attention）や意識のゲーティング機構を組み込む。
2.  **時間依存ネットワーク解析**: FFTでネットワーク全体の同期リズム（Gamma/Theta）を観測し、タスク実行中の状態遷移を分析。
3.  **多モーダル入力**: 視覚＝Rate符号、聴覚＝Temporal符号など、異なる符号化を組み合わせる。
4.  **Homeostatic Plasticity**: 個々のニューロンが発火率を一定（例: 5 Hz）に保つように、自身の閾値やシナプス重みを調整する長期的な可塑性。ネットワークの安定性を高める。
5.  **エネルギー制約学習**: 損失関数にスパイク活動のペナルティ項を追加し、精度とエネルギー効率をトレードオフする。
\[ L = L\_{task} + \lambda \sum\_i r\_i \]
（( r\_i ) はニューロンiの平均発火率）

-----

## 6\. 導入方針まとめ

| フェーズ | 実装方向 | 効果 |
|:---|:---|:---|
| 感覚入力 | Rate/Temporal/Latency符号化器 | 生理的入力表現、タスクに応じた使い分け |
| ニューロン | LIFセル（各種時定数 τm, 不応期を考慮） | 時系列スパイク処理の基本単位 |
| シナプス | STDP + Dopamine報酬（時定数 τs, τ+/τ- を考慮） | 自己組織的学習、生物学的妥当性の向上 |
| ネットワーク | FF + REC + E/I構造 (Dale's principle) + 階層構造 | 安定した動的表現、高次の脳機能の模倣 |
| 出力 | Spike Integration / Latency Decoder | 認知・運動表現化 |
| 評価 | 精度 + スパース性 + 同期性 | タスク性能とエネルギー効率の多面的評価 |
| Meta層 | ノイズ・エネルギー制約 | 生物的頑健性再現 |

-----

この構成を採用すれば、Mermaid図の概念構造をそのまま、より詳細で**生物学的に妥当性の高いSNNアーキテクチャ**として実装可能になります。$$