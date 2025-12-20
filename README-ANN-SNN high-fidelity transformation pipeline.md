# **SNN Project: ANN-SNN高忠実度変換 & Deep Bio-Calibration パイプライン (v14.0)**

## **🚀 プロジェクト概要**

本プロジェクトは、既存の高性能なANNモデル（LLMやCNN）を、エネルギー効率に優れたスパイクニューラルネットワーク（SNN）モデルに変換するための**高忠実度自動変換パイプライン**を提供します。

v14.0では、単なる重みの変換にとどまらず、**Deep Bio-Calibration（深層生物学的校正）** の概念を導入しました。これは、ANNの性能を「初期状態」としてSNNに移植し、その後、生物学的可塑性（STDP/BCM/Causal Trace）を用いて環境やハードウェア制約に適応させる、**ハイブリッド進化戦略**の中核を担う技術です。

### **🏆 主要な機能強化**

| 機能 | 目的 | 実装技術 |
| :---- | :---- | :---- |
| **Deep Bio-Calibration** | ANNの精度とSNNの適応性の完全融合 | **ECL (誤差補償学習)** の拡張、膜電位ダイナミクスの自動調整 |
| **高忠実度変換** | ANNと同等の精度をSNNで達成 | Scale-and-Fireニューロン（SFN）、$T=1$ 推論最適化 |
| **動的最適化** | モデルサイズとエネルギー消費の削減 | スパイク頻度に基づく**動的プルーニング**、**量子化メタデータ** |
| **スパイク注意機構** | LLM/TransformerのSNN化 | Softmax代替**Spiking Attention**モジュール |
| **Hardware Native Ready** | ニューロモーフィックOSへの展開準備 | Loihi/TrueNorth/FPGA向け**イベント駆動カーネル**への変換ヒント生成 |

## **🛠️ 変換ツールの使い方**

変換は、snn-cli.pyのconvertコマンドまたはscripts/convert\_model.pyスクリプトを使用して実行します。

### **1\. 前提条件: キャリブレーションデータの準備**

変換時の閾値キャリブレーション（ANNの活性化をSNNの閾値にマッピングする処理）には、**少量の代表的なデータ**（キャリブレーションセット）が必要です。

\# LLMタスクの場合: ダミーデータまたは検証データの一部

\# ANNモデルが処理できる形式のデータローダーを用意します

\# (scripts/convert\_model.py がデータローダーを自動生成することを想定)

### **2\. Deep Bio-Calibration を適用した LLM変換**

LLM変換 (convert\_llm\_weights) は、特に多くの最適化フラグをサポートします。--use-ecl フラグにより、Deep Bio-Calibration が有効になります。

\# LLM変換実行例:

python scripts/convert\_model.py \\

\--ann\_model\_path "gpt2" \\\\ \# HuggingFace ID または .pth / .safetensors ファイル

\--snn\_model\_config configs/models/spiking\_transformer.yaml \\

\--output\_snn\_path runs/converted\_spiking\_llm.pth \\

\--method llm-convert \\

\--calibration\_loader\_stub \<データローダー引数\> \\

\--use-ecl \\ \# Deep Bio-Calibration (ECL) を有効化

\--prune-low-activity 0.15 \\\\ \# スパイク頻度の低いニューロンを15%プルーニング

\--quantization-bits 4.0 \\\\ \# INT4/INT8量子化ヒントをメタデータに保存

\--hardware-target "Loihi" \# ハードウェア最適化ヒント

### **3\. CNNモデルの変換と最適化**

画像モデル（CNN）の変換は、**BatchNorm Folding**を自動的に行います。

\# CNN変換実行例:

python scripts/convert\_model.py \\

\--ann\_model\_path runs/ann\_resnet18\_cifar10.pth \\

\--snn\_model\_config configs/experiments/cifar10\_spikingcnn\_config.yaml \\

\--output\_snn\_path runs/converted\_spiking\_cnn.pth \\

\--method cnn-convert \\

\--use-ecl \\

\--prune-low-activity 0.05

## **⚙️ 変換後のモデルメタデータと活用（デプロイヒント）**

変換パイプラインは、SNNモデルの重みだけでなく、後続の学習およびデプロイメントプロセス（Phase 5, 6, 7）で必須となる**最適化ヒント**をconversion\_metadata辞書として保存します。

| メタデータキー | 説明 | 活用法（後続プロセス） |
| :---- | :---- | :---- |
| bio\_calibration\_status | Deep Bio-Calibration の適用状態。 | **Phase 5 (Neuro-Symbolic Evolution)** において、追加の生物学的学習（睡眠時の再学習など）が必要かどうかを判断。 |
| distillation.teacher\_model | 知識蒸留に使用すべき教師モデル名。 | 変換後SNNの**End-to-End微調整**時の損失関数設定に使用。 |
| pruning\_ratio | 変換時に達成された最終的なスパース性。 | **Phase 6 (Hardware Native)** のコンパイラが**メモリ配置と計算スキップ**を最適化するために参照。 |
| target\_quantization\_bits | 推奨される最終量子化ビット数（例: 4.0, 8.0）。 | **QAT** (Quantization-Aware Training) や**実行時カーネル**の低精度演算設定に利用。 |
| hardware\_optimization\_hint | ターゲットとするニューロモーフィックチップ（Loihi, TrueNorth）のヒント。 | **並列実行戦略**（タイムステップバッチングなど）や、カスタム**融合カーネル**の呼び出しをトリガー。 |
| normalization\_compensation | RMSNormなどの複雑な正規化に対する補正ヒント。 | SNN学習時に\*\*学習率や膜時定数（tau）\*\*をチャネルごとに動的に調整するための初期値として利用。 |
| evaluation\_metrics | ベンチマークで計測すべき必須指標。 | CI/CDパイプラインにおいて、**精度、スパイク数、レイテンシ**を自動計測し、SNNの総合性能を評価。 |

## **⚠️ 制限事項と技術的制約**

1. **初期精度の低下**: 変換直後はANNに比べて精度が低下する場合があります。Deep Bio-Calibration (ECL) と**Surrogate Gradient**を用いた微調整（ファインチューニング）により、本来の性能を引き出すことが推奨されます。  
2. **RMSNorm/SwiGLU**: RMSNormやSwiGLUの厳密なスパイク化は困難です。このパイプラインでは、変換後のメタデータに補正ヒントを保存することで、後の学習フェーズでSNNの学習能力を使って誤差を補償することを推奨しています。  
3. **動的プルーニング**: prune-low-activityは、モデルが**キャリブレーションデータ**上で推論を実行し、その活性化の傾向（Saliency）に基づいて剪定を行います。データが不十分な場合、不適切なシナプスが削除されるリスクがあります。  
4. **ハードウェア依存性**: 特定のハードウェア最適化ヒント（hardware\_optimization\_hint）は、ターゲットデバイスのSDKやコンパイラが対応している必要があります。

## **🔄 今後の展望 (Roadmap v14.0)**

* **Neuro-Symbolic Integration:** 変換されたSNNの隠れ層（アトラクタ）と、GraphRAGの概念ノードを動的にリンクさせるためのメタデータ拡張を計画しています。  
* **Event-Driven Kernel Support:** 変換時に、PyTorchモデルではなく、直接イベント駆動カーネル（CUDA/FPGAコード）を出力するオプションの開発を進めています。