# **SNN Project: ANN-SNN高忠実度変換パイプライン (v11.1)**

## **🚀 プロジェクト概要**

本プロジェクトは、既存の高性能なANNモデル（LLMやCNN）を、エネルギー効率に優れたスパイクニューラルネットワーク（SNN）モデルに変換するための**高忠実度自動変換パイプライン**を提供します。特に、変換時の精度劣化を最小限に抑え、$T=1$の低遅延推論を可能にするための最先端の最適化戦略を統合しています。

### **🏆 主要な機能強化**

| 機能 | 目的 | 実装技術 |
| :---- | :---- | :---- |
| **高忠実度変換** | ANNと同等の精度をSNNで達成 | ECL (誤差補償学習)、Scale-and-Fireニューロン（SFN） |
| **動的最適化** | モデルサイズとエネルギー消費の削減 | スパイク頻度に基づく**動的プルーニング**、**量子化メタデータ** |
| **スパイク注意機構** | LLM/TransformerのSNN化 | Softmax代替**Spiking Attention**モジュール |
| **柔軟な展開** | 様々なハードウェアへの対応 | Loihi/TrueNorth向け**ハードウェア最適化ヒント**の埋め込み |

## **🛠️ 変換ツールの使い方**

変換は、snn-cli.pyのconvertコマンドまたはscripts/convert\_model.pyスクリプトを使用して実行します。

### **1\. 前提条件: キャリブレーションデータの準備**

変換時の閾値キャリブレーション（ANNの活性化をSNNの閾値にマッピングする処理）には、**少量の代表的なデータ**（キャリブレーションセット）が必要です。

\# LLMタスクの場合: ダミーデータまたは検証データの一部  
\# ANNモデルが処理できる形式のデータローダーを用意します  
\# (scripts/convert\_model.py がデータローダーを自動生成することを想定)

### **2\. LLMモデルの変換と最適化**

LLM変換 (convert\_llm\_weights) は、特に多くの最適化フラグをサポートします。

\# LLM変換実行例:  
python scripts/convert\_model.py \\  
    \--ann\_model\_path "gpt2" \\\\  \# HuggingFace ID または .pth / .safetensors ファイル  
    \--snn\_model\_config configs/models/spiking\_transformer.yaml \\  
    \--output\_snn\_path runs/converted\_spiking\_llm.pth \\  
    \--method llm-convert \\  
    \--calibration\_loader\_stub \<データローダー引数\> \\  
    \--use-ecl \\  \# ECL（誤差補償学習）を有効化  
    \--prune-low-activity 0.15 \\\\ \# スパイク頻度の低いニューロンを15%プルーニング  
    \--quantization-bits 4.0 \\\\ \# INT4/INT8量子化ヒントをメタデータに保存  
    \--hardware-target "Loihi"  \# ハードウェア最適化ヒント

### **3\. CNNモデルの変換と最適化**

画像モデル（CNN）の変換は、**BatchNorm Folding**を自動的に行います。

\# CNN変換実行例:  
python scripts/convert\_model.py \\  
    \--ann\_model\_path runs/ann\_resnet18\_cifar10.pth \\  
    \--snn\_model\_config configs/cifar10\_spikingcnn\_config.yaml \\  
    \--output\_snn\_path runs/converted\_spiking\_cnn.pth \\  
    \--method cnn-convert \\  
    \--use-ecl \\  
    \--prune-low-activity 0.05

## **⚙️ 変換後のモデルメタデータと活用（デプロイヒント）**

変換パイプラインは、SNNモデルの重みだけでなく、後続の学習およびデプロイメントプロセスで必須となる**最適化ヒント**をconversion\_metadata辞書として保存します。

| メタデータキー | 説明 | 活用法（後続プロセス） |
| :---- | :---- | :---- |
| distillation.teacher\_model | 知識蒸留に使用すべき教師モデル名。 | 変換後SNNの**End-to-End微調整**時の損失関数設定に使用。 |
| pruning\_ratio | 変換時に達成された最終的なスパース性。 | ハードウェアコンパイラが**メモリ配置と計算スキップ**を最適化するために参照。 |
| target\_quantization\_bits | 推奨される最終量子化ビット数（例: 4.0, 8.0）。 | **QAT** (Quantization-Aware Training) や**実行時カーネル**の低精度演算設定に利用。 |
| hardware\_optimization\_hint | ターゲットとするニューロモーフィックチップ（Loihi, TrueNorth）のヒント。 | **並列実行戦略**（タイムステップバッチングなど）や、カスタム**融合カーネル**の呼び出しをトリガー。 |
| normalization\_compensation | RMSNormなどの複雑な正規化に対する補正ヒント。 | SNN学習時に\*\*学習率や膜時定数（tau）\*\*をチャネルごとに動的に調整するための初期値として利用。 |
| evaluation\_metrics | ベンチマークで計測すべき必須指標。 | CI/CDパイプラインにおいて、**精度、スパイク数、レイテンシ**を自動計測し、SNNの総合性能を評価。 |

## **⚠️ 制限事項と技術的制約**

1. **学習（Point 3）**: 変換直後は必ず精度が低下します。SNNの性能を最大限に引き出すには、変換後のモデルを--use-eclフラグと**Surrogate Gradient**を用いた微調整（ファインチューニング）が必要です。  
2. **RMSNorm/SwiGLU（Point 1）**: RMSNormやSwiGLUの厳密なスパイク化は困難です。このパイプラインでは、変換後のメタデータに補正ヒントを保存することで、後の学習フェーズでSNNの学習能力を使って誤差を補償することを推奨しています。  
3. **動的プルーニング（Point 7）**: prune-low-activityは、モデルが**キャリブレーションデータ**上で推論を実行し、その活性化の傾向（Saliency）に基づいて剪定を行います。データが不十分な場合、不適切なシナプスが削除されるリスクがあります。