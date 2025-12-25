# **SNNプロジェクト 網羅的テストマニュアル**

このドキュメントは、SNN（Spiking Neural Network）プロジェクトにおけるテスト実行手順、検証スクリプトの使用方法、および各テストの目的を網羅的にまとめたものです。

開発者は、コードの変更後にこれらのテストを実行し、リグレッションが発生していないことを確認してください。

## **1\. クイックスタート (全テスト実行)**

プロジェクトの健全性を全体的に確認するためのコマンドです。

### **プロジェクト全体の自動テスト**

scripts/run\_all\_tests.py スクリプトを使用すると、推奨される設定で全てのテストスイートを実行できます。

python scripts/run\_all\_tests.py

### **Pytestによる一括実行**

標準的な pytest コマンドを使用して、tests/ ディレクトリ以下の全てのテストを実行します。

pytest

## **2\. テストカテゴリと実行コマンド**

テストは目的ごとに分類されています。開発中の機能に合わせて、個別に実行することで効率的に開発を進めることができます。

### **2.1 スモークテスト (Smoke Tests)**

主要なコンポーネントがエラーなく起動・動作するかを素早く確認するためのテストです。詳細なロジック検証の前に実行することを推奨します。

| テスト対象 | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **全パラダイム** | 全ての主要な計算パラダイムの基本動作確認 | pytest tests/test\_smoke\_all\_paradigms.py |

### **2.2 ユニットテスト & 機能テスト (Unit & Functional Tests)**

個々のモジュールやクラスの動作を検証します。

#### **コアコンポーネント (Core Components)**

| テスト対象 | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **エンコーダー** | Universal Encoderの動作検証 | pytest tests/test\_universal\_encoder.py |
| **DSA層** | Dynamic Sparse Attention Layerの検証 | pytest tests/test\_dsa\_layer.py |
| **QK Norm** | Query-Key Normalizationの検証 | pytest tests/test\_qk\_norm.py |
| **Bit Spike** | Bit Spike Mambaアーキテクチャの検証 | pytest tests/test\_bit\_spike\_mamba.py |

#### **認知アーキテクチャ (Cognitive Architecture)**

| テスト対象 | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **認知モジュール** | 記憶、推論などの認知コンポーネント群 | pytest tests/cognitive\_architecture/ |
| **脳統合** | 各脳モジュールの統合動作検証 | pytest tests/test\_brain\_integration.py |
| **非同期カーネル** | 非同期処理カーネルの動作検証 | pytest tests/test\_async\_brain\_kernel.py |
| **視覚野** | Visual Cortexモデルの検証 | pytest tests/test\_visual\_cortex.py |
| **Liquid Association** | Liquid Association Cortexの検証 | pytest tests/test\_liquid\_association.py |

### **2.3 統合テスト & パイプラインテスト (Integration Tests)**

複数のコンポーネントが連携して動作することを確認するテストです。

| テスト対象 | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **DVSパイプライン** | Dynamic Vision Sensorデータ処理フロー | pytest tests/test\_dvs\_pipeline.py |
| **GRPOロジック** | Group Relative Policy Optimizationロジック | pytest tests/test\_grpo\_logic.py |
| **実世界統合** | センサー/アクチュエータを含む統合テスト | pytest tests/test\_integration\_real\_world.py |

## **3\. 検証・診断スクリプト (Verification Scripts)**

tests/ ディレクトリのテストコード以外にも、scripts/ ディレクトリにはシステムの健全性やパフォーマンスを検証するための有用なスクリプトが含まれています。

### **3.1 システム診断・ヘルスチェック**

| スクリプト | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **ヘルスチェック** | プロジェクト全体の依存関係と構成を確認 | python scripts/run\_project\_health\_check.py |
| **コンパイラテスト** | ハードウェアコンパイラの動作確認 | python scripts/runners/run\_compiler\_test.py |

### **3.2 パフォーマンス・学習検証**

| スクリプト | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **パフォーマンス検証** | 推論速度やメモリ効率の測定 | python scripts/verify\_performance.py |
| **Phase 3 検証** | プロジェクトフェーズ3の要件検証 | python scripts/verify\_phase3.py |
| **DSA学習検証** | DSAアルゴリズムの学習能力検証 | python scripts/verify\_dsa\_learning.py |
| **ベンチマーク** | 総合的なベンチマークスイートの実行 | python scripts/run\_benchmark\_suite.py |

### **3.3 デモ・シミュレーション実行**

特定の機能の動作を目視で確認するためのデモスクリプトです。

| スクリプト | 説明 | 実行コマンド |
| :---- | :---- | :---- |
| **Brain v14** | Artificial Brain v14の実行 | python scripts/run\_artificial\_brain\_v14.py |
| **視覚野デモ** | 視覚処理のデモンストレーション | python scripts/runners/run\_brain\_v20\_vision.py |
| **Web学習** | Webからの能動的学習デモ | python scripts/runners/run\_web\_learning.py |
| **睡眠学習** | 睡眠フェーズによる記憶定着デモ | python scripts/runners/run\_sleep\_learning\_demo.py |

## **4\. トラブルシューティング**

テストが失敗する場合の一般的な対処法です。

### **よくあるエラー**

1. **ModuleNotFoundError**:  
   * 仮想環境が有効になっているか確認してください。  
   * pip install \-r requirements.txt で依存関係がインストールされているか確認してください。  
   * 実行ディレクトリがプロジェクトルートであることを確認してください。  
2. **ImportError**:  
   * PYTHONPATH が正しく設定されていない可能性があります。プロジェクトルートで実行するか、export PYTHONPATH=$PYTHONPATH:. を実行してください。  
3. **CUDA/GPU エラー**:  
   * GPU環境がない場合、設定ファイル (configs/) で device: cpu に変更するか、コード内のデバイス判定ロジックがCPUにフォールバックしているか確認してください。

### **デバッグのヒント**

特定のテストの詳細なログを表示したい場合は、-v (verbose) や \-s (標準出力を表示) オプションを使用します。

pytest tests/test\_brain\_integration.py \-v \-s  
