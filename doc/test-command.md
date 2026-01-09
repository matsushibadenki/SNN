# **テストコマンド一覧 (v17.3 対応)**

このドキュメントでは、プロジェクトの各種テスト、検証スクリプト、実験、デモ、およびトレーニングを実行するためのコマンドを網羅的にまとめています。

すべてのコマンドは、**プロジェクトのルートディレクトリ**（pyproject.toml がある場所）で実行することを前提としています。

## **1\. ユニットテスト・統合テスト (tests/)**

pytest を使用して、tests/ ディレクトリ以下のテストを実行します。

### **基本コマンド**

* **全テスト実行**:  
  python \-m pytest tests/

* ヘルスチェック (推奨):  
  プロジェクトの健全性を手軽に確認できます。  
  snn-cli health-check  
  \# または  
  python scripts/tests/run\_project\_health\_check.py

* **クリーンアップ**:  
  snn-cli clean all

* **ログクリーンアップ**:  
  snn-cli clean logs 
* **キャッシュクリーンアップ**:  
  snn-cli clean cache

### **カテゴリ別テスト**

* **スモークテスト (簡易動作確認)**:  
  python \-m pytest tests/test\_smoke\_all\_paradigms.py

* **実世界統合テスト**:  
  python \-m pytest tests/test\_integration\_real\_world.py

* **認知アーキテクチャ (Brain Components)**:  
  python \-m pytest tests/cognitive\_architecture/

* **Brain Integration (脳全体)**:  
  python \-m pytest tests/test\_brain\_integration.py

* **特定機能のテスト**:  
  * Bit-Spike Mamba: python \-m pytest tests/test\_bit\_spike\_mamba.py  
  * ホメオスタシス: python \-m pytest tests/test\_homeostasis.py

## **2\. デモ・シミュレーション (scripts/demos/)**

特定の機能やシナリオを視覚的・対話的に確認するためのスクリプトです。

* **Brain v16 (統合デモ)**:  
  python scripts/demos/brain/run\_brain\_v16\_demo.py

* **睡眠学習 (Sleep Learning)**:  
  python scripts/demos/learning/run\_sleep\_learning\_demo.py

* **継続学習 (Continual Learning)**:  
  python scripts/demos/learning/run\_continual\_learning\_demo.py

* **視覚野 (Visual Cortex)**:  
  python scripts/demos/visual/run\_industrial\_eye\_demo.py

## **3\. 実験スクリプト (scripts/experiments/)**

研究開発フェーズごとの実験を実行します。

* **Brain Evolution (進化実験)**:  
  python scripts/experiments/brain/run\_brain\_evolution.py

* **Synesthesia (共感覚)**:  
  python scripts/experiments/brain/run\_synesthetic\_simulation.py

* **SCAL (統計的重心調整学習)**:  
  python scripts/experiments/scal/run\_scal\_spiking\_ff\_fashion.py

## **4\. エージェント実行 (scripts/agents/)**

自律エージェントとしてモデルを稼働させます。

* **自律学習エージェント**:  
  python scripts/agents/run\_autonomous\_learning.py

* **強化学習エージェント (RL)**:  
  python scripts/agents/run\_rl\_agent.py

## **5\. ベンチマーク (scripts/benchmarks/)**

性能や効率性を測定します。

* **レイテンシ測定**:  
  python scripts/benchmarks/benchmark\_latency.py

* **ベンチマークスイート実行**:  
  python scripts/benchmarks/run\_benchmark\_suite.py

## **6\. デバッグ・診断 (scripts/debug/)**

* **スパイク活動デバッグ**:  
  python scripts/debug/debug\_spike\_activity.py

* **信号診断**:  
  python scripts/debug/diagnose\_signal.py

## **7\. データ準備・ユーティリティ (scripts/data/, scripts/utils/)**

* **データ準備**:  
  python scripts/data/data\_preparation.py \--dataset wikitext-103

* **知識ベース構築**:  
  python scripts/data/build\_knowledge\_base.py

* **VLMダミーデータ生成**:  
  python scripts/data/generate\_vlm\_dummy\_data.py

* **モデル変換 (ANN \-\> SNN)**:  
  python scripts/utils/convert\_model.py

* **モデル管理**:  
  python scripts/utils/manage\_models.py

## **8\. 可視化・分析・最適化 (scripts/visualization/, scripts/optimization/)**

* **結果分析**:  
  python scripts/visualization/analyze\_results.py

* **HPO (ハイパーパラメータ最適化)**:  
  python scripts/optimization/run\_hpo.py  
  \# または Optuna版  
  python scripts/optimization/run\_optuna\_hpo.py

* **効率性自動チューニング**:  
  python scripts/optimization/auto\_tune\_efficiency.py

* **脳活動可視化**:  
  python scripts/visualization/visualize\_brain\_activity.py

* **スパイクパターン可視化**:  
  python scripts/visualization/visualize\_spike\_patterns.py

## **9\. システム実証実験 (Phase 6 \- 8 Simulations)**

Roadmap後半で実装された、高度な統合システムの動作検証コマンドです。これらはOSカーネル、社会性、自己進化を含みます。

### **Phase 6: AGIプロトタイプ (AGI Prototype)**

全脳アーキテクチャ、Ethical Guardrail（安全装置）、および自己修正機能の統合テスト。  
「危険な思考」に対するアストロサイトの物理的介入（Metabolic Block）もここで検証されます。  
python scripts/experiments/systems/run\_phase6\_agi\_prototype.py

* **検証内容**:  
  * OSカーネル (NeuromorphicOS) のブート  
  * 視床 (Thalamus) を介した知覚ループ  
  * EthicalGuardrail による危険思考の検知と遮断

### **Phase 7: 文明シミュレーション (Civilization & Consensus)**

複数の人工脳エージェントによる対話、合意形成（リキッドデモクラシー）、および文化の継承をシミュレートします。

python scripts/experiments/systems/run\_phase7\_civilization.py

* **検証内容**:  
  * マルチエージェントの生成  
  * ConsensusEngine による合意形成と信頼度更新  
  * CultureRepository への知識（ミーム）の保存と継承

### **Phase 8: 技術的特異点シミュレーション (Singularity / Omega Point)**

再帰的自己改善（Recursive Self-Improvement）ループを実行し、AIが自らのコード/パラメータを進化させる過程をシミュレートします。

python scripts/experiments/systems/run\_phase8\_singularity.py

* **検証内容**:  
  * OmegaPointSystem による進化ループの起動  
  * 遺伝的アルゴリズムによる脳の変異と選別  
  * ターゲット性能到達（特異点）のシミュレーション