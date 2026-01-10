# **テストコマンド一覧 (v17.3 対応)**

このドキュメントでは、プロジェクトの各種テスト、検証スクリプト、実験、デモを実行するためのコマンドを網羅的にまとめています。

重要: すべてのコマンドは、プロジェクトのルートディレクトリ（pyproject.toml がある場所）で実行してください。  
また、Pythonのモジュール検索パス問題を回避するため、必要に応じて PYTHONPATH=. を付与して実行してください。

## **1\. ユニットテスト・統合テスト (tests/)**

pytest を使用して、tests/ ディレクトリ以下のテストを実行します。

### **基本コマンド**

* **全テスト実行**:  
  python \-m pytest tests/

* ヘルスチェック (推奨):  
  プロジェクトの健全性を手軽に確認できます。  
  python scripts/tests/run\_project\_health\_check.py

* **全テストスクリプト実行**:  
  python scripts/tests/run\_all\_tests.py

### **カテゴリ別テスト**

* **スモークテスト (簡易動作確認)**:  
  python \-m pytest tests/test\_smoke\_all\_paradigms.py

* **実世界統合テスト**:  
  python \-m pytest tests/test\_integration\_real\_world.py

* **認知アーキテクチャ (Brain Components)**:  
  python \-m pytest tests/cognitive\_architecture/

* **Brain Integration (脳全体)**:  
  python \-m pytest tests/test\_brain\_integration.py

* **コンパイラテスト**:  
  python scripts/tests/run\_compiler\_test.py

* **検証スクリプト**:  
  python scripts/tests/verify\_phase3.py  
  python scripts/tests/verify\_dsa\_learning.py  
  python scripts/tests/verify\_performance.py  
  python scripts/tests/verify\_scalability.py

## **2\. デモ・シミュレーション (scripts/demos/)**

特定の機能やシナリオを視覚的・対話的に確認するためのスクリプトです。

### **Brain & Consciousness (意識・脳)**

* **Brain v16 (統合デモ)**: python scripts/demos/brain/run\_brain\_v16\_demo.py  
* **意識放送 (Global Workspace)**: python scripts/demos/brain/run\_conscious\_broadcast\_demo.py  
* **感情モデル**: python scripts/demos/brain/run\_emotional\_demo.py  
* **自由意志**: python scripts/demos/brain/run\_free\_will\_demo.py  
* **クオリア**: python scripts/demos/brain/run\_qualia\_demo.py  
* **自己修正**: python scripts/demos/brain/run\_self\_correction\_demo.py  
* **世界モデル**: python scripts/demos/brain/run\_world\_model\_demo.py

### **Learning (学習)**

* **継続学習**: python scripts/demos/learning/run\_continual\_learning\_demo.py  
* **蒸留学習**: python scripts/demos/learning/run\_distillation\_demo.py  
* **睡眠学習**: python scripts/demos/learning/run\_sleep\_learning\_demo.py  
* **睡眠サイクル**: python scripts/demos/learning/run\_sleep\_cycle\_demo.py

### **Visual & Spatial (視覚・空間)**

* **Forward-Forward**: python scripts/demos/visual/run\_forward\_forward\_demo.py  
* **Spiking FF**: python scripts/demos/visual/run\_spiking\_ff\_demo.py  
* **産業用視覚 (Industrial Eye)**: python scripts/demos/visual/run\_industrial\_eye\_demo.py  
* **空間認識**: python scripts/demos/visual/run\_spatial\_demo.py

### **Systems & Social (システム・社会性)**

* **能動的推論**: python scripts/demos/systems/run\_active\_inference\_demo.py  
* **マルチモーダル**: python scripts/demos/systems/run\_multimodal\_demo.py  
* **社会性認知**: python scripts/demos/social/run\_social\_cognition\_demo.py

## **3\. 実験スクリプト (scripts/experiments/)**

研究開発フェーズごとの実験を実行します。

### **Brain Evolution & Simulation**

* **脳の進化**: python scripts/experiments/brain/run\_brain\_evolution.py  
* **脳シミュレーション**: python scripts/experiments/brain/run\_brain\_simulation.py  
* **共感覚 (Synesthesia)**: python scripts/experiments/brain/run\_synesthetic\_simulation.py  
* **人工脳 v14**: python scripts/experiments/brain/run\_artificial\_brain\_v14.py  
* **Brain v20 Prototype**: python scripts/experiments/brain/run\_brain\_v20\_prototype.py

### **Learning & SCAL**

* **STDP学習**: python scripts/experiments/learning/run\_stdp\_learning.py  
* **継続学習**: python scripts/experiments/learning/run\_continual\_learning\_experiment.py  
* **オンチップ学習**: python scripts/experiments/learning/run\_on\_chip\_learning.py  
* **SCAL (Fashion-MNIST)**: python scripts/experiments/scal/run\_scal\_spiking\_ff\_fashion.py  
* **SCAL (Hybrid)**: python scripts/experiments/scal/run\_scal\_ff\_hybrid.py

### **Systems (Advanced)**

* **AGIプロトタイプ (Phase 6\)**: python scripts/experiments/systems/run\_phase6\_agi\_prototype.py  
* **文明シミュレーション (Phase 7\)**: python scripts/experiments/systems/run\_phase7\_civilization.py  
* **シンギュラリティ (Phase 8\)**: python scripts/experiments/systems/run\_phase8\_singularity.py  
* **集合知**: python scripts/experiments/systems/run\_collective\_intelligence.py  
* **ニューロモルフィックOS**: python scripts/experiments/systems/run\_neuromorphic\_os.py

### **Applications**

* **ECG解析**: python scripts/experiments/applications/run\_ecg\_analysis.py  
* **Web学習**: python scripts/experiments/applications/run\_web\_learning.py

## **4\. エージェント実行 (scripts/agents/)**

自律エージェントとしてモデルを稼働させます。

* **自律学習エージェント**: python scripts/agents/run\_autonomous\_learning.py  
* **強化学習エージェント (RL)**: python scripts/agents/run\_rl\_agent.py  
* **プランナー**: python scripts/agents/run\_planner.py  
* **Life Form**: python scripts/agents/run\_life\_form.py

## **5\. ベンチマーク (scripts/benchmarks/)**

性能や効率性を測定します。

* **レイテンシ測定**: python scripts/benchmarks/benchmark\_latency.py  
* **ベンチマークスイート実行**: python scripts/benchmarks/run\_benchmark\_suite.py

## **6\. その他ユーティリティ**

* **デバッグ (スパイク活動)**: python scripts/debug/debug\_spike\_activity.py  
* **デバッグ (信号診断)**: python scripts/debug/diagnose\_signal.py  
* **モデル変換**: python scripts/utils/convert\_model.py  
* **結果分析**: python scripts/visualization/analyze\_results.py  
* **可視化**: python scripts/visualization/visualize\_brain\_activity.py  
* **HPO (最適化)**: python scripts/optimization/run\_hpo.py