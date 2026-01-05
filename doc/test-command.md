# **テストコマンド一覧 (v17.3 対応)**

このドキュメントでは、プロジェクトの各種テスト、検証スクリプト、実験、デモ、およびトレーニングを実行するためのコマンドを網羅的にまとめています。

すべてのコマンドは、**プロジェクトのルートディレクトリ**（`pyproject.toml` がある場所）で実行することを前提としています。

---

## **1. ユニットテスト・統合テスト (tests/)**

`pytest` を使用して、`tests/` ディレクトリ以下のテストを実行します。

### **基本コマンド**

*   **全テスト実行**:
    ```bash
    python -m pytest tests/
    ```
*   **ヘルスチェック (推奨)**:
    プロジェクトの健全性を手軽に確認できます。
    ```bash
    snn-cli health-check
    # または
    python scripts/tests/run_project_health_check.py
    ```

### **カテゴリ別テスト**

*   **スモークテスト (簡易動作確認)**:
    ```bash
    python -m pytest tests/test_smoke_all_paradigms.py
    ```
*   **実世界統合テスト**:
    ```bash
    python -m pytest tests/test_integration_real_world.py
    ```
*   **認知アーキテクチャ (Brain Components)**:
    ```bash
    python -m pytest tests/cognitive_architecture/
    ```
*   **Brain Integration (脳全体)**:
    ```bash
    python -m pytest tests/test_brain_integration.py
    ```
*   **特定機能のテスト**:
    *   Bit-Spike Mamba: `python -m pytest tests/test_bit_spike_mamba.py`
    *   DVS Pipeline: `python -m pytest tests/test_dvs_pipeline.py`
    *   Visual Cortex: `python -m pytest tests/test_visual_cortex.py`
    *   Universal Encoder: `python -m pytest tests/test_universal_encoder.py`
    *   DSA Layer: `python -m pytest tests/test_dsa_layer.py`
    *   Async Brain Kernel: `python -m pytest tests/test_async_brain_kernel.py`
    *   Liquid Association: `python -m pytest tests/test_liquid_association.py`

---

## **2. 検証スクリプト (scripts/tests/)**

特定の機能やパフォーマンスを詳細に検証するためのスクリプト群です。

*   **プロジェクト健全性チェック**:
    ```bash
    python scripts/tests/run_project_health_check.py
    ```
*   **全テストランナー**:
    ```bash
    python scripts/tests/run_all_tests.py
    ```
*   **パフォーマンス検証**:
    ```bash
    python scripts/tests/verify_performance.py
    ```
*   **DSA学習検証**:
    ```bash
    python scripts/tests/verify_dsa_learning.py
    ```
*   **コンパイラテスト**:
    ```bash
    python scripts/tests/run_compiler_test.py
    ```
*   **スケーラビリティ検証**:
    ```bash
    python scripts/tests/verify_scalability.py
    ```
*   **Phase 3 検証**:
    ```bash
    python scripts/tests/verify_phase3.py
    ```

---

## **3. 実験ランナー (scripts/experiments/)**

研究開発フェーズごとの実験スクリプトです。機能別にサブディレクトリに整理されています。

### **Brain (脳モデル)**
*   **Brain v20 プロトタイプ**:
    ```bash
    python scripts/experiments/brain/run_brain_v20_prototype.py
    ```
*   **Brain v20 (視覚機能付き)**:
    ```bash
    python scripts/experiments/brain/run_brain_v20_vision.py
    ```
*   **Brain v21 (身体性・Embodiment)**:
    ```bash
    python scripts/experiments/brain/run_brain_v21_embodiment.py
    ```
*   **Brainとの対話 (CLI chat)**:
    ```bash
    python scripts/experiments/brain/talk_to_brain.py
    ```
*   **人工脳シミュレーション**:
    ```bash
    python scripts/experiments/brain/run_brain_simulation.py
    ```
*   **脳の進化 (Evolution)**:
    ```bash
    python scripts/experiments/brain/run_brain_evolution.py
    ```
*   **Brain v14 (Old Master)**:
    ```bash
    python scripts/experiments/brain/run_brain_v14.py
    ```
*   **マルチモーダル統合 (Brain v2.0 Integration)**:
    ```bash
    python scripts/experiments/brain/run_multimodal_brain.py
    ```

### **Learning (学習則・基礎実験)**
*   **STDP学習**:
    ```bash
    python scripts/experiments/learning/run_stdp_learning.py
    ```
*   **論理ゲート学習**:
    ```bash
    python scripts/experiments/learning/run_logic_gated_learning.py
    ```
*   **継続学習 (Continual Learning)**:
    ```bash
    python scripts/experiments/learning/run_continual_learning_experiment.py
    ```
*   **蒸留実験 (Distillation)**:
    ```bash
    python scripts/experiments/learning/run_distillation_experiment.py
    ```
*   **オンチップ学習**:
    ```bash
    python scripts/experiments/learning/run_on_chip_learning.py
    ```

### **Applications (応用)**
*   **ECG (心電図) 解析**:
    ```bash
    python scripts/experiments/applications/run_ecg_analysis.py
    ```
*   **VLM (Vision-Language Model) 適応**:
    ```bash
    python scripts/experiments/applications/run_vlm_adaptation.py
    ```
*   **VLM 睡眠学習**:
    ```bash
    python scripts/experiments/applications/run_vlm_sleep.py
    ```
*   **Web学習**:
    ```bash
    python scripts/experiments/applications/run_web_learning.py
    ```

### **Systems (システム・OS)**
*   **統合ミッション (The Odyssey)**:
    ```bash
    python scripts/experiments/systems/run_unified_mission.py
    ```
*   **集合知 (Collective Intelligence)**:
    ```bash
    python scripts/experiments/systems/run_collective_intelligence.py
    ```
*   **ニューロモルフィックOS**:
    ```bash
    python scripts/experiments/systems/run_neuromorphic_os.py
    ```
*   **ハードウェアシミュレーション**:
    ```bash
    python scripts/experiments/systems/run_hardware_simulation.py
    ```

### **SCAL (Statistical Centroid Alignment Learning)**
*   **SCAL Spiking FF (Fashion MNIST)**:
    ```bash
    python scripts/experiments/scal/run_scal_spiking_ff_fashion.py
    ```
*   **SCAL FF Hybrid v2**:
    ```bash
    python scripts/experiments/scal/run_scal_ff_hybrid_v2.py
    ```

---

## **4. デモ (scripts/demos/)**

特定の機能をわかりやすく実演するためのスクリプトです。

*   **マルチモーダル統合**: `python scripts/demos/run_multimodal_demo.py`
*   **睡眠学習**: `python scripts/demos/run_sleep_learning_demo.py`
*   **睡眠サイクル**: `python scripts/demos/run_sleep_cycle_demo.py`
*   **ニューロシンボリック推論**: `python scripts/demos/run_neuro_symbolic_demo.py`
*   **能動的推論 (Active Inference)**: `python scripts/demos/run_active_inference_demo.py`
*   **産業用Eye (Industrial Eye)**: `python scripts/demos/run_industrial_eye_demo.py`
*   **世界モデル (World Model)**: `python scripts/demos/run_world_model_demo.py`
*   **好奇心 (Curiosity)**: `python scripts/demos/run_curiosity_demo.py`
*   **感情モデル**: `python scripts/demos/run_emotional_demo.py`
*   **Forward-Forward**: `python scripts/demos/run_forward_forward_demo.py`
*   **継続学習**: `python scripts/demos/run_continual_learning_demo.py`
*   **空間認識**: `python scripts/demos/run_spatial_demo.py`
*   **Brain v16 デモ**: `python scripts/demos/run_brain_v16_demo.py`

---

## **5. エージェント (scripts/agents/)**

自律エージェント関連のスクリプトです。

*   **自律学習エージェント**:
    ```bash
    python scripts/agents/run_autonomous_learning.py
    ```
*   **プランナー (Planner)**:
    ```bash
    python scripts/agents/run_planner.py
    ```
*   **強化学習エージェント**:
    ```bash
    python scripts/agents/run_rl_agent.py
    ```
*   **デジタル生命体 (Life Form)**:
    ```bash
    python scripts/agents/run_life_form.py
    ```
*   **能動学習ループ**:
    ```bash
    python scripts/agents/run_active_learning_loop.py
    ```

---

## **6. トレーニング (scripts/training/)**

モデルの学習を行うためのスクリプトです。

*   **汎用トレーナー**:
    `--config` で設定ファイルを指定して実行します。
    ```bash
    python scripts/training/train.py --config configs/experiments/brain_v14_config.yaml
    ```
*   **MNIST SNN学習**:
    ```bash
    python scripts/training/train_mnist_snn.py
    ```
*   **CIFAR10 Bio-PC学習**:
    ```bash
    python scripts/training/train_bio_pc_cifar10.py
    ```
*   **Spiking VLM学習**:
    ```bash
    python scripts/training/train_spiking_vlm.py
    ```
*   **Planner学習**:
    ```bash
    python scripts/training/train_planner.py
    ```
*   **ダミートレーニング (動作確認用)**:
    ```bash
    python scripts/training/run_dummy_training.py
    ```

---

## **7. データ準備・ユーティリティ (scripts/data/, scripts/utils/)**

*   **データ準備**:
    ```bash
    python scripts/data/data_preparation.py --dataset wikitext-103
    ```
*   **知識ベース構築**:
    ```bash
    python scripts/data/build_knowledge_base.py
    ```
*   **VLMダミーデータ生成**:
    ```bash
    python scripts/data/generate_vlm_dummy_data.py
    ```
*   **モデル変換 (ANN -> SNN)**:
    ```bash
    python scripts/utils/convert_model.py
    ```
*   **モデル管理**:
    ```bash
    python scripts/utils/manage_models.py
    ```

---

## **8. 可視化・分析・最適化 (scripts/visualization/, scripts/optimization/)**

*   **結果分析**:
    ```bash
    python scripts/visualization/analyze_results.py
    ```
*   **HPO (ハイパーパラメータ最適化)**:
    ```bash
    python scripts/optimization/run_hpo.py
    # または Optuna版
    python scripts/optimization/run_optuna_hpo.py
    ```
*   **効率性自動チューニング**:
    ```bash
    python scripts/optimization/auto_tune_efficiency.py
    ```
*   **脳活動可視化**:
    ```bash
    python scripts/visualization/visualize_brain_activity.py
    ```
*   **スパイクパターン可視化**:
    ```bash
    python scripts/visualization/visualize_spike_patterns.py
    ```
*   **スパイク活動デバッグ**:
    ```bash
    python scripts/debug/debug_spike_activity.py
    ```

---

## **9. CLIツール (snn-cli)**

コマンドラインから主要機能を呼び出せます。

*   **GUI起動**: `snn-cli ui start`
*   **テスト実行**: `snn-cli test`
*   **ヘルスチェック**: `snn-cli health-check`
*   **クリーンアップ (全生成物)**: `snn-cli clean all`
*   **ログ削除**: `snn-cli clean logs`
```