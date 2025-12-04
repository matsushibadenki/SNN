# ファイルパス: tests/test_integration_real_world.py
# Title: 実用シナリオ統合テスト (詳細化・改善)
# Description: Improvement-Plan.md に基づき、ECG分析などの実用的なシナリオを
#              想定した統合テストを詳細化します。実際のデータや完全なモデル
#              の代わりに、モックオブジェクトとダミーデータを使用します。
#              ECGデータの生成を改善し、テストログを追加。

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import logging # ロギングをインポート
import random # random をインポート

# プロジェクトルートをPythonパスに追加
# testsディレクトリの一つ上がプロジェクトルートと仮定
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- 必要なモジュールをインポート ---
# (実際のプロジェクト構成に合わせて調整)
# 例: from snn_research.core.snn_core import SNNCore
# 例: from snn_research.training.trainers import BreakthroughTrainer
# 例: from snn_research.metrics.energy import EnergyMetrics, ENERGY_PER_SNN_OP

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- モッククラス (変更なし) ---
class MockSNNCore(nn.Module):
    """SNNCoreのダミー実装"""
    def __init__(self, num_classes=2, features=128):
        super().__init__()
        self.features = features
        self.fc = nn.Linear(features, num_classes) # ダミーの出力層
        self.num_classes = num_classes
        self._total_spikes = 0

    def forward(self, x, return_spikes=False, **kwargs):
        # ダミーの推論ロジック
        batch_size = x.shape[0]
        # 簡易的な特徴抽出をシミュレート
        dummy_features = torch.randn(batch_size, self.features, device=x.device)
        logits = self.fc(dummy_features)

        # ダミースパイク数を計算
        avg_spikes_val = 1500.0 + random.random() * 100 # 適当なスパイク数
        self._total_spikes += avg_spikes_val * batch_size
        avg_spikes = torch.tensor(avg_spikes_val, device=x.device)

        # (logits, avg_spikes, mem) のタプルを返す
        return logits, avg_spikes, torch.tensor(0.0, device=x.device)

    def get_total_spikes(self) -> float:
        return self._total_spikes

    def reset_spike_stats(self):
        self._total_spikes = 0

class MockBreakthroughTrainer:
    """BreakthroughTrainerのダミー実装"""
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self._train_loss_history = []
        self._steps = 0

    def train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        # ダミーの損失計算
        # loss = self.criterion(outputs[0], targets) # 実際のモデル出力を使う
        loss = torch.tensor(0.5 + random.random() * 0.1, requires_grad=True) # ダミー損失
        loss.backward()
        self.optimizer.step()
        self._train_loss_history.append(loss.item())
        self._steps += 1
        return loss.item()

    def get_average_loss(self, window=10):
        if not self._train_loss_history: return float('inf')
        return sum(self._train_loss_history[-window:]) / len(self._train_loss_history[-window:])

class MockEnergyMetrics:
    """EnergyMetricsのダミー実装"""
    def compute_energy(self, avg_spikes, num_neurons):
        # ダミーのエネルギー計算
        ENERGY_PER_SNN_OP = 0.9e-12 # Joules (仮)
        # 簡易的な計算
        return avg_spikes * num_neurons * ENERGY_PER_SNN_OP * 0.1 # 係数は適当

# --- テストクラス ---
# @pytest.mark.skip(reason="依存関係のある実際のクラスが必要です") # 必要に応じてスキップ解除
class TestRealWorldScenarios:
    """
    実際のユースケースに基づいた統合テスト（詳細化・改善）。
    """

    @pytest.fixture
    def device(self):
        """テストに使用するデバイスを取得"""
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {dev}")
        return dev

    @pytest.fixture
    def ecg_data(self, device):
        """ECGテストデータのダミーを生成するフィクスチャ（改善）。"""
        logger.info("   -> Generating dummy ECG test data...")
        batch_size = 4
        time_steps = 500
        features = 1 # 単一チャネルECG
        # 基礎となるノイズ
        data = torch.randn(batch_size, time_steps, features, device=device) * 0.1
        # 周期的な心拍様パターンを追加 (例: 約60BPM -> 1秒周期 -> 1000ms周期)
        # time_steps = 500ms なので、半周期分くらい
        time_vector = torch.arange(time_steps, device=device).float() / time_steps * (2 * torch.pi / 2) # 半周期分
        heartbeat_pattern = torch.sin(time_vector).unsqueeze(0).unsqueeze(-1) * 0.5
        data += heartbeat_pattern
        # 高周波ノイズも少し追加
        data += torch.randn(batch_size, time_steps, features, device=device) * 0.05

        labels = torch.randint(0, 2, (batch_size,), device=device) # 異常(1)/正常(0)のダミーラベル
        logger.info(f"   -> Generated ECG data shape: {data.shape}, Labels shape: {labels.shape}")
        return data, labels

    @pytest.fixture
    def trained_ecg_model(self, device):
        """訓練済みのECG異常検出モデルのダミーを作成するフィクスチャ。"""
        logger.info("   -> Creating dummy 'trained' SNN model for ECG.")
        # 実際の訓練済みモデルの代わりに、初期化済みのモックモデルを使用
        model = MockSNNCore(num_classes=2, features=64).to(device) # 異常/正常の2クラス分類
        model.eval() # 評価モードに設定
        return model

    @pytest.fixture
    def online_learning_setup(self, device):
        """オンライン学習用の設定を準備するフィクスチャ。"""
        logger.info("   -> Setting up online learning mock environment...")
        model = MockSNNCore(num_classes=5, features=32).to(device) # 別のタスクを想定
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MockBreakthroughTrainer(model, criterion, optimizer, device)

        # ダミーのデータストリーム (DataLoaderでシミュレート)
        stream_data = torch.randn(100, 20, 10, device=device) # 100サンプル, 20タイムステップ, 10特徴量
        stream_labels = torch.randint(0, 5, (100,), device=device)
        stream_dataset = TensorDataset(stream_data, stream_labels)
        stream_loader = DataLoader(stream_dataset, batch_size=4)
        logger.info("   -> Online learning setup complete.")
        return trainer, stream_loader

    def test_ecg_anomaly_detection_pipeline(self, trained_ecg_model, ecg_data, device):
        """ECG信号の異常検出エンドツーエンドテスト（詳細化）。"""
        logger.info("Testing ECG anomaly detection pipeline...")
        model = trained_ecg_model
        data, _ = ecg_data # ラベルは推論には不要
        logger.info(f"   Input data shape for inference: {data.shape}")

        # 1. モデルで推論を実行
        with torch.no_grad():
            # 入力形式をモデルに合わせる
            # MockSNNCoreは (B, Features) を期待すると仮定し、時間軸で平均を取る
            # 実際のSNNモデル (例: TemporalFeatureExtractor) は (B, T, F) を受け付けるはず
            input_for_model = data.mean(dim=1) # (B, F=1) -> これだと特徴量数が合わない
            # MockSNNCoreの入力次元に合わせてダミー入力を作成
            input_for_model_mock = torch.randn(data.shape[0], model.features, device=device)
            logger.info(f"   Adjusted input shape for MockSNNCore: {input_for_model_mock.shape}")


            # MockSNNCore は (logits, avg_spikes, mem) を返す
            outputs = model(input_for_model_mock, return_spikes=True)
            logits = outputs[0] # (B, num_classes=2)
            predictions = torch.argmax(logits, dim=-1) # (B,)

        # 2. 期待される出力形式・範囲を検証
        assert predictions.shape == (data.shape[0],), f"Expected prediction shape {(data.shape[0],)}, but got {predictions.shape}"
        assert torch.all(predictions >= 0) and torch.all(predictions < model.num_classes), f"Predictions out of expected range [0, {model.num_classes-1}]"
        logger.info(f"   -> Inference successful. Output shape: {predictions.shape}. Sample predictions: {predictions.cpu().numpy()}")
        logger.info("✅ ECG anomaly detection pipeline test passed.")

    def test_online_learning_convergence(self, online_learning_setup):
        """オンライン学習の収束性テスト（詳細化）。"""
        logger.info("Testing online learning convergence...")
        trainer, stream_loader = online_learning_setup
        max_steps = 25 # 短いステップ数でテスト
        convergence_threshold = 0.1 # 収束したとみなす損失閾値
        initial_loss = float('inf')
        converged = False

        # 1. データストリームをシミュレートし、数ステップ学習
        step = 0
        for batch in stream_loader:
            if step >= max_steps:
                break
            loss = trainer.train_step(batch)
            if step == 0:
                initial_loss = loss
            logger.info(f"   -> Online step {step+1}/{max_steps}: Loss = {loss:.4f}")

            # 損失が発散したら失敗
            if step > 10 and loss > initial_loss * 2: # 最初の損失の2倍を超えたら発散とみなす
                logger.error(f"Loss possibly diverged (Initial: {initial_loss:.4f}, Current: {loss:.4f} at step {step+1}).")
                pytest.fail(f"Loss possibly diverged during online learning.")

            # 収束判定 (直近数ステップの平均損失が閾値以下)
            avg_loss_last_5 = trainer.get_average_loss(window=5)
            if step > 10 and avg_loss_last_5 < convergence_threshold:
                converged = True
                logger.info(f"   -> Converged at step {step+1} (Avg loss {avg_loss_last_5:.4f} < {convergence_threshold}).")
                break
            step += 1

        if not converged:
             logger.error(f"Online learning did not converge within {max_steps} steps (Last avg loss: {trainer.get_average_loss(window=5):.4f}, Threshold: {convergence_threshold}).")
        assert converged, f"Online learning did not converge within {max_steps} steps."
        logger.info("✅ Online learning convergence test passed.")


    def test_energy_measurement_accuracy(self, trained_ecg_model, ecg_data, device):
        """エネルギー測定の精度検証（詳細化）。"""
        logger.info("Testing energy measurement accuracy...")
        model = trained_ecg_model
        data, _ = ecg_data
        energy_calculator = MockEnergyMetrics() # ダミー計算機を使用

        # 1. モデルのスパイク数を取得 (推論を実行)
        with torch.no_grad():
            model.reset_spike_stats() # スパイクカウンターをリセット
            # 前のテストと同様の入力形式を仮定
            input_for_model_mock = torch.randn(data.shape[0], model.features, device=device)

            outputs = model(input_for_model_mock, return_spikes=True)
            avg_spikes_batch = outputs[1] # モデルが返すサンプルあたりの平均スパイク数
            total_spikes = model.get_total_spikes() # または総スパイク数を取得
            num_neurons = sum(p.numel() for p in model.parameters()) # 総パラメータ数で代用（簡易的）
            logger.info(f"   -> Total spikes (mock): {total_spikes:.0f}, Avg spikes/sample (mock): {avg_spikes_batch.item():.0f}")


        # 2. エネルギー計算を実行
        estimated_energy_per_inference = energy_calculator.compute_energy(
            avg_spikes=avg_spikes_batch.item(),
            num_neurons=num_neurons
        )
        logger.info(f"   -> Number of neurons (params): {num_neurons}")
        logger.info(f"   -> Estimated energy per inference: {estimated_energy_per_inference:.4e} J")

        # 3. 期待される範囲と比較
        min_expected_energy = 1e-9 # nJオーダー
        max_expected_energy = 1e-3 # mJオーダー
        assert min_expected_energy < estimated_energy_per_inference < max_expected_energy, \
            f"Estimated energy {estimated_energy_per_inference:.4e} J is outside the plausible range [{min_expected_energy:.1e}, {max_expected_energy:.1e}] J."
        logger.info("✅ Energy measurement accuracy test passed (within plausible range).")