# ファイルパス: snn_research/io/neuromorphic_dataset.py
# Title: ニューロモルフィック(DVS)データセットローダー [Type Fixed]
# Description:
#   spikingjellyの型定義がないためのmypyエラーを修正。

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
import os

# SpikingJellyが利用可能かチェック
try:
    # type: ignore[import-untyped]
    from spikingjelly.datasets.n_mnist import NMNIST

    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    # 警告は呼び出し時に出すためここではスキップ


class MockDVSGenerator(Dataset):
    """
    DVSデータセットの代わりに使用するダミーデータジェネレータ。
    形状とスパース性のテスト用。
    """

    def __init__(self, root: str, train: bool = True, data_type: str = 'frame', frames_number: int = 10, split_by: str = 'number'):
        self.num_samples = 100  # テスト用サンプル数
        self.frames_number = frames_number
        self.height = 34
        self.width = 34
        self.channels = 2  # On/Off events
        self.num_classes = 10

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # ランダムなスパイク列を生成 (Time, C, H, W) -> (frames_number, 2, 34, 34)
        # スパース性を模倣 (90%は0)
        spikes = (torch.rand(self.frames_number, self.channels,
                  self.height, self.width) > 0.9).float()
        label = torch.randint(0, self.num_classes, (1,)).item()
        return spikes, label


class NeuromorphicDataFactory:
    """
    DVSデータセットの取得とDataLoaderの作成を行うファクトリークラス。
    """
    @staticmethod
    def create_dataloader(
        dataset_name: str,
        root_dir: str = "./data",  # デフォルトをプロジェクトルートのdataに変更
        batch_size: int = 32,
        time_steps: int = 16,
        split: str = 'train',
        use_mock: bool = False,
        num_workers: int = 2,
        pin_memory: Optional[bool] = None  # Added argument
    ) -> DataLoader:
        """
        DataLoaderを作成する。
        """
        # Determine pin_memory default if not provided
        if pin_memory is None:
            # MPS does not support pin_memory currently, so disable it on Mac if using MPS
            if torch.backends.mps.is_available():
                pin_memory = False
            else:
                pin_memory = True

        dataset: Dataset

        # ディレクトリ作成
        if not os.path.exists(root_dir):
            try:
                os.makedirs(root_dir, exist_ok=True)
            except OSError:
                pass  # 読み取り専用環境等の対策

        if use_mock or not SPIKINGJELLY_AVAILABLE:
            if not SPIKINGJELLY_AVAILABLE and not use_mock:
                print("⚠️ SpikingJelly not found. Falling back to Mock Mode.")

            print(f"Dataset [{dataset_name}] (Mock Mode) initialized.")
            dataset = MockDVSGenerator(root=root_dir, train=(
                split == 'train'), frames_number=time_steps)
        else:
            if dataset_name.lower() == 'n-mnist':
                # N-MNISTの実データ読み込み
                target_path = os.path.join(root_dir, 'N-MNIST')
                if not os.path.exists(target_path) and not os.path.exists(root_dir):
                    print(f"⚠️ Data dir {target_path} not found. Using Mock.")
                    dataset = MockDVSGenerator(root=root_dir, train=(
                        split == 'train'), frames_number=time_steps)
                else:
                    try:
                        dataset = NMNIST(root=root_dir, train=(
                            split == 'train'), data_type='frame', frames_number=time_steps, split_by='number')
                    except Exception as e:
                        print(f"⚠️ Failed to load NMNIST: {e}. Using Mock.")
                        dataset = MockDVSGenerator(root=root_dir, train=(
                            split == 'train'), frames_number=time_steps)
            else:
                raise ValueError(
                    f"Unknown dataset: {dataset_name}. Currently only supports 'n-mnist' or Mock.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

# ユーティリティ関数: 可視化用


def visualize_dvs_sample(spike_tensor: torch.Tensor, save_path: str = "dvs_sample.png"):
    """
    DVSスパイクデータを可視化して保存する (デバッグ用)。
    spike_tensor: (Time, C, H, W)
    """
    try:
        import matplotlib.pyplot as plt

        T, C, H, W = spike_tensor.shape
        # 時間方向に積分してヒートマップ化
        img = spike_tensor.sum(dim=0).cpu().numpy()  # (C, H, W)

        # チャンネル合成 (Green=On, Red=Off)
        composite = np.zeros((H, W, 3))
        if C == 2:
            composite[:, :, 1] = img[0]  # Green (On)
            composite[:, :, 0] = img[1]  # Red (Off)
        else:
            composite[:, :, 1] = img[0]

        # 正規化 (可視化のために明るさを調整)
        composite = np.clip(composite / (composite.max() + 1e-8), 0, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(composite)
        plt.title(f"Integrated DVS Spikes (T={T})")
        plt.axis('off')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"DVS sample saved to {save_path}")
    except ImportError:
        pass
