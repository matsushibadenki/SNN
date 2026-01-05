# ファイルパス: scripts/runners/run_scal_ff_hybrid.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from snn_research.training.trainers.forward_forward import ForwardForwardTrainer, ForwardForwardLayer
from snn_research.core.ensemble_scal import EnsembleSCAL

class SCALFeatureExtractor(nn.Module):
    """
    SCALを純粋な特徴抽出器としてラップするクラス
    """
    def __init__(self, in_features, out_features, n_models=3, device='cpu'):
        super().__init__()
        # SCAL本体
        self.scal = EnsembleSCAL(in_features=in_features, out_features=out_features, n_models=n_models).to(device)
        self.device = device
        
    def forward(self, x):
        # 画像 [B, 1, 28, 28] -> Flatten [B, 784]
        x_flat = x.view(x.size(0), -1).to(self.device)
        with torch.no_grad():
            out = self.scal(x_flat)
        return out['output'] # [B, out_features]

    def fit(self, data_loader, epochs=1):
        """
        SCALの教師なし学習を実行するメソッド
        """
        self.scal.train() # 学習モードかもしくは内部状態更新許可
        print(f"Pre-training SCAL for {epochs} epochs...")
        
        # SCALの実装によっては optimizer が不要で、forward時に統計更新するタイプがある
        # ここではデータを通すことで重心を更新させる
        for epoch in range(epochs):
            for data, _ in data_loader:
                data = data.view(data.size(0), -1).to(self.device)
                # forwardを通すことで内部の重心(centroids)等を更新させる想定
                # (EnsembleSCALの実装に依存するが、通常はこれで統計が蓄積される)
                self.scal(data)
        print("SCAL Pre-training Complete.")

class HybridFFTrainer(ForwardForwardTrainer):
    """
    SCALで特徴抽出した *後に* ラベルを埋め込むためのカスタムトレーナー
    """
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        
        # SCALはFFの学習ループ内では固定（事前学習済みとする）
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def generate_negative_data(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. Raw Image -> SCAL -> Features
        2. Features + Label -> Pos/Neg Embedding
        """
        # 1. 特徴抽出 & デバイス転送
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = features.to(self.device)
            
        # 2. ラベル埋め込み
        x_pos = self.overlay_y_on_x(features, y)
        
        y_fake = (y + torch.randint(1, self.num_classes, (len(y),)).to(self.device)) % self.num_classes
        x_neg = self.overlay_y_on_x(features, y_fake)
        
        return x_pos, x_neg

    def predict(self, data_loader: DataLoader, noise_level: float = 0.0) -> float:
        """
        推論メソッド。ノイズ注入機能付き。
        """
        self.model.eval()
        
        # モデル全体のデバイス確認
        self.model.to(self.device)
        for layer in self.execution_pipeline:
            layer.to(self.device)
            if hasattr(layer, 'block'):
                layer.block.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # ノイズ注入 (Robustness Test用)
                if noise_level > 0:
                    noise = torch.randn_like(data) * noise_level
                    data = data + noise
                    # クランプしないことでSCALの堅牢性を試す（あるいは0-1に収める）
                    # data = torch.clamp(data, -1.0, 1.0) 
                
                batch_size = data.size(0)
                
                # 1. 特徴抽出
                features = self.feature_extractor(data)
                features = features.to(self.device)
                
                class_goodness = torch.zeros(batch_size, self.num_classes, device=self.device)
                
                for label_idx in range(self.num_classes):
                    temp_labels = torch.full((batch_size,), label_idx, dtype=torch.long, device=self.device)
                    
                    # 2. ラベル埋め込み
                    h = self.overlay_y_on_x(features, temp_labels)
                    
                    # 3. FFレイヤーでの推論
                    for layer in self.execution_pipeline:
                        if isinstance(layer, ForwardForwardLayer):
                            h = layer.forward(h)
                            dims = list(range(1, h.dim()))
                            g = h.pow(2).mean(dim=dims)
                            class_goodness[:, label_idx] += g
                        else:
                            h = layer(h)
                
                predicted = class_goodness.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total

def run_scal_ff_hybrid_experiment():
    print("=== SCAL-FF Hybrid Architecture Experiment (Full Cycle) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. データセット
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../../../../data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 事前学習用ローダー（少なめでOK）
    pretrain_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 2. SCAL (Visual Cortex)
    print("\n--- Phase 1: Initializing & Pre-training SCAL (Unsupervised) ---")
    scal_out_dim = 1000 
    scal_extractor = SCALFeatureExtractor(in_features=784, out_features=scal_out_dim, n_models=3, device=device)
    
    # SCALの事前学習を実行 (重心をデータに適合させる)
    scal_extractor.fit(pretrain_loader, epochs=1)

    # 3. FF Layers (Association Cortex)
    print("\n--- Phase 2: Training Forward-Forward Layers (Supervised) ---")
    
    ff_model = nn.Sequential(
        nn.Linear(scal_out_dim, 2000), 
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU()
    ).to(device)

    # 4. Trainer
    config = {
        "learning_rate": 0.03,
        "ff_threshold": 2.0,
        "num_epochs": 5 # 時間短縮のため5エポック
    }
    
    trainer = HybridFFTrainer(
        feature_extractor=scal_extractor,
        model=ff_model,
        device=device,
        config=config,
        num_classes=10
    )

    # 5. Training
    for epoch in range(1, config['num_epochs'] + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Validating Epoch {epoch}...")
        acc = trainer.predict(test_loader)
        print(f"Epoch {epoch} - Loss: {metrics['train_loss']:.4f} - Accuracy: {acc:.2f}%")
        
    # 保存
    save_dir = os.path.join(os.path.dirname(__file__), '../../../../results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, "scal_ff_hybrid_trained.pth"))

    # 6. Robustness Test
    print("\n--- Phase 3: Noise Robustness Evaluation ---")
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    print(f"{'Noise Level':<15} | {'Accuracy':<10}")
    print("-" * 30)
    
    results = {}
    for noise in noise_levels:
        acc = trainer.predict(test_loader, noise_level=noise)
        print(f"{noise:<15.1f} | {acc:.2f}%")
        results[noise] = acc

    print("\nExperiment Finished.")

if __name__ == "__main__":
    run_scal_ff_hybrid_experiment()
