# ファイルパス: snn_research/training/trainers/forward_forward.py
# 修正: GoodnessをMeanに変更、Weight Decay削除、学習安定化
# mypy修正: 型アノテーションの追加とintキャスト

from snn_research.training.base_trainer import AbstractTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

# プロジェクトルートへのパス設定
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../..')))


logger = logging.getLogger(__name__)


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone() / (10.0 * torch.abs(input) + 1.0)**2
        return grad


class SpikingLayer(nn.Module):
    def __init__(self, layer: nn.Module, time_steps: int = 20, tau: float = 0.5, v_threshold: float = 1.0):
        super().__init__()
        self.layer = layer
        self.time_steps = time_steps
        self.tau = tau
        self.v_threshold = v_threshold
        self.spike_fn = SurrogateHeaviside.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dims = x.dim()
        if not ((input_dims == 3) or (input_dims == 5)):
            out_static = self.layer(x)
            current = out_static.unsqueeze(1).repeat(
                1, self.time_steps, *([1]*(out_static.dim()-1)))
        else:
            B, T = x.shape[0], x.shape[1]
            current = self.layer(x.flatten(0, 1)).view(
                [B, T] + list(self.layer(x.flatten(0, 1)).shape[1:]))
        v_mem = torch.zeros_like(current[:, 0])
        spikes = []
        for t in range(self.time_steps):
            v_mem = v_mem * self.tau + current[:, t]
            spike = self.spike_fn(v_mem - self.v_threshold)
            v_mem = v_mem * (1.0 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class SpikingForwardForwardLayer(nn.Module):
    def __init__(self, forward_block, threshold=2.0, learning_rate=0.001, time_steps=20):
        super().__init__()
        self.block = SpikingLayer(forward_block, time_steps)
        self.threshold = threshold
        # Weight decayを削除してGoodnessの成長を妨げないようにする
        self.optimizer = torch.optim.Adam(
            self.block.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.block(x)

    def train_step(self, x_pos, x_neg):
        self.optimizer.zero_grad()
        sp_pos, sp_neg = self(x_pos), self(x_neg)

        # Goodness calculation: Mean over Time, Mean over Features (Stability)
        g_pos = sp_pos.mean(dim=1).pow(2).mean(dim=1)
        g_neg = sp_neg.mean(dim=1).pow(2).mean(dim=1)

        # Loss: minimize neg goodness, maximize pos goodness
        loss = F.softplus(-(g_pos - self.threshold)).mean() + \
            F.softplus(g_neg - self.threshold).mean()

        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()


class ForwardForwardLayer(nn.Module):
    def __init__(self, forward_block, threshold=2.0, learning_rate=0.001):
        super().__init__()
        self.block = forward_block
        self.threshold = threshold
        # Weight decayを削除
        self.optimizer = torch.optim.Adam(
            self.block.parameters(), lr=learning_rate)
        self.is_conv = any(isinstance(m, nn.Conv2d)
                           for m in self.block.modules())

    def forward(self, x):
        if self.is_conv:
            norm = x.norm(2, dim=[1, 2, 3] if x.dim() ==
                          4 else [1], keepdim=True) + 1e-8
            return self.block(x / norm)
        x_flat = x.reshape(x.size(0), -1) if x.dim() > 2 else x
        # Normalize input length to 1
        return self.block(x_flat / (x_flat.norm(2, 1, keepdim=True) + 1e-8))

    def train_step(self, x_pos, x_neg):
        self.optimizer.zero_grad()
        out_pos, out_neg = self(x_pos), self(x_neg)

        # Goodness calculation: Mean of squared activities
        dims = list(range(1, out_pos.dim()))
        g_pos = out_pos.pow(2).mean(dim=dims)
        g_neg = out_neg.pow(2).mean(dim=dims)

        # Loss using Softplus for stability
        loss = F.softplus(-(g_pos - self.threshold)).mean() + \
            F.softplus(g_neg - self.threshold).mean()

        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()


class ForwardForwardTrainer(AbstractTrainer):
    def __init__(self, model: nn.Sequential, device: str = "cpu", config: Optional[Dict[str, Any]] = None, num_classes: int = 10, save_dir: str = "results"):
        super().__init__(model, None, None, device, save_dir)
        self.num_classes = num_classes
        from omegaconf import OmegaConf
        self.config = OmegaConf.create(
            config) if config else OmegaConf.create()
        self.ff_layers: List[Union[SpikingForwardForwardLayer,
                                   ForwardForwardLayer]] = []
        self.execution_pipeline: List[nn.Module] = []
        self.use_snn = self.config.get('use_snn', False)

        # Pipeline construction logic
        for layer in model.children():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # Adjust parameters from config
                lr = self.config.get("learning_rate", 0.001)
                threshold = self.config.get("ff_threshold", 2.0)

                # mypy修正: 型を明示
                ff: Union[SpikingForwardForwardLayer, ForwardForwardLayer]
                if self.use_snn:
                    ff = SpikingForwardForwardLayer(nn.Sequential(
                        layer), threshold=threshold, learning_rate=lr)
                else:
                    ff = ForwardForwardLayer(nn.Sequential(
                        layer), threshold=threshold, learning_rate=lr)

                self.ff_layers.append(ff)
                self.execution_pipeline.append(ff)
            else:
                self.execution_pipeline.append(layer.to(device))

    def overlay_y_on_x(self, x, y):
        x_mod = x.clone()
        y_oh = F.one_hot(y, self.num_classes).float().to(self.device) * 1.5
        if x_mod.dim() == 4:
            x_mod[:, 0, 0, :min(self.num_classes, x_mod.shape[3])] = y_oh[:, :min(
                self.num_classes, x_mod.shape[3])]
        elif x_mod.dim() == 2:
            x_mod[:, :self.num_classes] = y_oh
        return x_mod

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None) -> Dict[str, float]:
        if epoch:
            self.current_epoch = epoch
        total_loss = 0.0

        self.model.train()

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data, target = data.to(self.device), target.to(self.device)

            x_pos = self.overlay_y_on_x(data, target)
            x_neg = self.overlay_y_on_x(data, (target + 1) % self.num_classes)

            batch_loss = 0.0

            # Layer-wise training
            for layer in self.execution_pipeline:
                if isinstance(layer, (ForwardForwardLayer, SpikingForwardForwardLayer)):
                    layer_loss, _, _ = layer.train_step(x_pos, x_neg)
                    batch_loss += layer_loss

                    with torch.no_grad():
                        x_pos = layer(x_pos).detach()
                        x_neg = layer(x_neg).detach()
                else:
                    x_pos = layer(x_pos)
                    x_neg = layer(x_neg)

            total_loss += batch_loss

        return {"train_loss": total_loss / len(train_loader)}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        acc = self.predict(val_loader)
        return {"val_accuracy": acc}

    def predict(self, test_loader: DataLoader) -> float:
        """
        Forward-Forwardアルゴリズムを用いた推論。
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Predicting", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                goodness_scores = torch.zeros(
                    batch_size, self.num_classes, device=self.device)

                for label in range(self.num_classes):
                    y_candidate = torch.full(
                        (batch_size,), label, dtype=torch.long, device=self.device)
                    x = self.overlay_y_on_x(data, y_candidate)

                    layer_goodness = torch.zeros(
                        batch_size, device=self.device)
                    current_x = x
                    for layer in self.execution_pipeline:
                        if isinstance(layer, (ForwardForwardLayer, SpikingForwardForwardLayer)):
                            out = layer(current_x)

                            # Goodness計算: Meanを使用
                            if isinstance(layer, SpikingForwardForwardLayer):
                                g = out.mean(dim=1).pow(2).mean(dim=1)
                            else:
                                dims = list(range(1, out.dim()))
                                g = out.pow(2).mean(dim=dims)

                            layer_goodness += g
                            current_x = out.detach()
                        else:
                            current_x = layer(current_x)

                    goodness_scores[:, label] = layer_goodness

                preds = goodness_scores.argmax(dim=1)
                # mypy修正: intにキャスト
                correct += int(preds.eq(target).sum().item())
                total += batch_size

        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, filename: str = "checkpoint.pth", metric: Optional[float] = None) -> None:
        path = self.save_dir / filename
        state = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'ff_optimizers': [layer.optimizer.state_dict() for layer in self.ff_layers],
            'best_metric': self.best_metric if metric is None else metric
        }
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}")
            return

        state = torch.load(path, map_location=self.device)
        self.current_epoch = state.get('epoch', 0)
        self.model.load_state_dict(state['model_state'])
        if 'ff_optimizers' in state:
            for layer, s in zip(self.ff_layers, state['ff_optimizers']):
                layer.optimizer.load_state_dict(s)
        logger.info(f"Loaded checkpoint from {path}")
