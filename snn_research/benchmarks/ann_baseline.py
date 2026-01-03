# ファイルパス: snn_research/benchmark/ann_baseline.py
#
# SNNモデルとの性能比較を行うためのANNベースラインモデル
#
# (省略)
# - forwardメソッドの戻り値を3つのタプルに変更し、SNNモデルのインターフェースと統一した。
#
# 改善(snn_4_ann_parity_plan):
# - CIFAR-10などの画像分類タスクで、SpikingCNNとの直接比較および変換を行うための
#   シンプルなCNNベースラインモデル(SimpleCNN)を追加。

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple

class ANNBaselineModel(nn.Module):
    """
    シンプルなTransformerベースのテキスト分類モデル。
    BreakthroughSNNとの比較用。
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int, nlayers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformerエンコーダ層を定義
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # 分類ヘッド
        self.classifier = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            input_ids (torch.Tensor): 入力シーケンス (batch_size, seq_len)
            attention_mask (torch.Tensor): パディングマスク (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            分類ロジットと、SNNとの互換性のためのNone値2つ。
        """
        # 互換性のための引数名マッピング
        src = input_ids
        src_key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Transformerエンコーダに入力
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # パディングを考慮した平均プーリング
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(encoded)
            masked_encoded = encoded * mask.float()
            pooled = masked_encoded.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled)
        # SNN評価との互換性のため、3つのタプルで返す
        return logits, None, None

class SimpleCNN(nn.Module):
    """
    画像分類用のシンプルなCNN。SpikingCNNのANNベースラインとして機能する。
    Architecture: Conv -> ReLU -> AvgPool -> Conv -> ReLU -> AvgPool -> FC -> ReLU -> FC
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_images: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, None]:
        x = self.features(input_images)
        logits = self.classifier(x)
        return logits, None, None