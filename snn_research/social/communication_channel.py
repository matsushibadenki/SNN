# ファイルパス: snn_research/social/communication_channel.py
# 日本語タイトル: Noisy Communication Channel
# 目的: エージェント間の通信（発話・聴取）における物理的なノイズや劣化をシミュレートする。

import torch

class CommunicationChannel:
    """
    エージェントAとエージェントBをつなぐ通信チャネル。
    
    機能:
    1. 信号の伝送 (Transmission)
    2. ノイズの付加 (Noise Injection) - 誤字脱字や聞き間違いをシミュレート
    3. 信号のドロップアウト (Signal Loss)
    """
    def __init__(
        self, 
        noise_level: float = 0.0, 
        dropout_prob: float = 0.0,
        device: str = 'cpu'
    ):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.device = device

    def transmit_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        トークン列（言語）を伝送する。
        確率的にトークンが置き換わったり（聞き間違い）、欠落したりする。
        
        Args:
            token_ids: (B, SeqLen)
        Returns:
            noisy_tokens: (B, SeqLen)
        """
        if self.noise_level == 0 and self.dropout_prob == 0:
            return token_ids

        # クローンを作成
        noisy_ids = token_ids.clone()
        B, L = noisy_ids.shape
        
        # 1. Random Replacement (Noise)
        # noise_levelの確率で、ランダムなトークン(例: 0~1000)に置き換わる
        if self.noise_level > 0:
            mask = torch.rand(B, L, device=self.device) < self.noise_level
            random_tokens = torch.randint(0, 1000, (B, L), device=self.device) # 簡易語彙範囲
            noisy_ids[mask] = random_tokens[mask]

        # 2. Dropout (Signal Loss)
        # dropout_probの確率で、トークンが[PAD]または欠損扱いになる
        # ここでは簡易的に 0 (UNK/PAD) に置き換える
        if self.dropout_prob > 0:
            drop_mask = torch.rand(B, L, device=self.device) < self.dropout_prob
            noisy_ids[drop_mask] = 0
            
        return noisy_ids

    def transmit_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        アナログ/スパイク信号を伝送する（テレパシーや機械間通信用）。
        ガウシアンノイズを付加する。
        """
        if self.noise_level > 0:
            noise = torch.randn_like(spikes) * self.noise_level
            return spikes + noise
        return spikes