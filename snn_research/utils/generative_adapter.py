# ファイルパス: snn_research/utils/generative_adapter.py
# 日本語タイトル: SNN Generative Adapter
# 目的: SNNモデルにTransformersライクなgenerateメソッドを提供し、ReasoningEngineと接続する。

import torch
import torch.nn as nn
from typing import Optional
from spikingjelly.activation_based import functional as SJ_F


class SNNGenerativeAdapter(nn.Module):
    """
    SNNモデルをラップし、自己回帰的な生成機能(generate)を提供するアダプター。
    ReasoningEngineが要求するインターフェース(generate)を実装する。
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__()
        self.model = model
        self.device = device

        # モデルがembedding属性を持っているか確認、なければ層から探すなどのロジックが必要だが
        # ここではBitSpikeMambaを前提とする
        if hasattr(model, 'embedding'):
            self.embedding = model.embedding

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        自己回帰的な生成ループ。
        SNNの状態リセットと時間発展を各ステップで管理する。
        """
        self.model.eval()
        current_ids = input_ids.clone()

        # 生成ループ
        for _ in range(max_length - input_ids.shape[1]):
            # SNNはステートフルな挙動をするため、推論ごとにリセットするか、
            # 継続的な入力を与える必要がある。ここではシンプルに毎回リセットして
            # シーケンス全体を流す方式（低速だが確実）を採用する。
            # ※最適化するにはKVキャッシュのSNN版（膜電位キャッシュ）が必要
            SJ_F.reset_net(self.model)

            with torch.no_grad():
                # (Batch, Seq_Len, Vocab)
                logits, _, _ = self.model(current_ids)

                # 最後のトークンのロジットを取得
                next_token_logits = logits[:, -1, :]

                # Temperature適用
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # サンプリングまたはGreedy
                if do_sample:
                    # Top-k / Top-p filtering (簡易実装)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(
                        next_token_logits, dim=-1, keepdim=True)

            # 連結
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # 終了判定
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return current_ids
