# ファイルパス: snn_research/core/adaptive_attention_selector.py
# (新規作成)
# Title: Adaptive Attention Module
# Description: Improvement-Plan.md (改善案2, Part 2, Phase 2) に基づき、
#              学習中に標準的なSelf-AttentionとSpike-Driven Self-Attention (SDSA) を
#              動的に切り替える（または重み付けする）モジュールを実装します。
#              学習可能なパラメータを用いて、タスクや学習状況に応じて最適な
#              アテンションメカニズムを自動選択することを目指します。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

# 必要なアテンションモジュールとニューロンをインポート
# 標準Attention (PyTorch組み込み)
from torch.nn import MultiheadAttention as StandardAttention
# SDSA (スタブ実装)
from .attention import SpikeDrivenSelfAttention as SDSA
# ニューロン (FFN用) - FFN部分はここでは実装しない
# SNN用LayerNorm - Attentionの後処理として必要になる可能性がある
from .base import SNNLayerNorm # 必要に応じて使用

class AdaptiveAttentionModule(nn.Module):
    """
    標準的なSelf-AttentionとSpike-Driven Self-Attention (SDSA) を
    学習可能なパラメータに基づいて動的に混合または選択するモジュール。

    Improvement-Plan.md (Phase 2: 適応的SDSA) に基づく実装。
    """
    def __init__(self, d_model: int, nhead: int, time_steps: int, neuron_config: dict, dropout: float = 0.1):
        """
        Args:
            d_model (int): モデルの次元数。
            nhead (int): アテンションヘッド数。
            time_steps (int): SDSAが必要とするタイムステップ数。
            neuron_config (dict): SDSA内で使用するニューロンの設定。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.time_steps = time_steps

        # 1. 標準Attentionモジュール
        #    Note: 標準Attentionは内部でdropoutを持つ
        self.standard_attn = StandardAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 2. SDSAモジュール (スタブ実装を使用)
        self.sdsa_attn = SDSA(d_model, nhead, time_steps, neuron_config)

        # 3. 選択/混合パラメータ (学習可能)
        #    Sigmoidを通して 0~1 の重みαを生成する元のパラメータ。
        #    初期値は0.0 (α=0.5) で、どちらのアテンションにも均等に重み付け。
        self.attention_selector_logit = nn.Parameter(torch.tensor(0.0))

        # 4. Attention後の出力用Linear層 (SDSAには内部にあるが、標準にはないため追加)
        self.standard_out_proj = nn.Linear(d_model, d_model) # standard_attnの出力用
        # SDSAの出力層はsdsa_attn内部にある (to_out) と仮定

        # 5. ドロップアウト (最終出力に適用)
        self.dropout = nn.Dropout(dropout)

        print("✅ AdaptiveAttentionModule initialized.")
        print("   ⚠️ SDSA path uses a STUB implementation.")

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス。学習時と推論時で挙動が変わる。
        標準Attentionに合わせて query, key, value を受け取る。

        Args:
            query (torch.Tensor): クエリ (Batch, SeqLen_Q, Dim)。
            key (torch.Tensor): キー (Batch, SeqLen_K, Dim)。
            value (torch.Tensor): バリュー (Batch, SeqLen_K, Dim)。
            key_padding_mask (Optional[torch.Tensor]): 標準Attention用のキーパディングマスク (Batch, SeqLen_K)。
            attn_mask (Optional[torch.Tensor]): 標準Attention用のアテンションマスク。

        Returns:
            torch.Tensor: アテンション適用後の出力テンソル (Batch, SeqLen_Q, Dim)。
        """
        # --- 選択重み α の計算 ---
        # Sigmoidを通して 0 ~ 1 の値にする
        alpha = torch.sigmoid(self.attention_selector_logit)

        # --- 推論時の挙動 ---
        if not self.training:
            # 学習された選択に基づき、どちらか一方のアテンションのみを実行
            if alpha.item() > 0.5:
                # SDSAを選択
                # Note: SDSAスタブは現在 x のみを受け取る -> query を使用
                sdsa_output = self.sdsa_attn(query) # Assume SDSA handles Q,K,V internally based on query
                # SDSAは内部に出力射影層(to_out)を持つと仮定
                output = sdsa_output
            else:
                # 標準Attentionを選択
                # 標準Attentionは (attn_output, attn_weights) のタプルを返す
                standard_output_tuple = self.standard_attn(query, key, value,
                                                           key_padding_mask=key_padding_mask,
                                                           attn_mask=attn_mask,
                                                           need_weights=False) # 推論時は重み不要
                standard_output = standard_output_tuple[0]
                # 標準Attentionの後には出力射影層が必要
                output = self.standard_out_proj(standard_output)

            return self.dropout(output) # 最後にドロップアウト

        # --- 学習時の挙動 ---
        # 両方のアテンションパスを計算し、αで重み付け和を取る

        # 1. 標準Attentionパス
        standard_output_tuple = self.standard_attn(query, key, value,
                                                   key_padding_mask=key_padding_mask,
                                                   attn_mask=attn_mask,
                                                   need_weights=False) # 学習時も通常は重み不要
        standard_output = standard_output_tuple[0]
        standard_output_proj = self.standard_out_proj(standard_output) # 出力射影

        # 2. SDSAパス
        # Note: SDSAスタブは現在 x のみを受け取る -> query を使用
        sdsa_output = self.sdsa_attn(query) # Assume SDSA handles Q,K,V internally based on query
        # SDSAは内部に出力射影(to_out)を持つと仮定

        # 3. 重み付け和
        # alpha がSDSAの重み、(1 - alpha) が標準Attentionの重み
        # ⚠️ 注意: 出力される値のスケールが異なる可能性があるため、単純な重み付け和が
        #          最適とは限らない。LayerNorm等での調整が必要になる場合がある。
        output = alpha * sdsa_output + (1 - alpha) * standard_output_proj

        # 最後にドロップアウトを適用
        output = self.dropout(output)

        return output

    def get_current_preference(self) -> Dict[str, Any]:
        """
        現在どちらのAttentionが優先されているかを返すユーティリティ。
        """
        alpha = torch.sigmoid(self.attention_selector_logit).item()
        preference = "SDSA" if alpha > 0.5 else "Standard"
        # 閾値付近は混合と判断
        if abs(alpha - 0.5) < 0.1: # 例: 0.4 ~ 0.6 の範囲
             preference = "Mixed"

        return {
            "sdsa_weight": alpha,
            "standard_weight": 1 - alpha,
            "preference": preference,
            "selector_logit": self.attention_selector_logit.item() # 内部パラメータも返す
        }

    def reset(self):
        """SDSAモジュール内のニューロン状態をリセット"""
        if hasattr(self.sdsa_attn, 'reset'):
            self.sdsa_attn.reset()