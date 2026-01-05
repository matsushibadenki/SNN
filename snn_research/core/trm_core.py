# ファイルパス: snn_research/core/trm_core.py
# (改善案1: ゲート付き残差TRMを適用)
# Title: Tiny Recursive Model (TRM) コア実装 - ゲート付き残差版
# Description:
# - 単一の小型SNNブロックにGRU/LSTMライクなゲート機構と残差接続を導入し、
#   勾配フローの改善と深い思考ステップでの安定性を向上させる。
# - SNNのタイムステップを推論の再帰ステップとして活用する基本構造は維持。
#
# mypy --strict 準拠。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Type, cast

from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from .base import BaseModel, SNNLayerNorm
from spikingjelly.activation_based import functional # type: ignore[import-untyped]

class GatedTRMBlock(nn.Module):
    """
    ゲート機構と残差接続を持つTRMブロック
    """
    neuron: nn.Module

    def __init__(self, input_dim: int, hidden_dim: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()

        # 更新ゲート (どれだけ新しい情報を取り込むか)
        # combined_input (x + y_t + z_t) と h_recurrent (z_t) を入力とする
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid() # ゲートは通常Sigmoid
        )

        # リセットゲート (過去の状態(z_t)をどれだけ忘れるか)
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 候補状態の計算
        self.input_mix = nn.Linear(input_dim, hidden_dim)
        self.recurrent_fc = nn.Linear(hidden_dim, hidden_dim)

        neuron_params_copy = neuron_params.copy()
        neuron_params_copy.pop('type', None)
        # 明示的な型アノテーション
        self.neuron = neuron_class(features=hidden_dim, **neuron_params_copy)
        self.norm = SNNLayerNorm(hidden_dim)

    def forward(self, combined_input: torch.Tensor, h_recurrent: torch.Tensor) -> torch.Tensor:
        """
        潜在状態（h_recurrent = z_t）を1ステップ更新して z_{t+1} を計算する。
        """
        # ゲートの計算
        gate_input = torch.cat([combined_input, h_recurrent], dim=-1)
        update_g = self.update_gate(gate_input)  # どれだけ新しい候補状態を取り込むか
        reset_g = self.reset_gate(gate_input)     # 過去の状態 h_recurrent をどれだけリセットするか

        # リセットされた過去状態を使って候補状態を計算
        h_reset = reset_g * h_recurrent
        candidate_input = self.input_mix(combined_input) + self.recurrent_fc(h_reset)

        # SNN処理で候補状態を生成
        h_candidate_spikes, _ = self.neuron(candidate_input) # type: ignore[operator]
        h_candidate = self.norm(h_candidate_spikes)

        # ゲート制御された更新 (GRUと同様の更新式)
        # update_g が大きいほど新しい候補 h_candidate を取り込み、
        # (1 - update_g) が大きいほど過去の状態 h_recurrent を保持する（残差接続の役割）
        h_new = update_g * h_candidate + (1 - update_g) * h_recurrent

        return h_new

class TinyRecursiveModel(BaseModel):
    """
    Tiny Recursive Model (TRM): 単一のゲート付きブロックを時間軸で再帰させるモデル。
    """
    lif_out: nn.Module

    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, layer_dims: List[int], time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        # TRMではnum_layers, layer_dimsは使用しないが互換性のためコンストラクタに残す
        self.time_steps = time_steps # これが再帰の深さ N_sup (思考ステップ数) に相当
        self.d_model = d_model       # 質問埋め込み後の次元 (x)
        self.d_state = d_state       # 潜在状態 z と回答 y の次元

        neuron_type = neuron_config.get("type", "lif")
        neuron_class: Type[nn.Module] = AdaptiveLIFNeuron if neuron_type == 'lif' else IzhikevichNeuron

        # 1. 入力層
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. メインの再帰ブロック（ゲート付きに変更）
        # 入力次元 = d_model (x) + d_state (y_i) + d_state (z_i)
        input_dim_for_block = d_model + d_state + d_state
        self.recurrent_block = GatedTRMBlock(input_dim_for_block, d_state, neuron_class, neuron_config)

        # 3. 初期回答と潜在状態の初期化層
        self.init_state = nn.Linear(d_model, d_state * 2)

        # 4. 出力層（最終的なロジットを生成する）
        # 入力次元: d_state (更新された潜在状態 z)
        self.output_projection = nn.Linear(d_state, vocab_size)

        # 出力用LIF
        lif_out_config = neuron_config.copy()
        lif_out_config.pop('type', None)
        self.lif_out = neuron_class(features=vocab_size, **lif_out_config)

        self._init_weights()


    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device
        functional.reset_net(self)

        if L != 1:
            # TRMは単一の質問ベクトルを入力として受け取ることを想定
            # シーケンスの場合は最初のトークンのみ、あるいはPoolingが必要だが、ここでは簡易的に先頭を使用
            input_ids = input_ids[:, 0].unsqueeze(1)
            L = 1

        # 1. 初期埋め込み (x)
        x_emb = self.embedding(input_ids).squeeze(1) # B x D_model

        # 2. 初期状態の生成 (y_0, z_0)
        initial_state_and_answer = self.init_state(x_emb)
        # z_0 (潜在状態)とy_0 (初期回答/次の入力)を分離
        latent_z, answer_y = initial_state_and_answer.split(self.d_state, dim=-1) # 初期はアナログ値から開始

        # 再帰ブロック内のニューロンをstatefulモードに設定し状態をリセット
        trm_block_neuron = cast(Any, self.recurrent_block.neuron)
        if hasattr(trm_block_neuron, 'set_stateful'):
            trm_block_neuron.set_stateful(True)
            trm_block_neuron.reset()

        # 出力ニューロンもstatefulに設定し状態をリセット
        lif_out_neuron = cast(Any, self.lif_out)
        if hasattr(lif_out_neuron, 'set_stateful'):
            lif_out_neuron.set_stateful(True)
            lif_out_neuron.reset()

        # 3. 再帰的推論ループ
        for t in range(self.time_steps):
            # a) 再帰ブロックへの入力準備: x + y_t + z_t
            combined_input = torch.cat([x_emb, answer_y, latent_z], dim=-1) # B x (D_model + 2*D_state)

            # b) 潜在状態 z の更新 (ゲート付きブロックを使用)
            latent_z = self.recurrent_block(combined_input, latent_z) # B x D_state

            # c) 回答 y の更新 (更新された潜在状態 z_{t+1} から新しい回答 y_{t+1} を生成)
            logits_t_raw = self.output_projection(latent_z)

            # SNN化: ロジットをスパイクパターン（次のステップの回答 y）に変換
            answer_y_spikes, _ = self.lif_out(logits_t_raw) # type: ignore[operator]
            answer_y = answer_y_spikes # 次の再帰ステップへの入力となる

        # 再帰ループ終了後ニューロンの状態をリセット
        if hasattr(trm_block_neuron, 'set_stateful'):
            trm_block_neuron.set_stateful(False)
            trm_block_neuron.reset()
        if hasattr(lif_out_neuron, 'set_stateful'):
            lif_out_neuron.set_stateful(False)
            lif_out_neuron.reset()

        # 4. 最終出力 (最終ステップの潜在状態 z_N からロジットを生成)
        final_logits = self.output_projection(latent_z)

        # スパイク統計の計算
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (L * self.time_steps * B) if return_spikes else 0.0

        # 1. avg_spikes を作成 (1要素のスカラーテンソル)
        avg_spikes = torch.tensor(float(avg_spikes_val), dtype=torch.float32, device=device)

        # 2. mem を作成 (今回は未使用なので0)
        mem = torch.zeros(1, dtype=torch.float32, device=device).squeeze()
        
        # SNN Coreのインターフェースに合わせて (B, L=1, V) で返す
        return final_logits.unsqueeze(1), avg_spikes, mem