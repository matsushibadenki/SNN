# ファイルパス: snn_research/core/models/semm_model.py
# Title: Spiking Experts Mixture Mechanism (SEMM) with T=1 SFN Router
# Description:
#   Phase 3 実装の強化版。
#   ルーターに Scale-and-Fire Neuron (SFN) を採用し、T=1 での高速かつスパースな
#   エキスパート選択を実現。SFormerと組み合わせることで全体として T=1 動作を保証する。
#   負荷分散のための補助損失 (Load Balancing Loss) 用のロジット出力も完備。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple

from snn_research.core.base import BaseModel
from snn_research.core.neurons import ScaleAndFireNeuron
from snn_research.models.transformer.sformer import SFormerBlock
import logging

logger = logging.getLogger(__name__)

class SpikingRouter(nn.Module):
    """
    SFN (Scale-and-Fire Neuron) を使用したスパイクベースルーター。
    T=1 で動作し、入力に対してスパースなルーティング決定を行う。
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, sf_threshold: float = 4.0):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        
        # SFNを使用 (レベル数は少なくて良い、ここでは2値的な振る舞いを期待して小さめに設定、あるいは多値で重み付け)
        # レベル数=2なら 0, 1 のバイナリスパイクに近い
        self.sfn = ScaleAndFireNeuron(
            features=num_experts, 
            num_levels=4, 
            base_threshold=sf_threshold
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Seq, D) -> logits: (B, Seq, NumExperts)
        logits = self.gate(x)
        
        # Top-K マスキング (ソフトマックスの前段階として機能、ここではハードに選択)
        # 学習時は logits に勾配を流すために、gather/scatter トリックを使うか、
        # 選択された部分のみを通す。
        
        # Top-K のインデックスを取得
        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # マスク作成 (-inf で埋めることで選択されなかったエキスパートを無効化)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        
        # SFNに通す前に、選択されたロジットを正規化（またはそのまま通す）
        # ここではSoftmaxを通さず、生のロジット（ただしTopK以外はマスク）をSFNに通すことで
        # 「活動電位の大きさ＝エキスパートへの重み」とする
        
        # SFNは負の入力には発火しないため、TopK値をReLU的に扱う
        # マスクされた値(-inf)は発火しない
        routing_input = F.relu(mask) # 負の値は0に
        
        # SFNで量子化されたルーティング重みを取得
        # routing_weights: (B, Seq, NumExperts)
        routing_weights, _ = self.sfn(routing_input)
        
        # ルーターの生ロジット（損失計算用）も返す
        return routing_weights, logits

class SEMMBlock(nn.Module):
    """
    1つのSEMM層。SFormerBlockをエキスパートとして持つ。
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, dim_feedforward: int):
        super().__init__()
        self.router = SpikingRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            # エキスパートとして SFormerBlock を使用
            SFormerBlock(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=dim_feedforward
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. ルーティング
        # routing_weights: SFN出力 (量子化された重み)
        routing_weights, router_logits = self.router(x)
        
        # 加重和のためのバッファ
        final_output = torch.zeros_like(x)
        
        # 2. エキスパート実行 (スパイクゲーティング)
        # routing_weights が 0 のエキスパートは計算をスキップ（条件付き実行）できるが、
        # PyTorchのバッチ処理では全計算してからマスクする方が速い場合が多い。
        # ここでは論理的なスパース性を示す。
        
        # Top-K のインデックスのみ計算するのが真のMoEだが、実装の複雑さを避けるため
        # ここでは全エキスパートを実行し、SFN出力(0を含む)でマスクする
        for i, expert in enumerate(self.experts):
            # routing_weights[:, :, i]: (B, Seq)
            gate = routing_weights[:, :, i].unsqueeze(-1) # (B, Seq, 1)
            
            # ゲートが開いている（発火している）場合のみ寄与
            # (最適化: gateが全0ならexpert(x)をスキップするロジックを入れるとさらに高速)
            if gate.sum() > 0:
                expert_out = expert(x)
                final_output += expert_out * gate
            
        return final_output, router_logits

class SEMMModel(BaseModel):
    """
    Spiking Experts Mixture Mechanism (SEMM) Model.
    SFormerをベースにしたMoEモデル。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = 1 # T=1 保証
        self.num_experts = num_experts
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # エキスパートのFFN次元
        dim_feedforward = d_model * 4
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SEMMBlock(
                d_model=d_model, 
                num_experts=num_experts, 
                top_k=top_k,
                dim_feedforward=dim_feedforward
            ))
            
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()
        logger.info(f"✅ SEMM Model (T=1, SFN Router) initialized (Experts={num_experts}, TopK={top_k}).")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.embedding(input_ids)
        
        # 全レイヤーのルーターロジットを収集 (Load Balancing Loss計算用)
        all_router_logits = []
        
        for layer in self.layers:
            x, router_logits = layer(x)
            all_router_logits.append(router_logits)
            
        logits = self.output_projection(x)
        
        # ロジットをスタック: (Batch, Seq, Layers, NumExperts)
        aux_loss_logits = torch.stack(all_router_logits, dim=2)
        
        avg_spikes = torch.tensor(0.0, device=x.device)
        if return_spikes:
            # SFNなどの総スパイク数を取得
            avg_spikes = torch.tensor(self.get_total_spikes() / input_ids.numel(), device=x.device)
            
        # 戻り値に aux_logits を追加 (TrainerでLoss計算に使用)
        return logits, avg_spikes, torch.tensor(0.0, device=x.device), aux_loss_logits