# ファイルパス: snn_research/models/transformer/logic_gated_attention.py
# タイトル: Logic-Gated Spiking Self-Attention
# 内容: SCAL (Statistical Centroid Alignment Learning) を応用した論理ゲート付きAttention機構
# 目的: 単なる類似度ではなく、「論理的整合性」に基づいて情報の流れを制御する「推論するAttention」を実現

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from spikingjelly.activation_based import layer

# 内部モジュールインポート
from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
from snn_research.core.layers.logic_gated_snn_v2_1 import SCALPerceptionLayer

class LogicGateController(nn.Module):
    """
    AttentionのQueryとKeyの関係性を監視し、
    「文脈的に不自然な接続」を抑制する論理ゲートコントローラ。
    Phase-Critical SCALを使用して、動的にゲートの開閉閾値を調整する。
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 論理判断を行うためのSCALコア
        # 入力: QueryとKeyの結合特徴量
        # 出力: ゲート開閉信号 (Sigmoid like)
        self.logic_core = SCALPerceptionLayer(
            in_features=dim, # ヘッドごとの次元ではなくモデル次元全体を見る（グローバルコンテキスト）
            out_features=num_heads, # ヘッドごとにゲートを持つ
            time_steps=1, # Attentionは瞬時的な判断とする（あるいは時系列展開も可）
            v_th_init=0.5,
            gain=10.0
        )
        
        self.proj_q = nn.Linear(dim, dim // 2)
        self.proj_k = nn.Linear(dim, dim // 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (Batch, Num_Heads, Seq_Len, Head_Dim) -> 平均化して使う
            k: (Batch, Num_Heads, Seq_Len, Head_Dim)
        Returns:
            gate: (Batch, Num_Heads, 1, 1) -> Attention Mapに乗算するマスク
        """
        B, H, N, D = q.shape
        
        # グローバルな文脈抽出 (Global Context Pooling)
        # ヘッド次元とシーケンス次元を平均化して、大域的な「問い(Query)」と「知識(Key)」の状態を見る
        q_ctx = q.mean(dim=[2, 3]) # (B, H) -> 本来はもっとリッチな情報を渡すべきだが軽量化のため
        
        # 簡易化のため、入力次元を合わせるダミー変換 (実際はH*Head_Dim等を扱う)
        # ここではSCALを「ヘッドごとの信頼度推定器」として使う
        # 入力として現在のAttentionの「エネルギー総量」のようなものを渡す
        
        energy = (q * k).sum(dim=-1).mean(dim=-1) # (B, H)
        
        # SCALに入力するために次元拡張・変換が必要だが、
        # ここではConceptとして、SCALが「エネルギー分布の異常」を検知してゲートを閉じる動作を模倣
        
        # SCAL Forward
        # 期待する入力: (Batch, In_Features)
        # ここでは energy を特徴量として拡張して渡す
        energy_feat = energy.repeat(1, self.dim // H) # (B, Dim)
        
        gate_signals = self.logic_core(energy_feat)['output'] # (B, H)
        
        # ゲート値は 0.0 (Block) ~ 1.0 (Pass)
        # SCALの出力はスパイクレートなので、それをゲート係数として使う
        gate = gate_signals.view(B, H, 1, 1)
        
        return gate


class LogicGatedSpikingSelfAttention(nn.Module):
    """
    Logic-Gated Spiking Self-Attention (LG-SSA).
    従来の (Q @ K^T) * Scale に加えて、LogicGateによるフィルタリングを行う。
    これにより、「相関はあるが論理的に矛盾する」アテンションを排除できる。
    """
    def __init__(self, d_model: int, num_heads: int, tau_m: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.125

        # Standard Spiking Attention Components
        self.q_linear = layer.Linear(d_model, d_model)
        self.q_bn = nn.BatchNorm1d(d_model)
        self.q_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.k_linear = layer.Linear(d_model, d_model)
        self.k_bn = nn.BatchNorm1d(d_model)
        self.k_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.v_linear = layer.Linear(d_model, d_model)
        self.v_bn = nn.BatchNorm1d(d_model)
        self.v_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        # --- Logic Gate Component ---
        # これが "Logic-Gated SNN" の核心
        self.logic_gate = LogicGateController(d_model, num_heads)

        self.attn_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(d_model, d_model)
        self.proj_bn = nn.BatchNorm1d(d_model)
        self.proj_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape

        # 1. Q, K, V Generation (Spiking)
        q = self.q_lif(self.q_bn(self.q_linear(x).transpose(1, 2)).transpose(1, 2))
        k = self.k_lif(self.k_bn(self.k_linear(x).transpose(1, 2)).transpose(1, 2))
        v = self.v_lif(self.v_bn(self.v_linear(x).transpose(1, 2)).transpose(1, 2))

        # 2. Reshape for Multi-Head
        # (B, N, H, D_h) -> (B, H, N, D_h)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Compute Logic Gate (Contextual Filtering)
        # QueryとKeyの全体的な整合性をチェックし、不適切なヘッドを抑制する
        # Gate: (B, H, 1, 1)
        logic_mask = self.logic_gate(q, k)

        # 4. Attention Calculation with Logic Gating
        # Attn = ((Q @ K^T) * scale) * LogicGate
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        
        # 論理ゲートの適用 (Soft Gating)
        # SCALが「このヘッドの判断は怪しい」と思ったら logic_mask が小さくなり、寄与が減る
        attn_gated = attn_score * logic_mask

        # 5. Aggregate
        x_attn = attn_gated @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        # 6. Output
        x_attn = self.attn_lif(x_attn)
        x_attn = self.proj_lif(self.proj_bn(self.proj_linear(x_attn).transpose(1, 2)).transpose(1, 2))

        return x_attn