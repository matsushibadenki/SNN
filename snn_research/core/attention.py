# ファイルパス: snn_research/core/attention.py
# Title: Spike-Driven Self-Attention (SDSA) (完成版)
# Description: Improvement-Plan.mdに基づき、乗算を使用しないスパイクベースの
#              自己注意メカニズム (SDSA) を実装。
#
# 特徴:
# - XNORベースの類似度計算による省メモリ化とSNNネイティブな演算の実装。
# - SpikingTransformerV2等からの呼び出しに対応した単一タイムステップ処理 (SDSA)。
# - DTA-SNN (Dynamic Temporal Attention) の試験実装を含む。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
import logging

# SDSAで使用するスパイクニューロン
from .neurons import AdaptiveLIFNeuron as LIFNeuron
# 代理勾配関数 / 基底クラス
from spikingjelly.activation_based import base # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpikeDrivenSelfAttention(base.MemoryModule):
    """
    Spike-Driven Self-Attention (SDSA) の改善・詳細化版実装。
    単純な積和演算の代わりに、XNORベースのビット演算的類似度計算を用いることで、
    SNNハードウェアでの効率的な実装を模擬します。
    """
    lif_q: LIFNeuron
    lif_k: LIFNeuron
    lif_v: LIFNeuron

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 time_steps: int, # 互換性のために残すが、SDSA内部ループでは使用しない
                 neuron_config: dict,
                 add_noise_if_silent: bool = True,
                 noise_prob: float = 0.01
                ):
        """
        Args:
            dim (int): モデルの次元数。
            num_heads (int): アテンションヘッド数。
            time_steps (int): (廃止予定) 外部ループで制御されるため内部では未使用。
            neuron_config (dict): スパイクニューロンの設定。
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.add_noise_if_silent = add_noise_if_silent
        self.noise_prob = noise_prob

        # 線形変換層 (入力 -> Q, K, V)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # スパイク生成ニューロン
        # 必要なパラメータのみを抽出
        lif_params = {k: v for k, v in neuron_config.items() 
                      if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 
                               'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        
        # SDSA専用の閾値設定があれば優先使用
        lif_params['base_threshold'] = neuron_config.get("sdsa_threshold", lif_params.get('base_threshold', 1.0))

        self.lif_q = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_k = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_v = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))

        # 出力層
        self.to_out = nn.Linear(dim, dim)

        logging.info("✅ SpikeDrivenSelfAttention initialized.")
        logging.info("   - Mechanism: XNOR-based similarity")

    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        """
        XNORベースの類似度計算 (省メモリ版)。
        
        Binary Spike (0/1) 前提:
        Popcount(XNOR(q, k)) = Dh - Popcount(q) - Popcount(k) + 2 * DotProduct(q, k)
        
        Args:
            q_spikes: (B, H, N, Dh)
            k_spikes: (B, H, N, Dh)
        Returns:
            attn_scores: (B, H, N, N)
        """
        Dh = q_spikes.shape[-1]
        
        # 1. DotProduct(q, k) -> (B, H, N, N)
        qk_dot = torch.matmul(q_spikes, k_spikes.transpose(-1, -2))
        
        # 2. Popcount(q) -> (B, H, N, 1)
        q_popcount = q_spikes.sum(dim=-1, keepdim=True)
        
        # 3. Popcount(k) -> (B, H, 1, N)
        k_popcount = k_spikes.sum(dim=-1, keepdim=True).transpose(-1, -2)
        
        # 4. Combine
        attn_scores = Dh - q_popcount - k_popcount + (2 * qk_dot)
        
        return attn_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SDSAのフォワードパス（単一タイムステップ）。
        
        Args:
            x (torch.Tensor): 現在のタイムステップの入力 (Batch, Num_Tokens, Dim)。
                             (アナログ電流として扱われる)
        Returns:
            torch.Tensor: 出力テンソル (Batch, Num_Tokens, Dim)。
        """
        B, N, C = x.shape
        
        # 1. アナログ電流を計算
        q_lin = self.to_q(x) # (B, N, C)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # 2. スパイクを生成 (単一ステップ)
        # Flatten for neuron processing: (B*N, C)
        s_q_t, _ = self.lif_q(q_lin.reshape(B * N, C)) 
        s_k_t, _ = self.lif_k(k_lin.reshape(B * N, C))
        s_v_t, _ = self.lif_v(v_lin.reshape(B * N, C)) 

        # Reshape back: (B, N, C)
        s_q_agg = s_q_t.reshape(B, N, C)
        s_k_agg = s_k_t.reshape(B, N, C)
        s_v_agg = s_v_t.reshape(B, N, C)

        # 3. ヘッド分割とXNOR類似度計算
        # (B, N, C) -> (B, N, H, Dh) -> (B, H, N, Dh)
        s_q = s_q_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        s_k = s_k_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        s_v = s_v_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores_xnor = self._xnor_similarity(s_q, s_k) 
        
        # スパイクベースのAttention Weight (Sigmoidで確率的に解釈)
        attn_weights = torch.sigmoid(attn_scores_xnor) 

        # Valueとの積 (現状は浮動小数点演算を使用、完全スパイク化の余地あり)
        attention_out = torch.matmul(attn_weights, s_v)

        # 4. 出力の整形
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(attention_out)

        return out

    def set_stateful(self, stateful: bool):
        """内部ニューロンのステートフルモードを設定する。"""
        self.stateful = stateful
        self.lif_q.set_stateful(stateful)
        self.lif_k.set_stateful(stateful)
        self.lif_v.set_stateful(stateful)

    def reset(self):
        """ニューロンの状態をリセット"""
        super().reset()
        self.lif_q.reset()
        self.lif_k.reset()
        self.lif_v.reset()


class DynamicTemporalAttention(base.MemoryModule):
    """
    DTA-SNN (Dynamic Temporal Attention) の実装。
    入力に対して内部的に時間方向(T)のスパイク列を生成し、
    GRU等を用いて時間的ダイナミクスを集約する方式。
    """
    lif_q: LIFNeuron
    lif_k: LIFNeuron
    lif_v: LIFNeuron
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 time_steps: int,
                 neuron_config: dict
                ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.time_steps = time_steps

        # 線形変換層
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # スパイク生成ニューロン (パラメータ共有なし)
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold']}
        
        self.lif_q = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_k = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_v = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        
        # 時間ダイナミクスエンコーダ (GRU)
        self.temporal_encoder = nn.GRU(
            input_size=self.head_dim, 
            hidden_size=self.head_dim, 
            batch_first=True
        )

        self.to_out = nn.Linear(dim, dim)
        
        logging.info("✅ DynamicTemporalAttention (DTA-SNN Stub) initialized.")

    def _generate_spikes(self, x: torch.Tensor, neuron: LIFNeuron) -> torch.Tensor:
        """
        静的入力 (B, N, C) からスパイク時系列 (B, T, N, C) を生成する。
        """
        B, N, C = x.shape
        x_lin_flat = x.reshape(B * N, C)
        
        s_list = []
        
        # 非ステートフルモードの場合、この生成プロセス用に一時的にリセット＆ステートフル化
        was_stateful = self.stateful
        if not was_stateful:
            neuron.reset()
            neuron.set_stateful(True)

        for _ in range(self.time_steps):
            s_t, _ = neuron(x_lin_flat) # (B*N, C)
            s_list.append(s_t.reshape(B, N, C))
            
        # 状態を元に戻す
        if not was_stateful:
            neuron.set_stateful(False)
            neuron.reset()
            
        return torch.stack(s_list, dim=1) # (B, T, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DTA のフォワードパス。
        """
        B, N, C = x.shape
        
        # 1. Q, K, V の電流を計算
        q_lin = self.to_q(x)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # 2. スパイク時系列を生成 (B, T, N, C)
        # 各自のニューロンを使用するように修正
        s_q_time = self._generate_spikes(q_lin, self.lif_q)
        s_k_time = self._generate_spikes(k_lin, self.lif_k)
        s_v_time = self._generate_spikes(v_lin, self.lif_v)
        
        # 3. ヘッド分割 (B, T, N, H, Dh) -> (B, H, N, T, Dh)
        # 注意: DTAでは時間軸(T)を埋め込みとして扱う
        s_q_heads = s_q_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        s_k_heads = s_k_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        s_v_heads = s_v_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        
        # フラット化してGRUへ: (Batch * Heads * Tokens, Time, Dim_Head)
        q_flat = s_q_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)
        k_flat = s_k_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)
        v_flat = s_v_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)

        # 4. 時間ダイナミクスのエンコード
        # GRUの最終状態を取得 (1, BatchSize, HiddenDim)
        _, q_temporal = self.temporal_encoder(q_flat) 
        _, k_temporal = self.temporal_encoder(k_flat)
        _, v_temporal = self.temporal_encoder(v_flat)
        
        # 形状を戻す (B, H, N, Dh)
        q_out = q_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)
        k_out = k_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)
        v_out = v_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)

        # 5. Scaled Dot-Product Attention
        attn_scores = torch.matmul(q_out, k_out.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attention_out = torch.matmul(attn_weights, v_out) # (B, H, N, Dh)

        # 6. 出力整形
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(attention_out)

        return out

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        self.lif_q.set_stateful(stateful)
        self.lif_k.set_stateful(stateful)
        self.lif_v.set_stateful(stateful)

    def reset(self):
        super().reset()
        self.lif_q.reset()
        self.lif_k.reset()
        self.lif_v.reset()
        self.temporal_encoder.flatten_parameters()
