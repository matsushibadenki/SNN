# ファイルパス: snn_research/core/cortical_column.py
# Title: Cortical Column with Causal Plasticity (Phase 5 Impl / Type Fixed)
# Description:
#   3層構造 (L4, L2/3, L5/6) を持つ大脳皮質カラムモデル。
#   AbstractSNNNetworkを継承し、Causal Trace Learning (V2) による
#   自律的なシナプス可塑性（学習機能）を実装している。

import torch
import torch.nn as nn
from typing import Dict, Any, Type, Tuple, Optional, cast

from .base import SNNLayerNorm
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from .networks.abstract_snn_network import AbstractSNNNetwork
from snn_research.learning_rules import get_bio_learning_rule, BioLearningRule

class CorticalLayer(nn.Module):
    """
    皮質カラム内の1つの層 (例: L4, L2/3)。
    """
    def __init__(self, features: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any], name: str):
        super().__init__()
        self.name = name
        self.neuron = neuron_class(features=features, **neuron_params)
        self.norm = SNNLayerNorm(features)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (Batch, Features)
        spikes, mem = self.neuron(x) # type: ignore
        return spikes, mem

class CorticalColumn(AbstractSNNNetwork):
    """
    3層構造 (L4, L2/3, L5/6) を持ち、生物学的学習則によって自己組織化する皮質カラム。
    """
    def __init__(
        self, 
        input_dim: int, 
        column_dim: int, 
        output_dim: int, 
        neuron_config: Dict[str, Any],
        learning_rule_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.column_dim = column_dim
        
        # 1. ニューロン設定の解決
        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[nn.Module]
        if neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
            valid_keys = ['a', 'b', 'c', 'd', 'dt']
            neuron_params = {k: v for k, v in neuron_params.items() if k in valid_keys}
        else:
            neuron_class = AdaptiveLIFNeuron
            # AdaptiveLIF用のパラメータフィルタリング
            valid_keys = [
                'tau_mem', 'base_threshold', 'adaptation_strength', 
                'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset'
            ]
            neuron_params = {k: v for k, v in neuron_params.items() if k in valid_keys}

        # 2. 層の構築
        self.L4 = CorticalLayer(column_dim, neuron_class, neuron_params, "L4")
        self.L23 = CorticalLayer(column_dim, neuron_class, neuron_params, "L23")
        self.L56 = CorticalLayer(column_dim, neuron_class, neuron_params, "L56")

        # 3. シナプス結合 (重み) の定義
        # 層間結合
        self.proj_input_L4 = nn.Linear(input_dim, column_dim)
        self.proj_L4_L23 = nn.Linear(column_dim, column_dim)
        self.proj_L23_L56 = nn.Linear(column_dim, column_dim)
        self.proj_L56_L4 = nn.Linear(column_dim, column_dim) # Feedback
        
        # 再帰結合 (Recurrent)
        self.rec_L4 = nn.Linear(column_dim, column_dim)
        self.rec_L23 = nn.Linear(column_dim, column_dim)
        self.rec_L56 = nn.Linear(column_dim, column_dim)

        # 出力投影
        self.proj_out_ff = nn.Linear(column_dim, output_dim)
        self.proj_out_fb = nn.Linear(column_dim, output_dim)

        # 4. 学習則の初期化 (Causal Trace V2 推奨)
        self.synaptic_rules: Dict[str, BioLearningRule] = {}
        
        if learning_rule_config:
            rule_name = learning_rule_config.get("learning_rule", "CAUSAL_TRACE_V2")
            # 各投影に対して学習則をインスタンス化
            self._setup_learning_rule("proj_input_L4", rule_name, learning_rule_config)
            self._setup_learning_rule("proj_L4_L23", rule_name, learning_rule_config)
            self._setup_learning_rule("proj_L23_L56", rule_name, learning_rule_config)
            self._setup_learning_rule("proj_L56_L4", rule_name, learning_rule_config)
            self._setup_learning_rule("rec_L4", rule_name, learning_rule_config)
            self._setup_learning_rule("rec_L23", rule_name, learning_rule_config)
            self._setup_learning_rule("rec_L56", rule_name, learning_rule_config)

        self._init_weights()
        print(f"🧠 CorticalColumn initialized (Plasticity: {'ON' if self.synaptic_rules else 'OFF'}).")

    def _init_weights(self) -> None:
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _setup_learning_rule(self, projection_name: str, rule_name: str, config: Dict[str, Any]):
        """指定された結合のための学習則インスタンスを生成"""
        self.synaptic_rules[projection_name] = get_bio_learning_rule(rule_name, config)

    def forward( # type: ignore[override]
        self, 
        input_signal: torch.Tensor, 
        prev_states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        順伝播処理。学習則のための活動履歴も記録する。
        入力が (Batch, Time, Dim) の場合は時間ループを実行する。
        """
        # 時間軸対応
        if input_signal.dim() == 3:
            # (Batch, Time, InputDim)
            B, T, D = input_signal.shape
            device = input_signal.device
            
            if prev_states is None:
                current_states = {
                    "L4": torch.zeros(B, self.column_dim, device=device),
                    "L23": torch.zeros(B, self.column_dim, device=device),
                    "L56": torch.zeros(B, self.column_dim, device=device)
                }
            else:
                current_states = prev_states

            out_ff_list = []
            out_fb_list = []
            
            # 時間ループ
            for t in range(T):
                input_t = input_signal[:, t, :] # (Batch, InputDim)
                out_ff_t, out_fb_t, current_states = self._forward_step(input_t, current_states)
                out_ff_list.append(out_ff_t)
                out_fb_list.append(out_fb_t)
            
            # (Batch, Time, OutputDim)
            out_ff_stacked = torch.stack(out_ff_list, dim=1)
            out_fb_stacked = torch.stack(out_fb_list, dim=1)
            
            return out_ff_stacked, out_fb_stacked, current_states
            
        elif input_signal.dim() == 2:
            # (Batch, InputDim) - 単一ステップ
            return self._forward_step(input_signal, prev_states)
        else:
            raise ValueError(f"CorticalColumn received input with unexpected shape: {input_signal.shape}")

    def _forward_step(
        self, 
        input_signal: torch.Tensor, 
        prev_states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        1タイムステップ分の処理。
        """
        batch_size = input_signal.shape[0]
        device = input_signal.device
        
        if prev_states is None:
            spikes_L4_prev = torch.zeros(batch_size, self.column_dim, device=device)
            spikes_L23_prev = torch.zeros(batch_size, self.column_dim, device=device)
            spikes_L56_prev = torch.zeros(batch_size, self.column_dim, device=device)
        else:
            spikes_L4_prev = prev_states["L4"]
            spikes_L23_prev = prev_states["L23"]
            spikes_L56_prev = prev_states["L56"]

        # --- Layer 4 ---
        # 入力: Sensory Input + Feedback from L56 + Recurrent
        in_L4_ff = self.proj_input_L4(input_signal)
        in_L4_fb = self.proj_L56_L4(spikes_L56_prev)
        in_L4_rec = self.rec_L4(spikes_L4_prev)
        
        spikes_L4, _ = self.L4(in_L4_ff + in_L4_fb + in_L4_rec)
        
        # 学習用状態記録 (Pre -> Post)
        self.model_state["proj_input_L4_pre"] = input_signal.detach()
        self.model_state["proj_input_L4_post"] = spikes_L4.detach()
        self.model_state["rec_L4_pre"] = spikes_L4_prev.detach()
        self.model_state["rec_L4_post"] = spikes_L4.detach()
        self.model_state["proj_L56_L4_pre"] = spikes_L56_prev.detach()
        self.model_state["proj_L56_L4_post"] = spikes_L4.detach()

        # --- Layer 2/3 ---
        # 入力: Feedforward from L4 + Recurrent
        in_L23_ff = self.proj_L4_L23(spikes_L4)
        in_L23_rec = self.rec_L23(spikes_L23_prev)
        
        spikes_L23, _ = self.L23(in_L23_ff + in_L23_rec)
        
        # 学習用状態記録
        self.model_state["proj_L4_L23_pre"] = spikes_L4.detach()
        self.model_state["proj_L4_L23_post"] = spikes_L23.detach()
        self.model_state["rec_L23_pre"] = spikes_L23_prev.detach()
        self.model_state["rec_L23_post"] = spikes_L23.detach()

        # --- Layer 5/6 ---
        # 入力: Feedforward from L2/3 + Recurrent
        in_L56_ff = self.proj_L23_L56(spikes_L23)
        in_L56_rec = self.rec_L56(spikes_L56_prev)
        
        spikes_L56, _ = self.L56(in_L56_ff + in_L56_rec)

        # 学習用状態記録
        self.model_state["proj_L23_L56_pre"] = spikes_L23.detach()
        self.model_state["proj_L23_L56_post"] = spikes_L56.detach()
        self.model_state["rec_L56_pre"] = spikes_L56_prev.detach()
        self.model_state["rec_L56_post"] = spikes_L56.detach()

        # 出力
        out_ff = self.proj_out_ff(spikes_L23)
        out_fb = self.proj_out_fb(spikes_L56)
        
        current_states = {
            "L4": spikes_L4,
            "L23": spikes_L23,
            "L56": spikes_L56
        }
        
        return out_ff, out_fb, current_states

    def run_learning_step(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        記録された活動に基づいてシナプス重みを更新する。
        """
        if not self.training or not self.synaptic_rules:
            return {}

        metrics = {}
        total_delta = 0.0
        optional_params: Dict[str, Any] = {}

        # 定義されたすべての学習則を実行
        target_projections = [
            ("proj_input_L4", self.proj_input_L4),
            ("proj_L4_L23", self.proj_L4_L23),
            ("proj_L23_L56", self.proj_L23_L56),
            ("proj_L56_L4", self.proj_L56_L4),
            ("rec_L4", self.rec_L4),
            ("rec_L23", self.rec_L23),
            ("rec_L56", self.rec_L56)
        ]

        for name, layer in target_projections:
            if name in self.synaptic_rules:
                rule = self.synaptic_rules[name]
                
                # model_state から Pre/Post 活動を取得
                pre_spikes = self.model_state.get(f"{name}_pre")
                post_spikes = self.model_state.get(f"{name}_post")
                
                if pre_spikes is not None and post_spikes is not None:
                    # 重み更新の計算
                    dw, _ = rule.update(
                        pre_spikes=pre_spikes,
                        post_spikes=post_spikes,
                        weights=layer.weight,
                        optional_params=optional_params
                    )
                    
                    # 重みの更新適用
                    with torch.no_grad():
                        layer.weight += dw
                        # 重みのクリッピング (発散防止)
                        layer.weight.clamp_(-1.0, 1.0)
                    
                    delta_mag = dw.abs().mean().item()
                    metrics[f"{name}_update"] = delta_mag
                    total_delta += delta_mag

        metrics["total_update_magnitude"] = total_delta
        return metrics

    def reset_state(self) -> None:
        super().reset_state()
        # 各ニューロン層のリセット
        if hasattr(self.L4.neuron, 'reset'):
            cast(Any, self.L4.neuron).reset()
        if hasattr(self.L23.neuron, 'reset'):
            cast(Any, self.L23.neuron).reset()
        if hasattr(self.L56.neuron, 'reset'):
            cast(Any, self.L56.neuron).reset()