# ファイルパス: snn_research/models/experimental/predictive_coding_model.py
# Title: Predictive Coding Model Wrapper (Fixed: Mypy Tensor Casting)
# Description:
#   SequentialPCNetwork を包含し、テキスト(input_ids)と連続値(x)の両方に対応するハイレベルモデル。
#   修正: total_spikes へのアクセス時に cast(torch.Tensor) を追加し、mypyエラー "Tensor not callable" を解消。

import torch
import torch.nn as nn
from typing import List, Dict, Any, cast, Optional, Tuple, Union

from snn_research.core.networks.sequential_pc_network import SequentialPCNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.learning_rules.predictive_coding_rule import PredictiveCodingRule
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class PredictiveCodingModel(nn.Module):
    """
    画像分類・テキスト処理兼用の予測符号化SNNモデル。
    入力データを d_model 次元に射影し、PC Network で処理を行う。
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        neuron_params: Dict[str, Any],
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # d_model は hidden_dims の最初の要素、または input_dim
        self.d_model = hidden_dims[0] if hidden_dims else input_dim
        
        # --- 入力層の定義 ---
        # テキスト用: Embedding
        self.token_embedding: Optional[nn.Embedding]
        
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, self.d_model)
        else:
            self.token_embedding = None
            
        # 連続値用: Linear Projection (次元合わせ)
        self.input_projector = nn.Linear(input_dim, self.d_model)
        
        # --- PC Network の構築 ---
        layers: List[PredictiveCodingLayer] = []
        current_dim = self.d_model
        
        # 隠れ層の構築
        for h_dim in hidden_dims:
            layer = PredictiveCodingLayer(
                d_model=current_dim, # Bottom-up input dimension
                d_state=h_dim,       # Top-down state dimension
                neuron_class=AdaptiveLIFNeuron,
                neuron_params=neuron_params,
                weight_tying=True
            )
            layers.append(layer)
            current_dim = h_dim
            
        self.network = SequentialPCNetwork(layers)
        
        # 最終分類層 (Readout)
        self.classifier = nn.Linear(current_dim, output_dim)
        
        # 学習ルールの登録
        self._register_learning_rules()

    def _register_learning_rules(self) -> None:
        """
        各層に対応する学習ルール（PredictiveCodingRule）をネットワークに登録する。
        """
        # network.pc_layers が Iterable であることを前提
        if hasattr(self.network, 'pc_layers'):
            for i, layer_module in enumerate(self.network.pc_layers):
                layer_name = f"layer_{i}"
                layer = cast(PredictiveCodingLayer, layer_module)
                
                # mypy対策: パラメータのキャスト
                params: List[nn.Parameter] = [cast(nn.Parameter, layer.generative_fc.weight)]
                if layer.generative_fc.bias is not None:
                    params.append(cast(nn.Parameter, layer.generative_fc.bias))
                
                rule = PredictiveCodingRule(
                    params=params,
                    learning_rate=0.005,
                    layer_name=layer_name,
                    error_weight=1.0,
                    weight_decay=1e-4
                )
                self.network.add_learning_rule(rule)

    def forward(
        self, 
        x: Optional[torch.Tensor] = None, 
        input_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 連続値入力 (Batch, Dim) or (Batch, C, H, W) 
               OR 整数型の場合は Token IDs とみなす
            input_ids: テキスト入力 (Batch, SeqLen)
            return_spikes: スパイク列を返すかどうか (Trainer互換性のため)
        """
        inputs: torch.Tensor
        
        # 入力形状を保存して後で復元するための変数
        original_shape: Optional[Tuple[int, int]] = None
        
        # --- 入力処理分岐 (SNNCore対策) ---
        target_ids = input_ids
        target_x = x
        
        # x が整数型なら input_ids として扱う
        if target_x is not None and not target_x.is_floating_point():
            if target_ids is None:
                target_ids = target_x
                target_x = None
        
        if target_ids is not None:
            # テキスト入力 (Embedding)
            if self.token_embedding is None:
                raise ValueError("Model initialized without vocab_size, cannot handle input_ids.")
            
            inputs = self.token_embedding(target_ids) # (B, Seq, d_model)
            
            # シーケンスがある場合、バッチ次元に展開して処理する (Batch * Seq, Dim)
            if inputs.ndim == 3:
                b, s, d = inputs.shape
                original_shape = (b, s)
                inputs = inputs.view(b * s, d)
                
        elif target_x is not None:
            # 連続値入力 (Projector)
            # 画像 (B, C, H, W) ならフラット化
            if target_x.ndim > 2:
                target_x = target_x.view(target_x.size(0), -1)
            
            # 次元チェック (不一致なら無理やり合わせるかエラーだが、ここではそのまま渡す)
            if not target_x.is_floating_point():
                target_x = target_x.float()
                
            inputs = self.input_projector(target_x)
            
        else:
            raise ValueError("Either x (continuous) or input_ids (discrete) must be provided.")

        # 2. PC Network 順伝播
        # inputs は (Batch', d_model)
        final_state = self.network(inputs)
        
        # 3. 分類/出力
        logits = self.classifier(final_state) # (Batch', OutputDim)
        
        # シーケンス次元の復元 (Batch * Seq, Vocab) -> (Batch, Seq, Vocab)
        if original_shape is not None:
            b, s = original_shape
            logits = logits.view(b, s, -1)
        
        # 4. 戻り値の整形
        if return_spikes:
            # トレーナーが要求する場合、ダミーのスパイク情報を返す
            # (get_total_spikes で統計は取れるため、ここでは互換性維持)
            if original_shape is not None:
                b, s = original_shape
                dummy_spikes = torch.zeros(b, s, self.output_dim, device=logits.device)
            else:
                batch_size = logits.size(0)
                dummy_spikes = torch.zeros(batch_size, self.output_dim, device=logits.device)
            return logits, dummy_spikes
            
        return logits

    def reset_state(self) -> None:
        """状態のリセット"""
        # 1. Networkのリセット
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
            
        # 2. SpikingJellyのリセット
        SJ_F.reset_net(self)
        
        # 3. 統計のリセット
        self.reset_spike_stats()

    def get_total_spikes(self) -> float:
        """
        モデル全体の総スパイク数を取得する。
        各層のニューロンが持つ 'total_spikes' バッファを集計する。
        """
        total_spikes = 0.0
        if hasattr(self.network, 'pc_layers'):
            for layer in self.network.pc_layers:
                # Generative Neuron
                if hasattr(layer, 'generative_neuron') and hasattr(layer.generative_neuron, 'total_spikes'):
                    # Cast to Tensor explicitly to avoid mypy 'Tensor not callable' confusion on .item()
                    gen_spikes = cast(torch.Tensor, layer.generative_neuron.total_spikes)
                    total_spikes += float(gen_spikes.item())
                # Inference Neuron
                if hasattr(layer, 'inference_neuron') and hasattr(layer.inference_neuron, 'total_spikes'):
                    inf_spikes = cast(torch.Tensor, layer.inference_neuron.total_spikes)
                    total_spikes += float(inf_spikes.item())
        return total_spikes

    def reset_spike_stats(self) -> None:
        """スパイク統計カウンターをリセットする"""
        if hasattr(self.network, 'pc_layers'):
            for layer in self.network.pc_layers:
                if hasattr(layer, 'generative_neuron') and hasattr(layer.generative_neuron, 'total_spikes'):
                    gen_spikes = cast(torch.Tensor, layer.generative_neuron.total_spikes)
                    gen_spikes.zero_()
                if hasattr(layer, 'inference_neuron') and hasattr(layer.inference_neuron, 'total_spikes'):
                    inf_spikes = cast(torch.Tensor, layer.inference_neuron.total_spikes)
                    inf_spikes.zero_()