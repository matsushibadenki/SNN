# ファイルパス: snn_research/models/experimental/moe_model.py
# Title: Spiking FrankenMoE & Router (Path Logic Fixed)
# 機能説明:
#   複数の学習済みSNNモデル（エキスパート）を統合するMoEモデル。
#   修正: SNNCoreのインポートを遅延させ、循環参照を回避。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, cast
import logging
import os
from pathlib import Path

from snn_research.core.base import BaseModel
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore

logger = logging.getLogger(__name__)

class ContextAwareSpikingRouter(BaseModel):
    """
    テキスト入力と視覚コンテキストの両方を考慮してルーティングを行うSNNルーター。
    """
    def __init__(self, input_dim: int, num_experts: int, neuron_config: Dict[str, Any]):
        super().__init__()
        self.num_experts = num_experts
        router_neuron_params = neuron_config.copy()
        router_neuron_params.pop('type', None)
        self.text_proj = nn.Linear(input_dim, input_dim // 2)
        self.text_lif = AdaptiveLIFNeuron(features=input_dim // 2, **router_neuron_params)
        self.ctx_proj = nn.Linear(input_dim, input_dim // 2)
        self.ctx_lif = AdaptiveLIFNeuron(features=input_dim // 2, **router_neuron_params)
        self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.fusion_lif = AdaptiveLIFNeuron(features=input_dim // 2, **router_neuron_params)
        self.routing_head = nn.Linear(input_dim // 2, num_experts)
        self.output_lif = AdaptiveLIFNeuron(features=num_experts, **router_neuron_params)
        self._init_weights()

    def forward(self, x_text: torch.Tensor, x_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_out, _ = self.text_lif(self.text_proj(x_text))
        combined: torch.Tensor
        if x_context is not None:
            c_out, _ = self.ctx_lif(self.ctx_proj(x_context))
            combined = torch.cat([t_out, c_out], dim=-1)
        else:
            zeros = torch.zeros_like(t_out)
            combined = torch.cat([t_out, zeros], dim=-1)
        fused, _ = self.fusion_lif(self.fusion_layer(combined))
        r_out, _ = self.output_lif(self.routing_head(fused))
        routing_weights = F.softmax(r_out, dim=-1)
        return routing_weights

class SpikingFrankenMoE(BaseModel):
    """
    複数の学習済みSNNモデル（エキスパート）を統合するMoEモデル。
    """
    experts: nn.ModuleList
    router: ContextAwareSpikingRouter
    router_embedding: Optional[nn.Embedding]

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        expert_configs: List[Dict[str, Any]],
        expert_checkpoints: List[str],
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = time_steps
        self.num_experts = len(expert_configs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}
            
        self.router = ContextAwareSpikingRouter(d_model, self.num_experts, neuron_config)
        self.experts = nn.ModuleList()
        self.router_embedding = None
        
        # --- 修正: 遅延インポートで循環参照を回避 ---
        try:
            from snn_research.core.snn_core import SNNCore
        except ImportError:
            raise ImportError("Failed to import SNNCore in SpikingFrankenMoE.")
        # ----------------------------------------
        
        # プロジェクトルートの推定 (このファイルの位置から遡る)
        try:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        except Exception:
            project_root = Path(".")

        for i, (cfg, ckpt_path) in enumerate(zip(expert_configs, expert_checkpoints)):
            logger.info(f"🧟 FrankenMoE: Building Expert {i}...")
            try:
                expert_container = SNNCore(config=cfg, vocab_size=vocab_size)
                
                # チェックポイントのロード
                if ckpt_path and str(ckpt_path).lower() != "none":
                    load_path = None
                    
                    # 1. そのままのパスで確認
                    if os.path.exists(ckpt_path):
                        load_path = ckpt_path
                    else:
                        # 2. プロジェクトルートからの相対パスとして試行
                        potential_path = project_root / ckpt_path
                        if potential_path.exists():
                            load_path = str(potential_path)
                    
                    if load_path:
                        try:
                            state_dict = torch.load(load_path, map_location='cpu')
                            if 'model_state_dict' in state_dict:
                                state_dict = state_dict['model_state_dict']
                            
                            # キーのクリーニング (model. プレフィックスの削除)
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                k = k.replace("model.", "") 
                                new_state_dict[k] = v
                                
                            expert_container.model.load_state_dict(new_state_dict, strict=False)
                            logger.info(f"   -> Loaded weights from {load_path}")
                        except Exception as e:
                            logger.error(f"   -> Error loading weights from {load_path}: {e}. Using random weights.")
                    else:
                        logger.warning(f"   -> Checkpoint path '{ckpt_path}' not found (Tried abs and relative to {project_root}). Using random weights.")
                else:
                    logger.info(f"   -> No checkpoint specified. Using random weights for Expert {i}.")
                
                self.experts.append(expert_container.model)
                
            except Exception as e:
                logger.error(f"Failed to initialize expert {i}: {e}")
                raise e

        self._init_weights()
        logger.info(f"✅ SpikingFrankenMoE initialized with {self.num_experts} experts.")

    def forward(self, 
                input_ids: torch.Tensor, 
                return_spikes: bool = False, 
                visual_context: Optional[torch.Tensor] = None,
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        device = input_ids.device
        
        # 全エキスパートの内部状態をリセット
        SJ_F.reset_net(self)
        
        # 1. ルーティング入力の準備
        first_expert = self.experts[0]
        text_embeds: torch.Tensor
        
        if hasattr(first_expert, 'embedding'):
             emb_layer = cast(nn.Module, getattr(first_expert, 'embedding'))
             text_embeds = emb_layer(input_ids)
        elif hasattr(first_expert, 'token_embedding'):
             emb_layer = cast(nn.Module, getattr(first_expert, 'token_embedding'))
             text_embeds = emb_layer(input_ids)
        else:
             # エキスパートがEmbeddingを持たない場合（稀）、ルーター用のEmbeddingを使う
             if self.router_embedding is None:
                 num_embeddings = self.vocab_size
                 self.router_embedding = nn.Embedding(num_embeddings, self.d_model).to(device)
             text_embeds = self.router_embedding(input_ids)

        # ルーティングには「最後のトークン」を使用する
        router_text_input = text_embeds[:, -1, :]
        
        router_visual_input: Optional[torch.Tensor] = None
        if visual_context is not None:
             router_visual_input = visual_context.mean(dim=1)
        
        # 2. Routing
        routing_weights = self.router(router_text_input, router_visual_input)
        
        # 3. Experts Execution
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            # Expert呼び出し
            out_any = expert(
                input_ids, 
                return_spikes=False, # エキスパート内部での集計は不要
                context_embeds=visual_context, 
                **kwargs
            )
            
            # タプル戻り値のハンドリング
            if isinstance(out_any, tuple):
                expert_outputs.append(out_any[0]) # (logits, spikes, mem) -> logits
            else:
                expert_outputs.append(out_any)
            
        # 4. Aggregation
        # (Batch, SeqLen, Dim)
        try:
            stacked_outputs = torch.stack(expert_outputs, dim=-1) 
            rw_expanded = routing_weights.unsqueeze(1).unsqueeze(1)
            final_output = (stacked_outputs * rw_expanded).sum(dim=-1)
        except Exception as e:
            logger.error(f"Error stacking expert outputs. Shapes might mismatch: {[o.shape for o in expert_outputs]}")
            raise e
        
        # 5. Statistics
        total_spikes = 0.0
        for expert in self.experts:
            if hasattr(expert, 'get_total_spikes'):
                total_spikes += expert.get_total_spikes() # type: ignore
        
        if hasattr(self.router, 'get_total_spikes'):
            total_spikes += self.router.get_total_spikes()
            
        avg_spikes_val = total_spikes / (input_ids.shape[0] * input_ids.shape[1] * self.time_steps)
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return final_output, avg_spikes, mem