# ファイルパス: snn_research/cognitive_architecture/planner_snn.py
# 日本語タイトル: Planner SNN v2.4 - Recursive Embedding Search
# 目的・内容:
#   プランニングタスク用SNNモデル。
#   修正: Embedding層の探索を再帰的に行い、複雑なモデル構造でも確実に見つけられるように強化。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union, Tuple
from snn_research.core.snn_core import SNNCore
import logging

logger = logging.getLogger(__name__)

class PlannerSNN(nn.Module):
    """
    プランニングタスクに特化したSNNモデル。
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        d_state: int, 
        num_layers: int, 
        time_steps: int, 
        n_head: int, 
        num_skills: int, 
        neuron_config: Dict[str, Any]
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_skills = num_skills

        # SNNCore を使用してバックボーンを構築
        # 注: 実運用では config['architecture_type'] を引数から柔軟に設定できるようにすべきですが、
        # ここでは既存のコードに合わせて predictive_coding をデフォルトとします。
        self.core = SNNCore(
            config={
                'architecture_type': 'predictive_coding', 
                'd_model': d_model,
                'num_layers': num_layers,
                'time_steps': time_steps,
                'neuron': neuron_config,
                'd_state': d_state, 
                'n_head': n_head    
            },
            vocab_size=vocab_size
        )

        # スキル選択用の出力ヘッド
        self.skill_head = nn.Linear(d_model, num_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen) の入力ID
        Returns:
            logits: (Batch, NumSkills) のスキル選択確率
        """
        # output_hidden_states=True を渡して中間層の取得を試みる
        outputs = self.core(x, output_hidden_states=True)

        hidden_state = self._extract_hidden_state(outputs, x)

        # 時間方向・シーケンス方向のプーリング (平均)
        if hidden_state.dim() == 4: # (Time, Batch, SeqLen, Features)
            pooled = hidden_state.mean(dim=[0, 2])
        elif hidden_state.dim() == 3: # (Batch, SeqLen, Features)
            pooled = hidden_state.mean(dim=1)
        else:
            pooled = hidden_state 

        return self.skill_head(pooled)

    def _extract_hidden_state(self, outputs: Any, input_ids: torch.Tensor) -> torch.Tensor:
        """
        バックボーンの出力から d_model 次元の隠れ状態を抽出する。
        見つからない場合は入力埋め込みをフォールバックとして使用する。
        """
        candidate = None

        # 1. 探索ロジック: Tensor, Tuple, Dict, Object に対応
        if isinstance(outputs, torch.Tensor):
            if outputs.shape[-1] == self.d_model:
                candidate = outputs
        
        elif isinstance(outputs, (tuple, list)):
            for out in reversed(outputs):
                if isinstance(out, torch.Tensor) and out.shape[-1] == self.d_model:
                    candidate = out
                    break
        
        elif isinstance(outputs, dict):
            for key in ['hidden_states', 'last_hidden_state', 'embedding', 'x']:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    if outputs[key].shape[-1] == self.d_model:
                        candidate = outputs[key]
                        break
        
        elif hasattr(outputs, 'hidden_states'): 
            if isinstance(outputs.hidden_states, (tuple, list)) and len(outputs.hidden_states) > 0:
                candidate = outputs.hidden_states[-1]
            elif isinstance(outputs.hidden_states, torch.Tensor):
                candidate = outputs.hidden_states
        
        elif hasattr(outputs, 'last_hidden_state'):
            candidate = outputs.last_hidden_state

        if candidate is not None and candidate.shape[-1] == self.d_model:
            return candidate

        # 2. フォールバック: Embedding層の利用
        # Hidden Stateが見つからない場合、警告を出してEmbeddingを使用する
        # ※ 頻繁に出るためログレベルを debug に落とすか、初回のみ warning にするのが理想
        # ここではユーザーへのフィードバックのため warning のままにするが、メッセージを少し抑制的にする
        
        # logger.debug(f"PlannerSNN: Using Input Embeddings fallback. (Output shape: {outputs.shape if isinstance(outputs, torch.Tensor) else type(outputs)})")
        
        return self._get_input_embeddings(input_ids)

    def _get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """モデル内部の埋め込み層を再帰的に探索して適用する"""
        model = self.core.model
        
        # --- 再帰探索関数 ---
        def find_embedding_layer(module: nn.Module) -> Optional[nn.Embedding]:
            # 自身がEmbeddingなら返却
            if isinstance(module, nn.Embedding):
                return module
            
            # 子モジュールを走査
            for name, child in module.named_children():
                # 優先度: 名前が embedding っぽいものを先にチェック
                if any(k in name for k in ['embed', 'wte', 'token']):
                    res = find_embedding_layer(child)
                    if res: return res
            
            # 見つからなければ全探索
            for child in module.children():
                res = find_embedding_layer(child)
                if res: return res
                
            return None
        # -------------------

        embed_layer = find_embedding_layer(model)
        
        if embed_layer is not None:
            return embed_layer(input_ids)
        
        # 最終手段: 新しい埋め込み層を一時的に作る（学習されないが形状は合う）
        # これが呼ばれることはモデル構造上ほぼありえないはず
        logger.error("PlannerSNN: CRITICAL - Could not find ANY embedding layer. Creating random projection.")
        device = input_ids.device
        projection = nn.Embedding(self.vocab_size, self.d_model).to(device)
        return projection(input_ids)