# ファイルパス: snn_research/training/trainers/forward_forward.py
# 日本語タイトル: Forward-Forward アルゴリズムトレーナー
# 機能説明: 
#   誤差逆伝播法(Backpropagation)を使わずに、推論(Forward pass)のみで学習を行うトレーナー。
#   Geoffrey Hinton (2022) の手法に基づき、Positive/NegativeデータのGoodnessを最適化する。
#   GPUに依存しない局所的な学習更新則を実現し、省エネ化を推進する。

from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Any, Optional, List, Tuple, Union, Iterable

from snn_research.training.base_trainer import AbstractTrainer, DataLoader, MetricsMap
from snn_research.core.network import AbstractNetwork
from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork

logger = logging.getLogger(__name__)

class ForwardForwardTrainer(AbstractTrainer):
    """
    Forward-Forward Algorithm (Hinton, 2022) を実装したトレーナー。
    
    ネットワーク全体のエラーを逆伝播させるのではなく、層ごとに
    「Positive Pass (本物)」と「Negative Pass (偽物)」を行い、
    局所的な重み更新を行う。
    """

    def __init__(
        self,
        model: Union[SequentialSNNNetwork, nn.Module],
        learning_rate: float = 0.001,
        threshold: float = 2.0,
        logger_client: Optional[Any] = None
    ) -> None:
        super().__init__(model, logger_client)
        self.learning_rate = learning_rate
        self.threshold = threshold
        
        # 層ごとのオプティマイザを保持するリスト
        self.layer_optimizers: List[torch.optim.Optimizer] = []
        self._setup_layer_optimizers()

    def _setup_layer_optimizers(self) -> None:
        """
        モデルの各層に対して個別のオプティマイザを設定する。
        SequentialSNNNetwork または nn.Sequential に対応。
        """
        layers = self._get_layers()
        if not layers:
            logger.warning("No trainable layers found for Forward-Forward training.")
            return

        for layer in layers:
            # パラメータがある層のみオプティマイザを作成
            params = list(layer.parameters())
            if params:
                optimizer = Adam(params, lr=self.learning_rate)
                self.layer_optimizers.append(optimizer)
            else:
                self.layer_optimizers.append(None) # type: ignore

    def _get_layers(self) -> List[nn.Module]:
        """モデルから層のリストを抽出する"""
        if isinstance(self.model, SequentialSNNNetwork):
            # layer_order に基づいて層を取得
            return [self.model.layers_map[name] for name in self.model.layer_order]
        elif isinstance(self.model, nn.Sequential):
            return list(self.model)
        elif isinstance(self.model, nn.Module):
             # 修正: 明示的に nn.Module であることを確認してから children() を呼ぶ
             # これにより AbstractNetwork などの Union 型チェックエラーを回避
            return list(self.model.children())
        else:
            # nn.Module でない場合は層を取得できないため空リストを返す
            return []

    def _generate_negative_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Negative Data (偽データ) を生成する。
        シンプルな手法として、バッチ内でラベルと画像の組み合わせをランダムに入れ替える、
        あるいは教師あり学習の場合はラベル情報を入力に埋め込んでマスクする手法が一般的。
        
        ここでは、教師なし/自己教師あり的なアプローチとして、
        バッチ内の入力をランダムにシャッフルしたものを「偽データ」として扱う簡易実装を行う。
        （より高度な実装では、入力画像に誤ったラベルのワンホットベクトルを重畳する）
        """
        # バッチサイズ方向の置換
        perm_indices = torch.randperm(inputs.size(0), device=inputs.device)
        return inputs[perm_indices]

    def _calculate_goodness(self, h: torch.Tensor) -> torch.Tensor:
        """
        Goodness (良さ) 関数。
        通常はニューロン活動の二乗和などを利用する。
        LIFニューロンの場合はスパイク率や膜電位のエネルギーを使用。
        shape: (batch_size, )
        """
        # 時間次元がある場合 (Batch, Time, Features) -> (Batch, Features)
        if h.dim() == 3:
            h = h.mean(dim=1)
        
        return h.pow(2).mean(dim=1)

    def _loss_fn(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """
        Forward-Forward 損失関数。
        Positiveデータに対するGoodnessは閾値より高く、
        Negativeデータに対するGoodnessは閾値より低くなるように学習する。
        
        Loss = log(1 + exp(-(G_pos - threshold))) + log(1 + exp(G_neg - threshold))
        """
        # log(1 + exp(x)) は softplus
        loss_pos = torch.nn.functional.softplus(-(pos_goodness - self.threshold))
        loss_neg = torch.nn.functional.softplus(neg_goodness - self.threshold)
        return (loss_pos + loss_neg).mean()

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Forward-Forward アルゴリズムによる1エポックの学習。
        """
        logger.info(f"Starting Forward-Forward training epoch {self.current_epoch}...")
        
        layers = self._get_layers()
        if len(layers) != len(self.layer_optimizers):
            # 動的変更に対応するため再セットアップ
            self.layer_optimizers = []
            self._setup_layer_optimizers()
        
        total_loss = 0.0
        batch_count = 0
        
        for batch in data_loader:
            if isinstance(batch, dict):
                 inputs = batch.get('input_ids', batch.get('input_images')) # type: ignore
                 targets = batch.get('labels') # type: ignore
            else:
                 inputs, targets = batch
            
            if inputs is None: continue
            
            # Negative Data の生成
            neg_inputs = self._generate_negative_data(inputs, targets)
            
            # 入力を層ごとに伝播させるための変数
            # 層間の勾配を切るため、detach() しながら渡していく
            pos_h = inputs
            neg_h = neg_inputs
            
            batch_loss = 0.0
            
            # --- Layer-wise Training Loop ---
            for i, layer in enumerate(layers):
                optimizer = self.layer_optimizers[i]
                
                # 学習対象でない層は単に通過させる
                if optimizer is None:
                    with torch.no_grad():
                        pos_h = layer(pos_h)
                        neg_h = layer(neg_h)
                        # 出力がdictやtupleの場合の処理 (SequentialSNNNetwork準拠)
                        if isinstance(pos_h, dict): pos_h = pos_h.get('activity', list(pos_h.values())[0])
                        if isinstance(neg_h, dict): neg_h = neg_h.get('activity', list(neg_h.values())[0])
                        if isinstance(pos_h, tuple): pos_h = pos_h[0]
                        if isinstance(neg_h, tuple): neg_h = neg_h[0]
                    continue
                
                # --- Forward Pass (Local) ---
                # Positive Pass
                pos_out = layer(pos_h)
                # Negative Pass
                neg_out = layer(neg_h)
                
                # 出力の正規化処理（次層への入力準備）
                # ここでは活動(activity)を取り出し、勾配計算用とは別に次層用にdetachする
                
                def extract_activity(out: Any) -> torch.Tensor:
                    if isinstance(out, dict): return out.get('activity', list(out.values())[0])
                    if isinstance(out, tuple): return out[0]
                    return out

                pos_act = extract_activity(pos_out)
                neg_act = extract_activity(neg_out)
                
                # --- Loss Calculation & Update ---
                pos_goodness = self._calculate_goodness(pos_act)
                neg_goodness = self._calculate_goodness(neg_act)
                
                loss = self._loss_fn(pos_goodness, neg_goodness)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                
                # --- Detach for next layer ---
                # 次の層へは勾配を伝播させない (FFの肝)
                pos_h = pos_act.detach()
                neg_h = neg_act.detach()
            
            total_loss += batch_loss
            batch_count += 1
            
        avg_loss = total_loss / max(1, batch_count)
        
        metrics = {'loss': avg_loss}
        logger.info(f"Forward-Forward Epoch {self.current_epoch} finished. Loss: {avg_loss:.4f}")
        
        if self.logger_client:
            self.logger_client.log({f"train/ff_loss": avg_loss}, step=self.current_epoch)
            
        self.current_epoch += 1
        return metrics
