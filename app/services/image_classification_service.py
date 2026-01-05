# ファイルパス: app/services/image_classification_service.py
# (動的ロードUI対応 v4)
# Title: 画像分類サービス
# Description:
# - GradioのImageコンポーネントからの入力を受け取り、
#   画像分類SNNモデル（SpikingCNNなど）で推論を実行する。
# - SNNInferenceEngine（動的ロード）に依存。
# - 'Union' の NameError を修正。
# - CIFAR-10のクラス名をハードコード。
#
# 修正 (v5):
# - mypyエラー [import-untyped] [index] を解消。

import numpy as np
from PIL import Image
# --- ▼ 修正: Union をインポート ▼ ---
from typing import Dict, Union
# --- ▲ 修正 ▲ ---
from torchvision import transforms  # type: ignore[import-untyped]
import torch

from app.deployment import SNNInferenceEngine

class ImageClassificationService:
    """
    画像分類SNNモデルの推論を実行し、Gradioに適した形式で結果を返すサービス。
    """
    # CIFAR-10のクラス名
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, engine: SNNInferenceEngine):
        """
        Args:
            engine (SNNInferenceEngine):
                画像分類モデル（例: SpikingCNN）をロード済みの推論エンジン。
        """
        self.snn_engine = engine
        self.model = engine.model # SNNCoreラッパー
        
        # モデルのアーキテクチャタイプを取得
        self.architecture_type = engine.config.get("model", {}).get("architecture_type", "unknown")
        
        # モデルに適した画像前処理を定義
        self.transform = transforms.Compose([
            # SpikingCNN (snn_core.py) は 224x224 を想定している
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # CIFAR-10学習時の正規化 (一般的)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, image_input: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Gradioから受け取った画像で分類を実行し、ラベル辞書を返す。

        Args:
            image_input (Union[np.ndarray, Image.Image]): GradioのImageコンポーネントからの入力。

        Returns:
            Dict[str, float]: GradioのLabelコンポーネント用の辞書 (例: {'cat': 0.8, 'dog': 0.2})
        """
        if image_input is None:
            return {"Error": 1.0, "No image provided": 0.0}

        # 1. PIL Imageに変換 (Gradioがndarrayを渡す場合があるため)
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
            
        # 2. 前処理
        # (B, C, H, W) の形状にする
        input_tensor = self.transform(image).unsqueeze(0).to(self.snn_engine.device)
        
        # 3. モデルタイプに応じて入力キーを決定
        if self.architecture_type == "spiking_cnn":
            model_input_key = "input_images"
        else:
            # 他のモデルタイプ（例: Hybrid）も "input_images" を期待すると仮定
            model_input_key = "input_images"
            
        model_kwargs = {
            model_input_key: input_tensor,
            "return_spikes": True # スパイク数も取得
        }

        # 4. 推論
        with torch.no_grad():
            outputs = self.model(**model_kwargs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # スパイク数も取得 (オプション)
            # avg_spikes = outputs[1].item() if isinstance(outputs, tuple) and len(outputs) > 1 else 0.0

        # 5. 結果のフォーマット
        probabilities = torch.softmax(logits.squeeze(), dim=0)
        
        # 上位3クラスを取得
        topk_probs, topk_indices = torch.topk(probabilities, 3)

        # GradioのLabel形式 (Dict[str, float]) に変換
        results: Dict[str, float] = {}
        for i in range(topk_probs.size(0)):
            # --- ▼ 修正: mypy [index] ▼ ---
            class_idx_int = int(topk_indices[i].item())
            class_name = self.CIFAR10_CLASSES[class_idx_int] if 0 <= class_idx_int < len(self.CIFAR10_CLASSES) else f"Class_{class_idx_int}"
            # --- ▲ 修正 ▲ ---
            prob = topk_probs[i].item()
            results[class_name] = prob
            
        return results