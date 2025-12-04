# ファイルパス: snn_research/io/sensory_receptor.py
# (更新: 画像入力対応)
#
# Title: Sensory Receptor (感覚受容器)
#
# Description:
# - 人工脳アーキテクチャの「入力層」を担うコンポーネント。
# - 外部環境からの多様な感覚情報（テキスト、数値、画像）を受け取る。
# - 修正: 画像ファイルパスまたはPILオブジェクトを検出し、'image' タイプとして処理する機能を追加。

from typing import Dict, Any, Union
import os
try:
    from PIL import Image # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class SensoryReceptor:
    """
    外部からの感覚情報を受け取り、内部表現に変換するモジュール。
    """
    def __init__(self):
        print("👁️ 感覚受容器モジュールが初期化されました。")

    def receive(self, data: Union[str, float, Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        外部からのデータを受け取り、標準化された辞書形式で返す。

        Args:
            data: 入力される感覚データ。テキスト、数値、辞書、画像パス、またはPIL画像。

        Returns:
            Dict[str, Any]: 標準化された感覚情報。
                            例: {'type': 'text', 'content': 'hello'}
                                {'type': 'image', 'content': <PIL.Image>}
        """
        data_type = "unknown"
        content = data

        if isinstance(data, str):
            # 画像パスかどうかの簡易判定
            if PIL_AVAILABLE and os.path.exists(data) and data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    content = Image.open(data).convert('RGB')
                    data_type = "image"
                except Exception as e:
                    print(f"⚠️ 画像ロード失敗: {e}")
                    data_type = "text" # ロードできなければテキストとして扱う
            else:
                data_type = "text"
                
        elif isinstance(data, (int, float)):
            data_type = "numeric"
            
        elif isinstance(data, dict):
            data_type = data.get("type", "dict")
            content = data.get("content", data)
            
        elif PIL_AVAILABLE and isinstance(data, Image.Image):
            data_type = "image"

        print(f"📬 感覚受容器: '{data_type}' タイプの情報を受信しました。")
        return {"type": data_type, "content": content}