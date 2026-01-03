# ファイルパス: snn_research/io/sensory_receptor.py
# タイトル: 感覚受容器モジュール (Multimodal Receptor)
#
# 目的:
# - 人工脳の「入力層」として、外部環境からのデータを取り込む。
# - Phase 8/9 "Unified Perception" に対応し、画像・音声・テキスト・DVS等の
#   多様なモダリティを識別・前処理して Universal Spike Encoder へ渡す準備を行う。

from typing import Dict, Any, Union, Optional, Tuple
import os
import mimetypes

try:
    from PIL import Image # type: ignore
    import numpy as np # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ [SensoryReceptor] PIL/numpy not found. Image processing will be limited.")

class SensoryReceptor:
    """
    外部からの感覚情報を受け取り、内部表現に変換するモジュール。
    Universal Spike Encoderの前段として機能し、生データのロードとメタデータの付与を行う。
    """
    def __init__(self, default_image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            default_image_size: 画像読み込み時の標準リサイズサイズ。
        """
        self.default_image_size = default_image_size
        print("👁️ 感覚受容器モジュール (Multimodal Ready) が初期化されました。")

    def receive(self, data: Union[str, float, Dict[str, Any], Any], preprocess: bool = True) -> Dict[str, Any]:
        """
        外部からのデータを受け取り、標準化された辞書形式で返す。
        
        Args:
            data: 入力データ。パス(str)、数値、辞書、PIL画像など。
            preprocess: Trueの場合、基本的な前処理（画像リサイズ等）を行う。

        Returns:
            Dict[str, Any]: {
                'type': 'image' | 'text' | 'audio' | 'video' | 'dvs' | 'numeric' | ...,
                'content': <Loadable Object or Raw Data>,
                'metadata': { ... }
            }
        """
        data_type = "unknown"
        content = data
        # mypyが Dict[str, str] など狭い型に推論しないよう、Any型を明示する
        metadata: Dict[str, Any] = {}

        # 1. 文字列入力の処理 (パス判定 or テキスト)
        if isinstance(data, str):
            if os.path.exists(data):
                # ファイルパスとして処理
                mime_type, _ = mimetypes.guess_type(data)
                ext = os.path.splitext(data)[1].lower()
                metadata['file_path'] = data
                metadata['extension'] = ext
                
                # 画像判定
                if (mime_type and mime_type.startswith('image')) or ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
                    data_type = "image"
                    if PIL_AVAILABLE:
                        try:
                            img = Image.open(data).convert('RGB')
                            if preprocess:
                                img = self._preprocess_image(img)
                            content = img
                            metadata['size'] = img.size
                        except Exception as e:
                            print(f"⚠️ [SensoryReceptor] 画像ロード失敗: {e}")
                            data_type = "text" # エラー時はパス文字列そのものを扱う
                
                # 音声判定
                elif (mime_type and mime_type.startswith('audio')) or ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    data_type = "audio"
                    # Audio loading logic would go here (e.g. librosa)
                    # 今回はパスとメタデータのみ渡す
                    
                # 動画判定
                elif (mime_type and mime_type.startswith('video')) or ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    data_type = "video"
                    
                # DVS (Dynamic Vision Sensor) データ判定 (プロジェクト固有の拡張子など)
                elif ext in ['.aedat', '.aedat4', '.hdf5', '.evt']:
                    data_type = "dvs"
                    print(f"⚡️ [SensoryReceptor] Neuromorphic data detected: {os.path.basename(data)}")
                    
                else:
                    # その他のファイルはテキストとして読み込みを試みる、またはパスとして扱う
                    try:
                        with open(data, 'r', encoding='utf-8') as f:
                            content = f.read()
                        data_type = "text"
                        metadata['source'] = 'file'
                    except:
                        data_type = "text" # 内容を読めなければパス文字列
                        
            else:
                # ファイルではない文字列 -> テキスト入力
                data_type = "text"
                metadata['length'] = len(data)
                
        # 2. 数値入力
        elif isinstance(data, (int, float)):
            data_type = "numeric"
            
        # 3. 辞書入力 (すでに構造化されている場合)
        elif isinstance(data, dict):
            data_type = data.get("type", "dict")
            content = data.get("content", data)
            metadata = data.get("metadata", {})
            
        # 4. オブジェクト直接入力
        elif PIL_AVAILABLE and isinstance(data, Image.Image):
            data_type = "image"
            # mypyに対してcontentがImage型であることを保証するための変数を使用
            img_content: Image.Image = data
            if preprocess:
                img_content = self._preprocess_image(data)
            content = img_content
            metadata['size'] = img_content.size

        elif PIL_AVAILABLE and isinstance(data, np.ndarray):
            # Numpy配列 -> 画像またはその他信号とみなす
            if data.ndim in [2, 3]: # おそらく画像
                data_type = "image" # または 'tensor'
                try:
                    img_from_arr = Image.fromarray(data)
                    if preprocess:
                        content = self._preprocess_image(img_from_arr)
                    else:
                        content = img_from_arr
                except:
                    data_type = "tensor"
            else:
                data_type = "tensor"

        # ログ出力 (大量のデータの場合は省略などの工夫が必要)
        info = f"size={metadata.get('size')}" if 'size' in metadata else f"type={type(content)}"
        print(f"📬 [SensoryReceptor] Received '{data_type}': {info}")
        
        return {
            "type": data_type, 
            "content": content, 
            "metadata": metadata
        }

    def _preprocess_image(self, img: 'Image.Image') -> 'Image.Image':
        """画像の標準化処理"""
        # リサイズ
        if img.size != self.default_image_size:
            img = img.resize(self.default_image_size, Image.Resampling.LANCZOS)
        return img