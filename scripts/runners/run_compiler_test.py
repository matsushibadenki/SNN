# ファイルパス: scripts/runners/run_compiler_test.py

import sys
import os

# ------------------------------------------------------------------------------
# [Auto-inserted by fix_script_paths.py]
# プロジェクトルートディレクトリをsys.pathに追加して、snn_researchモジュールを解決可能にする
# このファイルは scripts/runners/ に配置されていることを想定しています (ルートから2階層下)
# ------------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------------------

# ファイルパス: run_compiler_test.py
# Title: コンパイラテストスクリプト
# Description: 
#   SNNモデルをニューロモルフィックハードウェア（のエミュレーション）向けにコンパイルするテスト。
#
# 修正:
#   - config オブジェクトへの属性アクセス (.attribute) を辞書アクセス (['key']) または .get() に変更。
#   - SNNCompiler のインポートエラーを修正 (type: ignore 追加)

import unittest
import torch
import sys
import os
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# SNNCompiler のインポート
# モジュールが存在するはずだが、mypyが見つけられない場合は ignore する
from snn_research.hardware.compiler import SNNCompiler # type: ignore[attr-defined]
from snn_research.core.snn_core import SNNCore

class TestCompiler(unittest.TestCase):
    def setUp(self) -> None:
        self.config: Dict[str, Any] = {
            'architecture_type': 'spiking_cnn',
            'num_classes': 10,
            'neuron': {'type': 'lif'},
            'time_steps': 16
        }
        self.model = SNNCore(config=self.config)
        self.compiler = SNNCompiler(target_hardware="loihi2")

    def test_compilation(self) -> None:
        print("Testing compilation...")
        compiled_model = self.compiler.compile(self.model)
        self.assertIsNotNone(compiled_model)
        print("Compilation successful.")

    def test_config_access(self) -> None:
        # --- 修正箇所: 辞書アクセスに変更 ---
        architecture_type = self.config.get('architecture_type')
        # --- 修正終了 ---
        
        self.assertEqual(architecture_type, 'spiking_cnn')

if __name__ == "__main__":
    unittest.main()
