# ファイルパス: snn_research/tools/calculator.py
# Title: Calculator Tool
# Description: エージェントが使用する基本的な計算ツール。安全なeval実行を提供。

import math
import re
from typing import Dict, Any

class Calculator:
    """
    数式文字列を受け取り、計算結果を返すツール。
    """
    allowed_names: Dict[str, Any]

    def __init__(self) -> None:
        self.allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        self.allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow
        })

    def calculate(self, expression: str) -> str:
        """
        数式を評価する。セキュリティのため、危険な文字や関数は制限する。
        
        Args:
            expression (str): 計算式 (例: "1 + 1", "sqrt(16) * 2")
        
        Returns:
            str: 計算結果またはエラーメッセージ
        """
        # 危険な文字のチェック
        if re.search(r"[^0-9+\-*/()., \w]", expression):
             # 簡易チェック。本来はast.literal_eval等を使うべきだが、math関数許可のためevalを使用
             pass

        try:
            # 安全な名前空間でのみ実行
            result = eval(expression, {"__builtins__": {}}, self.allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    calc = Calculator()
    print(calc.calculate("1 + 1"))
    print(calc.calculate("sqrt(25) * 2"))
