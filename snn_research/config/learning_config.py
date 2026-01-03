# ファイルパス: snn_research/config/learning_config.py
# タイトル: 学習規則ハイパーパラメータ設定
# 機能説明: 
#   Project SNN4のロードマップ (Phase 1, P1-3) に基づき、
#   学習規則 (LearningRule) のハイパーパラメータを管理するための
#   設定クラスを定義します。
#   dataclasses を使用して型安全性を確保し、ハイパーパラメータ感度分析の
#   実験設定を容易にします。

from dataclasses import dataclass, field
# 修正: Iterable をインポート
from typing import Dict, Any, Optional, Iterable

@dataclass(frozen=True)
class BaseLearningConfig:
    """
    全ての学習規則に共通する基本ハイパーパラメータ。
    (P1-4 AbstractLearningRule に対応)
    
    Args:
        learning_rate (float): 基本となる学習率。
    """
    learning_rate: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """学習規則のコンストラクタに渡すための辞書を返します。"""
        return {"learning_rate": self.learning_rate}


@dataclass(frozen=True)
class PredictiveCodingConfig(BaseLearningConfig):
    """
    予測符号化 (PredictiveCodingRule) 固有のハイパーパラメータ。
    (P1-1, P1-2 の実装に対応)

    Args:
        learning_rate (float): 基本学習率 (継承)。
        error_weight (float): 予測誤差の重み更新への寄与度。
                              P1-3 の感度分析の主要ターゲットの一つ。
    """
    error_weight: float = 1.0

    # (mypy) 継承した learning_rate も含めて辞書を返すようにオーバーライド
    def to_dict(self) -> Dict[str, Any]:
        """学習規則のコンストラクタに渡すための辞書を返します。"""
        return {
            "learning_rate": self.learning_rate,
            "error_weight": self.error_weight
        }

# --- デモ: P1-3 の感度分析でこの Config をどう使うか ---

# 修正:
# ダミーのクラス定義を関数の外（グローバルスコープ）に移動
# これにより、if __name__ == '__main__' ブロックからも参照可能になる

# (ダミーのインポートと型定義)
# from ..core.learning_rule import AbstractLearningRule, Parameters
# from ..core.learning_rules.predictive_coding_rule import PredictiveCodingRule

# (mypy --strict のためのダミー実装)
class AbstractLearningRule:
    """ (デモ用のダミークラス) """
    def __init__(self, params: Iterable[Any], **kwargs: Any) -> None:
        print(f"Rule created with params: {kwargs}")
        
class PredictiveCodingRule(AbstractLearningRule): # type: ignore[misc]
    """ (デモ用のダミークラス) """
    pass
    
# -----------------------------------------------

def create_learning_rule(
    config: BaseLearningConfig,
    params: Iterable[Any] # (ダミー: 本来は Parameters[Tensor])
) -> AbstractLearningRule: # 修正: 戻り値の型ヒントを Any から変更
    """
    P1-3 の実験ループなどで使用するファクトリ関数の例。
    
    Config オブジェクトを受け取り、対応する学習規則をインスタンス化する。
    """
    
    rule_kwargs: Dict[str, Any] = config.to_dict()

    if isinstance(config, PredictiveCodingConfig):
        print(f"Creating PredictiveCodingRule with config: {config}")
        # (mypy) **rule_kwargs は型安全
        return PredictiveCodingRule(params, **rule_kwargs)
    
    elif isinstance(config, BaseLearningConfig):
        # (将来的に他のルールが追加された場合)
        print(f"Creating Base (default) Rule with config: {config}")
        # (ダミー: Baseは抽象なので実際はインスタンス化不可)
        # return SomeOtherRule(params, **rule_kwargs)
        raise TypeError(f"Config type {type(config)} not fully supported yet.")
    
    else:
        raise TypeError(f"Unknown config type: {type(config)}")


if __name__ == '__main__':
    # P1-3 ハイパーパラメータ感度分析のシミュレーション
    
    # 1. デフォルト設定
    default_pc_config: PredictiveCodingConfig = PredictiveCodingConfig()
    
    # 2. 感度分析用の設定 (学習率を変更)
    high_lr_config: PredictiveCodingConfig = PredictiveCodingConfig(
        learning_rate=0.1, 
        error_weight=1.0
    )
    
    # 3. 感度分析用の設定 (誤差の重みを変更)
    low_error_config: PredictiveCodingConfig = PredictiveCodingConfig(
        learning_rate=0.01, 
        error_weight=0.5
    )
    
    # 修正: Iterable エラーは 'typing' からのインポートで解決済み
    dummy_params: Iterable[float] = [0.1, 0.2] # ダミーのモデルパラメータ

    print("--- P1-3 Hyperparameter Sensitivity Test ---")
    
    # (mypy) config オブジェクトを渡すだけでよくなり、型安全
    
    # 修正: AbstractLearningRule はグローバルスコープのダミークラスを参照
    rule_default: AbstractLearningRule = create_learning_rule(
        default_pc_config, dummy_params
    )
    
    rule_high_lr: AbstractLearningRule = create_learning_rule(
        high_lr_config, dummy_params
    )
    
    rule_low_error: AbstractLearningRule = create_learning_rule(
        low_error_config, dummy_params
    )