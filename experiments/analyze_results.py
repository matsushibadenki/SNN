# ファイルパス: experiments/analyze_results.py
# タイトル: P5-5 実験結果の分析と可視化
# 機能説明: 
#   Project SNN4のロードマップ (Phase 5, P5-5) に基づき、
#   P5-4 (Optuna HPO) で実行された実験結果を分析・可視化します。
#   
#   P5-5 の主な機能:
#   1. 'optuna' と 'plotly' (可視化バックエンド) をインポートします。
#   2. P5-4 で生成されたと仮定される Optuna のストレージ
#      (例: 'sqlite:///hpo_study.db') から 'study' をロードします。
#   3. 'optuna.visualization' を使用して、最適化履歴や
#      パラメータ重要度をプロットし、HTML ファイルとして保存します。
#   4. P5-3 (TensorBoard) のログを分析するためのコマンドライン例を示します。

import sys
import os
import logging
from typing import Any, Optional

# (mypy) 'snn_research' パッケージをパスに追加 (P5-4 と同様)
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- P5-4: Optuna ---
OPTUNA_AVAILABLE: bool = False
try:
    import optuna  # type: ignore[import-not-found]
    OPTUNA_AVAILABLE = True
except ImportError:
    pass

# --- P5-5: Plotly (Optuna 可視化に必要) ---
PLOTLY_AVAILABLE: bool = False
if OPTUNA_AVAILABLE:
    try:
        # Optuna の可視化バックエンド (plotly) が利用可能かチェック
        # (mypy) optuna がない場合 [attr-defined]
        if optuna.visualization.is_available(): # type: ignore[attr-defined]
            import plotly.io as pio # type: ignore[import-not-found]
            PLOTLY_AVAILABLE = True
    except ImportError:
        pass

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("P5-5_Analysis_Script")

def analyze_hpo_results(
    study_name: str = "snn4_hpo_study",
    storage_url: str = "sqlite:///hpo_study.db",
    output_dir: str = "hpo_analysis"
) -> None:
    """
    P5-5: HPO (Optuna) の結果をロードして可視化します。
    """
    
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not found. 'pip install optuna'")
        return
        
    if not PLOTLY_AVAILABLE:
        logger.error(
            "Plotly not found. 'pip install plotly' "
            "is required for Optuna visualization."
        )
        return

    logger.info(f"Loading Optuna study '{study_name}' from '{storage_url}'...")
    
    try:
        # (mypy) optuna がない場合 [attr-defined]
        study: optuna.Study = optuna.load_study( # type: ignore[attr-defined]
            study_name=study_name, 
            storage=storage_url
        )
    except ImportError:
        # (sqlite3 がない場合など)
        logger.error(f"Could not load study. Ensure database driver is installed.")
        return
    except Exception as e:
        logger.error(f"Failed to load study '{study_name}'.")
        logger.error(
            "Did 'run_optuna_hpo.py' (P5-4) run successfully and "
            "was it configured to use this storage URL?"
        )
        logger.error(f"(Error: {e})")
        return

    logger.info(f"Study loaded. Number of trials: {len(study.trials)}")
    if not study.trials:
        logger.warning("No trials found in study. Skipping analysis.")
        return
        
    logger.info(f"Best trial (value): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # --- P5-5: 可視化 & 保存 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 最適化履歴
    try:
        # (mypy) optuna がない場合 [attr-defined]
        fig_history = optuna.visualization.plot_optimization_history(study) # type: ignore[attr-defined]
        output_path: str = os.path.join(
            output_dir, "1_optimization_history.html"
        )
        fig_history.write_html(output_path)
        logger.info(f"Saved optimization history plot to {output_path}")

    except Exception as e:
        logger.warning(f"Failed to plot optimization history: {e}")

    # 2. パラメータ重要度
    try:
        # (mypy) optuna がない場合 [attr-defined]
        fig_importance = optuna.visualization.plot_param_importances(study) # type: ignore[attr-defined]
        output_path = os.path.join(
            output_dir, "2_param_importances.html"
        )
        fig_importance.write_html(output_path)
        logger.info(f"Saved parameter importance plot to {output_path}")
        
    except ValueError as e:
        # (トライアル数が少ないと重要度を計算できない場合がある)
        logger.warning(f"Could not calculate parameter importance: {e}")
    except Exception as e:
        logger.warning(f"Failed to plot parameter importance: {e}")

    logger.info("HPO analysis finished.")

def show_tensorboard_instructions() -> None:
    """
    P5-3: TensorBoard のログを分析する方法 (コマンド) を表示します。
    """
    logger.info("--- Analyzing TensorBoard Logs (P5-3) ---")
    logger.info(
        "To analyze results logged by TensorBoardLogger (P5-3) or "
        "Optuna HPO (P5-4, if TensorBoard was used):"
    )
    logger.info("\n1. Open your terminal.")
    logger.info("2. Run the following command in the project root directory:")
    logger.info("\n  tensorboard --logdir=runs\n")
    logger.info("3. Open the URL provided in your browser (e.g., http://localhost:6006/).")
    logger.info("---------------------------------------------")


if __name__ == "__main__":
    # P5-4 (Optuna) の結果を分析
    analyze_hpo_results(
        # (P5-4 のスクリプトがこの設定で実行されたと仮定)
        study_name="snn4_hpo_study",
        storage_url="sqlite:///hpo_study.db"
    )
    
    print("\n" + "="*50 + "\n")
    
    # P5-3 (TensorBoard) の分析方法を表示
    show_tensorboard_instructions()