# ファイルパス: scripts/observe_brain_thought_process.py
#
# Title: 思考の観察（人工脳との対話）
#
# Description:
# 統合されたArtificialBrainが、多様な感情的テキスト入力に対し、
# どのように感じ、記憶し、意思決定するのか、その「思考プロセス」を
# 詳細に観察するための対話型スクリプト。
#
# 修正 (v2):
# - ArtificialBrainの属性変更に対応 (global_context -> workspace/modules)。
# - AttributeError を解消。

import sys
from pathlib import Path
import argparse

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import BrainContainer

def main():
    """
    DIコンテナを使って人工脳を初期化し、思考プロセスを観察しながら
    対話形式のシミュレーションを実行する。
    """
    parser = argparse.ArgumentParser(
        description="人工脳 思考プロセス観察ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/micro.yaml", # デフォルトをmicroに変更
        help="モデルアーキテクチャ設定ファイルのパス。"
    )
    args = parser.parse_args()

    # 1. DIコンテナを初期化し、設定ファイルをロード
    print("🏗️ システムを初期化しています...")
    container = BrainContainer()
    container.config.from_yaml("configs/templates/base_config.yaml")
    container.config.from_yaml(args.model_config)

    # 2. コンテナから完成品の人工脳インスタンスを取得
    brain = container.artificial_brain()

    # 3. 対話ループの開始
    print("\n" + "="*70)
    print("🧠 人工脳との対話を開始します。'exit' と入力すると終了します。")
    print("   喜び、怒り、悲しみなど感情豊かな文章や、複雑な質問を入力して、AIの思考を探ってみましょう。")
    print("="*70)

    try:
        while True:
            # ユーザーからの入力を受け付け
            print("\n" + "-"*30)
            input_text = input("あなた: ")
            if input_text.lower() == 'exit':
                break
            if not input_text:
                continue

            # --- 認知サイクルの実行 ---
            # この内部で print 出力が多数行われる
            brain.run_cognitive_cycle(input_text)

            # --- 思考プロセスの観察 (内部状態の確認) ---
            print("\n" + "="*20 + " 🔍 思考プロセスの事後分析 " + "="*20)
            
            # 1. 感情 (Amygdala) の状態
            # AmygdalaがWorkspaceにアップロードした情報を確認する
            amygdala_info = brain.workspace.get_information("amygdala")
            if amygdala_info:
                # 辞書形式かオブジェクトかを確認して表示
                print(f"❤️ 感情評価 (Amygdala output): {amygdala_info}")
            else:
                print("❤️ 感情評価: (今回は情動反応なし)")

            # 2. 意識 (Global Workspace) の内容
            conscious = brain.workspace.conscious_broadcast_content
            if conscious:
                source = conscious.get('source_module', 'Unknown')
                print(f"💡 意識に上った情報 (Consciousness): Source='{source}'")
            else:
                print("💡 意識: (意識レベルに達した情報はなし)")

            # 3. 意思決定 (Basal Ganglia)
            action = brain.basal_ganglia.selected_action
            if action:
                print(f"⚡ 決定された行動 (Basal Ganglia): '{action}'")
            else:
                print("⚡ 行動: (実行なし)")
            
            # 4. 短期記憶 (Hippocampus)
            # working_memoryへのアクセス (リストの長さを確認)
            wm_size = len(brain.hippocampus.working_memory)
            print(f"📖 短期記憶 (Hippocampus): 保持中のエピソード数 = {wm_size}")
            
            # 5. 記憶の固定化状況
            if brain.cycle_count % 5 == 0:
                print("\n💾 [Long-Term Memory Update]")
                print("   長期記憶への固定化が実行されました。")
                # Cortexの知識の一部を表示 (APIがあれば)
                if hasattr(brain.cortex, 'knowledge_graph'):
                     kg_size = len(brain.cortex.knowledge_graph)
                     print(f"   現在の知識グラフノード数: {kg_size}")

            print("="*64)


    except KeyboardInterrupt:
        print("\n👋 対話ループを終了しました。")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧠 システムをシャットダウンします。")


if __name__ == "__main__":
    main()
