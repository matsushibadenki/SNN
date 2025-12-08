# ファイルパス: snn_research/cognitive_architecture/neuromorphic_scheduler.py
# Title: Neuromorphic Scheduler (Phase 7 Brain OS Kernel)
# Description:
#   脳を「OS」と見なし、各認知モジュールを「プロセス」として管理するスケジューラ。
#   静的な制御フローではなく、各プロセスの「入札（Bid）」と「エネルギー残量」に基づく
#   動的な実行権の割り当て（調停）を行う。

import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import uuid

from .astrocyte_network import AstrocyteNetwork
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

@dataclass
class ProcessBid:
    """
    プロセスが提出する実行リクエスト（入札）。
    """
    process_name: str
    priority: float       # 実行の重要度 (Salience)
    energy_cost: float    # 実行に必要なエネルギー予測値
    intent: str           # 実行内容の概要

class BrainProcess:
    """
    認知モジュールをラップするプロセスクラス。
    OS上での実行単位となる。
    """
    def __init__(
        self, 
        name: str, 
        module_ref: Any, 
        bid_strategy: Callable[[Any, Dict[str, Any]], ProcessBid],
        execution_func: Callable[[Any, Any], Dict[str, Any]]
    ):
        self.pid = str(uuid.uuid4())[:8]
        self.name = name
        self.module = module_ref
        self._bid_strategy = bid_strategy
        self._execution_func = execution_func
        self.state = "IDLE" # IDLE, RUNNING, SUSPENDED

    def make_bid(self, sensory_input: Any, context: Dict[str, Any]) -> ProcessBid:
        """現在の状況に基づいて入札を行う"""
        return self._bid_strategy(self.module, sensory_input, context)

    def run(self, input_data: Any) -> Dict[str, Any]:
        """プロセスを実行する"""
        self.state = "RUNNING"
        try:
            result = self._execution_func(self.module, input_data)
            return result
        finally:
            self.state = "IDLE"

class NeuromorphicScheduler:
    """
    脳内プロセスの競合を調停するカーネルスケジューラ。
    AstrocyteNetwork (Resource) と GlobalWorkspace (Attention) を統合して判断する。
    """
    def __init__(self, astrocyte: AstrocyteNetwork, workspace: GlobalWorkspace):
        self.astrocyte = astrocyte
        self.workspace = workspace
        self.processes: Dict[str, BrainProcess] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        logger.info("🧠 Neuromorphic OS Scheduler initialized.")

    def register_process(self, process: BrainProcess):
        self.processes[process.name] = process
        logger.info(f"   - Registered process: {process.name} (PID: {process.pid})")

    def step(self, sensory_input: Any) -> Dict[str, Any]:
        """
        OSの1クロックサイクル。
        1. 全プロセスから入札を収集
        2. アストロサイトによるエネルギー審査
        3. ワークスペースによる注意フィルタリング
        4. 勝者プロセスの実行
        """
        # コンテキスト（現在の意識状態など）の取得
        context = {
            "consciousness": self.workspace.conscious_broadcast_content,
            "energy": self.astrocyte.current_energy,
            "fatigue": self.astrocyte.fatigue_toxin
        }
        
        # 1. 入札収集 (Bidding Phase)
        bids: List[ProcessBid] = []
        for proc in self.processes.values():
            try:
                # 柔軟な引数対応 (module, input, context などを想定)
                # 簡易化のため bid_strategy には input と context を渡す
                bid = proc.make_bid(sensory_input, context)
                if bid.priority > 0: # 実行意思がある場合のみ
                    bids.append(bid)
            except Exception as e:
                logger.error(f"Error collecting bid from {proc.name}: {e}")

        # 優先度でソート (降順)
        bids.sort(key=lambda x: x.priority, reverse=True)
        
        executed_processes = []
        denied_processes = []
        
        # 2. 調停と実行 (Arbitration & Execution Phase)
        # 高優先度のものから順に、リソースが許す限り実行
        for bid in bids:
            process = self.processes[bid.process_name]
            
            # アストロサイトにリソース許可を申請
            # 優先度が高いほど、エネルギーが少なくても許可されやすいロジックはAstrocyte側にあると想定
            # ここでは単純にリクエストを投げる
            allowed = self.astrocyte.request_resource(process.name, bid.energy_cost)
            
            if allowed:
                logger.info(f"   ▶️ EXEC: {process.name} (Pri:{bid.priority:.2f}, Cost:{bid.energy_cost:.1f})")
                
                # 実行
                try:
                    result = process.run(sensory_input)
                    
                    # 結果をWorkspaceへアップロード（プロセスの責任だが、ここではOSが代行しても良い）
                    # 簡易的に、結果に 'upload' キーがあればアップロード
                    if isinstance(result, dict) and "upload_content" in result:
                        self.workspace.upload_to_workspace(
                            source=process.name,
                            data=result["upload_content"],
                            salience=bid.priority # 優先度をそのまま顕著性として扱う
                        )
                        
                    executed_processes.append(process.name)
                except Exception as e:
                    logger.error(f"Error executing {process.name}: {e}")
            else:
                # リソース不足による却下
                logger.warning(f"   🛑 DENY: {process.name} (Low Energy / Low Priority)")
                denied_processes.append(process.name)

        # 3. アストロサイトの定常更新 (代謝など)
        self.astrocyte.step()
        
        # 4. 意識のブロードキャスト (GWT)
        self.workspace.conscious_broadcast_cycle()

        summary = {
            "executed": executed_processes,
            "denied": denied_processes,
            "energy": self.astrocyte.current_energy,
            "consciousness": self.workspace.conscious_broadcast_content
        }
        self.execution_log.append(summary)
        return summary