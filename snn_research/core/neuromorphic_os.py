# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic OS Kernel v1.0
# 目的: SNNハードウェア(ArtificialBrain)のリソース管理、タスクスケジューリング、システムコール処理を行うOS層の実装。

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, cast
from dataclasses import dataclass, field

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logger = logging.getLogger(__name__)


@dataclass
class ProcessControlBlock:
    """プロセス制御ブロック (PCB)"""
    pid: int
    name: str
    status: str  # "READY", "RUNNING", "WAITING", "TERMINATED"
    priority: int
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class NeuromorphicOS:
    """
    Neuromorphic Operating System Kernel.
    人工脳のリソースを管理し、上位アプリケーションからの要求を調整する。
    """

    def __init__(self, brain: ArtificialBrain):
        self.brain: ArtificialBrain = brain
        self.scheduler_tick = 0.01  # 10ms

        # プロセス管理
        self.process_table: Dict[int, ProcessControlBlock] = {}
        self.ready_queue: List[int] = []
        self.next_pid = 1
        self.current_pid: Optional[int] = None

        # カーネル状態
        self.is_running = False
        self.system_load = 0.0

        logger.info("🖥️ Neuromorphic OS Kernel initialized.")

    async def boot(self):
        """OSの起動シーケンス"""
        logger.info("🟢 Booting Neuromorphic OS...")
        self.is_running = True

        # 初期診断
        brain_instance = cast(ArtificialBrain, self.brain)
        status = brain_instance.get_brain_status()
        logger.info(f"   Hardware Check: {status['status']}")

        # アイドルプロセスの生成
        self.spawn_process("SystemIdle", priority=0)

        # メインループ開始
        await self.kernel_loop()

    def spawn_process(self, name: str, priority: int = 1) -> int:
        """新しい思考プロセスの生成"""
        pid = self.next_pid
        self.next_pid += 1

        pcb = ProcessControlBlock(
            pid=pid,
            name=name,
            status="READY",
            priority=priority
        )
        self.process_table[pid] = pcb
        self.ready_queue.append(pid)
        # 優先度順にソート
        self.ready_queue.sort(
            key=lambda x: self.process_table[x].priority, reverse=True)

        logger.info(f"   [OS] Process spawned: {name} (PID: {pid})")
        return pid

    async def kernel_loop(self):
        """カーネルのメインループ (スケジューラ)"""
        while self.is_running:
            # 1. スケジューリング
            if self.ready_queue:
                next_pid = self.ready_queue[0]  # 最も優先度の高いプロセス
                self._context_switch(next_pid)

            # 2. ハードウェアリソース監視 (Astrocyte連携)
            brain_instance = cast(ArtificialBrain, self.brain)
            if brain_instance.astrocyte:
                # 型チェックとキャスト
                astrocyte = cast(Any, brain_instance.astrocyte)
                if hasattr(astrocyte, 'get_energy_level'):
                    energy_level = astrocyte.get_energy_level()
                    if energy_level < 0.2:
                        logger.warning(
                            "   [OS] Critical Energy! Throttling processes...")
                        await asyncio.sleep(0.1)  # スローダウン

            # 3. 実行中のプロセスがある場合の処理 (シミュレーション)
            if self.current_pid:
                _ = self.process_table[self.current_pid]
                # ここでBrainにタスクを実行させる
                pass

            await asyncio.sleep(self.scheduler_tick)

    def _context_switch(self, target_pid: int):
        """コンテキストスイッチ"""
        if self.current_pid == target_pid:
            return

        prev_proc = self.process_table[self.current_pid] if self.current_pid else None
        next_proc = self.process_table[target_pid]

        if prev_proc:
            prev_proc.status = "READY"
            # 実際にはここでBrainの状態(短期記憶など)を退避する

        self.current_pid = target_pid
        next_proc.status = "RUNNING"

        # logger.debug(f"   [OS] Context Switch: {prev_proc.name if prev_proc else 'None'} -> {next_proc.name}")

    def shutdown(self):
        """システムシャットダウン"""
        logger.info("🔴 Shutting down Neuromorphic OS...")
        self.is_running = False

    # --- System Calls (API) ---

    async def sys_perceive_and_act(self, sensory_input: Any) -> Dict[str, Any]:
        """
        システムコール: 知覚と行動の実行
        OSがハードウェア(Brain)へのアクセスを仲介する。
        """
        if not self.is_running:
            return {"error": "OS not running"}

        brain_instance = cast(ArtificialBrain, self.brain)

        # 割り込み禁止などの排他制御がここに入る想定

        # 安全装置のチェック (Brain内部でも行われるが、OSレベルでも事前チェック可能)
        if isinstance(sensory_input, str):
            if hasattr(brain_instance, 'guardrail') and brain_instance.guardrail:
                guardrail = cast(Any, brain_instance.guardrail)
                safe, msg = guardrail.check_input(sensory_input)
                if not safe:
                    return {"status": "blocked", "reason": msg}

        # ハードウェア実行
        result = brain_instance.run_cognitive_cycle(sensory_input)

        return result

    async def sys_sleep(self):
        """システムコール: 睡眠モードへの移行"""
        logger.info("   [OS] System Call: SLEEP requested.")
        # 優先度の低いプロセスを一時停止するなどの処理
        brain_instance = cast(ArtificialBrain, self.brain)
        brain_instance.sleep_cycle()

    def sys_get_diagnostics(self) -> Dict[str, Any]:
        """システムコール: 診断情報の取得"""
        brain_instance = cast(ArtificialBrain, self.brain)
        brain_status = brain_instance.get_brain_status()
        os_status = {
            "running_processes": len(self.process_table),
            "current_pid": self.current_pid,
            "load": self.system_load
        }
        return {**brain_status, "os": os_status}
