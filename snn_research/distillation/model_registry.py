# -*- coding: utf-8 -*-
# ファイルパス: snn_research/distillation/model_registry.py
# (修正: 循環インポート回避)
#
# Title: モデルレジストリ
# Description:
# - モデルの登録、検索、管理、および動的インスタンス化を行う。
# - 修正: SNNCoreのトップレベルインポートを廃止し、遅延インポートに変更。

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, cast
import json
from pathlib import Path
import time
import shutil
import os
import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Windows対応: fcntl は Unix系のみ
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

# SNNCore のトップレベルインポートを削除
# from snn_research.core.snn_core import SNNCore

logger = logging.getLogger(__name__)


class ModelRegistry(ABC):
    """
    専門家モデルを管理するためのインターフェース。
    """
    registry_path: Optional[Path]

    @abstractmethod
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass

    @abstractmethod
    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """特定のタスクに最適なモデルを検索する。"""
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """モデルIDに基づいてモデル情報を取得する。"""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """登録されているすべてのモデルのリストを取得する。"""
        pass

    @staticmethod
    def get_model(model_config: DictConfig, load_weights: bool = False) -> nn.Module:
        """
        設定ファイル (DictConfig) に基づいてモデルのインスタンスを構築するファクトリメソッド。
        SNNCoreを使用してモデルを構築する。
        """
        # --- 修正: 遅延インポートで循環参照を回避 ---
        try:
            from snn_research.core.snn_core import SNNCore
        except ImportError:
            raise ImportError("Failed to import SNNCore in ModelRegistry.")
        # ----------------------------------------

        model_name = model_config.get("name", "unknown_model")
        architecture_type = model_config.get("architecture_type")

        if not architecture_type:
            raise ValueError(
                f"モデル設定 '{model_name}' に 'architecture_type' が指定されていません。")

        logger.info(
            f"Building model '{model_name}' (Type: {architecture_type}) using SNNCore...")

        # DictConfig を辞書に変換
        config_dict: Dict[str, Any]
        if isinstance(model_config, DictConfig):
            container = OmegaConf.to_container(model_config, resolve=True)
            if isinstance(container, dict):
                config_dict = cast(Dict[str, Any], container)
            else:
                logger.warning(
                    f"Model config converted to non-dict type: {type(container)}. Using empty dict.")
                config_dict = {}
        else:
            config_dict = dict(model_config)

        vocab_size_val = config_dict.get(
            "vocab_size", config_dict.get("num_classes", 1000))
        vocab_size = int(
            vocab_size_val) if vocab_size_val is not None else 1000

        try:
            model = SNNCore(config=config_dict, vocab_size=vocab_size)
        except Exception as e:
            logger.error(f"モデル '{model_name}' の構築に失敗しました: {e}")
            raise

        # 重みのロード
        if load_weights:
            model_path_str = model_config.get("path")
            if model_path_str:
                model_path = Path(model_path_str)
                if model_path.exists():
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']

                        missing, unexpected = model.load_state_dict(
                            state_dict, strict=False)

                        if missing:
                            if not any(k.startswith('model.') for k in missing):
                                logger.warning(
                                    f"Missing keys during load: {missing[:5]}...")
                        if unexpected:
                            logger.warning(
                                f"Unexpected keys during load: {unexpected[:5]}...")

                        logger.info(
                            f"モデル '{model_name}' の重みをロードしました: {model_path}")
                    except Exception as e:
                        logger.error(f"重みのロード中にエラーが発生しました ({model_path}): {e}")
                else:
                    logger.warning(f"重みファイルが見つかりません: {model_path}")
            else:
                logger.warning("モデル設定に 'path' がないため、重みをロードできません。")

        return model


class SimpleModelRegistry(ModelRegistry):
    """
    JSONファイルを使用したシンプルなモデルレジストリの実装。
    """

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = "workspace/runs/model_registry.json"
            logger.debug(
                f"Registry path not provided. Using default: {registry_path}")

        self.registry_path = Path(registry_path)
        self.project_root = self.registry_path.resolve().parent.parent
        self.models: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.registry_path and self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save(self) -> None:
        if self.registry_path:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.registry_path.with_suffix(
                f"{self.registry_path.suffix}.tmp")
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.models, f, indent=4, ensure_ascii=False)
                os.replace(temp_path, self.registry_path)
            except Exception as e:
                logger.error(f"Failed to save registry: {e}")
                if temp_path.exists():
                    os.remove(temp_path)

    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        if task_description not in self.models:
            self.models[task_description] = []

        model_info = {
            "model_id": model_id,
            "model_path": model_path,
            "metrics": metrics,
            "config": config,
            "task_description": task_description,
            "registration_date": time.time()
        }

        self.models[task_description].insert(0, model_info)
        self._save()
        logger.info(
            f"Registered model '{model_id}' for task '{task_description}'.")

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if task_description in self.models:
            models_for_task = self.models[task_description]
            models_for_task.sort(
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True
            )

            resolved_models = []
            for model_info in models_for_task[:top_k]:
                # 絶対パス解決
                relative_path_str = model_info.get(
                    'model_path') or model_info.get('path')
                if relative_path_str:
                    try:
                        absolute_path = Path(relative_path_str).resolve()
                        model_info['model_path'] = str(absolute_path)
                    except Exception:
                        pass
                resolved_models.append(model_info)

            return resolved_models
        return []

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        for task_models in self.models.values():
            for model in task_models:
                if model.get('model_id') == model_id:
                    return model
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for task_description, model_list in self.models.items():
            for model_info in model_list:
                model_info['task_description'] = task_description
                all_models.append(model_info)
        return all_models


class DistributedModelRegistry(SimpleModelRegistry):
    """
    ファイルロックを使用して、複数のプロセスからの安全なアクセスを保証する
    分散環境向けのモデルレジストリ。
    """

    def __init__(self, registry_path: Optional[str] = None, timeout: int = 10, shared_skill_dir: str = "workspace/runs/shared_skills"):
        super().__init__(registry_path)
        self.timeout = timeout
        self.shared_skill_dir = Path(shared_skill_dir)
        self.shared_skill_dir.mkdir(parents=True, exist_ok=True)

        if not FCNTL_AVAILABLE:
            logger.warning(
                "DistributedModelRegistry: 'fcntl' module not found. File locking is disabled (not safe for concurrent writes).")

    def _execute_with_lock(self, mode: str, operation, *args, **kwargs) -> Any:
        """ファイルロックを取得して操作を実行する (Windows対応版)。"""
        if self.registry_path is None:
            raise ValueError("Registry path is not set.")

        # Windowsなどfcntlがない環境ではロックをスキップ
        if not FCNTL_AVAILABLE:
            if mode == 'w':
                # 書き込みモードでもロックなしで実行
                # 読み込みが必要なら読む
                if os.path.exists(self.registry_path):
                    with open(self.registry_path, 'r', encoding='utf-8') as f:
                        try:
                            pass
                        except Exception:
                            pass

                # 書き込み用ファイルを開く
                with open(self.registry_path, 'w', encoding='utf-8') as f:
                    return operation(f, *args, **kwargs)
            else:
                # 読み込みモード
                if not os.path.exists(self.registry_path):
                    return operation(None, *args, **kwargs)
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return operation(f, *args, **kwargs)

        start_time = time.time()
        # 'a+' で開くことでファイルの作成も兼ねる
        with open(self.registry_path, 'a+', encoding='utf-8') as f:
            while time.time() - start_time < self.timeout:
                try:
                    lock_type = fcntl.LOCK_EX if mode == 'w' else fcntl.LOCK_SH
                    fcntl.flock(f, lock_type | fcntl.LOCK_NB)

                    # 読み書き位置を先頭に
                    f.seek(0)
                    result = operation(f, *args, **kwargs)

                    fcntl.flock(f, fcntl.LOCK_UN)
                    return result
                except (IOError, BlockingIOError):
                    time.sleep(0.1)

        raise IOError(
            f"レジストリの{'書き込み' if mode == 'w' else '読み取り'}ロックの取得に失敗しました (Timeout)。")

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        """ロックを取得してレジストリファイルを読み込む。"""
        if self.registry_path is None or not self.registry_path.exists():
            return {}

        def read_operation(f) -> Dict[str, List[Dict[str, Any]]]:
            if f is None:
                return {}  # ファイルがない場合
            try:
                content = f.read()
                if not content:
                    return {}
                return json.loads(content)
            except json.JSONDecodeError:
                return {}

        return self._execute_with_lock('r', read_operation)

    def _save(self) -> None:
        """ロックを取得してレジストリファイルに書き込む。"""
        if self.registry_path is None:
            return

        models_to_save = self.models

        def write_operation(f, models_data):
            # fcntlがない場合の単純書き込み (上の _execute_with_lock で 'w' モードなら f は 'w' で開かれている)
            if not FCNTL_AVAILABLE:
                json.dump(models_data, f, indent=4, ensure_ascii=False)
                return

            # ロックありの場合の処理
            # 1. アトミック書き込み
            temp_path = self.registry_path.with_suffix(
                f"{self.registry_path.suffix}.tmp")  # type: ignore
            try:
                with open(temp_path, 'w', encoding='utf-8') as temp_f:
                    json.dump(models_data, temp_f,
                              indent=4, ensure_ascii=False)
                os.replace(temp_path, self.registry_path)
            except Exception as e:
                logger.error(f"Atomic save failed: {e}")
                if temp_path.exists():
                    os.remove(temp_path)

            # 2. ロック中のファイルディスクリプタの内容も更新
            f.seek(0)
            f.truncate()
            json.dump(models_data, f, indent=4, ensure_ascii=False)

        self._execute_with_lock('w', write_operation, models_to_save)

    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        self.models = self._load()
        await super().register_model(model_id, task_description, metrics, model_path, config)

    async def publish_skill(self, model_id: str) -> bool:
        """学習済みモデル（スキル）を共有ディレクトリに公開する。"""
        self.models = self._load()
        target_info = await self.get_model_info(model_id)

        if not target_info:
            return False

        src_path_str = target_info.get('model_path')
        if not src_path_str:
            return False

        src_path = Path(src_path_str)
        if not src_path.exists():
            return False

        dest_path = self.shared_skill_dir / f"{model_id}.pth"
        try:
            shutil.copy(src_path, dest_path)
        except Exception as e:
            logger.error(f"コピー失敗: {e}")
            return False

        target_info['published'] = True
        target_info['shared_path'] = str(dest_path)

        task_desc = target_info.get('task_description')
        if task_desc and task_desc in self.models:
            for idx, m in enumerate(self.models[task_desc]):
                if m.get('model_id') == model_id:
                    self.models[task_desc][idx] = target_info
                    break

        self._save()
        logger.info(f"スキル '{model_id}' を公開しました: {dest_path}")
        return True

    async def download_skill(self, model_id: str, destination_dir: str) -> Optional[Dict[str, Any]]:
        """共有ディレクトリからスキルをダウンロードする。"""
        self.models = self._load()

        target_skill = None
        for tasks in self.models.values():
            for m in tasks:
                if m.get('model_id') == model_id and m.get('published'):
                    target_skill = m
                    break
            if target_skill:
                break

        if not target_skill or not target_skill.get('shared_path'):
            return None

        src_path = Path(target_skill['shared_path'])  # type: ignore
        if not src_path.exists():
            return None

        dest_dir_path = Path(destination_dir)
        dest_dir_path.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir_path / f"{model_id}.pth"

        shutil.copy(src_path, dest_path)

        new_info = target_skill.copy()
        new_info['model_path'] = str(dest_path)
        del new_info['shared_path']
        del new_info['published']

        await self.register_model(
            model_id=model_id,
            task_description=new_info.get(
                'task_description', 'imported_skill'),
            metrics=new_info.get('metrics', {}),
            model_path=str(dest_path),
            config=new_info.get('config', {})
        )

        return new_info
