"""会话文件存储。

每个会话一个目录 ``<data_dir>/<session_id>/``：

* ``config.json``  —— 阵列/更新/估计配置与元数据（原子覆盖写）；
* ``state.npz``    —— 增量协方差状态（C、窗口快拍、有效快拍数、已计入
  批次号集合等，写临时文件后 ``os.replace``，崩溃不撕裂）；
* ``results.jsonl``—— 每次估计一行 JSON，``version`` 从 1 单调递增，
  只追加。
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np


class StorageError(RuntimeError):
    """会话落盘/读取失败。"""


def _atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, obj: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        .encode("utf-8"),
    )


class FileStore:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    # ---- 目录/配置 ----
    def session_dir(self, session_id: str) -> str:
        self._validate_id(session_id)
        return os.path.join(self.base_dir, session_id)

    @staticmethod
    def _validate_id(session_id: str) -> None:
        if not session_id or not all(
            c.isalnum() or c in "-_" for c in session_id
        ) or len(session_id) > 128:
            raise StorageError(f"非法会话 ID：{session_id!r}")

    def ensure_session_dir(self, session_id: str) -> str:
        d = self.session_dir(session_id)
        os.makedirs(d, exist_ok=True)
        return d

    def list_sessions(self) -> list[str]:
        return [
            name for name in os.listdir(self.base_dir)
            if os.path.isfile(os.path.join(self.base_dir, name, "config.json"))
        ]

    def save_config(self, session_id: str, config: dict) -> None:
        _atomic_write_json(
            os.path.join(self.ensure_session_dir(session_id), "config.json"),
            config,
        )

    def load_config(self, session_id: str) -> dict:
        with open(os.path.join(self.session_dir(session_id), "config.json"),
                  "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- 增量状态 ----
    def save_state(self, session_id: str, state: dict) -> None:
        d = self.ensure_session_dir(session_id)
        scalars = {k: v for k, v in state.items()
                   if not isinstance(v, np.ndarray)}
        arrays = {k: v for k, v in state.items()
                  if isinstance(v, np.ndarray)}
        fd, tmp = tempfile.mkstemp(prefix=".tmp-state-", suffix=".npz", dir=d)
        os.close(fd)
        try:
            np.savez(tmp, _scalars=json.dumps(scalars), **arrays)
            os.replace(tmp, os.path.join(d, "state.npz"))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load_state(self, session_id: str) -> dict | None:
        path = os.path.join(self.session_dir(session_id), "state.npz")
        if not os.path.isfile(path):
            return None
        with np.load(path) as z:
            scalars = json.loads(str(z["_scalars"]))
            arrays = {k: z[k] for k in z.files if k != "_scalars"}
        scalars.update(arrays)
        return scalars

    # ---- 已计入批次号（与协方差状态同存于 state.npz，保证原子一致） ----

    # ---- 估计结果（只追加） ----
    def append_result(self, session_id: str, result: dict) -> None:
        path = os.path.join(self.ensure_session_dir(session_id), "results.jsonl")
        line = (json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                + "\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_results(self, session_id: str,
                     since_version: int = 0) -> list[dict]:
        path = os.path.join(self.session_dir(session_id), "results.jsonl")
        if not os.path.isfile(path):
            return []
        out: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec["version"]) > int(since_version):
                    out.append(rec)
        return out
