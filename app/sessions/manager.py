"""实验会话管理器：创建、增量追加、估计、持久化恢复、轮询与订阅。

并发语义（单进程多线程，FastAPI sync 路由跑在线程池里）：

* 每个会话一把 ``threading.RLock``，同会话的追加/估计串行化。两批并发
  追加的最终状态严格等价于按某一先后顺序串行追加；
* 追加请求带 ``batch_id``，已计入的批次重复提交直接幂等返回，不丢批
  也不重复计入。批次号集合与协方差状态在同一个 ``state.npz`` 里原子
  落盘；
* 不同会话互不影响。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from ..doa.covariance import (
    Accumulator,
    accumulator_from_state,
    build_accumulator,
)
from ..doa.engine import (
    EstimateConfig,
    EstimationError,
    MAX_ELEMENTS,
    MAX_SNAPSHOTS_PER_BATCH,
    run_doa,
)
from ..doa.geometry import GeometryError, ULA
from .models import (
    ArraySpec,
    EstimateSpec,
    SessionMeta,
    UpdateSpec,
)
from .pubsub import PubSub
from .storage import FileStore

# 估计结果里不进 jsonl 的调试重字段
_NON_PERSISTED_KEYS = ("_spectrum",)


class SessionNotFound(KeyError):
    """会话不存在。"""


class Session:
    def __init__(self, meta: SessionMeta, store: FileStore):
        self.meta = meta
        self._store = store
        self.lock = threading.RLock()
        self.array: ULA = meta.array.build()
        self.acc: Accumulator | None = None
        self.batch_ids: set[str] = set()

    # ---- 状态装载/恢复 ----
    def _load_state_if_needed(self) -> None:
        if self.acc is not None:
            return
        state = self._store.load_state(self.meta.session_id)
        if state is None:
            self.acc = build_accumulator(
                self.meta.array.m,
                self.meta.update.method,
                self.meta.update.window,
                self.meta.update.alpha,
            )
            self.batch_ids = set()
        else:
            self.acc = accumulator_from_state(state)
            self.batch_ids = set(state.get("_batch_ids", []).tolist()) \
                if isinstance(state.get("_batch_ids"), np.ndarray) \
                else set(state.get("_batch_ids", []))

    def _persist_state(self) -> None:
        state = self.acc.to_state()
        state["_batch_ids"] = np.asarray(sorted(self.batch_ids))
        self._store.save_state(self.meta.session_id, state)

    # ---- 追加 ----
    def append(self, batch_id: str, X: np.ndarray) -> dict:
        sid = self.meta.session_id
        with self.lock:
            self._load_state_if_needed()
            if batch_id in self.batch_ids:
                return {
                    "session_id": sid,
                    "batch_id": batch_id,
                    "deduplicated": True,
                    "accepted_snapshots": 0,
                    "total_snapshots": self.acc.total_snapshots,
                    "n_effective_snapshots": float(self.acc.n_eff),
                }
            accepted = self.acc.add_batch(X)
            self.batch_ids.add(batch_id)
            self._persist_state()
            self.meta.updated_at = time.time()
            self._store.save_config(self.meta.session_id,
                                   self.meta.to_dict())
            return {
                "session_id": sid,
                "batch_id": batch_id,
                "deduplicated": False,
                "accepted_snapshots": int(accepted),
                "total_snapshots": self.acc.total_snapshots,
                "n_effective_snapshots": float(self.acc.n_eff),
            }

    # ---- 估计 ----
    def estimate(self, overrides: dict | None = None,
                 true_angles: list[float] | None = None,
                 include_spectrum: bool = False) -> dict:
        with self.lock:
            self._load_state_if_needed()
            m = self.meta.array.m
            if self.acc.total_snapshots < m:
                raise EstimationError(
                    f"第一次出估计时累计快拍数为 {self.acc.total_snapshots}，"
                    f"还没有阵元数 {m} 多；协方差秩不足，"
                    f"请至少累计 {m} 条快拍后再估计。"
                )
            cfg = self._build_config(overrides or {})
            result = run_doa(
                self.array, self.acc.C, float(self.acc.n_eff),
                cfg, true_angles=true_angles,
            )
            version = len(self._store.read_results(self.meta.session_id)) + 1
            persisted = {
                k: v for k, v in result.items()
                if k not in _NON_PERSISTED_KEYS
            }
            rec = {
                "version": version,
                "session_id": self.meta.session_id,
                "created_at": time.time(),
                "batch_count": len(self.batch_ids),
                "total_snapshots": self.acc.total_snapshots,
                "config_used": {
                    "source_mode": _jsonable(cfg.source_mode),
                    "angle_min": cfg.angle_min,
                    "angle_max": cfg.angle_max,
                    "step_deg": cfg.step_deg,
                    "smoothing": cfg.smoothing,
                    "sub_len": cfg.sub_len,
                },
                "result": persisted,
            }
            self._store.append_result(self.meta.session_id, rec)
            self.meta.updated_at = time.time()
            self._store.save_config(self.meta.session_id,
                                   self.meta.to_dict())
            if include_spectrum:
                rec["result"]["spectrum"] = result["_spectrum"]
            return rec

    def _build_config(self, overrides: dict) -> EstimateConfig:
        base = self.meta.estimate
        merged = {
            "source_mode": overrides.get("source_mode", base.source_mode),
            "angle_min": overrides.get("angle_min", base.angle_min),
            "angle_max": overrides.get("angle_max", base.angle_max),
            "step_deg": overrides.get("step_deg", base.step_deg),
            "smoothing": overrides.get("smoothing", base.smoothing),
            "sub_len": overrides.get("sub_len", base.sub_len),
        }
        return EstimateConfig(**merged)

    # ---- 读取 ----
    def results_since(self, since_version: int = 0) -> list[dict]:
        return self._store.read_results(self.meta.session_id, since_version)

    def latest(self) -> dict | None:
        all_recs = self._store.read_results(self.meta.session_id, 0)
        return all_recs[-1] if all_recs else None

    def status(self) -> dict:
        with self.lock:
            self._load_state_if_needed()
            latest = self.latest()
            return {
                **self.meta.to_dict(),
                "n_total_snapshots": self.acc.total_snapshots,
                "n_effective_snapshots": float(self.acc.n_eff),
                "n_batches": len(self.batch_ids),
                "latest_version": latest["version"] if latest else 0,
            }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


class SessionManager:
    def __init__(self, data_dir: str):
        self.store = FileStore(data_dir)
        self.pubsub = PubSub()
        self._sessions: dict[str, Session] = {}
        self._registry_lock = threading.Lock()
        self._recover_all()

    def _recover_all(self) -> None:
        """启动时扫描数据目录，把已有会话元数据装入内存（状态懒加载）。"""
        for sid in self.store.list_sessions():
            try:
                meta = SessionMeta.from_dict(self.store.load_config(sid))
            except (KeyError, ValueError, OSError):  # pragma: no cover
                continue
            sess = Session(meta, self.store)
            self._sessions[sid] = sess

    # ---- 创建 ----
    def create_session(self, array: ArraySpec,
                       update: UpdateSpec | None = None,
                       estimate: EstimateSpec | None = None,
                       name: str = "") -> Session:
        # 提前触发几何校验，错误在这里抛出
        ula = array.build()
        if ula.m > MAX_ELEMENTS:
            raise GeometryError(
                f"阵元数 {ula.m} 超过规模上限 {MAX_ELEMENTS}。"
            )
        update = update or UpdateSpec()
        estimate = estimate or EstimateSpec()
        # 让更新器/估计配置的参数校验也尽早失败
        build_accumulator(ula.m, update.method, update.window, update.alpha)
        EstimateConfig(
            source_mode=estimate.source_mode,
            angle_min=estimate.angle_min,
            angle_max=estimate.angle_max,
            step_deg=estimate.step_deg,
            smoothing=estimate.smoothing,
            sub_len=estimate.sub_len,
        ).validated(ula.m)

        meta = SessionMeta.new(array, update, estimate, name=name)
        sess = Session(meta, self.store)
        with self._registry_lock:
            self.store.save_config(meta.session_id, meta.to_dict())
            self._sessions[meta.session_id] = sess
        return sess

    def get(self, session_id: str) -> Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise SessionNotFound(session_id)
        return sess

    def list_sessions(self) -> list[dict]:
        out = []
        with self._registry_lock:
            sids = sorted(self._sessions)
        for sid in sids:
            out.append(self._sessions[sid].status())
        return out

    # ---- 追加（带形状/数值校验） ----
    def append(self, session_id: str, batch_id: str,
               real: np.ndarray, imag: np.ndarray) -> dict:
        sess = self.get(session_id)
        self._validate_batch_id(batch_id)
        real = np.asarray(real, dtype=np.float64)
        imag = np.asarray(imag, dtype=np.float64)
        if real.ndim != 2 or imag.ndim != 2:
            raise EstimationError(
                "实部与虚部都必须是二维 M×K 矩阵。"
            )
        if real.shape != imag.shape:
            raise EstimationError(
                f"实部形状 {real.shape} 与虚部形状 {imag.shape} 不一致。"
            )
        m, k = real.shape
        if m != sess.meta.array.m:
            raise EstimationError(
                f"数据阵元维 {m} 与会话阵元数 {sess.meta.array.m} 不一致。"
            )
        if k == 0:
            raise EstimationError("单批快拍数不能为 0。")
        if k > MAX_SNAPSHOTS_PER_BATCH:
            raise EstimationError(
                f"单批快拍数 {k} 超过上限 {MAX_SNAPSHOTS_PER_BATCH}。"
            )
        if not (np.all(np.isfinite(real)) and np.all(np.isfinite(imag))):
            raise EstimationError("快拍数据中含有 NaN 或无穷大。")
        X = real + 1j * imag
        resp = sess.append(batch_id, X)
        return resp

    @staticmethod
    def _validate_batch_id(batch_id: str) -> None:
        if not isinstance(batch_id, str) or not batch_id:
            raise EstimationError(
                "批次号必须是非空字符串，用于追加幂等去重。"
            )
        if len(batch_id) > 200:
            raise EstimationError("批次号长度不能超过 200。")

    # ---- 估计并推送 ----
    def estimate(self, session_id: str, overrides: dict | None = None,
                 true_angles: list[float] | None = None,
                 include_spectrum: bool = False) -> dict:
        sess = self.get(session_id)
        rec = sess.estimate(overrides, true_angles=true_angles,
                            include_spectrum=include_spectrum)
        self.pubsub.publish(session_id, rec)
        return rec

    def results_since(self, session_id: str, since_version: int) -> list[dict]:
        return self.get(session_id).results_since(since_version)
