"""FastAPI 应用：会话管理、流式追加、估计、版本轮询与 SSE 订阅。

同步业务函数（``def`` 路由）由 FastAPI 放到线程池执行，会话内部的
``RLock`` 保证同会话并发追加的串行等价性。数据目录由环境变量
``DOA_DATA_DIR`` 指定（容器内默认 ``/data``，可挂载）。
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import StreamingResponse

from ..doa.covariance import batch_covariance
from ..doa.engine import EstimateConfig, run_doa
from ..doa.geometry import ULA
from ..doa.synthesis import synthesize
from ..sessions.manager import SessionManager
from ..sessions.models import ArraySpec, EstimateSpec, UpdateSpec
from .errors import APIError, install_exception_handlers
from .schemas import (
    AppendBatchIn,
    CreateSessionIn,
    EstimateIn,
    EstimateSpecIn,
    SynthesizeIn,
)


def create_app(data_dir: str | None = None) -> FastAPI:
    app = FastAPI(
        title="八阵元线阵常驻测向后端",
        version="1.0.0",
        description=(
            "MUSIC 测向常驻服务：实验会话、快拍流式增量协方差更新、"
            "MDL/AIC 源数判定、前后向空间平滑、谱峰细化与 CRB。"
        ),
    )
    install_exception_handlers(app)
    manager = SessionManager(
        data_dir or os.environ.get("DOA_DATA_DIR", "./data")
    )
    app.state.manager = manager

    def _to_specs(body: CreateSessionIn) -> tuple[ArraySpec, UpdateSpec,
                                                  EstimateSpec]:
        a, u, e = body.array, body.update, body.estimate
        array_spec = ArraySpec(
            m=a.m, spacing=a.spacing, wavelength=a.wavelength,
            position_errors=a.position_errors,
        )
        update_spec = UpdateSpec(
            method=u.method, window=u.window, alpha=u.alpha,
        )
        estimate_spec = EstimateSpec(
            source_mode=e.source_mode,
            angle_min=e.angle_min, angle_max=e.angle_max,
            step_deg=e.step_deg, smoothing=e.smoothing,
            sub_len=e.sub_len,
        )
        return array_spec, update_spec, estimate_spec

    # ---------------- 健康检查 ----------------
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # ---------------- 会话 ----------------
    @app.post("/sessions", status_code=201)
    def create_session(body: CreateSessionIn) -> dict:
        specs = _to_specs(body)
        sess = manager.create_session(*specs, name=body.name)
        return sess.status()

    @app.get("/sessions")
    def list_sessions() -> dict:
        return {"sessions": manager.list_sessions()}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        return manager.get(session_id).status()

    @app.post("/sessions/{session_id}/batches")
    def append_batch(session_id: str, body: AppendBatchIn) -> dict:
        real = np.asarray(body.real, dtype=np.float64)
        imag = np.asarray(body.imag, dtype=np.float64)
        return manager.append(session_id, body.batch_id, real, imag)

    @app.post("/sessions/{session_id}/estimate")
    def estimate(session_id: str, body: EstimateIn) -> dict:
        overrides = body.model_dump(exclude_none=True)
        truth = overrides.pop("true_angles", None)
        include_spectrum = overrides.pop("include_spectrum", False)
        return manager.estimate(session_id, overrides, true_angles=truth,
                                include_spectrum=include_spectrum)

    @app.get("/sessions/{session_id}/results")
    def get_results(
        session_id: str,
        since_version: int = Query(
            0, ge=0,
            description="只返回版本号严格大于该值的结果",
        ),
    ) -> dict:
        recs = manager.results_since(session_id, since_version)
        latest = recs[-1]["version"] if recs else since_version
        return {
            "session_id": session_id,
            "since_version": since_version,
            "latest_version": latest,
            "results": recs,
        }

    @app.get("/sessions/{session_id}/results/stream")
    async def stream_results(
        session_id: str,
        after_version: int = Query(
            0, ge=0, description="先补发版本号大于该值的历史结果，再推送新结果"
        ),
    ) -> StreamingResponse:
        # 会话不存在要在订阅前报 404
        sess = manager.get(session_id)

        async def event_gen():
            sub, _ = manager.pubsub.subscribe(session_id, after_version)
            try:
                # 已落盘但没推送过的历史结果先补发
                for rec in sess.results_since(after_version):
                    yield _sse("result", rec)
                while True:
                    try:
                        rec = await asyncio.wait_for(sub.queue.get(),
                                                    timeout=15.0)
                        yield _sse("result", rec)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
            except asyncio.CancelledError:
                raise
            finally:
                manager.pubsub.unsubscribe(session_id, sub)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"},
        )

    # ---------------- 无状态合成 + 即时估计（闭环自测） ----------------
    @app.post("/synthesize")
    def synthesize_data(body: SynthesizeIn) -> dict:
        from ..doa.engine import MAX_ELEMENTS, MAX_SCAN_POINTS, \
            MAX_SNAPSHOTS_PER_BATCH
        if body.array.m > MAX_ELEMENTS:
            raise APIError(
                f"阵元数 {body.array.m} 超过规模上限 {MAX_ELEMENTS}。",
                error_type="limit_error",
            )
        if body.n_snapshots > MAX_SNAPSHOTS_PER_BATCH:
            raise APIError(
                f"快拍条数 {body.n_snapshots} 超过单次上限 "
                f"{MAX_SNAPSHOTS_PER_BATCH}。",
                error_type="limit_error",
            )
        a = body.array
        array = ULA(
            a.m, a.spacing, a.wavelength,
            None if a.position_errors is None
            else np.asarray(a.position_errors, dtype=np.float64),
        )
        angles = [s.angle_deg for s in body.sources]
        powers = [s.power for s in body.sources]
        coherent = [s.coherent for s in body.sources]
        X = synthesize(
            array, angles, powers, body.snr_db, body.n_snapshots,
            body.seed, coherent_flags=coherent,
        )
        out: dict[str, Any] = {
            "array": body.array.model_dump(),
            "true_angles_deg": angles,
            "n_snapshots": body.n_snapshots,
            "seed": body.seed,
        }
        if body.estimate is not None:
            e = body.estimate
            cfg = EstimateConfig(
                source_mode=e.source_mode, angle_min=e.angle_min,
                angle_max=e.angle_max, step_deg=e.step_deg,
                smoothing=e.smoothing, sub_len=e.sub_len,
            )
            C, k = batch_covariance(X)
            out["estimate"] = run_doa(array, C, k, cfg,
                                      true_angles=angles)
        if body.include_data:
            out["real"] = X.real.tolist()
            out["imag"] = X.imag.tolist()
        return out

    return app


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


app = create_app()
