"""人读错误的统一 HTTP 出口。

业务异常（阵列几何、协方差、估计、会话/存储问题）统一回
``400``（会话不存在回 ``404``），JSON 体为
``{"error": {"type": ..., "message": ...}}``，消息直接面向使用者。
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..doa.covariance import CovarianceError
from ..doa.engine import EstimationError
from ..doa.geometry import GeometryError
from ..doa.synthesis import SynthesisError
from ..sessions.manager import SessionNotFound
from ..sessions.storage import StorageError


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400,
                 error_type: str = "api_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


def _detail(exc: Exception, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": str(exc)}},
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GeometryError)
    async def _geom(_: Request, exc: GeometryError) -> JSONResponse:
        return _detail(exc, 400, "geometry_error")

    @app.exception_handler(CovarianceError)
    async def _cov(_: Request, exc: CovarianceError) -> JSONResponse:
        return _detail(exc, 400, "covariance_error")

    @app.exception_handler(EstimationError)
    async def _est(_: Request, exc: EstimationError) -> JSONResponse:
        return _detail(exc, 400, "estimation_error")

    @app.exception_handler(SynthesisError)
    async def _syn(_: Request, exc: SynthesisError) -> JSONResponse:
        return _detail(exc, 400, "synthesis_error")

    @app.exception_handler(StorageError)
    async def _sto(_: Request, exc: StorageError) -> JSONResponse:
        return _detail(exc, 500, "storage_error")

    @app.exception_handler(APIError)
    async def _api(_: Request, exc: APIError) -> JSONResponse:
        return _detail(exc, exc.status_code, exc.error_type)

    @app.exception_handler(SessionNotFound)
    async def _nf(_: Request, exc: SessionNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "session_not_found",
                    "message": f"会话 {exc.args[0]!r} 不存在。",
                }
            },
        )
