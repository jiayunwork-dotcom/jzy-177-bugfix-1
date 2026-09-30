"""会话相关的数据模型（阵列/更新/估计配置、估计结果）。

这些类型刻意不依赖 FastAPI/Pydantic，存储层与 HTTP 层都能直接使用；
API 的请求模型在 :mod:`app.api.schemas` 里单独定义并转换到这里。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..doa.geometry import ULA

CONFIG_VERSION = 1


@dataclass(frozen=True)
class ArraySpec:
    """会话级阵列参数。"""

    m: int
    spacing: float
    wavelength: float
    position_errors: list[float] | None = None

    def build(self) -> ULA:
        return ULA(
            self.m, self.spacing, self.wavelength,
            None if self.position_errors is None
            else np.asarray(self.position_errors, dtype=np.float64),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArraySpec":
        pe = d.get("position_errors")
        return cls(
            m=int(d["m"]),
            spacing=float(d["spacing"]),
            wavelength=float(d["wavelength"]),
            position_errors=(None if pe is None else [float(x) for x in pe]),
        )


@dataclass(frozen=True)
class UpdateSpec:
    """增量协方差更新方式。

    method: "sliding"（定长滑动窗口，默认）或 "forgetting"（指数遗忘）。
    window: sliding 模式窗口长度，默认 256。
    alpha: forgetting 模式遗忘因子，默认 0.9922（与 W=256 等效记忆
        alpha=(W-1)/(W+1)）。
    """

    method: str = "sliding"
    window: int = 256
    alpha: float = 0.9922

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UpdateSpec":
        return cls(
            method=str(d.get("method", "sliding")),
            window=int(d.get("window", 256)),
            alpha=float(d.get("alpha", 0.9922)),
        )


@dataclass(frozen=True)
class EstimateSpec:
    """会话级默认估计配置（单次请求可覆盖部分字段）。"""

    source_mode: Any = "mdl"          # "mdl" / "aic" / int
    angle_min: float = -90.0
    angle_max: float = 90.0
    step_deg: float = 0.5
    smoothing: bool = False
    sub_len: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EstimateSpec":
        return cls(
            source_mode=d.get("source_mode", "mdl"),
            angle_min=float(d.get("angle_min", -90.0)),
            angle_max=float(d.get("angle_max", 90.0)),
            step_deg=float(d.get("step_deg", 0.5)),
            smoothing=bool(d.get("smoothing", False)),
            sub_len=(None if d.get("sub_len") is None
                     else int(d["sub_len"])),
        )


@dataclass
class SessionMeta:
    session_id: str
    name: str
    created_at: float
    updated_at: float
    array: ArraySpec
    update: UpdateSpec
    estimate: EstimateSpec

    @classmethod
    def new(cls, array: ArraySpec, update: UpdateSpec,
            estimate: EstimateSpec, name: str = "") -> "SessionMeta":
        now = time.time()
        return cls(
            session_id=uuid.uuid4().hex,
            name=name,
            created_at=now,
            updated_at=now,
            array=array,
            update=update,
            estimate=estimate,
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "array": self.array.to_dict(),
            "update": self.update.to_dict(),
            "estimate": self.estimate.to_dict(),
            "version": CONFIG_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionMeta":
        return cls(
            session_id=str(d["session_id"]),
            name=str(d.get("name", "")),
            created_at=float(d["created_at"]),
            updated_at=float(d["updated_at"]),
            array=ArraySpec.from_dict(d["array"]),
            update=UpdateSpec.from_dict(d.get("update", {})),
            estimate=EstimateSpec.from_dict(d.get("estimate", {})),
        )
