"""HTTP 请求/响应的 Pydantic 模型。

实部/虚部分开传（嵌套列表），全部数值校验由 FastAPI 422 处理；
业务语义校验（形状一致、不含 NaN 等）在内核/会话层给出人读错误。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArraySpecIn(BaseModel):
    # 上限 64 在业务层校验，以便返回人读的 400 错误
    m: int = Field(..., ge=1, description="阵元数（≤64）")
    spacing: float = Field(..., description="阵元间距（米），必须为正")
    wavelength: float = Field(..., description="载波波长（米），必须为正")
    position_errors: list[float] | None = Field(
        None, description="长度 M 的阵元位置误差（米），缺省为理想线阵"
    )


class UpdateSpecIn(BaseModel):
    method: Literal["sliding", "forgetting"] = Field(
        "sliding", description="增量更新方式，默认定长滑动窗口"
    )
    window: int = Field(256, ge=1, description="sliding 模式窗口长度")
    alpha: float = Field(
        0.9922, gt=0.0, le=1.0,
        description="forgetting 模式遗忘因子（0,1]；0.9922 等效窗口 256",
    )


class EstimateSpecIn(BaseModel):
    source_mode: int | Literal["mdl", "aic"] = Field(
        "mdl", description="源数判定：mdl / aic，或直接给整数源数"
    )
    angle_min: float = -90.0
    angle_max: float = 90.0
    step_deg: float = Field(0.5, gt=0.0, description="粗扫描步长（度）")
    smoothing: bool = Field(False, description="是否启用前后向空间平滑")
    sub_len: int | None = Field(None, ge=1, description="平滑子阵长度")


class CreateSessionIn(BaseModel):
    name: str = ""
    array: ArraySpecIn
    update: UpdateSpecIn = UpdateSpecIn()
    estimate: EstimateSpecIn = EstimateSpecIn()


class AppendBatchIn(BaseModel):
    batch_id: str = Field(..., description="批次号，同号重复提交只计一次")
    real: list[list[float]] = Field(..., description="实部，M 行 K 列")
    imag: list[list[float]] = Field(..., description="虚部，M 行 K 列")


class EstimateIn(BaseModel):
    source_mode: int | Literal["mdl", "aic"] | None = None
    angle_min: float | None = None
    angle_max: float | None = None
    step_deg: float | None = None
    smoothing: bool | None = None
    sub_len: int | None = None
    true_angles: list[float] | None = Field(
        None, description="合成自测时填真实方位，返回 |估计-真值|/CRB"
    )
    include_spectrum: bool = Field(
        False, description="是否返回粗网格 MUSIC 谱（不随结果落盘）"
    )


class SourceSpec(BaseModel):
    angle_deg: float
    power: float = Field(1.0, ge=0.0)
    coherent: bool = Field(False, description="是否属于相干源组")


class SynthesizeIn(BaseModel):
    array: ArraySpecIn
    sources: list[SourceSpec] = Field(..., min_length=1)
    snr_db: float
    n_snapshots: int = Field(..., ge=1, description="快拍条数（≤10000）")
    seed: int = Field(..., ge=0)
    # 合成后可直接顺手估计，便于闭环自测
    estimate: EstimateSpecIn | None = None
    include_data: bool = Field(
        True, description="是否把生成的实部/虚部数据一并返回（默认返回）"
    )
