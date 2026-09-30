"""Uniform-linear-array geometry.

The nominal array lies on the x-axis.  Element ``m`` is located at
``(m - (M-1)/2) * spacing`` (the centering convention cancels out of all
covariance based estimation).  Optional position errors (in metres, along
the axis) are added to the nominal positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GeometryError(ValueError):
    """阵列参数不合法。"""


@dataclass(frozen=True)
class ULA:
    """M 阵元均匀线阵定义。

    Attributes:
        m: 阵元数。
        spacing: 相邻阵元名义间距（米），必须为正。
        wavelength: 载波波长（米），必须为正。
        position_errors: 长度为 M 的可选阵元位置误差（米）；缺省为理想线阵。
    """

    m: int
    spacing: float
    wavelength: float
    position_errors: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.m, (int, np.integer)) or isinstance(self.m, bool):
            raise GeometryError("阵元数必须是整数。")
        if self.m < 1:
            raise GeometryError("阵元数必须为正。")
        if not (isinstance(self.spacing, (int, float, np.floating, np.integer))
                and np.isfinite(self.spacing)) or self.spacing <= 0:
            raise GeometryError("阵元间距必须为正数。")
        if not (isinstance(self.wavelength, (int, float, np.floating, np.integer))
                and np.isfinite(self.wavelength)) or self.wavelength <= 0:
            raise GeometryError("载波波长必须为正数。")
        pe = None
        if self.position_errors is not None:
            pe = np.asarray(self.position_errors, dtype=np.float64).reshape(-1)
            if pe.shape != (self.m,):
                raise GeometryError(
                    f"阵元位置误差长度必须为 {self.m}，实际为 {pe.shape[0]}。"
                )
            if not np.all(np.isfinite(pe)):
                raise GeometryError("阵元位置误差中含有 NaN 或无穷大。")
        object.__setattr__(self, "position_errors", pe)

    @property
    def positions(self) -> np.ndarray:
        """各阵元沿阵列轴的实际坐标（米），长度 M。"""
        nominal = (np.arange(self.m) - (self.m - 1) / 2.0) * float(self.spacing)
        if self.position_errors is not None:
            nominal = nominal + self.position_errors
        return nominal

    @property
    def is_ideal(self) -> bool:
        return self.position_errors is None or not np.any(self.position_errors)

    def steering(self, theta_deg: float | np.ndarray) -> np.ndarray:
        """方位角（度）对应的导向矢量。

        ``theta`` 为与阵列法线（broadside）的夹角；单个角度返回形状 ``(M,)``，
        多个角度返回形状 ``(M, len(theta))``。
        """
        scalar = np.isscalar(theta_deg)
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        if not np.all(np.isfinite(theta)):
            raise GeometryError("方位角中含有 NaN 或无穷大。")
        # k * x * sin(theta), k = 2*pi/lambda
        phase = (2.0 * np.pi / float(self.wavelength)) * np.outer(
            self.positions, np.sin(np.deg2rad(theta))
        )
        a = np.exp(1j * phase)
        if scalar:
            return a[:, 0]
        return a

    def grating_aliases(self, theta_deg: float) -> list[float]:
        """理想均匀线阵下与 ``theta_deg`` 栅瓣混淆的所有可见角度（度）。

        导向矢量相位里只出现 ``(d/lambda) sin(theta)``，因此
        ``sin(theta') = sin(theta) + n * lambda/d`` 时完全混淆。
        只在间距大于半个波长时才会出现落在可见域 (-90, 90) 内的别名。
        带位置误差的阵列没有严格栅瓣，返回空表。
        """
        if not self.is_ideal:
            return []
        ratio = float(self.wavelength) / float(self.spacing)
        s0 = np.sin(np.deg2rad(float(theta_deg)))
        aliases: list[float] = []
        n = 1
        while True:
            hit = False
            for cand in (s0 + n * ratio, s0 - n * ratio):
                if -1.0 < cand < 1.0:
                    ang = float(np.rad2deg(np.arcsin(cand)))
                    if abs(ang - float(theta_deg)) > 1e-9:
                        aliases.append(ang)
                    hit = True
            if not hit and n * ratio > 2.0:
                break
            n += 1
            if n > 1000:  # pragma: no cover - 安全上限
                break
        return sorted(aliases)
