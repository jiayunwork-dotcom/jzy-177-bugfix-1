"""单信源、白噪声、确定性（条件）模型下的克拉美–罗界。

窄带远场、阵列输出 x(t) = a(theta) s(t) + n(t)，n 为空间白高斯噪声。
对方位角（弧度）：

    CRB(theta) = 1 / ( 2 N * SNR * (D^H P_A^{\\perp} D) )

其中 D = da/dtheta，P_A^{\\perp} = I - A (A^H A)^{-1} A^H。
单信源时 A 即导向矢量，P_A^{\\perp} D 为其在 a 正交补上的投影。
"""
from __future__ import annotations

import numpy as np

from .geometry import ULA


def steering_derivative(array: ULA, theta_deg: float) -> np.ndarray:
    """导向矢量对方位角（弧度）的导数 D = da/dtheta。"""
    theta = np.deg2rad(float(theta_deg))
    x = array.positions  # 沿轴坐标（米）
    k = 2.0 * np.pi / array.wavelength
    a = array.steering(theta_deg)
    # a_m = exp(j k x_m sin θ)，对 θ 求导：j k x_m cos θ * a_m
    return 1j * k * x * np.cos(theta) * a


def crb_rad(array: ULA, theta_deg: float, snr_linear: float,
            n_snapshots: float) -> float:
    """单信源确定性 CRB，单位弧度²。"""
    a = array.steering(theta_deg)
    d = steering_derivative(array, theta_deg)
    # P_A^\perp D = D - a (a^H D)/(a^H a)
    proj_d = d - a * np.vdot(a, d) / np.vdot(a, a)
    fisher = 2.0 * float(n_snapshots) * float(snr_linear) * float(
        np.vdot(proj_d, proj_d).real
    )
    if fisher <= 0.0:
        return np.inf
    return 1.0 / fisher


def crb_deg(array: ULA, theta_deg: float, snr_linear: float,
            n_snapshots: float) -> float:
    """单信源确定性 CRB 的角度标准差，单位度。"""
    return float(np.rad2deg(np.sqrt(crb_rad(
        array, theta_deg, snr_linear, n_snapshots
    ))))
