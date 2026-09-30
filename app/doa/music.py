"""MUSIC 空间谱扫描、谱峰细化与信源功率估计。

约定角度为与阵列法线（broadside）夹角，单位度；扫描区间 [-90, 90]。
细化用黄金分割一维搜索，在粗网格步长区间内把峰位精度推进到比步长
细至少一个数量级（默认 50 次迭代，精度远超要求）。
"""
from __future__ import annotations

import numpy as np

from .geometry import ULA

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...
_REFINE_ITERS = 50
_EPS = 1e-12


def _music_denom_chunk(array: ULA, En: np.ndarray,
                       theta_grid: np.ndarray) -> np.ndarray:
    """网格各点的 MUSIC 分母 a^H En En^H a。"""
    A = array.steering(theta_grid)  # (M, G)
    proj = En.conj().T @ A  # (M-D, G)
    return np.sum(np.abs(proj) ** 2, axis=0)


def music_spectrum(array: ULA, En: np.ndarray,
                   theta_min: float = -90.0,
                   theta_max: float = 90.0,
                   step_deg: float = 0.5,
                   max_points: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """粗网格 MUSIC 谱，返回 (角度网格, 谱值)。点数超过上限抛错。"""
    if step_deg <= 0:
        raise ValueError("扫描步长必须为正。")
    n_span = theta_max - theta_min
    if n_span <= 0:
        raise ValueError("扫描角度范围必须满足 theta_max > theta_min。")
    n_points = int(np.floor(n_span / step_deg + 1e-9)) + 1
    if n_points > max_points:
        raise ValueError(
            f"扫描点数 {n_points} 超过上限 {max_points}，请增大步长或缩小范围。"
        )
    grid = theta_min + np.arange(n_points) * step_deg
    # 分块防止大网格时一次性复制导向矩阵（上限 20 万点时其实也不大）
    denom = np.empty(n_points)
    chunk = 20_000
    for i in range(0, n_points, chunk):
        denom[i:i + chunk] = _music_denom_chunk(
            array, En, grid[i:i + chunk]
        )
    return grid, 1.0 / np.maximum(denom, _EPS)


def _find_local_peaks(grid: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """返回严格局部极大点的网格下标（边界不算峰）。"""
    idx = np.where(
        (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])
    )[0] + 1
    # 谱值高的峰优先
    return idx[np.argsort(spec[idx])[::-1]]


def _golden_refine(array: ULA, En: np.ndarray,
                   lo: float, hi: float) -> tuple[float, float]:
    """在 [lo, hi] 内黄金分割最大化 MUSIC 谱，返回 (峰角, 峰谱)。"""
    def f(t: float) -> float:
        a = array.steering(t)
        # MUSIC 峰 = 分母 a^H En En^H a 的极小点，直接最小化分母
        return float(np.sum(np.abs(En.conj().T @ a) ** 2))

    a, b = lo, hi
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(_REFINE_ITERS):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    t_best = (a + b) / 2.0
    a_vec = array.steering(t_best)
    peak = 1.0 / max(float(np.sum(np.abs(En.conj().T @ a_vec) ** 2)), _EPS)
    return float(t_best), peak


def estimate_angles(array: ULA, En: np.ndarray, n_sources: int,
                    theta_min: float = -90.0,
                    theta_max: float = 90.0,
                    step_deg: float = 0.5,
                    max_points: int = 200_000) -> dict:
    """扫描 + 细化，返回 n_sources 个最强峰。

    返回：
        angles: 按角度升序排列的峰位（度）
        peaks: 对应的细化后谱值
        grid/spectrum: 粗网格谱（调试/绘图用）
        warning: 找不到足够谱峰时的人读提示（否则为 None）
    """
    grid, spec = music_spectrum(
        array, En, theta_min, theta_max, step_deg, max_points
    )
    peak_idx = _find_local_peaks(grid, spec)
    refined: list[tuple[float, float]] = []
    for p in peak_idx:
        lo = grid[p - 1] if p - 1 >= 0 else grid[p] - step_deg
        hi = grid[p + 1] if p + 1 < grid.size else grid[p] + step_deg
        refined.append(_golden_refine(array, En, lo, hi))
        if len(refined) == n_sources:
            break
    warning = None
    if len(refined) < n_sources:
        warning = (
            f"在 {theta_min:g}° 到 {theta_max:g}° 范围内只找到 "
            f"{len(refined)} 个 MUSIC 谱峰，少于期望信源数 {n_sources}；"
            f"可能是相干源未开空间平滑、信源数过大或角度落到扫描边界。"
        )
    refined.sort(key=lambda t: t[0])
    return {
        "angles": [t[0] for t in refined],
        "peaks": [t[1] for t in refined],
        "grid": grid,
        "spectrum": spec,
        "warning": warning,
    }


def source_powers_ls(C: np.ndarray, array: ULA,
                     angles: list[float], sigma2: float) -> np.ndarray:
    """由协方差做最小二乘功率拟合：C ≈ A diag(p) A^H + sigma2 I。

    向量化（去掉重复的上三角）后按最小二乘解 p = pinv(M) y，负值裁 0。
    """
    d = len(angles)
    if d == 0:
        return np.zeros(0)
    A = array.steering(np.asarray(angles, dtype=np.float64))  # (M, D)
    m = C.shape[0]
    # 取厄米矩阵的独立元素：上三角（含对角）
    iu = np.triu_indices(m)
    y = (C - sigma2 * np.eye(m))[iu]
    M = np.empty((y.size, d), dtype=np.complex128)
    for j in range(d):
        M[:, j] = np.outer(A[:, j], A[:, j].conj())[iu]
    p, *_ = np.linalg.lstsq(M, y, rcond=None)
    p = p.real
    return np.maximum(p, 0.0)
