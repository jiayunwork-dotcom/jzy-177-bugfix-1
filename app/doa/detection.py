"""信源个数判定：MDL 与 AIC（Wax & Kailath 特征值准则）。

对样本协方差的 M 个降序特征值 lambda_i，对候选 k（噪声子空间维数
M-k）比较噪声特征值的算术平均与几何平均：

    L(k) = (M-k) * ln( arithmetic_mean / geometric_mean )
    AIC(k) = 2 * N * L(k) + 2 * k * (2M - k)
    MDL(k) = N * L(k) + (1/2) * k * (2M - k) * ln(N)

取使准则最小的 k。N 为有效快拍数（FB 平滑时取等效快拍数）。
"""
from __future__ import annotations

import numpy as np


def noise_likelihood_ratio(eigvals: np.ndarray, k: int) -> float:
    """候选信源数 k 下，噪声特征值算术/几何均值之比的对数乘噪声维数。"""
    noise = np.asarray(eigvals, dtype=np.float64)[k:]
    q = noise.size
    if q == 0:
        return np.inf
    amean = float(np.mean(noise))
    # 数值保护：特征值理论上为正
    gmean = float(np.exp(np.mean(np.log(np.maximum(noise, np.finfo(np.float64).tiny)))))
    if gmean <= 0.0:
        return np.inf
    return q * float(np.log(amean / gmean))


def source_count_aic(eigvals: np.ndarray, n_snapshots: float) -> int:
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    eigvals = np.maximum(eigvals, np.finfo(np.float64).tiny)
    m = eigvals.size
    best_k, best_cost = 0, np.inf
    for k in range(m):  # 候选 0 .. M-1
        penalty = 2.0 * k * (2 * m - k)
        cost = 2.0 * float(n_snapshots) * noise_likelihood_ratio(eigvals, k) + penalty
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k


def source_count_mdl(eigvals: np.ndarray, n_snapshots: float) -> int:
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    eigvals = np.maximum(eigvals, np.finfo(np.float64).tiny)
    m = eigvals.size
    n = max(float(n_snapshots), 1.0)
    best_k, best_cost = 0, np.inf
    for k in range(m):  # 候选 0 .. M-1
        penalty = 0.5 * k * (2 * m - k) * np.log(n)
        cost = n * noise_likelihood_ratio(eigvals, k) + penalty
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k


def both_criteria(eigvals: np.ndarray, n_snapshots: float) -> dict:
    """一次返回 MDL、AIC 两种判定。"""
    return {
        "mdl": source_count_mdl(eigvals, n_snapshots),
        "aic": source_count_aic(eigvals, n_snapshots),
    }
