"""闭环自测用的复基带快拍合成。

模型：X = A(theta) S + N，信源窄带、远场、平面波。

* 非相干源：各源复高斯包络独立 CN(0, P_i)；
* 相干源：全部标记 coherent=True 的源共用同一组 CN(0,1) 包络，各自乘
  固定随机相位（由种子决定），因此协方差秩为 1。

噪声为空间白 CN(0, sigma2 I)。信噪比按功率最大的源定义（dB）：
sigma2 = P_max / 10^(SNR_dB/10)。

随机数只走 numpy ``default_rng(seed)``，绘制顺序固定（先噪声后信源），
同样 (参数, 种子) 生成的复数据逐位相同。
"""
from __future__ import annotations

import numpy as np

from .geometry import ULA


class SynthesisError(ValueError):
    """合成参数不合法。"""


def synthesize(
    array: ULA,
    angles_deg: list[float],
    powers: list[float],
    snr_db: float,
    n_snapshots: int,
    seed: int,
    coherent_flags: list[bool] | None = None,
) -> np.ndarray:
    """生成 ``(M, K)`` 复基带快拍。

    Args:
        array: 阵列定义。
        angles_deg: 各源方位角（度）。
        powers: 各源功率（线性值，必须非负且至少一个为正）。
        snr_db: 最大功率源相对白噪声的信噪比（dB）。
        n_snapshots: 快拍条数 K。
        seed: 随机种子（非负整数）；同种子逐位复现。
        coherent_flags: 长度等于源数；True 表示该源加入相干组。
            缺省全部非相干。
    """
    angles = np.asarray(angles_deg, dtype=np.float64).reshape(-1)
    powers = np.asarray(powers, dtype=np.float64).reshape(-1)
    d = angles.size
    if d == 0:
        raise SynthesisError("至少要给定一个信源。")
    if powers.shape != (d,):
        raise SynthesisError("功率列表长度必须与方位角列表一致。")
    if coherent_flags is None:
        coherent_flags = [False] * d
    coherent_flags = np.asarray(coherent_flags, dtype=bool).reshape(-1)
    if coherent_flags.shape != (d,):
        raise SynthesisError("相干标记长度必须与源数一致。")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(powers)):
        raise SynthesisError("方位角与功率必须是有限数值。")
    if np.any(powers < 0) or np.max(powers) <= 0:
        raise SynthesisError("信源功率必须非负，且至少有一个功率为正。")
    if not isinstance(n_snapshots, (int, np.integer)) or n_snapshots <= 0:
        raise SynthesisError("快拍条数必须为正整数。")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise SynthesisError("随机种子必须为非负整数。")

    rng = np.random.default_rng(int(seed))
    m = array.m
    k = int(n_snapshots)
    p_max = float(np.max(powers))
    sigma2 = p_max / (10.0 ** (float(snr_db) / 10.0))

    # 绘制顺序固定：先噪声，再信源包络。
    N = (rng.standard_normal((m, k)) +
         1j * rng.standard_normal((m, k))) * np.sqrt(sigma2 / 2.0)

    S = np.zeros((d, k), dtype=np.complex128)
    if np.any(coherent_flags):
        base = (rng.standard_normal(k) +
                1j * rng.standard_normal(k)) / np.sqrt(2.0)
        phases = np.exp(1j * 2.0 * np.pi * rng.random(d))
    for i in range(d):
        if coherent_flags[i]:
            S[i] = phases[i] * np.sqrt(powers[i]) * base
        else:
            S[i] = ((rng.standard_normal(k) +
                     1j * rng.standard_normal(k))
                    * np.sqrt(powers[i] / 2.0))

    A = array.steering(angles)  # (M, D)
    return A @ S + N
