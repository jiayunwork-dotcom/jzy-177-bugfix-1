"""测向内核总装：协方差 -> 源数 -> 平滑 -> MUSIC -> 功率 -> CRB。

引擎是无状态的纯函数；会话层负责攒协方差和持久化。所有校验也集中在
这里，HTTP 层与离线自测共用同一套人读错误。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .covariance import forward_backward_smoothing, hermitianize
from .crb import crb_deg
from .detection import both_criteria
from .geometry import ULA
from .music import estimate_angles, source_powers_ls

MAX_ELEMENTS = 64
MAX_SNAPSHOTS_PER_BATCH = 10_000
MAX_SCAN_POINTS = 200_000


class EstimationError(ValueError):
    """估计配置或状态不合法，消息面向调用方（人读）。"""


@dataclass(frozen=True)
class EstimateConfig:
    """一次估计的配置（随会话持久化）。

    Attributes:
        source_mode: "mdl" / "aic" 或直接给定的整数源数。
        angle_min/angle_max/step_deg: MUSIC 扫描范围与粗步长（度）。
        smoothing: 是否启用前后向空间平滑。
        sub_len: 平滑子阵长度；None 时取 ceil(2M/3)。
        max_scan_points: 扫描点数硬上限。
    """

    source_mode: Any = "mdl"
    angle_min: float = -90.0
    angle_max: float = 90.0
    step_deg: float = 0.5
    smoothing: bool = False
    sub_len: int | None = None
    max_scan_points: int = MAX_SCAN_POINTS

    def validated(self, m: int) -> "EstimateConfig":
        """返回填好默认值并完成全部校验的配置。"""
        cfg = self
        if not (np.isfinite(cfg.angle_min) and np.isfinite(cfg.angle_max)):
            raise EstimationError("扫描角度范围必须是有限数值。")
        if cfg.angle_min < -90.0 or cfg.angle_max > 90.0:
            raise EstimationError("扫描范围必须落在 [-90°, 90°] 之内。")
        if cfg.angle_min >= cfg.angle_max:
            raise EstimationError("扫描范围必须满足 angle_min < angle_max。")
        if cfg.step_deg <= 0 or not np.isfinite(cfg.step_deg):
            raise EstimationError("扫描步长必须为正数。")
        n_points = int(np.floor((cfg.angle_max - cfg.angle_min)
                                / cfg.step_deg + 1e-9)) + 1
        if n_points > cfg.max_scan_points:
            raise EstimationError(
                f"扫描点数 {n_points} 超过上限 {cfg.max_scan_points}，"
                f"请增大步长或缩小扫描范围。"
            )
        mode = cfg.source_mode
        if isinstance(mode, str):
            if mode not in ("mdl", "aic"):
                raise EstimationError(
                    f"源数指定方式 {mode!r} 不支持，"
                    f"应为 'mdl'、'aic' 或非负整数。"
                )
        elif isinstance(mode, (int, np.integer)) and not isinstance(mode, bool):
            if mode < 0:
                raise EstimationError("指定的信源个数不能为负。")
        else:
            raise EstimationError("源数指定方式必须是 'mdl'、'aic' 或整数。")

        sub_len = cfg.sub_len
        if cfg.smoothing:
            if sub_len is None:
                sub_len = (2 * m + 2) // 3  # ceil(2M/3)
            if not (1 <= int(sub_len) <= m):
                raise EstimationError(
                    f"子阵长度必须在 [1, {m}] 之间，实际为 {sub_len}。"
                )
            sub_len = int(sub_len)
        elif sub_len is not None:
            if not (1 <= int(sub_len) <= m):
                raise EstimationError(
                    f"子阵长度必须在 [1, {m}] 之间，实际为 {sub_len}。"
                )
        return replace(cfg, sub_len=sub_len)


def _match_truth(angles: list[float], truth: list[float]) -> list[float | None]:
    """按最近角距把真值贪心配给各估计角；真值不重复使用。"""
    remaining = list(map(float, truth))
    out: list[float | None] = []
    for est in angles:
        if not remaining:
            out.append(None)
            continue
        j = int(np.argmin([abs(est - t) for t in remaining]))
        out.append(remaining.pop(j))
    return out


def run_doa(
    array: ULA,
    C: np.ndarray,
    n_eff: float,
    config: EstimateConfig,
    true_angles: list[float] | None = None,
) -> dict:
    """对给定样本协方差执行完整测向流程。

    Args:
        array: 全阵列定义。
        C: ``(M, M)`` 样本协方差（MLE 归一化，与 n_eff 对应）。
        n_eff: 有效快拍数（窗口内条数，或遗忘因子权重和）。
        config: 估计配置。
        true_angles: 合成自测时可填真实方位，用于算 |误差|/CRB。
    """
    m = array.m
    cfg = config.validated(m)

    C = hermitianize(np.asarray(C, dtype=np.complex128))
    if C.shape != (m, m):
        raise EstimationError(f"协方差必须是 {m}×{m}，实际为 {C.shape}。")
    if not np.all(np.isfinite(C)):
        raise EstimationError("协方差中含有 NaN 或无穷大。")
    if n_eff <= 0:
        raise EstimationError("有效快拍数为 0，无法估计。")

    # --- （可选）前后向空间平滑，得到子阵维协方差 ---
    snap_multiplier = 1
    if cfg.smoothing:
        Cs, snap_multiplier = forward_backward_smoothing(C, cfg.sub_len)
        dim = cfg.sub_len
        # 平滑后导向按名义均匀线阵的子阵几何（取前 dim 个名义阵元）
        eff_array = ULA(dim, array.spacing, array.wavelength)
    else:
        Cs, dim = C, m
        eff_array = array
    n_eff_sm = n_eff * snap_multiplier

    # --- 特征分解（升序返回，反转成降序） ---
    eigvals_v, eigvecs = np.linalg.eigh(Cs)
    order = np.argsort(eigvals_v)[::-1]
    eigvals = np.maximum(eigvals_v[order].real, np.finfo(np.float64).tiny)
    eigvecs = eigvecs[:, order]

    counts = both_criteria(eigvals, n_eff_sm)

    mode = cfg.source_mode
    if mode == "mdl":
        d = counts["mdl"]
    elif mode == "aic":
        d = counts["aic"]
    else:
        d = int(mode)
    if d >= dim:
        if cfg.smoothing:
            raise EstimationError(
                f"信源数 {d} 不少于平滑子阵阵元数 {dim}，MUSIC 要求噪声子空间"
                f"非空；请增大子阵长度或减少信源数。"
            )
        raise EstimationError(
            f"信源数 {d} 不少于阵元数 {dim}，MUSIC 要求噪声子空间非空。"
        )

    # --- MUSIC 扫描 + 细化 ---
    if d > 0:
        En = eigvecs[:, d:]
        found = estimate_angles(
            eff_array, En, d,
            theta_min=cfg.angle_min, theta_max=cfg.angle_max,
            step_deg=cfg.step_deg, max_points=cfg.max_scan_points,
        )
    else:
        found = {
            "angles": [], "peaks": [], "grid": None, "spectrum": None,
            "warning": "MDL/AIC 判定信源数为 0，未执行 MUSIC。",
        }
    angles = found["angles"]

    # --- 噪声功率、各源功率/SNR、CRB ---
    sigma2 = float(np.mean(eigvals[d:])) if d < dim else float("nan")
    powers = source_powers_ls(Cs, eff_array, angles, sigma2) if d > 0 \
        else np.zeros(0)

    matched = _match_truth(angles, true_angles) if true_angles is not None \
        else [None] * len(angles)
    per_angle = []
    for i, ang in enumerate(angles):
        snr_lin = float(powers[i] / sigma2) if sigma2 > 0 else np.inf
        crb = crb_deg(eff_array, ang, snr_lin, n_eff_sm)
        t = matched[i]
        ratio = (abs(ang - t) / crb) if (t is not None and np.isfinite(crb)
                                         and crb > 0) else None
        per_angle.append({
            "angle_deg": float(ang),
            "music_peak": float(found["peaks"][i]),
            "power": float(powers[i]),
            "snr_db": float(10.0 * np.log10(snr_lin))
            if np.isfinite(snr_lin) and snr_lin > 0 else None,
            "crb_std_deg": float(crb) if np.isfinite(crb) else None,
            "true_angle_deg": (float(t) if t is not None else None),
            "error_over_crb": (float(ratio) if ratio is not None else None),
        })

    # --- 栅瓣模糊（d > lambda/2 才有可见别名） ---
    ambiguous = array.spacing / array.wavelength > 0.5 + 1e-12
    grating_pairs = []
    if ambiguous and array.is_ideal:
        for item in per_angle:
            aliases = array.grating_aliases(item["angle_deg"])
            if aliases:
                grating_pairs.append({
                    "angle_deg": item["angle_deg"],
                    "confused_with_deg": [float(a) for a in aliases],
                })

    return {
        "angles_deg": [float(a) for a in angles],
        "n_sources": d,
        "source_counts": {"mdl": int(counts["mdl"]), "aic": int(counts["aic"])},
        "per_angle": per_angle,
        "noise_power": sigma2,
        "n_effective_snapshots": float(n_eff_sm),
        "smoothing": {
            "enabled": bool(cfg.smoothing),
            "sub_len": int(dim) if cfg.smoothing else None,
            "n_subarrays": int(m - dim + 1) if cfg.smoothing else None,
            "snapshot_multiplier": int(snap_multiplier),
        },
        "grating_ambiguous": bool(grating_pairs),
        "grating_pairs": grating_pairs,
        "warning": found["warning"],
        "_spectrum": {
            "grid": None if found["grid"] is None
            else found["grid"].tolist(),
            "spectrum": None if found["spectrum"] is None
            else found["spectrum"].tolist(),
        },
    }
