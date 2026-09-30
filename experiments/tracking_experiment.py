"""滑动窗口 vs 指数遗忘：信源角度阶跃的跟踪实验。

场景（八阵元半波长 ULA，单源，白噪声）：

* 第 0..T0-1 条快拍来波在 −20°，第 T0 条起跳变到 +20°；
* sliding：窗口长度 W；forgetting：alpha = (W-1)/(W+1)，使两者等效
  记忆深度同为 W 条快拍；
* 每 16 条快拍做一次 MUSIC 估计（MDL/AIC 关闭，直接 D=1）。

指标（多组种子平均）：

* 跟拍延迟 lag_1.0°：阶跃后估计首次持续进入新角 ±1° 时，距阶跃点的
  快拍数；
* 稳态 RMSE：阶跃后 3W .. 6W 区间内估计相对 +20° 的均方根误差（度）。

用法：python -m experiments.tracking_experiment
结果写到 experiments/results/tracking_results.json，并打印 Markdown 表。
统计使用固定种子。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.doa.covariance import (  # noqa: E402
    ForgettingAccumulator,
    SlidingWindowAccumulator,
)
from app.doa.engine import EstimateConfig, run_doa  # noqa: E402
from app.doa.geometry import ULA  # noqa:E402
from app.doa.synthesis import synthesize  # noqa: E402

M = 8
THETA0, THETA1 = -20.0, 20.0
SNR_DB = 15.0
# 两种阶跃前历史长度：
#   warm  ：阶跃前只有 2W 条（服务刚跑起来）；
#   settled：阶跃前已有 ≥10W 条（稳态常驻后端），暴露遗忘因子长拖尾。
THORIZON = 14_000
EST_EVERY = 16
SEEDS = [101, 202, 303, 404, 505]
WINDOWS = [128, 256, 512]
LATCH_TOL_DEG = 1.0


def _stream(array, seed: int, t0: int, horizon: int):
    """阶跃前后两段合成后拼接（同一种子流的连续两段）。"""
    x0 = synthesize(array, [THETA0], [1.0], SNR_DB, t0, seed)
    x1 = synthesize(array, [THETA1], [1.0], SNR_DB, horizon - t0,
                    seed + 7919)
    return np.concatenate([x0, x1], axis=1)


def _run_one(method: str, w: int, seed: int, t0: int, horizon: int):
    array = ULA(M, 0.5, 1.0)
    X = _stream(array, seed, t0, horizon)
    if method == "sliding":
        acc = SlidingWindowAccumulator(M, window=w)
    else:
        alpha = (w - 1) / (w + 1)
        acc = ForgettingAccumulator(M, alpha=alpha)
    cfg = EstimateConfig(source_mode=1)
    times, angles = [], []
    for end in range(EST_EVERY, horizon + 1, EST_EVERY):
        acc.add_batch(X[:, end - EST_EVERY:end])
        if end < M:
            continue
        r = run_doa(array, acc.C, float(acc.n_eff), cfg)
        if r["angles_deg"]:
            times.append(end)
            angles.append(r["angles_deg"][0])
    times = np.asarray(times)
    angles = np.asarray(angles)

    # 跟拍延迟：阶跃后连续 3 次估计都在新角 ±tol 内的首次时刻
    lag = None
    post = times >= t0
    tp, ap = times[post], angles[post]
    ok = np.abs(ap - THETA1) < LATCH_TOL_DEG
    for i in range(len(ok) - 2):
        if ok[i] and ok[i + 1] and ok[i + 2]:
            lag = int(tp[i] - t0)
            break

    # 暂态后稳态 RMSE：阶跃后 6W..12W（遗忘因子要多个时间常数才沉底）
    sel = (tp >= t0 + 6 * w) & (tp <= t0 + 12 * w)
    rmse = float(np.sqrt(np.mean(
        np.square(ap[sel] - THETA1)
    ))) if np.any(sel) else float("nan")

    # 另外记录阶跃后 0..3W 的暂态 RMSE（体现拖尾）
    sel_tr = (tp >= t0) & (tp <= t0 + 3 * w)
    rmse_transient = float(np.sqrt(np.mean(
        np.square(ap[sel_tr] - THETA1)
    ))) if np.any(sel_tr) else float("nan")

    # 阶跃前稳态偏差 RMSE（对照）
    pre = (times >= t0 - min(3 * w, t0)) & (times < t0)
    rmse_pre = float(np.sqrt(np.mean(
        np.square(angles[pre] - THETA0)
    ))) if np.any(pre) else float("nan")
    return lag, rmse, rmse_pre, rmse_transient


def main() -> None:
    scenarios = [
        ("warm", lambda w: (2 * w, min(THORIZON, 16 * w))),
        ("settled", lambda w: (10 * w, min(THORIZON, 24 * w))),
    ]
    rows = []
    for scenario, t0_fn in scenarios:
        for w in WINDOWS:
            t0, horizon = t0_fn(w)
            for method, label in (
                    ("sliding", f"滑动窗口 W={w}"),
                    ("forgetting",
                     f"遗忘因子 α={(w-1)/(w+1):.4f}（记忆{w}）")):
                lags, rmses, rmses_pre, rmses_tr = [], [], [], []
                for seed in SEEDS:
                    lag, rmse, rmse_pre, rmse_tr = _run_one(
                        method, w, seed, t0, horizon)
                    lags.append(lag)
                    rmses.append(rmse)
                    rmses_pre.append(rmse_pre)
                    rmses_tr.append(rmse_tr)
                rows.append({
                    "scenario": scenario,
                    "pre_step_snapshots": t0,
                    "method": method,
                    "memory": w,
                    "alpha": (w - 1) / (w + 1),
                    "lag_snapshots_p90": int(np.nanpercentile(
                        [l for l in lags if l is not None], 90))
                    if any(l is not None for l in lags) else None,
                    "lag_snapshots_max": int(np.nanmax(
                        [l for l in lags if l is not None]))
                    if any(l is not None for l in lags) else None,
                    "steady_rmse_deg_post_mean": float(np.mean(rmses)),
                    "steady_rmse_deg_pre_mean": float(np.mean(rmses_pre)),
                    "transient_rmse_deg_0_3w_mean": float(np.mean(rmses_tr)),
                    "steady_window_note": "阶跃后 6W..12W",
                    "seeds": SEEDS,
                    "snr_db": SNR_DB,
                    "theta0": THETA0,
                    "theta1": THETA1,
                    "label": label,
                })

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tracking_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("| 场景 | 方式 | 记忆 | 跟拍延迟 P90/最大 | 暂态RMSE 0–3W | "
          "稳态RMSE 阶跃后 | 阶跃前 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {r['label']} | {r['memory']} | "
              f"{r['lag_snapshots_p90']} / {r['lag_snapshots_max']} | "
              f"{r['transient_rmse_deg_0_3w_mean']:.4f} | "
              f"{r['steady_rmse_deg_post_mean']:.4f} | "
              f"{r['steady_rmse_deg_pre_mean']:.4f} |")


if __name__ == "__main__":
    main()
