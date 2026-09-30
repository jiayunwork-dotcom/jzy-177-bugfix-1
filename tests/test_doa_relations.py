"""逐条核对需求里的测向关系（统计用例全部固定种子）。"""
from __future__ import annotations

import numpy as np

from app.doa.covariance import batch_covariance
from app.doa.engine import EstimateConfig, run_doa
from app.doa.geometry import ULA
from app.doa.synthesis import synthesize

from .conftest import append_synth, create_session, do_estimate


def _angles_of(result):
    return sorted(item["angle_deg"] for item in result["per_angle"])


def _matched_errors(est_angles, truth):
    """最近邻贪心配对，返回与 truth 等长的绝对误差。"""
    remaining = list(est_angles)
    errs = []
    for t in truth:
        j = int(np.argmin([abs(x - t) for x in remaining]))
        errs.append(abs(remaining.pop(j) - t))
    return errs


# ---------------------------------------------------------------- 1. 偏差
def test_high_snr_well_separated_bias_under_0p1_deg(array8, make_data):
    truth = [-25.0, 30.0]
    X = make_data(truth, snr=20.0, k=500, seed=42)
    C, n = batch_covariance(X)
    res = run_doa(array8, C, n, EstimateConfig(source_mode=2))
    errs = _matched_errors(_angles_of(res), truth)
    assert max(errs) < 0.1, errs


# ------------------------------------------------------- 2. 复标量不变性
def test_complex_scaling_leaves_angles_unchanged(array8, make_data):
    X = make_data([-10.0, 20.0], snr=15.0, k=400, seed=11)
    C1, n1 = batch_covariance(X)
    gamma = -3.2 + 7.5j  # 任意非零复数
    C2, n2 = batch_covariance(gamma * X)
    r1 = run_doa(array8, C1, n1, EstimateConfig(source_mode=2))
    r2 = run_doa(array8, C2, n2, EstimateConfig(source_mode=2))
    a1, a2 = _angles_of(r1), _angles_of(r2)
    assert np.allclose(a1, a2, atol=1e-7), (a1, a2)


def test_complex_scaling_invariant_over_http(client, make_data):
    X = make_data([-10.0, 20.0], snr=15.0, k=400, seed=11)
    s1 = create_session(client)
    s2 = create_session(client)
    gamma = -3.2 + 7.5j
    append_synth(client, s1, X, "x")
    append_synth(client, s2, gamma * X, "gx")
    a1 = do_estimate(client, s1, source_mode=2).json()["result"]["angles_deg"]
    a2 = do_estimate(client, s2, source_mode=2).json()["result"]["angles_deg"]
    assert np.allclose(sorted(a1), sorted(a2), atol=1e-7)


# ----------------------------------------------------------- 3. 栅瓣模糊
def test_spacing_over_half_wavelength_reports_grating_pairs():
    arr = ULA(8, 1.0, 1.0)  # d = lambda
    X = synthesize(arr, [10.0], [1.0], 30.0, 400, seed=3)
    C, n = batch_covariance(X)
    res = run_doa(arr, C, n, EstimateConfig(source_mode=1))
    assert res["grating_ambiguous"] is True
    assert len(res["grating_pairs"]) == 1
    pair = res["grating_pairs"][0]
    # sin(theta') = sin(theta) +/- lambda/d = sin(10°) +/- 1
    expected = np.rad2deg(np.arcsin(np.sin(np.deg2rad(10.0)) - 1.0))
    assert any(abs(a - expected) < 0.05
               for a in pair["confused_with_deg"]), pair


def test_half_wavelength_no_grating_ambiguity(array8, make_data):
    X = make_data([10.0], snr=30.0, k=400, seed=3)
    C, n = batch_covariance(X)
    res = run_doa(array8, C, n, EstimateConfig(source_mode=1))
    assert res["grating_ambiguous"] is False
    assert res["grating_pairs"] == []


# ------------------------------------------------------------- 4. 变号
def test_sign_flip_of_angles(array8, make_data):
    """所有信源方位变号等价于快拍取共轭（阵列坐标为实数）：

    conj(a(theta)) = a(-theta)，因此 C' = conj(C) 时 MUSIC 谱满足
    P(-theta; C') = P(theta; C)，估计角应精确互为相反数。
    """
    X = make_data([-25.0, 30.0], snr=20.0, k=500, seed=42)
    C1, n1 = batch_covariance(X)
    C2, n2 = batch_covariance(np.conj(X))
    r1 = run_doa(array8, C1, n1, EstimateConfig(source_mode=2))
    r2 = run_doa(array8, C2, n2, EstimateConfig(source_mode=2))
    ap = np.array(_angles_of(r1))
    an = np.array(_angles_of(r2))
    assert np.allclose(an, -ap[::-1], atol=1e-9), (ap, an)

    # 合成接口层面同样成立：直接造一组 -theta 数据做交叉校验（统计容差）
    Xn = make_data([25.0, -30.0], snr=20.0, k=500, seed=42)
    Cn, nn = batch_covariance(Xn)
    rn = run_doa(array8, Cn, nn, EstimateConfig(source_mode=2))
    assert max(_matched_errors(_angles_of(rn), [25.0, -30.0])) < 0.1


# -------------------------------------- 5. 相干源：不开平滑失败/开了成功
def test_coherent_sources_need_smoothing(array8):
    X = synthesize(array8, [-8.0, 8.0], [1.0, 1.0], 20.0, 600, seed=123,
                   coherent_flags=[True, True])
    C, n = batch_covariance(X)
    r_no = run_doa(array8, C, n,
                   EstimateConfig(source_mode=2, smoothing=False))
    # 不开平滑：要么不足两个峰，要么偏差很大
    a_no = _angles_of(r_no)
    unresolved = len(a_no) < 2 or max(_matched_errors(a_no, [-8.0, 8.0])) > 2.0
    assert unresolved, a_no

    r_fb = run_doa(array8, C, n,
                   EstimateConfig(source_mode=2, smoothing=True))
    a_fb = _angles_of(r_fb)
    assert len(a_fb) == 2
    assert max(_matched_errors(a_fb, [-8.0, 8.0])) < 1.0, a_fb


# ---------------------- 6. RMSE 随 SNR 下降，高 SNR 段 ≤3×CRB
def test_rmse_decreases_with_snr_and_within_3crb(array8):
    truth = [-25.0, 30.0]
    snrs = [0, 5, 10, 15, 20]
    n_trials = 40
    rmses = []
    ratios_high = []
    for snr in snrs:
        all_err = []
        crbs = []
        for seed in range(n_trials):
            X = synthesize(array8, truth, [1.0, 1.0], float(snr), 300,
                           1000 + seed)
            C, n = batch_covariance(X)
            r = run_doa(array8, C, n, EstimateConfig(source_mode=2))
            used, remaining = set(), _angles_of(r)
            for t in truth:
                j = int(np.argmin([abs(x - t) for x in remaining]))
                x = remaining.pop(j)
                item = next(p for p in r["per_angle"]
                            if p["angle_deg"] == x)
                all_err.append(x - t)
                crbs.append(item["crb_std_deg"])
        rmse = float(np.sqrt(np.mean(np.square(all_err))))
        rmses.append(rmse)
        if snr >= 15:
            ratios_high.append(rmse / float(np.mean(crbs)))
    # 严格单调下降
    for a, b in zip(rmses, rmses[1:]):
        assert b < 0.75 * a, rmses
    assert max(ratios_high) < 3.0, ratios_high


# ----------------------------------------------------------- 7. MDL 源数
def test_mdl_correct_at_high_snr(array8, make_data):
    X = make_data([-20.0, 5.0, 25.0], snr=20.0, k=800, seed=99,
                  powers=[1.0, 1.0, 1.0])
    C, n = batch_covariance(X)
    res = run_doa(array8, C, n, EstimateConfig(source_mode="mdl"))
    assert res["source_counts"]["mdl"] == 3
    assert res["n_sources"] == 3


# --------------------------------------------- 8. 整批 vs 分批估计一致
def test_whole_batch_vs_streamed_batches(client, make_data):
    X = make_data([-15.0, 25.0], snr=18.0, k=600, seed=77)
    sid = create_session(client)
    r = append_synth(client, sid, X, "b0")
    assert r.status_code == 200, r.text
    whole = do_estimate(client, sid, source_mode=2).json()

    sid2 = create_session(client)
    cuts = [0, 37, 200, 213, 600]
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        rr = append_synth(client, sid2, X[:, lo:hi], f"b{i}")
        assert rr.status_code == 200, rr.text
    chunked = do_estimate(client, sid2, source_mode=2).json()

    a1 = sorted(x["angle_deg"] for x in whole["result"]["per_angle"])
    a2 = sorted(x["angle_deg"] for x in chunked["result"]["per_angle"])
    assert np.allclose(a1, a2, atol=1e-6), (a1, a2)
