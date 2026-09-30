"""人读错误、规模上限与其他边界条件。"""
from __future__ import annotations

import numpy as np
import pytest

from app.doa.covariance import check_snapshots, CovarianceError
from app.doa.engine import EstimateConfig, EstimationError, run_doa
from app.doa.geometry import GeometryError, ULA

from .conftest import append_synth, create_session, do_estimate


def _err(resp):
    assert resp.status_code == 400, resp.text
    return resp.json()["error"]["message"]


# ----------------------------------------------------- 信源数 >= 阵元数
def test_source_count_ge_elements_rejected(client):
    sid = create_session(client)
    X = np.random.default_rng(0).standard_normal((8, 32))
    append_synth(client, sid, X + 0j, "b")
    msg = _err(do_estimate(client, sid, source_mode=8))
    assert "信源数" in msg and "阵元数" in msg


# ------------------------------------------- 首次估计累计快拍 < 阵元数
def test_first_estimate_needs_at_least_m_snapshots(client):
    sid = create_session(client)
    X = np.random.default_rng(1).standard_normal((8, 5))
    append_synth(client, sid, X + 0j, "b")
    msg = _err(do_estimate(client, sid, source_mode=2))
    assert "阵元数" in msg and "快拍" in msg


# ------------------------------------------------- 波长/间距不为正
@pytest.mark.parametrize("spacing,wavelength", [(0.0, 1.0), (-0.5, 1.0),
                                                (0.5, 0.0), (0.5, -2.0)])
def test_nonpositive_spacing_or_wavelength(client, spacing, wavelength):
    r = client.post("/sessions", json={
        "array": {"m": 8, "spacing": spacing, "wavelength": wavelength}
    })
    assert r.status_code == 400, r.text
    msg = r.json()["error"]["message"]
    assert "间距" in msg or "波长" in msg


# ------------------------------------------------------------- NaN
def test_nan_snapshots_rejected(client):
    sid = create_session(client)
    real = np.ones((8, 10)); real[3, 4] = np.nan
    imag = np.zeros((8, 10))
    r = append_synth(client, sid, real + 1j * imag, "nan")
    assert r.status_code == 400
    assert "NaN" in r.json()["error"]["message"]


def test_nan_kernel_rejected():
    X = np.full((4, 8), np.nan, dtype=np.complex128)
    with pytest.raises(CovarianceError, match="NaN"):
        check_snapshots(X)


# ----------------------------------------------- 实部虚部形状不一致
def test_real_imag_shape_mismatch(client):
    sid = create_session(client)
    r = client.post(f"/sessions/{sid}/batches", json={
        "batch_id": "shape",
        "real": np.zeros((8, 10)).tolist(),
        "imag": np.zeros((8, 9)).tolist(),
    })
    assert r.status_code == 400
    assert "形状" in r.json()["error"]["message"]


# -------------------------------------------------------- 规模上限
def test_more_than_64_elements_rejected(client):
    r = client.post("/sessions", json={
        "array": {"m": 65, "spacing": 0.5, "wavelength": 1.0}
    })
    assert r.status_code == 400
    assert "64" in r.json()["error"]["message"]


def test_more_than_10000_snapshots_per_batch_rejected(client):
    sid = create_session(client)
    r = client.post(f"/sessions/{sid}/batches", json={
        "batch_id": "big",
        "real": np.zeros((8, 10001)).tolist(),
        "imag": np.zeros((8, 10001)).tolist(),
    })
    assert r.status_code == 400
    assert "10000" in r.json()["error"]["message"]


def test_scan_points_over_limit_rejected(client):
    sid = create_session(client)
    X = np.random.default_rng(2).standard_normal((8, 20))
    append_synth(client, sid, X + 0j, "b")
    # 180° / 1e-4 ≈ 180 万点
    msg = _err(do_estimate(client, sid, source_mode=2, step_deg=1e-4))
    assert "扫描点数" in msg


def test_smoothing_source_limit_enforced(client):
    sid = create_session(client, estimate={"smoothing": True, "sub_len": 4})
    X = np.random.default_rng(3).standard_normal((8, 40))
    append_synth(client, sid, X + 0j, "b")
    msg = _err(do_estimate(client, sid, source_mode=4))
    assert "子阵" in msg


# ------------------------------------------------------- 会话不存在
def test_unknown_session_404(client):
    r = client.get("/sessions/deadbeef")
    assert r.status_code == 404


# ----------------------------------------------------- 位置错误长度
def test_position_error_wrong_length(client):
    r = client.post("/sessions", json={
        "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0,
                  "position_errors": [0.0, 0.1]}
    })
    assert r.status_code == 400
    assert "位置误差" in r.json()["error"]["message"]


# ------------------------------------------------- 细化比步长细一阶
def test_refinement_resolution_finer_than_step():
    """真值偏离粗网格时，细化后误差远小于步长（至少细一个数量级）。"""
    arr = ULA(8, 0.5, 1.0)
    from app.doa.synthesis import synthesize
    # 18.37° 不落在 0.5° 网格上
    X = synthesize(arr, [18.37], [1.0], 30.0, 2000, seed=2024)
    n = X.shape[1]
    C = (X @ X.conj().T) / n
    res = run_doa(arr, C, n, EstimateConfig(source_mode=1, step_deg=0.5))
    err = abs(res["angles_deg"][0] - 18.37)
    assert err < 0.05, err  # 步长 0.5° 的十分之一


def test_position_error_geometry_direct():
    with pytest.raises(GeometryError):
        ULA(8, 0.5, 1.0, position_errors=np.zeros(7))
