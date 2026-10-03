"""遗忘因子口径修复的端到端验收（题面复现场景）。

8 阵元、间距 0.5、波长 1，alpha=0.98，源数直接指定 1；两段单源 15 dB：
-20°/300 条/种子 1 与 +20°/300 条/种子 2，拼成 600 条。整块推、每包 16
条推、逐条推三种分包方式，最终角度 / 有效快拍数 / CRB 比值必须一致，且
与按定义（最新一条权重 1、往前每条乘 alpha）直接写出的 NumPy 协方差吻合。
另覆盖并发追加、去重、重启续追加、口径可见性与极端数值配置。
"""
from __future__ import annotations

import threading

import numpy as np
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.doa.covariance import (
    RULE_FORGETTING,
    ForgettingAccumulator,
    forgetting_covariance,
    hermitianize,
)
from app.doa.geometry import ULA
from app.doa.synthesis import synthesize

from .conftest import append_synth, create_session, do_estimate

ALPHA = 0.98


def _step_data():
    arr = ULA(8, 0.5, 1.0)
    x0 = synthesize(arr, [-20.0], [1.0], 15.0, 300, 1)
    x1 = synthesize(arr, [20.0], [1.0], 15.0, 300, 2)
    return arr, np.concatenate([x0, x1], axis=1)


def _push(client, X, sizes):
    sid = create_session(
        client, update={"method": "forgetting", "alpha": ALPHA}
    )
    pos = 0
    for i, k in enumerate(sizes):
        r = append_synth(client, sid, X[:, pos:pos + k], f"b{i}")
        assert r.status_code == 200, r.text
        pos += k
    assert pos == X.shape[1]
    rec = do_estimate(client, sid, source_mode=1,
                      true_angles=[20.0]).json()
    return sid, rec


def test_three_packetizations_give_identical_estimate(client):
    _, X = _step_data()
    sizes_16 = [16] * (600 // 16) + [600 % 16]
    _, whole = _push(client, X, [600])
    _, packs = _push(client, X, sizes_16)
    _, ones = _push(client, X, [1] * 600)

    for rec in (whole, packs, ones):
        assert len(rec["result"]["angles_deg"]) == 1
    # 角度经过 eigh + 黄金细化，浮点累积顺序差异体现在 1e-8 度量级；
    # 协方差层面的 1e-9 口径在下方逐元素核对。
    a = [rec["result"]["angles_deg"][0] for rec in (whole, packs, ones)]
    assert max(a) - min(a) < 1e-7, a
    n_effs = [rec["result"]["n_effective_snapshots"]
              for rec in (whole, packs, ones)]
    assert max(n_effs) - min(n_effs) < 1e-9, n_effs
    ratios = [rec["result"]["per_angle"][0]["error_over_crb"]
              for rec in (whole, packs, ones)]
    assert all(r is not None for r in ratios)
    # 比值由细化角度/CRB 导出，允许 1e-6 的传递误差
    assert max(ratios) - min(ratios) < 1e-6, ratios
    # 源数判定也用同一有效快拍数，三者 MDL/AIC 计数一致
    counts = [(rec["result"]["source_counts"]["mdl"],
               rec["result"]["source_counts"]["aic"])
              for rec in (whole, packs, ones)]
    assert counts[0] == counts[1] == counts[2]
    # 新数据主导：估计角在 +20° 一侧（旧整块口径曾报 -18.97°）
    assert a[0] > 19.0


def test_result_matches_external_numpy_definition(client):
    """服务内三种推法与外部 NumPy 按定义直接算的协方差/有效快拍一致。"""
    _, X = _step_data()
    _, rec = _push(client, X, [16] * 37 + [8])
    w = ALPHA ** np.arange(600, dtype=np.float64)[::-1]
    C_ref = hermitianize((X * w) @ X.conj().T / w.sum())

    sid = rec["session_id"]
    # 直接读会话内的协方差状态做逐元素比对
    from app.sessions.manager import SessionManager
    mgr = client.app.state.manager
    sess = mgr.get(sid)
    sess._load_state_if_needed()
    rel = np.linalg.norm((sess.acc.C - C_ref).ravel()) \
        / np.linalg.norm(C_ref.ravel())
    assert rel < 1e-9
    assert abs(sess.acc.n_eff - float(w.sum())) / float(w.sum()) < 1e-9
    assert abs(rec["result"]["n_effective_snapshots"] - float(w.sum())) \
        / float(w.sum()) < 1e-9
    # 与一次性批处理参考实现一致
    C_batch, n_batch = forgetting_covariance(X, ALPHA)
    assert np.allclose(sess.acc.C, C_batch, rtol=1e-12, atol=1e-14)
    assert abs(sess.acc.n_eff - n_batch) < 1e-9


def test_status_exposes_accumulation_rule(client):
    sid = create_session(
        client, update={"method": "forgetting", "alpha": ALPHA}
    )
    st = client.get(f"/sessions/{sid}").json()
    assert st["update"]["rule"] == RULE_FORGETTING
    assert st["accumulation_rule"] == RULE_FORGETTING
    assert st["append_frozen"] is False


def _push_alpha1(client, X, sizes):
    sid = create_session(
        client, update={"method": "forgetting", "alpha": 1.0}
    )
    pos = 0
    for i, k in enumerate(sizes):
        r = append_synth(client, sid, X[:, pos:pos + k], f"b{i}")
        assert r.status_code == 200, r.text
        pos += k
    return sid


def test_alpha_equals_one_chunked_matches_whole(client):
    _, X = _step_data()
    s1 = _push_alpha1(client, X, [600])
    s2 = _push_alpha1(client, X, [16] * 37 + [8])
    a = do_estimate(client, s1, source_mode=1).json()
    b = do_estimate(client, s2, source_mode=1).json()
    assert abs(a["result"]["angles_deg"][0]
               - b["result"]["angles_deg"][0]) < 1e-6
    assert abs(a["result"]["n_effective_snapshots"]
               - b["result"]["n_effective_snapshots"]) < 1e-9


def test_extreme_alpha_half_10k_batch_served_without_nan(client):
    """HTTP 路径：alpha=0.5、单批 10000 条，不 NaN/除零，可正常出估计。"""
    rng = np.random.default_rng(0)
    X = ((rng.standard_normal((8, 10_000)) +
          1j * rng.standard_normal((8, 10_000))) / np.sqrt(2))
    sid = create_session(
        client, update={"method": "forgetting", "alpha": 0.5}
    )
    r = append_synth(client, sid, X, "big")
    assert r.status_code == 200, r.text
    st = client.get(f"/sessions/{sid}").json()
    assert np.isfinite(st["n_effective_snapshots"])
    assert 0 < st["n_effective_snapshots"] < 3.0
    rec = do_estimate(client, sid, source_mode=1).json()
    assert np.isfinite(rec["result"]["angles_deg"][0])


def test_concurrent_forgetting_appends_match_some_serial_order(tmp_path):
    """同一会话两批并发追加：落盘协方差严格等于 AB 或 BA 某一串行顺序。"""
    from app.sessions.manager import SessionManager
    from app.sessions.models import (
        ArraySpec, EstimateSpec, UpdateSpec,
    )

    mgr = SessionManager(str(tmp_path / "fc"))
    spec = ArraySpec(m=8, spacing=0.5, wavelength=1.0)
    sess = mgr.create_session(
        spec, update=UpdateSpec(method="forgetting", alpha=ALPHA,
                                rule=RULE_FORGETTING),
        estimate=EstimateSpec(source_mode=1),
    )
    sid = sess.meta.session_id
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((8, 11)) +
         1j * rng.standard_normal((8, 11))) / np.sqrt(2)
    B = (rng.standard_normal((8, 13)) +
         1j * rng.standard_normal((8, 13))) / np.sqrt(2)

    barrier = threading.Barrier(2)
    errors = []

    def w(batch, bid):
        try:
            barrier.wait()
            sess.append(bid, batch)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=w, args=(A, "A")),
          threading.Thread(target=w, args=(B, "B"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors

    # 两种串行顺序的参考结果
    def serial(first, second):
        acc = ForgettingAccumulator(8, alpha=ALPHA)
        acc.add_batch(first)
        acc.add_batch(second)
        return acc.C, acc.n_eff

    C_ab, n_ab = serial(A, B)
    C_ba, n_ba = serial(B, A)

    mgr2 = SessionManager(str(tmp_path / "fc"))
    s2 = mgr2.get(sid)
    s2._load_state_if_needed()
    assert s2.batch_ids == {"A", "B"}
    matches_ab = np.allclose(s2.acc.C, C_ab, rtol=1e-12) \
        and abs(s2.acc.n_eff - n_ab) < 1e-9
    matches_ba = np.allclose(s2.acc.C, C_ba, rtol=1e-12) \
        and abs(s2.acc.n_eff - n_ba) < 1e-9
    assert matches_ab or matches_ba


def test_restart_then_continue_appending_keeps_one_definition(tmp_path):
    """重启后续追加：与不重启一直追加的会话协方差/有效快拍相同。"""
    data_dir = str(tmp_path / "rc")
    rng = np.random.default_rng(123)
    X = ((rng.standard_normal((8, 200)) +
          1j * rng.standard_normal((8, 200))) / np.sqrt(2))

    with TestClient(create_app(data_dir)) as c:
        sid = create_session(c, update={"method": "forgetting",
                                        "alpha": ALPHA})
        append_synth(c, sid, X[:, :120], "b0")
        append_synth(c, sid, X[:, 120:150], "b1")

    with TestClient(create_app(data_dir)) as c:
        st = c.get(f"/sessions/{sid}").json()
        assert st["accumulation_rule"] == RULE_FORGETTING
        # 重复批次号仍然幂等
        r = append_synth(c, sid, X[:, :120], "b0")
        assert r.json()["deduplicated"] is True
        append_synth(c, sid, X[:, 150:], "b2")
        rec = do_estimate(c, sid, source_mode=1).json()

    ref = ForgettingAccumulator(8, alpha=ALPHA)
    ref.add_batch(X)
    assert abs(rec["result"]["n_effective_snapshots"] - ref.n_eff) < 1e-9
    sess = c.app.state.manager.get(sid)
    sess._load_state_if_needed()
    assert np.allclose(sess.acc.C, ref.C, rtol=1e-12, atol=1e-14)
