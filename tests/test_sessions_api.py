"""会话 API：追加幂等、并发串行等价、会话隔离、持久化恢复、轮询。"""
from __future__ import annotations

import threading

import numpy as np

from app.doa.synthesis import synthesize

from .conftest import append_synth, create_session, do_estimate


def _make(snr=15.0, k=64, seed=1):
    from app.doa.geometry import ULA
    return synthesize(ULA(8, 0.5, 1.0), [-10.0, 20.0], [1.0, 1.0],
                      snr, k, seed)


def test_create_session_and_status(client):
    sid = create_session(client)
    st = client.get(f"/sessions/{sid}").json()
    assert st["array"]["m"] == 8
    assert st["n_total_snapshots"] == 0
    assert st["update"]["method"] == "sliding"


def test_duplicate_batch_id_counted_once(client):
    X = _make(k=64)
    sid = create_session(client)
    r1 = append_synth(client, sid, X, "dup-1")
    r2 = append_synth(client, sid, X, "dup-1")
    assert r1.json()["deduplicated"] is False
    assert r2.json()["deduplicated"] is True
    assert r2.json()["accepted_snapshots"] == 0
    st = client.get(f"/sessions/{sid}").json()
    assert st["n_total_snapshots"] == 64
    assert st["n_batches"] == 1


def test_concurrent_same_batch_id_counted_once(client):
    X = _make(k=32)
    sid = create_session(client)
    results = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        results.append(append_synth(client, sid, X, "same-id").json())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    accepted = [r for r in results if not r["deduplicated"]]
    assert len(accepted) == 1
    assert client.get(f"/sessions/{sid}").json()["n_total_snapshots"] == 32


def test_concurrent_distinct_batches_no_loss_no_double_count(client):
    sid = create_session(client)
    Xs = [_make(k=10, seed=i) for i in range(8)]
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        append_synth(client, sid, Xs[i], f"b{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = client.get(f"/sessions/{sid}").json()
    # 8 批 × 10 条，不丢不重
    assert st["n_total_snapshots"] == 80
    assert st["n_batches"] == 8
    # 重放（同 batch_id）不改变状态
    for i in range(8):
        append_synth(client, sid, Xs[i], f"b{i}")
    assert client.get(f"/sessions/{sid}").json()["n_total_snapshots"] == 80


def test_sessions_are_isolated(client):
    s1, s2 = create_session(client), create_session(client)
    X1 = _make(k=40, seed=1)
    X2 = _make(k=55, seed=2)
    append_synth(client, s1, X1, "a")
    append_synth(client, s2, X2, "z")
    st1 = client.get(f"/sessions/{s1}").json()
    st2 = client.get(f"/sessions/{s2}").json()
    assert st1["n_total_snapshots"] == 40
    assert st2["n_total_snapshots"] == 55
    assert set(s["session_id"] for s in client.get("/sessions").json()["sessions"]) >= {s1, s2}


def test_concurrent_batches_state_matches_a_serial_order(tmp_path):
    """两批并发追加：落盘窗口严格等于 AB 或 BA 中某一种串行顺序。"""
    from app.sessions.manager import SessionManager
    from app.sessions.models import (
        ArraySpec, EstimateSpec, UpdateSpec,
    )

    mgr = SessionManager(str(tmp_path / "ser"))
    spec = ArraySpec(m=8, spacing=0.5, wavelength=1.0)
    sess = mgr.create_session(
        spec, update=UpdateSpec(method="sliding", window=12),
        estimate=EstimateSpec(source_mode=2),
    )
    sid = sess.meta.session_id
    rng = np.random.default_rng(7)
    seed_b = (rng.standard_normal((8, 9)) +
              1j * rng.standard_normal((8, 9))) / np.sqrt(2)
    A = (rng.standard_normal((8, 5)) +
         1j * rng.standard_normal((8, 5))) / np.sqrt(2)
    B = (rng.standard_normal((8, 3)) +
         1j * rng.standard_normal((8, 3))) / np.sqrt(2)
    mgr.append(sid, "seed", seed_b.real, seed_b.imag)

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

    # 恢复后直接读落盘窗口快拍
    mgr2 = SessionManager(str(tmp_path / "ser"))
    s2 = mgr2.get(sid)
    s2._load_state_if_needed()
    buf = np.stack(list(s2.acc._buffer), axis=1)

    full = np.concatenate([seed_b, A, B], axis=1)
    win_ab = full[:, -12:]
    full_ba = np.concatenate([seed_b, B, A], axis=1)
    win_ba = full_ba[:, -12:]
    assert (np.allclose(buf, win_ab) and s2.batch_ids == {"seed", "A", "B"}) \
        or np.allclose(buf, win_ba)
    # 协方差与窗口内容自洽
    assert np.allclose(s2.acc.C,
                       buf @ buf.conj().T / buf.shape[1])


def test_persistence_reload_keeps_state_and_results(tmp_path):
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    data_dir = str(tmp_path / "persist")
    with TestClient(create_app(data_dir)) as c:
        sid = create_session(c)
        append_synth(c, sid, _make(k=64), "b0")
        r = do_estimate(c, sid, source_mode=2)
        assert r.status_code == 200, r.text
        version_before = r.json()["version"]
        angles_before = r.json()["result"]["angles_deg"]

    # 服务重启：重建 manager，扫描目录恢复
    with TestClient(create_app(data_dir)) as c:
        st = c.get(f"/sessions/{sid}").json()
        assert st["n_total_snapshots"] == 64
        recs = c.get(f"/sessions/{sid}/results",
                     params={"since_version": 0}).json()
        assert len(recs["results"]) == 1
        assert recs["results"][0]["version"] == version_before
        assert recs["results"][0]["result"]["angles_deg"] == angles_before
        # 继续追加，版本号接着涨
        append_synth(c, sid, _make(k=16, seed=9), "b1")
        r2 = do_estimate(c, sid, source_mode=2).json()
        assert r2["version"] == version_before + 1


def test_polling_returns_only_newer_versions(client):
    sid = create_session(client)
    append_synth(client, sid, _make(k=64), "b0")
    do_estimate(client, sid, source_mode=2)
    append_synth(client, sid, _make(k=16, seed=2), "b1")
    do_estimate(client, sid, source_mode=2)
    body = client.get(f"/sessions/{sid}/results",
                      params={"since_version": 1}).json()
    assert len(body["results"]) == 1
    assert body["results"][0]["version"] == 2
    assert body["latest_version"] == 2
    body0 = client.get(f"/sessions/{sid}/results",
                       params={"since_version": 2}).json()
    assert body0["results"] == []


def test_estimate_version_monotonic_and_config_overrides(client):
    sid = create_session(client)
    append_synth(client, sid, _make(k=128), "b0")
    r = do_estimate(client, sid, source_mode="aic").json()
    assert r["config_used"]["source_mode"] == "aic"
    r2 = do_estimate(client, sid, step_deg=0.2,
                     angle_min=-60, angle_max=60).json()
    assert r2["version"] == r["version"] + 1
    assert r2["config_used"]["step_deg"] == 0.2


def test_estimate_with_true_angles_returns_crb_ratio(client):
    sid = create_session(client)
    append_synth(client, sid, _make(k=300), "b0")
    r = do_estimate(client, sid, source_mode=2,
                    true_angles=[-10.0, 20.0]).json()
    for item in r["result"]["per_angle"]:
        assert item["crb_std_deg"] is not None
        assert item["error_over_crb"] is not None
        assert item["error_over_crb"] >= 0.0
        assert item["true_angle_deg"] in (-10.0, 20.0)


def test_growing_average_chunked_matches_whole(client):
    """alpha=1（无遗忘，等价增长样本协方差）：整批与分批逐元素一致。"""
    X = _make(k=200)
    kw = {"method": "forgetting", "alpha": 1.0}
    s1 = create_session(client, update=kw)
    append_synth(client, s1, X, "all")
    s2 = create_session(client, update=kw)
    for i, (lo, hi) in enumerate([(0, 23), (23, 100), (100, 200)]):
        append_synth(client, s2, X[:, lo:hi], f"b{i}")
    a = do_estimate(client, s1, source_mode=2).json()["result"]["angles_deg"]
    b = do_estimate(client, s2, source_mode=2).json()["result"]["angles_deg"]
    assert np.allclose(sorted(a), sorted(b), atol=1e-9)


def test_position_error_array_runs(client):
    sid = create_session(client,
                         position_errors=[0.0, 0.005, -0.004, 0.0,
                                          0.003, 0.0, -0.002, 0.001])
    append_synth(client, sid, _make(k=128), "b0")
    r = do_estimate(client, sid, source_mode=2)
    assert r.status_code == 200, r.text
    assert len(r.json()["result"]["angles_deg"]) == 2
