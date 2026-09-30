"""合成接口与 SSE 订阅推送。"""
from __future__ import annotations

import json
import threading

import numpy as np

from app.doa.synthesis import synthesize
from app.doa.geometry import ULA

from .conftest import append_synth, create_session, do_estimate


def _synth_body(**over):
    body = {
        "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0},
        "sources": [
            {"angle_deg": -20.0, "power": 1.0},
            {"angle_deg": 25.0, "power": 1.0},
        ],
        "snr_db": 20.0,
        "n_snapshots": 400,
        "seed": 1234,
        "estimate": {"source_mode": 2},
        "include_data": True,
    }
    body.update(over)
    return body


def test_synthesis_bit_identical_for_seed(client):
    r1 = client.post("/synthesize", json=_synth_body())
    r2 = client.post("/synthesize", json=_synth_body())
    assert r1.status_code == 200, r1.text
    a1 = np.array(r1.json()["real"]) + 1j * np.array(r1.json()["imag"])
    a2 = np.array(r2.json()["real"]) + 1j * np.array(r2.json()["imag"])
    assert np.array_equal(a1, a2)
    # 与本地直接合成逐位一致（同一 RNG 流程）
    local = synthesize(ULA(8, 0.5, 1.0), [-20.0, 25.0], [1.0, 1.0],
                       20.0, 400, 1234)
    assert np.array_equal(a1, local)


def test_synthesis_different_seed_differs(client):
    r1 = client.post("/synthesize", json=_synth_body(seed=1))
    r2 = client.post("/synthesize", json=_synth_body(seed=2))
    a1 = np.array(r1.json()["real"])
    a2 = np.array(r2.json()["real"])
    assert not np.array_equal(a1, a2)


def test_synthesis_estimate_and_crb_ratio(client):
    r = client.post("/synthesize", json=_synth_body())
    est = r.json()["estimate"]
    angles = sorted(p["angle_deg"] for p in est["per_angle"])
    # 真值通过 true_angles 配对：合成接口内部自动传真值
    for p in est["per_angle"]:
        assert p["error_over_crb"] is not None
        assert p["crb_std_deg"] is not None and p["crb_std_deg"] > 0
    assert abs(angles[0] - (-20.0)) < 0.1
    assert abs(angles[1] - 25.0) < 0.1


def test_synthesis_coherent_flag(client):
    body = _synth_body(
        sources=[{"angle_deg": -8.0, "power": 1.0, "coherent": True},
                 {"angle_deg": 8.0, "power": 1.0, "coherent": True}],
        n_snapshots=600, seed=5,
        estimate={"source_mode": 2, "smoothing": True},
        include_data=False,
    )
    r = client.post("/synthesize", json=body)
    assert r.status_code == 200, r.text
    angles = sorted(p["angle_deg"]
                    for p in r.json()["estimate"]["per_angle"])
    assert abs(angles[0] + 8.0) < 1.0 and abs(angles[1] - 8.0) < 1.0


def test_synthesis_too_many_snapshots(client):
    r = client.post("/synthesize", json=_synth_body(n_snapshots=10001,
                                                   include_data=False))
    assert r.status_code == 400
    assert "10000" in r.json()["error"]["message"]


def test_sse_stream_pushes_new_estimates(tmp_path):
    """SSE 用真实 uvicorn 服务测（TestClient 单门户开流时不能并发请求）。"""
    import httpx
    import uvicorn
    import socket

    from app.api.main import create_app

    app = create_app(str(tmp_path / "data"))
    config = uvicorn.Config(app, host="127.0.0.1", port=0,
                            log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        for _ in range(100):
            if server.started:
                break
            assert th.is_alive(), "uvicorn 启动失败"
            threading.Event().wait(0.05)
        sock = server.servers[0].sockets[0]
        port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        with httpx.Client(base_url=base, timeout=10.0) as http, \
                httpx.Client(base_url=base, timeout=10.0) as stream_http:
            r = http.post("/sessions", json={
                "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0}
            })
            sid = r.json()["session_id"]
            rng = np.random.default_rng(0)
            X = rng.standard_normal((8, 64)) + \
                1j * rng.standard_normal((8, 64))

            received: list[dict] = []

            def reader():
                with stream_http.stream("GET",
                                 f"/sessions/{sid}/results/stream"
                                 f"?after_version=0") as s:
                    assert s.status_code == 200
                    event_lines = []
                    for line in s.iter_lines():
                        if line is None or line == "":
                            event_lines = []
                            continue
                        if line.startswith(":"):  # keepalive
                            continue
                        if line.startswith("data: "):
                            received.append(json.loads(line[6:]))
                            if len(received) >= 2:
                                return

            rt = threading.Thread(target=reader, daemon=True)
            rt.start()

            def post_batch(Xpart, bid):
                rr = http.post(f"/sessions/{sid}/batches", json={
                    "batch_id": bid,
                    "real": Xpart.real.tolist(),
                    "imag": Xpart.imag.tolist(),
                })
                assert rr.status_code == 200, rr.text

            post_batch(X, "b0")
            r1 = http.post(f"/sessions/{sid}/estimate",
                           json={"source_mode": 2})
            assert r1.status_code == 200, r1.text
            # 确认订阅者已经收到第一条再发第二批（避免顺序竞态误判）
            for _ in range(100):
                if received:
                    break
                threading.Event().wait(0.05)
            post_batch(X[:, :8], "b1")
            r2 = http.post(f"/sessions/{sid}/estimate",
                           json={"source_mode": 2})
            assert r2.status_code == 200, r2.text

            rt.join(timeout=10)
            assert not rt.is_alive()
            assert [r["version"] for r in received] == [1, 2]

            # 带版本订阅：历史结果先补发
            hist = http.get(f"/sessions/{sid}/results",
                            params={"since_version": 0}).json()
            assert len(hist["results"]) == 2
    finally:
        server.should_exit = True
        th.join(timeout=10)
