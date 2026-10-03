"""升级前（v1）遗忘因子会话的迁移：冻结只读、口径可见、历史逐字不变。

数据目录严格按升级前格式落盘：config.json 带 ``"version": 1`` 且 update
里没有 rule 字段；state.npz 由旧版 to_state 写出（无 rule 键，C 按旧的
“批次等权、批间乘 alpha”递推得到）；results.jsonl 预先写入确定内容。
随后用真实 FastAPI 应用对该目录启动一次，验证：

* 会话仍能被列出、查询；状态里看得出按哪种规则累积、是否冻结；
* state.npz 与 results.jsonl 的旧内容逐字节不变，历史结果按版本号读出
  逐字相同（轮询 / SSE 补发路径同样成立）；
* 旧遗忘会话追加被拒（409 session_frozen），新批次不计入；
* 旧滑动窗口会话不受影响，升级后可正常继续追加；
* 用冻结前的旧状态仍可出估计（只读），不改动任何既有状态。
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.doa.covariance import (
    RULE_FORGETTING_LEGACY,
    RULE_SLIDING,
)
from app.sessions.storage import FileStore

from .conftest import append_synth, do_estimate


def _old_forgetting_update(C, n_eff, X, alpha):
    """升级前旧递推：整批等权，仅批间乘 alpha。"""
    k = X.shape[1]
    new_sum = alpha * n_eff * C + X @ X.conj().T
    n_eff = alpha * n_eff + k
    C = 0.5 * (new_sum / n_eff + (new_sum / n_eff).conj().T)
    return C, n_eff


def _build_legacy_forgetting_session(store: FileStore, sid: str,
                                     alpha: float, seed: int,
                                     n_packs: int) -> str:
    """按升级前的真实格式写一个旧遗忘因子会话（含一条历史结果）。"""
    rng = np.random.default_rng(seed)
    X = ((rng.standard_normal((8, 16 * n_packs)) +
          1j * rng.standard_normal((8, 16 * n_packs))) / np.sqrt(2.0))
    C = np.zeros((8, 8), dtype=np.complex128)
    n_eff = 0.0
    batch_ids = []
    for i in range(n_packs):
        part = X[:, 16 * i:16 * (i + 1)]
        C, n_eff = _old_forgetting_update(C, n_eff, part, alpha)
        batch_ids.append(f"frame-{i:04d}")

    config = {
        "session_id": sid,
        "name": f"legacy-{sid[:6]}",
        "created_at": 1_700_000_000.0 + seed,
        "updated_at": 1_700_100_000.0 + seed,
        "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0,
                  "position_errors": None},
        "update": {"method": "forgetting", "window": 256,
                   "alpha": alpha},  # 旧配置：没有 rule 字段
        "estimate": {"source_mode": "mdl", "angle_min": -90.0,
                     "angle_max": 90.0, "step_deg": 0.5,
                     "smoothing": False, "sub_len": None},
        "version": 1,
    }
    store.save_config(sid, config)
    # 旧版 state.npz：没有 rule 键
    store.save_state(sid, {
        "kind": "forgetting",
        "m": 8,
        "alpha": alpha,
        "C": C,
        "n_eff": float(n_eff),
        "total_snapshots": 16 * n_packs,
        "_batch_ids": np.asarray(sorted(batch_ids)),
    })
    # 预先写死一条历史结果（逐字节校验对象）
    record = {
        "version": 1,
        "session_id": sid,
        "created_at": 1_700_100_001.0 + seed,
        "batch_count": n_packs,
        "total_snapshots": 16 * n_packs,
        "config_used": {"source_mode": 1, "angle_min": -90.0,
                        "angle_max": 90.0, "step_deg": 0.5,
                        "smoothing": False, "sub_len": None},
        "result": {
            "angles_deg": [12.3456789 + 0.01 * seed],
            "n_sources": 1,
            "source_counts": {"mdl": 1, "aic": 1},
            "n_effective_snapshots": float(n_eff),
            "noise_power": 0.03162277660168379,
            "smoothing": {"enabled": False, "sub_len": None,
                          "n_subarrays": None, "snapshot_multiplier": 1},
            "grating_ambiguous": False,
            "grating_pairs": [],
            "warning": None,
            "per_angle": [],
        },
    }
    line = (json.dumps(record, ensure_ascii=False,
                       separators=(",", ":")) + "\n")
    d = store.ensure_session_dir(sid)
    with open(os.path.join(d, "results.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(line)
    return line


def _build_legacy_sliding_session(store: FileStore, sid: str) -> None:
    config = {
        "session_id": sid,
        "name": "legacy-sliding",
        "created_at": 1_700_000_000.0,
        "updated_at": 1_700_100_000.0,
        "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0,
                  "position_errors": None},
        "update": {"method": "sliding", "window": 256, "alpha": 0.9922},
        "estimate": {"source_mode": "mdl", "angle_min": -90.0,
                     "angle_max": 90.0, "step_deg": 0.5,
                     "smoothing": False, "sub_len": None},
        "version": 1,
    }
    store.save_config(sid, config)
    # 滑动旧会话从未追加过，不落 state.npz（冷会话也能被恢复）


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_legacy_forgetting_sessions_freeze_and_history_verbatim(tmp_path):
    data_dir = str(tmp_path / "data")
    store = FileStore(data_dir)
    legacy = [("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", 0.98, 101, 40),
              ("11112222333344445555666677778888", 0.9922, 202, 60),
              ("9999aaaabbbbccccddddeeeeffff0000", 0.95, 303, 20)]
    expected_lines = {}
    for sid, alpha, seed, packs in legacy:
        expected_lines[sid] = _build_legacy_forgetting_session(
            store, sid, alpha, seed, packs
        )
    sliding_sid = "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"
    _build_legacy_sliding_session(store, sliding_sid)

    state_hashes_before = {
        sid: _sha(os.path.join(store.session_dir(sid), "state.npz"))
        for sid, *_ in legacy
    }
    results_bytes_before = {
        sid: open(os.path.join(store.session_dir(sid), "results.jsonl"),
                  "rb").read()
        for sid, *_ in legacy
    }

    # ---- 用升级后的服务对升级前数据目录真实启动 ----
    with TestClient(create_app(data_dir)) as c:
        listed = {s["session_id"]: s
                  for s in c.get("/sessions").json()["sessions"]}
        assert set(listed) == {sid for sid, *_ in legacy} | {sliding_sid}

        for sid, alpha, _, packs in legacy:
            st = listed[sid]
            # 调用方一眼能看出当前按什么规则累积、能不能追加
            assert st["accumulation_rule"] == RULE_FORGETTING_LEGACY
            assert st["append_frozen"] is True
            assert st["update"]["rule"] == RULE_FORGETTING_LEGACY
            assert st["n_total_snapshots"] == 16 * packs

        # config 已升级为 v2 并带口径标记
        cfg = store.load_config(legacy[0][0])
        assert cfg["version"] == 2
        assert cfg["update"]["rule"] == RULE_FORGETTING_LEGACY

        # state.npz 在列出/查询后逐字节不变
        for sid, *_ in legacy:
            assert _sha(os.path.join(store.session_dir(sid), "state.npz")) \
                == state_hashes_before[sid]

        sid = legacy[0][0]
        # 轮询路径：历史结果逐字读回
        body = c.get(f"/sessions/{sid}/results",
                     params={"since_version": 0}).json()
        assert len(body["results"]) == 1
        returned_line = json.dumps(body["results"][0],
                                   ensure_ascii=False,
                                   separators=(",", ":")) + "\n"
        assert returned_line == expected_lines[sid]
        # 磁盘上的 results.jsonl 仍是原始字节
        assert open(os.path.join(store.session_dir(sid), "results.jsonl"),
                    "rb").read() == results_bytes_before[sid]

        # 冻结会话拒绝追加：409，且批次未计入
        rng = np.random.default_rng(0)
        fresh = (rng.standard_normal((8, 8)) +
                 1j * rng.standard_normal((8, 8)))
        r = append_synth(c, sid, fresh, "after-upgrade")
        assert r.status_code == 409, r.text
        assert r.json()["error"]["type"] == "session_frozen"
        assert c.get(f"/sessions/{sid}").json()["n_batches"] == 40

        # 旧批次号重放也不允许（冻结检查先于去重，状态不变）
        r = append_synth(c, sid, fresh, "frame-0000")
        assert r.status_code == 409

        # 用冻结的旧状态仍可只读地出估计；不改动既有状态/历史结果
        rec = do_estimate(c, sid, source_mode=1)
        assert rec.status_code == 200, rec.text
        assert rec.json()["version"] == 2
        assert open(os.path.join(store.session_dir(sid), "results.jsonl"),
                    "rb").read().startswith(results_bytes_before[sid])
        assert _sha(os.path.join(store.session_dir(sid), "state.npz")) \
            == state_hashes_before[sid]

        # 旧滑动窗口会话不受冻结影响，正常继续追加
        r = append_synth(c, sliding_sid, fresh + 0j, "post-upgrade")
        assert r.status_code == 200, r.text
        st = c.get(f"/sessions/{sliding_sid}").json()
        assert st["accumulation_rule"] == RULE_SLIDING
        assert st["append_frozen"] is False

    # ---- 再重启一次：冻结/口径标记持续生效 ----
    with TestClient(create_app(data_dir)) as c:
        st = c.get(f"/sessions/{sid}").json()
        assert st["accumulation_rule"] == RULE_FORGETTING_LEGACY
        assert st["append_frozen"] is True
        body = c.get(f"/sessions/{sid}/results",
                     params={"since_version": 0}).json()
        assert len(body["results"]) == 2
        assert json.dumps(body["results"][0], ensure_ascii=False,
                          separators=(",", ":")) + "\n" \
            == expected_lines[sid]


def test_legacy_session_history_served_over_sse(tmp_path):
    """SSE 补发路径同样逐字读出旧结果（真实 uvicorn 服务）。"""
    import httpx
    import socket
    import threading
    import uvicorn

    data_dir = str(tmp_path / "data")
    store = FileStore(data_dir)
    sid = "abcdef0123456789abcdef0123456789"
    expected_line = _build_legacy_forgetting_session(
        store, sid, 0.97, 55, 10
    )

    app = create_app(data_dir)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        for _ in range(100):
            if server.started:
                break
            threading.Event().wait(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",
                          timeout=10.0) as http:
            with http.stream("GET",
                             f"/sessions/{sid}/results/stream"
                             f"?after_version=0") as s:
                assert s.status_code == 200
                payload = None
                for line in s.iter_lines():
                    if line and line.startswith("data: "):
                        payload = line[6:]
                        break
            assert payload is not None
            assert json.dumps(json.loads(payload), ensure_ascii=False,
                              separators=(",", ":")) + "\n" == expected_line
    finally:
        server.should_exit = True
        th.join(timeout=10)
