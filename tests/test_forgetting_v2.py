"""遗忘因子按快拍口径（v2）的端到端验收与旧口径会话升级行为。

* 同一批快拍整批/每包 16 条/逐条推入，角度、有效快拍数、CRB 比值一致，
  且与外部 NumPy 按定义直接算出的协方差吻合；
* 极端配置（alpha=0.5、单批一万条）不出 NaN；
* 升级前格式（无 rule 标记、按批遗忘 v1）的数据目录真实启动：旧会话
  可列出、历史结果逐字保留、状态标明旧口径且冻结只读、追加回 409。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.doa.covariance import hermitianize
from app.doa.engine import EstimateConfig, run_doa
from app.doa.geometry import ULA
from app.doa.synthesis import synthesize
from app.sessions.storage import FileStore

from .conftest import append_synth, create_session, do_estimate

ALPHA = 0.98
FORGETTING = {"method": "forgetting", "alpha": ALPHA}


def _step_data():
    """题面复现数据：-20° 300 条（种子 1）接 +20° 300 条（种子 2）。"""
    array = ULA(8, 0.5, 1.0)
    x1 = synthesize(array, [-20.0], [1.0], 15.0, 300, 1)
    x2 = synthesize(array, [20.0], [1.0], 15.0, 300, 2)
    return np.concatenate([x1, x2], axis=1)


def _numpy_reference(X, alpha):
    """按定义直接算：最新一条权重 1，每往前一条乘 alpha。"""
    n = X.shape[1]
    w = alpha ** np.arange(n - 1, -1, -1, dtype=np.float64)
    return (X * w) @ X.conj().T / w.sum(), float(w.sum())


def test_three_push_patterns_agree_with_numpy_reference(client):
    """题面复现：整块 / 每包 16 条 / 逐条，三种推法结果一致且对得上参考。"""
    X = _step_data()
    sids = {}
    for name, chunk in (("whole", 600), ("by16", 16), ("by1", 1)):
        sid = create_session(client, update=FORGETTING,
                             estimate={"source_mode": 1})
        for i, lo in enumerate(range(0, 600, chunk)):
            r = append_synth(client, sid, X[:, lo:lo + chunk], f"b{i}")
            assert r.status_code == 200, r.text
        sids[name] = sid

    outs = {}
    for name, sid in sids.items():
        rec = do_estimate(client, sid, true_angles=[20.0]).json()
        assert rec["update_rule"] == "forgetting-snapshot-v2"
        outs[name] = rec

    whole, by16, by1 = (outs[k]["result"] for k in ("whole", "by16", "by1"))
    ang = [o["angles_deg"][0] for o in (whole, by16, by1)]
    neff = [o["n_effective_snapshots"] for o in (whole, by16, by1)]
    ratio = [o["per_angle"][0]["error_over_crb"] for o in (whole, by16, by1)]

    # 三种推法互相一致（源数判定也吃有效快拍数，一并核对）
    assert ang[1] == pytest.approx(ang[0], abs=1e-6)
    assert ang[2] == pytest.approx(ang[0], abs=1e-6)
    assert neff[1] == pytest.approx(neff[0], rel=1e-9)
    assert neff[2] == pytest.approx(neff[0], rel=1e-9)
    assert ratio[1] == pytest.approx(ratio[0], rel=1e-6)
    assert ratio[2] == pytest.approx(ratio[0], rel=1e-6)
    assert whole["source_counts"] == by16["source_counts"] == \
        by1["source_counts"]

    # 与外部 NumPy 参考一致：有效快拍数、由参考协方差推出的角度
    C_ref, n_ref = _numpy_reference(X, ALPHA)
    assert neff[0] == pytest.approx(n_ref, rel=1e-9)
    assert neff[0] == pytest.approx((1 - ALPHA ** 600) / (1 - ALPHA), rel=1e-9)
    ref = run_doa(ULA(8, 0.5, 1.0), C_ref, n_ref,
                  EstimateConfig(source_mode=1), true_angles=[20.0])
    assert ang[0] == pytest.approx(ref["angles_deg"][0], abs=1e-6)
    # 阶跃后段占绝对权重，估计应贴住 +20° 一侧
    assert abs(ang[0] - 20.0) < 0.2

    # 会话状态里能直接读出当前累积口径
    st = client.get(f"/sessions/{sids['whole']}").json()
    assert st["update_rule"] == "forgetting-snapshot-v2"
    assert st["update_frozen"] is False
    assert st["n_effective_snapshots"] == pytest.approx(n_ref, rel=1e-9)


def test_extreme_alpha_single_huge_batch_over_api(client):
    """alpha=0.5、单批一万条：不出 NaN、不出除零，估计正常返回。"""
    rng = np.random.default_rng(31)
    X = (rng.standard_normal((8, 10_000)) +
         1j * rng.standard_normal((8, 10_000))) / np.sqrt(2.0)
    sid = create_session(client, update={"method": "forgetting", "alpha": 0.5},
                         estimate={"source_mode": 1})
    r = append_synth(client, sid, X, "huge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted_snapshots"] == 10_000
    assert body["n_effective_snapshots"] == pytest.approx(2.0, rel=1e-9)
    rec = do_estimate(client, sid).json()
    assert np.all(np.isfinite(rec["result"]["angles_deg"]))
    assert np.isfinite(rec["result"]["n_effective_snapshots"])


def test_status_exposes_accumulation_rule(client):
    sid_f = create_session(client, update=FORGETTING)
    st = client.get(f"/sessions/{sid_f}").json()
    assert st["update_rule"] == "forgetting-snapshot-v2"
    assert st["update_frozen"] is False
    sid_s = create_session(client)  # 默认滑动窗口
    st = client.get(f"/sessions/{sid_s}").json()
    assert st["update_rule"] == "sliding-window-v1"
    assert st["update_frozen"] is False
    # 列表接口同样带口径标记
    listed = {s["session_id"]: s for s in client.get("/sessions").json()["sessions"]}
    assert listed[sid_f]["update_rule"] == "forgetting-snapshot-v2"
    assert listed[sid_s]["update_rule"] == "sliding-window-v1"


# ---------------------------------------------------------------- 旧会话升级

def _legacy_forgetting_run(batches, alpha):
    """升级前旧实现的逐字复刻：整批权重 1，每批旧和只乘一次 alpha。"""
    m = batches[0].shape[0]
    C = np.zeros((m, m), np.complex128)
    n_eff = 0.0
    total = 0
    for X in batches:
        k = X.shape[1]
        C = hermitianize((alpha * n_eff * C + X @ X.conj().T)
                         / (alpha * n_eff + k))
        n_eff = alpha * n_eff + k
        total += k
    return C, n_eff, total


def _write_pre_upgrade_session(store: FileStore, sid: str) -> dict:
    """用升级前的磁盘格式落一个遗忘因子会话（跑了一两周的那种）。

    config.json 无新字段；state.npz 的标量里没有 rule 标记；results.jsonl
    是旧版估计记录（没有 update_rule 字段）。返回用于核对的字典。
    """
    array = ULA(8, 0.5, 1.0)
    batches = [
        synthesize(array, [-12.0], [1.0], 18.0, 40, 901),
        synthesize(array, [-12.0], [1.0], 18.0, 24, 902),
        synthesize(array, [-12.0], [1.0], 18.0, 40, 903),
    ]
    C, n_eff, total = _legacy_forgetting_run(batches, 0.98)

    config = {
        "session_id": sid,
        "name": "legacy-wang",
        "created_at": 1727000000.0,
        "updated_at": 1727000100.0,
        "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0,
                  "position_errors": None},
        "update": {"method": "forgetting", "window": 256, "alpha": 0.98},
        "estimate": {"source_mode": 1, "angle_min": -90.0,
                     "angle_max": 90.0, "step_deg": 0.5,
                     "smoothing": False, "sub_len": None},
        "version": 1,
    }
    store.save_config(sid, config)
    store.save_state(sid, {
        "kind": "forgetting", "m": 8, "alpha": 0.98,
        "C": C, "n_eff": n_eff, "total_snapshots": total,
        "_batch_ids": np.asarray(["f-001", "f-002", "f-003"]),
    })

    # 旧版估计记录：按旧口径的 C/n_eff 真实跑出来的结果
    result = run_doa(array, C, n_eff, EstimateConfig(source_mode=1),
                     true_angles=[-12.0])
    result = {k: v for k, v in result.items() if k != "_spectrum"}
    recs = []
    for version in (1, 2):
        rec = {
            "version": version,
            "session_id": sid,
            "created_at": 1727000200.0 + version,
            "batch_count": 3,
            "total_snapshots": total,
            "config_used": {"source_mode": 1, "angle_min": -90.0,
                            "angle_max": 90.0, "step_deg": 0.5,
                            "smoothing": False, "sub_len": None},
            "result": result,
        }
        store.append_result(sid, rec)
        recs.append(rec)
    return {"C": C, "n_eff": n_eff, "total": total, "records": recs}


def test_pre_upgrade_data_dir_boots_and_freezes(tmp_path):
    """升级前格式的数据目录真实启动：列出、逐字保留、标旧口径、冻结。"""
    data_dir = str(tmp_path / "data")
    sid = "a1b2c3d4e5f60708"
    store = FileStore(data_dir)
    saved = _write_pre_upgrade_session(store, sid)
    results_path = os.path.join(data_dir, sid, "results.jsonl")
    with open(results_path, "rb") as f:
        bytes_before = f.read()

    with TestClient(create_app(data_dir)) as c:
        # 1) 旧会话能列出来，状态标明旧口径且冻结
        listed = {s["session_id"]: s
                  for s in c.get("/sessions").json()["sessions"]}
        assert sid in listed
        entry = listed[sid]
        assert entry["update_rule"] == "forgetting-batch-v1"
        assert entry["update_frozen"] is True
        assert entry["n_total_snapshots"] == saved["total"]
        assert entry["n_effective_snapshots"] == pytest.approx(
            saved["n_eff"], rel=1e-12)
        assert entry["latest_version"] == 2

        st = c.get(f"/sessions/{sid}").json()
        assert st["update_rule"] == "forgetting-batch-v1"
        assert st["update_frozen"] is True

        # 2) 历史结果按版本号读出来与原来逐字相同
        recs = c.get(f"/sessions/{sid}/results",
                     params={"since_version": 0}).json()["results"]
        assert recs == saved["records"]
        recs1 = c.get(f"/sessions/{sid}/results",
                      params={"since_version": 1}).json()["results"]
        assert recs1 == saved["records"][1:]
        with open(results_path, "rb") as f:
            assert f.read() == bytes_before

        # 3) 新的快拍追加被拒绝（409，人读消息指明已冻结、另开会话）
        X = synthesize(ULA(8, 0.5, 1.0), [5.0], [1.0], 15.0, 16, 777)
        r = append_synth(c, sid, X, "f-004")
        assert r.status_code == 409, r.text
        err = r.json()["error"]
        assert err["type"] == "session_frozen"
        assert "只读" in err["message"] and "新建会话" in err["message"]
        # 历史批次号的重复提交仍是幂等回报（不改变任何状态）
        r_dup = append_synth(c, sid, X, "f-003")
        assert r_dup.status_code == 200
        assert r_dup.json()["deduplicated"] is True

        # 4) 冻结不影响在既有协方差上继续出估计；新结果带旧口径标记，
        #    且历史行逐字不动
        r_est = do_estimate(c, sid, true_angles=[-12.0])
        assert r_est.status_code == 200, r_est.text
        est = r_est.json()
        assert est["version"] == 3
        assert est["update_rule"] == "forgetting-batch-v1"
        assert est["result"]["n_effective_snapshots"] == pytest.approx(
            saved["n_eff"], rel=1e-12)
        with open(results_path, "rb") as f:
            after = f.read()
        assert after[:len(bytes_before)] == bytes_before

        # 5) 同一数据目录里新开的会话走新口径，行为完全正常
        sid2 = create_session(c, update=FORGETTING,
                              estimate={"source_mode": 1})
        X2 = synthesize(ULA(8, 0.5, 1.0), [10.0], [1.0], 20.0, 64, 555)
        assert append_synth(c, sid2, X2, "n-1").status_code == 200
        st2 = c.get(f"/sessions/{sid2}").json()
        assert st2["update_rule"] == "forgetting-snapshot-v2"
        assert st2["update_frozen"] is False

    # 6) 再次重启：旧会话仍列出、仍冻结，历史结果依旧逐字可读
    with TestClient(create_app(data_dir)) as c2:
        st = c2.get(f"/sessions/{sid}").json()
        assert st["update_rule"] == "forgetting-batch-v1"
        assert st["update_frozen"] is True
        recs = c2.get(f"/sessions/{sid}/results",
                      params={"since_version": 0}).json()["results"]
        assert recs[:2] == saved["records"]
        assert len(recs) == 3
        r = append_synth(c2, sid, X, "f-005")
        assert r.status_code == 409
