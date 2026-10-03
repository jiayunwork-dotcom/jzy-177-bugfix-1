"""增量协方差更新：与一次性定义的数值一致性、厄米性、长程不漂移。"""
from __future__ import annotations

import numpy as np
import pytest

from app.doa.covariance import (
    RULE_FORGETTING_BATCH_V1,
    RULE_FORGETTING_SNAPSHOT_V2,
    RULE_SLIDING_V1,
    ForgettingAccumulator,
    LegacyForgettingAccumulator,
    SlidingWindowAccumulator,
    accumulator_from_state,
    batch_covariance,
    forgetting_covariance,
    hermitianize,
)

from .conftest import append_synth, create_session, do_estimate

M = 8
RTOL = 1e-12


def _rng_data(k, seed=0, m=M):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal((m, k)) +
             1j * rng.standard_normal((m, k)))
            / np.sqrt(2.0))


def test_sliding_incremental_matches_batch_before_window_full():
    rng = np.random.default_rng(5)
    batches = [(rng.standard_normal((M, k)) +
                1j * rng.standard_normal((M, k))) / np.sqrt(2)
               for k in rng.integers(1, 30, size=10)]
    acc = SlidingWindowAccumulator(M, window=10_000)
    for b in batches:
        acc.add_batch(b)
    C_ref, n_ref = batch_covariance(np.concatenate(batches, axis=1))
    assert acc.n_eff == n_ref
    assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_sliding_window_matches_windowed_batch():
    rng = np.random.default_rng(6)
    window = 50
    data = (rng.standard_normal((M, 300)) +
            1j * rng.standard_normal((M, 300))) / np.sqrt(2.0)
    acc = SlidingWindowAccumulator(M, window=window)
    # 先灌 120 条，再继续灌；随时与“窗口内一次算”对齐
    acc.add_batch(data[:, :120])
    rest = np.split(data[:, 120:], range(13, 180, 13), axis=1)
    cursor = 120
    for part in rest:
        if part.shape[1] == 0:
            continue
        acc.add_batch(part)
        cursor += part.shape[1]
        lo = max(0, cursor - window)
        C_ref, n_ref = batch_covariance(data[:, lo:cursor])
        assert abs(acc.n_eff - n_ref) < 1e-9
        assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_forgetting_incremental_matches_one_shot():
    rng = np.random.default_rng(7)
    alpha = 0.9
    batches = [(rng.standard_normal((M, k)) +
                1j * rng.standard_normal((M, k))) / np.sqrt(2)
               for k in (1, 5, 2, 11, 3, 1)]
    acc = ForgettingAccumulator(M, alpha=alpha)
    for b in batches:
        acc.add_batch(b)
    C_ref, n_ref = forgetting_covariance(batches, alpha)
    assert abs(acc.n_eff - n_ref) < n_ref * 1e-12
    assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_forgetting_chunking_invariant_matches_numpy_definition():
    """按快拍逐条遗忘的定义：整批/小批/逐条推入，与外部 NumPy 参考一致。

    参考定义（与增量实现独立的直接计算）：第 t 条权重 alpha^(N-1-t)，
    C = sum w_k x_k x_k^H / sum w_k。
    """
    rng = np.random.default_rng(21)
    alpha = 0.98
    X = (rng.standard_normal((M, 600)) +
         1j * rng.standard_normal((M, 600))) / np.sqrt(2.0)
    w = alpha ** np.arange(599, -1, -1, dtype=np.float64)
    C_ref = (X * w) @ X.conj().T / w.sum()
    n_ref = float(w.sum())

    for chunk in (600, 16, 1, 7):  # 7 不整除 600，覆盖不整齐分包
        acc = ForgettingAccumulator(M, alpha=alpha)
        for lo in range(0, 600, chunk):
            acc.add_batch(X[:, lo:lo + chunk])
        scale = float(np.max(np.abs(C_ref)))
        assert abs(acc.n_eff - n_ref) < n_ref * 1e-9
        assert np.max(np.abs(acc.C - C_ref)) < scale * 1e-9
        assert acc.total_snapshots == 600
    # 一次性参考实现同样对齐该定义
    C1, n1 = forgetting_covariance([X], alpha)
    assert n1 == pytest.approx(n_ref, rel=1e-12)
    assert np.allclose(C1, C_ref, rtol=RTOL, atol=1e-14)


def test_forgetting_extreme_alpha_huge_batch_no_nan():
    """alpha=0.5、单批 10000 条：alpha^K 下溢到 0 也不许出 NaN/除零。"""
    rng = np.random.default_rng(22)
    X = (rng.standard_normal((M, 10_000)) +
         1j * rng.standard_normal((M, 10_000))) / np.sqrt(2.0)
    acc = ForgettingAccumulator(M, alpha=0.5)
    acc.add_batch(X)
    assert np.all(np.isfinite(acc.C.real)) and np.all(np.isfinite(acc.C.imag))
    assert np.isfinite(acc.n_eff) and acc.n_eff > 0.0
    # 权重和 sum 0.5^k 收敛到 2
    assert acc.n_eff == pytest.approx(2.0, rel=1e-12)
    assert np.allclose(acc.C, acc.C.conj().T, atol=0.0)
    # 极端状态下继续追加仍然正常
    acc.add_batch(X[:, :5])
    assert np.all(np.isfinite(acc.C.real))
    assert acc.n_eff == pytest.approx(2.0, rel=1e-12)
    assert acc.total_snapshots == 10_005


def test_forgetting_alpha1_equals_plain_batch_covariance():
    """alpha=1 时遗忘退化为增长平均：任意分包都与 XX^H/K 逐元素一致。"""
    rng = np.random.default_rng(23)
    X = (rng.standard_normal((M, 200)) +
         1j * rng.standard_normal((M, 200))) / np.sqrt(2.0)
    acc = ForgettingAccumulator(M, alpha=1.0)
    for lo, hi in [(0, 23), (23, 100), (100, 200)]:
        acc.add_batch(X[:, lo:hi])
    C_ref, n_ref = batch_covariance(X)
    assert acc.n_eff == n_ref
    assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_legacy_forgetting_state_loads_frozen():
    """升级前落盘（无 rule 标记）的遗忘状态：加载为只读旧口径对象。"""
    rng = np.random.default_rng(24)
    X = (rng.standard_normal((M, 40)) +
         1j * rng.standard_normal((M, 40))) / np.sqrt(2.0)
    C0, _ = batch_covariance(X)
    legacy_state = {  # 旧实现写出的字段，没有 "rule"
        "kind": "forgetting", "m": M, "alpha": 0.97,
        "C": C0, "n_eff": 40.0, "total_snapshots": 40,
    }
    acc = accumulator_from_state(legacy_state)
    assert isinstance(acc, LegacyForgettingAccumulator)
    assert acc.rule == RULE_FORGETTING_BATCH_V1
    assert not acc.accepts_appends
    assert acc.n_eff == 40.0 and acc.total_snapshots == 40
    assert np.array_equal(acc.C, hermitianize(C0))
    with pytest.raises(Exception, match="只读"):
        acc.add_batch(X)
    # 即使被再次序列化也固化 v1 标记，恢复后仍是只读旧口径
    acc2 = accumulator_from_state(acc.to_state())
    assert isinstance(acc2, LegacyForgettingAccumulator)
    assert not acc2.accepts_appends


def test_new_states_carry_rule_markers():
    acc = SlidingWindowAccumulator(M, window=8)
    assert acc.rule == RULE_SLIDING_V1
    assert accumulator_from_state(acc.to_state()).rule == RULE_SLIDING_V1
    facc = ForgettingAccumulator(M, alpha=0.9)
    assert facc.rule == RULE_FORGETTING_SNAPSHOT_V2
    assert accumulator_from_state(facc.to_state()).rule == \
        RULE_FORGETTING_SNAPSHOT_V2
    # 旧滑动窗口状态（无 rule）照常恢复、可继续追加
    st = acc.to_state()
    del st["rule"]
    assert accumulator_from_state(st).accepts_appends
    # 未知口径标记明确报错
    bad = facc.to_state()
    bad["rule"] = "forgetting-weird-v9"
    with pytest.raises(Exception, match="口径"):
        accumulator_from_state(bad)


def test_forgetting_state_roundtrip_preserves_covariance():
    """遗忘因子状态序列化/恢复后继续追加，与不中断的增量结果一致。"""
    rng = np.random.default_rng(11)
    a = ForgettingAccumulator(M, alpha=0.97)
    b = ForgettingAccumulator(M, alpha=0.97)
    batches = [(rng.standard_normal((M, k)) +
                1j * rng.standard_normal((M, k))) / np.sqrt(2)
               for k in (4, 9, 2)]
    a.add_batch(batches[0])
    # b 从 a 的落盘状态恢复
    state = a.to_state()
    b = ForgettingAccumulator.from_state(state)
    a.add_batch(batches[1]); b.add_batch(batches[1])
    a.add_batch(batches[2]); b.add_batch(batches[2])
    assert abs(a.n_eff - b.n_eff) < 1e-12
    assert np.allclose(a.C, b.C, rtol=RTOL, atol=1e-14)


def test_long_run_keeps_hermitian_and_no_drift():
    """逐条追加很久：始终精确厄米；窗口稳态协方差与重算一致。"""
    rng = np.random.default_rng(8)
    window = 64
    acc = SlidingWindowAccumulator(M, window=window)
    total = 20_000
    for i in range(total):
        x = ((rng.standard_normal((M, 1)) +
              1j * rng.standard_normal((M, 1))) / np.sqrt(2.0))
        acc.add_batch(x)
        if i % 997 == 0:
            assert np.allclose(acc.C, acc.C.conj().T, atol=0.0)
            eig = np.linalg.eigvalsh(acc.C)
            assert eig[0] >= -1e-12
    # 与从窗口缓冲重算的样本协方差一致
    buf = np.stack(list(acc._buffer), axis=1)
    C_ref, n_ref = batch_covariance(buf)
    assert acc.n_eff == n_ref
    assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_forgetting_hermitian_and_steady_n_eff():
    rng = np.random.default_rng(9)
    alpha = 0.9922
    acc = ForgettingAccumulator(M, alpha=alpha)
    for i in range(5000):
        acc.add_batch((rng.standard_normal((M, 1)) +
                       1j * rng.standard_normal((M, 1))) / np.sqrt(2.0))
        if i % 500 == 0:
            assert np.allclose(acc.C, acc.C.conj().T, atol=0.0)
    # 单条连续追加的稳态有效快拍 1/(1-alpha)
    assert abs(acc.n_eff - 1.0 / (1.0 - alpha)) < 1.0


def test_hermitianize_is_exact():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = hermitianize(C)
    assert np.array_equal(H, H.conj().T)


def test_api_batch_vs_one_shot_sliding_covariance(client, make_data):
    """同一会话：一次整批 vs 多次小批，窗口未满，协方差定义相同。"""
    X = make_data([-10.0, 20.0], snr=15.0, k=300, seed=5)
    s1 = create_session(client, update={"method": "sliding", "window": 4096})
    append_synth(client, s1, X, "all")
    s2 = create_session(client, update={"method": "sliding", "window": 4096})
    for i, (lo, hi) in enumerate(zip(range(0, 300, 37),
                                     list(range(37, 300, 37)) + [300])):
        r = append_synth(client, s2, X[:, lo:hi], f"b{i}")
        assert r.status_code == 200, r.text
    a = do_estimate(client, s1, source_mode=2).json()["result"]["angles_deg"]
    b = do_estimate(client, s2, source_mode=2).json()["result"]["angles_deg"]
    assert np.allclose(sorted(a), sorted(b), atol=1e-6)
