"""增量协方差更新：与一次性定义的数值一致性、厄米性、长程不漂移。"""
from __future__ import annotations

import numpy as np

from app.doa.covariance import (
    ForgettingAccumulator,
    SlidingWindowAccumulator,
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
    """遗忘因子：分批增量结果与“全部快拍按条加权一次算”逐元素一致。"""
    rng = np.random.default_rng(7)
    alpha = 0.9
    batches = [(rng.standard_normal((M, k)) +
                1j * rng.standard_normal((M, k))) / np.sqrt(2)
               for k in (1, 5, 2, 11, 3, 1)]
    X = np.concatenate(batches, axis=1)
    acc = ForgettingAccumulator(M, alpha=alpha)
    for b in batches:
        acc.add_batch(b)
    C_ref, n_ref = forgetting_covariance(X, alpha)
    assert abs(acc.n_eff - n_ref) < n_ref * 1e-12
    assert np.allclose(acc.C, C_ref, rtol=RTOL, atol=1e-14)


def test_forgetting_independent_of_packetization():
    """同一份快拍任意切分（整块/16 条包/逐条），C 与 n_eff 严格同口径。"""
    rng = np.random.default_rng(77)
    alpha = 0.98
    X = (rng.standard_normal((M, 600)) +
         1j * rng.standard_normal((M, 600))) / np.sqrt(2)

    def run(sizes):
        acc = ForgettingAccumulator(M, alpha=alpha)
        pos = 0
        for k in sizes:
            acc.add_batch(X[:, pos:pos + k])
            pos += k
        assert pos == 600
        return acc.C, acc.n_eff

    sizes_16 = [16] * (600 // 16) + [600 % 16]
    C_ref, n_ref = forgetting_covariance(X, alpha)
    for sizes in ([600], sizes_16, [1] * 600,
                  [37, 200, 213, 150]):
        C, n = run(sizes)
        assert abs(n - n_ref) < n_ref * 1e-12
        assert np.allclose(C, C_ref, rtol=1e-12, atol=1e-14)


def test_forgetting_matches_explicit_numpy_weights():
    """增量结果与调用方在外部按定义直接写出的 NumPy 权重和吻合（1e-9）。"""
    rng = np.random.default_rng(31)
    alpha = 0.98
    X = (rng.standard_normal((M, 600)) +
         1j * rng.standard_normal((M, 600))) / np.sqrt(2)
    acc = ForgettingAccumulator(M, alpha=alpha)
    acc.add_batch(X[:, :300])
    acc.add_batch(X[:, 300:])  # 中途换批次边界
    # 最新一条权重 1，往前每条乘 alpha
    w = alpha ** np.arange(600, dtype=np.float64)[::-1]
    C_ref = (X * w) @ X.conj().T / w.sum()
    C_ref = hermitianize(C_ref)
    rel = np.linalg.norm((acc.C - C_ref).ravel()) \
        / np.linalg.norm(C_ref.ravel())
    assert rel < 1e-9
    assert abs(acc.n_eff - w.sum()) / w.sum() < 1e-12


def test_forgetting_alpha_half_huge_batch_no_nan():
    """alpha=0.5、单批 10000 条的极端配置不出 NaN/除零，C 仍是合法协方差。"""
    rng = np.random.default_rng(2)
    X = (rng.standard_normal((M, 10_000)) +
         1j * rng.standard_normal((M, 10_000))) / np.sqrt(2)
    acc = ForgettingAccumulator(M, alpha=0.5)
    acc.add_batch(X)
    assert np.all(np.isfinite(acc.C))
    assert acc.n_eff > 1.0
    eig = np.linalg.eigvalsh(acc.C)
    assert np.all(np.isfinite(eig)) and eig[0] >= -1e-12
    C_ref, n_ref = forgetting_covariance(X, 0.5)
    assert np.allclose(acc.C, C_ref, rtol=1e-12, atol=1e-14)
    assert abs(acc.n_eff - n_ref) < 1e-12
    # 继续追加也不出 NaN
    acc.add_batch(X[:, :100])
    assert np.all(np.isfinite(acc.C)) and acc.n_eff > 0


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
