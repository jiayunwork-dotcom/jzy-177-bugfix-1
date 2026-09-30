"""样本协方差、前后向空间平滑与增量更新。

两种增量更新器维护同一个语义量：

* 样本协方差（MLE 归一化） ``C = (1/N) sum_k x_k x_k^H``；
* 有效快拍数 ``N``。

定长滑动窗口把窗口内全部 N 个快拍等权；指数遗忘因子令旧数据权重按
alpha 指数衰减，``N`` 取权重和 ``sum alpha^k``。两种定义都给出一次性
批处理参考实现 ``batch_covariance`` / ``forgetting_covariance``，
增量结果与之逐元素对齐（见 tests/test_incremental.py）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


class CovarianceError(ValueError):
    """快拍数据或更新参数不合法。"""


def check_snapshots(X: np.ndarray, m: int | None = None) -> np.ndarray:
    """校验并返回 ``(M, K)`` complex128 快拍矩阵。"""
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.complexfloating):
        X = X.astype(np.complex128, copy=False)
    if X.ndim != 2:
        raise CovarianceError(f"快拍数据必须是二维 M×K 矩阵，实际为 {X.ndim} 维。")
    if m is not None and X.shape[0] != m:
        raise CovarianceError(
            f"快拍的阵元维为 {X.shape[0]}，与会话阵元数 {m} 不一致。"
        )
    if not np.all(np.isfinite(X.real)) or not np.all(np.isfinite(X.imag)):
        raise CovarianceError("快拍数据中含有 NaN 或无穷大。")
    return np.ascontiguousarray(X, dtype=np.complex128)


def hermitianize(C: np.ndarray) -> np.ndarray:
    """返回精确厄米对称的协方差 ``(C + C^H) / 2``。"""
    return 0.5 * (C + C.conj().T)


def batch_covariance(X: np.ndarray) -> tuple[np.ndarray, int]:
    """一次性样本协方差 ``X X^H / K``（MLE 归一化），返回 (C, K)。"""
    X = check_snapshots(X)
    k = X.shape[1]
    if k == 0:
        raise CovarianceError("快拍数为 0，无法估计协方差。")
    C = (X @ X.conj().T) * (1.0 / k)
    return hermitianize(C), k


def forgetting_covariance(
    batches: list[np.ndarray], alpha: float
) -> tuple[np.ndarray, float]:
    """按提交顺序、遗忘因子 alpha 一次性计算协方差。

    最新批次权重为 1，再往前每批乘 alpha。归一化用权重和 N，与
    :class:`ForgettingAccumulator` 的增量定义完全相同。
    """
    if not (0.0 < alpha <= 1.0):
        raise CovarianceError("遗忘因子必须满足 0 < alpha <= 1。")
    if not batches:
        raise CovarianceError("没有任何快拍。")
    m = np.asarray(batches[0]).shape[0]
    acc = np.zeros((m, m), dtype=np.complex128)
    weight_sum = 0.0
    weight = 1.0
    # 从最新批次向旧批次遍历
    for X in reversed(batches):
        X = check_snapshots(X, m)
        k = X.shape[1]
        acc += weight * (X @ X.conj().T)
        weight_sum += weight * k
        weight *= alpha
    if weight_sum <= 0.0:
        raise CovarianceError("有效快拍权重为 0。")
    return hermitianize(acc / weight_sum), weight_sum


def forward_backward_smoothing(C: np.ndarray, sub_len: int) -> tuple[np.ndarray, int]:
    """对 ``(M, M)`` 协方差做前后向空间平滑。

    前向子阵协方差 ``C_f[p:p+L, p:p+L]`` 共 ``M-L+1`` 个；后向用交换阵
    ``J C_f^H J``。等效快拍数乘 ``2(M-L+1)``（相对原始每条快拍，
    FB 平滑里一条原始快拍贡献 2 个子阵快拍）。
    """
    C = np.asarray(C, dtype=np.complex128)
    m = C.shape[0]
    if not (1 <= sub_len <= m):
        raise CovarianceError(f"子阵长度必须在 [1, {m}] 之间，实际为 {sub_len}。")
    n_sub = m - sub_len + 1
    Cs = np.zeros((sub_len, sub_len), dtype=np.complex128)
    J = np.fliplr(np.eye(sub_len))
    for p in range(n_sub):
        Cf = C[p:p + sub_len, p:p + sub_len]
        Cb = J @ Cf.conj() @ J
        Cs += Cf + Cb
    Cs /= (2.0 * n_sub)
    return hermitianize(Cs), 2 * n_sub


@dataclass
class Accumulator:
    """增量协方差更新器基类。

    ``C`` 始终是厄米的样本协方差，``n_eff`` 为其分母（有效快拍数）。
    """

    m: int
    C: np.ndarray
    n_eff: float
    total_snapshots: int

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def add_batch(self, X: np.ndarray) -> int:
        """追加一批 ``(M, K)`` 快拍，返回本批实际新计入的快拍数。"""
        raise NotImplementedError

    def to_state(self) -> dict:
        raise NotImplementedError


class SlidingWindowAccumulator(Accumulator):
    """定长滑动窗口。

    保留最近 ``window`` 条快拍，``C = (1/N) sum 窗口内 x x^H``，N 为
    窗口内实际条数。窗口未满前 N 随追加线性增长。
    """

    def __init__(self, m: int, window: int,
                 C: np.ndarray | None = None,
                 n_eff: float = 0.0,
                 total_snapshots: int = 0,
                 buffer: deque | None = None):
        if not isinstance(window, (int, np.integer)) or window <= 0:
            raise CovarianceError("滑动窗口长度必须为正整数。")
        super().__init__(
            m=m,
            C=np.zeros((m, m), np.complex128) if C is None
            else hermitianize(np.asarray(C, dtype=np.complex128)),
            n_eff=float(n_eff),
            total_snapshots=int(total_snapshots),
        )
        self.window = int(window)
        # 每条快拍一个 (M,1) 数组；总条数上限 window（默认配置下 256×64，
        # 内存可忽略）。
        self._buffer: deque[np.ndarray] = (
            deque(np.asarray(v, dtype=np.complex128).reshape(m) for v in buffer)
            if buffer is not None else deque()
        )

    @property
    def kind(self) -> str:
        return "sliding"

    def add_batch(self, X: np.ndarray) -> int:
        X = check_snapshots(X, self.m)
        k = X.shape[1]
        # 未归一化和 S = C * n_eff，避免与归一化量混合。
        S = self.C * self.n_eff
        for i in range(k):
            x = X[:, i]
            if len(self._buffer) >= self.window:
                old = self._buffer[0]
                S -= np.outer(old, old.conj())
                self._buffer.popleft()
                self.n_eff -= 1.0
            self._buffer.append(x.copy())
            S += np.outer(x, x.conj())
            self.n_eff += 1.0
        self.total_snapshots += k
        if self.n_eff > 0:
            self.C = hermitianize(S / self.n_eff)
        else:  # pragma: no cover - n_eff 必 > 0
            self.C = np.zeros((self.m, self.m), np.complex128)
        return k

    def to_state(self) -> dict:
        return {
            "kind": "sliding",
            "m": self.m,
            "window": self.window,
            "C": self.C,
            "n_eff": self.n_eff,
            "total_snapshots": self.total_snapshots,
            "buffer": np.stack(self._buffer, axis=1) if self._buffer
            else np.zeros((self.m, 0), np.complex128),
        }

    @classmethod
    def from_state(cls, state: dict) -> "SlidingWindowAccumulator":
        buf = state["buffer"]
        buffer = deque()
        if buf.shape[1] > 0:
            for i in range(buf.shape[1]):
                buffer.append(buf[:, i])
        return cls(
            m=int(state["m"]),
            window=int(state["window"]),
            C=state["C"],
            n_eff=float(state["n_eff"]),
            total_snapshots=int(state["total_snapshots"]),
            buffer=buffer,
        )


class ForgettingAccumulator(Accumulator):
    """指数遗忘因子更新。

    新批进入：``C <- (alpha * n_eff * C + X X^H) / (alpha*n_eff + K)``，
    ``n_eff <- alpha*n_eff + K``。稳态下 ``n_eff -> K/(1-alpha)``（对
    连续单条追加）。无需保留历史快拍。
    """

    def __init__(self, m: int, alpha: float,
                 C: np.ndarray | None = None,
                 n_eff: float = 0.0,
                 total_snapshots: int = 0):
        if not (0.0 < float(alpha) <= 1.0):
            raise CovarianceError("遗忘因子必须满足 0 < alpha <= 1。")
        super().__init__(
            m=m,
            C=np.zeros((m, m), np.complex128) if C is None
            else hermitianize(np.asarray(C, dtype=np.complex128)),
            n_eff=float(n_eff),
            total_snapshots=int(total_snapshots),
        )
        self.alpha = float(alpha)

    @property
    def kind(self) -> str:
        return "forgetting"

    def add_batch(self, X: np.ndarray) -> int:
        X = check_snapshots(X, self.m)
        k = X.shape[1]
        new_sum = self.alpha * self.n_eff * self.C + X @ X.conj().T
        self.n_eff = self.alpha * self.n_eff + k
        self.C = hermitianize(new_sum / self.n_eff)
        self.total_snapshots += k
        return k

    def to_state(self) -> dict:
        return {
            "kind": "forgetting",
            "m": self.m,
            "alpha": self.alpha,
            "C": self.C,
            "n_eff": self.n_eff,
            "total_snapshots": self.total_snapshots,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ForgettingAccumulator":
        return cls(
            m=int(state["m"]),
            alpha=float(state["alpha"]),
            C=state["C"],
            n_eff=float(state["n_eff"]),
            total_snapshots=int(state["total_snapshots"]),
        )


def build_accumulator(m: int, method: str, window: int | None = None,
                      alpha: float | None = None) -> Accumulator:
    """按配置构造增量更新器；默认定长滑动窗口。"""
    if method == "sliding":
        return SlidingWindowAccumulator(m, window if window is not None else 256)
    if method == "forgetting":
        return ForgettingAccumulator(m, alpha if alpha is not None else 0.9922)
    raise CovarianceError(f"未知的增量更新方式：{method!r}（支持 sliding / forgetting）。")


def accumulator_from_state(state: dict) -> Accumulator:
    if state["kind"] == "sliding":
        return SlidingWindowAccumulator.from_state(state)
    if state["kind"] == "forgetting":
        return ForgettingAccumulator.from_state(state)
    raise CovarianceError(f"未知的增量更新状态：{state['kind']!r}。")
