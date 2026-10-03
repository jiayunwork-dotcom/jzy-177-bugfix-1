"""样本协方差、前后向空间平滑与增量更新。

两种增量更新器维护同一个语义量：

* 样本协方差（MLE 归一化） ``C = (1/N) sum_k w_k x_k x_k^H``；
* 有效快拍数 ``N = sum_k w_k``。

定长滑动窗口把窗口内全部 N 个快拍等权（``w_k = 1``）；指数遗忘因子按
**快拍**逐条加权：最新一条权重为 1，每往前一条乘 alpha，即第 t 条权重
``alpha^(N-t)``。两种定义都给出一次性批处理参考实现
``batch_covariance`` / ``forgetting_covariance``，增量结果与之逐元素
对齐（见 tests/test_incremental.py），且与分包方式无关。

每个更新器都带口径标记 ``rule`` 随状态落盘：``sliding-window-v1``、
``forgetting-snapshot-v2``（按快拍遗忘，现行定义）。升级前落盘的遗忘
因子状态没有该标记，是按**批**加权的旧口径，加载为只读的
:class:`LegacyForgettingAccumulator`（``forgetting-batch-v1``）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# 累积口径标记：随 state.npz 落盘，调用方可从会话状态读出当前按哪种
# 规则累积。v1 遗忘（按批加权）只用于识别升级前的旧状态，不再能创建。
RULE_SLIDING_V1 = "sliding-window-v1"
RULE_FORGETTING_SNAPSHOT_V2 = "forgetting-snapshot-v2"
RULE_FORGETTING_BATCH_V1 = "forgetting-batch-v1"


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
    """按快拍顺序、遗忘因子 alpha 一次性计算协方差。

    全部批次的快拍按提交顺序拼接，最新**一条**权重为 1，每往前一条乘
    alpha（第 t 条权重 ``alpha^(N-1-t)``）。归一化用权重和
    ``N = sum alpha^k``，与 :class:`ForgettingAccumulator` 的增量定义
    完全相同（增量结果与分包方式无关）。
    """
    if not (0.0 < alpha <= 1.0):
        raise CovarianceError("遗忘因子必须满足 0 < alpha <= 1。")
    if not batches:
        raise CovarianceError("没有任何快拍。")
    m = np.asarray(batches[0]).shape[0]
    X = np.concatenate([check_snapshots(b, m) for b in batches], axis=1)
    n = X.shape[1]
    if n == 0:
        raise CovarianceError("没有任何快拍。")
    # 最新一条权重 1，逐条往前乘 alpha；很旧的快拍权重下溢到 0 是定义
    # 允许的（其贡献本就为 0），不会产生 NaN。
    w = alpha ** np.arange(n - 1, -1, -1, dtype=np.float64)
    weight_sum = float(w.sum())
    if weight_sum <= 0.0:  # pragma: no cover - 最新一条权重恒为 1
        raise CovarianceError("有效快拍权重为 0。")
    return hermitianize(((X * w) @ X.conj().T) / weight_sum), weight_sum


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
    ``rule`` 标识累积口径并随状态落盘；``accepts_appends`` 为 False
    的更新器（旧口径只读状态）拒绝新的快拍追加。
    """

    m: int
    C: np.ndarray
    n_eff: float
    total_snapshots: int

    # 旧口径只读状态（LegacyForgettingAccumulator）置 False
    accepts_appends: ClassVar[bool] = True

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def rule(self) -> str:
        """累积口径标记（随状态落盘，会话状态里对调用方可见）。"""
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

    @property
    def rule(self) -> str:
        return RULE_SLIDING_V1

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
            "rule": self.rule,
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
    """指数遗忘因子更新（按快拍逐条加权，口径 v2）。

    定义：已累积 N 条快拍时第 t 条权重为 ``alpha^(N-t)``——最新一条
    权重 1，每往前一条乘 alpha。一批 K 条进入时的递推：

    * ``S <- alpha^K * S + sum_j alpha^(K-j) x_j x_j^H``（未归一化和，
      由 ``C * n_eff`` 还原）；
    * ``N <- alpha^K * N + sum_j alpha^(K-j)``；
    * ``C = S / N``。

    结果只取决于快拍序列本身，与分包方式无关：同一批快拍整批推、切
    小批推、逐条推得到的 ``C`` 与 ``n_eff`` 相同（浮点 1e-12 量级，
    见 tests/test_incremental.py），并与一次性参考
    :func:`forgetting_covariance` 对齐。稳态 ``N -> 1/(1-alpha)``。
    无需保留历史快拍，内存 O(M^2)。

    ``alpha`` 很小或批很大时 ``alpha^K`` 会下溢到 0——这正确定义了
    “旧数据权重为 0”，不会产生 NaN 或除零（最新一条权重恒为 1，
    ``n_eff >= 1``）。
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

    @property
    def rule(self) -> str:
        return RULE_FORGETTING_SNAPSHOT_V2

    def add_batch(self, X: np.ndarray) -> int:
        X = check_snapshots(X, self.m)
        k = X.shape[1]
        # 批内第 j 条（按到达顺序）权重 alpha^(k-j)：批内最新一条为 1，
        # 每往前一条乘 alpha；旧未归一化和整体再乘 alpha^k。
        w = self.alpha ** np.arange(k - 1, -1, -1, dtype=np.float64)
        decay = self.alpha ** k
        S = decay * (self.n_eff * self.C) + (X * w) @ X.conj().T
        self.n_eff = decay * self.n_eff + float(w.sum())
        self.total_snapshots += k
        if self.n_eff > 0.0:
            self.C = hermitianize(S / self.n_eff)
        else:  # pragma: no cover - 仅当 K=0 且此前无任何快拍
            self.C = np.zeros((self.m, self.m), np.complex128)
        return k

    def to_state(self) -> dict:
        return {
            "kind": "forgetting",
            "rule": self.rule,
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


class LegacyForgettingAccumulator(Accumulator):
    """升级前按**批**遗忘口径（v1）的只读状态。

    旧实现把遗忘作用在批上：每来一批，整批权重 1、旧和乘一次 alpha，
    因此累积结果随分包方式变化，不是快拍序列的确定函数，也无法换算
    成按快拍口径（服务从不保存遗忘方式下的原始快拍）。升级后这类
    落盘状态被冻结：可读取、可继续在其协方差上出估计，但拒绝再追加
    快拍（会话层在追加前检查 :attr:`accepts_appends` 并回 409）。
    """

    accepts_appends: ClassVar[bool] = False

    def __init__(self, m: int, alpha: float,
                 C: np.ndarray | None = None,
                 n_eff: float = 0.0,
                 total_snapshots: int = 0):
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

    @property
    def rule(self) -> str:
        return RULE_FORGETTING_BATCH_V1

    def add_batch(self, X: np.ndarray) -> int:
        raise CovarianceError(
            "该会话是升级前按旧口径（按批遗忘 v1）累积的，已冻结为只读，"
            "不能再追加快拍；请新建会话（按快拍遗忘 v2）继续推送数据。"
        )

    def to_state(self) -> dict:
        # 冻结会话不会再落盘；保留完整序列化以便防御性拷贝，且口径标记
        # 固化为 v1，即使被写出也永远按旧口径识别。
        return {
            "kind": "forgetting",
            "rule": self.rule,
            "m": self.m,
            "alpha": self.alpha,
            "C": self.C,
            "n_eff": self.n_eff,
            "total_snapshots": self.total_snapshots,
        }

    @classmethod
    def from_state(cls, state: dict) -> "LegacyForgettingAccumulator":
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
        # 升级前落盘的遗忘状态没有 rule 标记，按旧口径（按批 v1）识别
        rule = state.get("rule", RULE_FORGETTING_BATCH_V1)
        if rule == RULE_FORGETTING_SNAPSHOT_V2:
            return ForgettingAccumulator.from_state(state)
        if rule == RULE_FORGETTING_BATCH_V1:
            return LegacyForgettingAccumulator.from_state(state)
        raise CovarianceError(f"未知的遗忘因子口径标记：{rule!r}。")
    raise CovarianceError(f"未知的增量更新状态：{state['kind']!r}。")
