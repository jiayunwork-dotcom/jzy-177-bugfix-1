"""样本协方差、前后向空间平滑与增量更新。

两种增量更新器维护同一个语义量：

* 样本协方差（MLE 归一化） ``C = (1/N) sum_k w_k x_k x_k^H``；
* 有效快拍数 ``N = sum_k w_k``。

定长滑动窗口把窗口内全部 N 个快拍等权；指数遗忘因子按**快拍**赋权：
按提交顺序，最新一条权重为 1，往前每条乘 alpha，``N`` 为权重和
``sum alpha^j``。两种定义都给出一次性批处理参考实现
``batch_covariance`` / ``forgetting_covariance``，增量结果与分批方式
无关、与一次性定义逐元素对齐（见 tests/test_incremental.py）。

历史口径（rule 标记）：

* ``sliding_window_v1``：定长滑动窗口，语义从未变化；
* ``forgetting_snapshot_v2``：按快拍指数加权（本文件实现的当前口径）；
* ``forgetting_batch_v1_legacy``：升级前的旧实现，按**批次**等权、仅
  批间乘 alpha——同一批数据切法不同结果就不同。该口径不再接受追加，
  见 :class:`LegacyForgettingAccumulator`。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


class CovarianceError(ValueError):
    """快拍数据或更新参数不合法。"""


# 累积口径标识（写入会话状态与状态接口，调用方可显式区分）。
RULE_SLIDING = "sliding_window_v1"
RULE_FORGETTING = "forgetting_snapshot_v2"
RULE_FORGETTING_LEGACY = "forgetting_batch_v1_legacy"


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
    snapshots: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    """按提交顺序、遗忘因子 alpha 对全部快拍一次性计算协方差。

    定义与分批方式无关：把 ``snapshots`` 视作按时间排列的 ``(M, K)``
    快拍序列，最新一条（最后一列）权重为 1，往前每条乘 alpha，即第 j 条
    （从最新一条往旧数，j=0..K-1）权重 ``alpha**j``：

    ``C = (sum_j alpha^j x_j x_j^H) / (sum_j alpha^j)``，``N`` 取权重和。

    与 :class:`ForgettingAccumulator` 的增量定义完全相同：无论一次性
    整批提交还是切成任意多小批逐条追加，结果一致（相对误差 1e-12 量级）。
    """
    if not (0.0 < alpha <= 1.0):
        raise CovarianceError("遗忘因子必须满足 0 < alpha <= 1。")
    X = check_snapshots(snapshots)
    k = X.shape[1]
    if k == 0:
        raise CovarianceError("没有任何快拍。")
    # 最旧 -> 最新：alpha^(K-1), ..., alpha, 1
    weights = alpha ** np.arange(k - 1, -1, -1, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise CovarianceError("有效快拍权重为 0。")
    acc = (X * weights) @ X.conj().T
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

    @property
    def rule(self) -> str:
        """该更新器实际遵循的累积口径标识（见模块级 RULE_* 常量）。"""
        raise NotImplementedError

    @property
    def append_frozen(self) -> bool:
        """True 表示该会话因口径升级被冻结，拒绝继续追加（只读）。"""
        return False

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
        return RULE_SLIDING

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
    """指数遗忘因子更新（按快拍加权，``forgetting_snapshot_v2`` 口径）。

    定义与切批方式无关：按提交顺序，最新一条快拍权重为 1，往前每条乘
    alpha。设当前有效快拍数为 ``n_eff``（已存快拍的权重和），追加一批
    ``K`` 条快拍 ``X = [x_1,...,x_K]``（x_K 最新）后：

    ```
    S' = alpha^K * n_eff * C
         + sum_{i=1..K} alpha^(K-i) * x_i x_i^H
    n_eff' = alpha^K * n_eff + sum_{i=1..K} alpha^(K-i)
    C' = S' / n_eff'
    ```

    无论 K 取 1 还是一次 10000 条，都等价于逐条递推，因此与一次性
    :func:`forgetting_covariance` 的结果逐元素相同。连续单条追加时稳态
    有效快拍数 ``n_eff -> 1/(1-alpha)``。权重由 ``alpha ** 0..K`` 一次性
    生成，alpha 很小、K 很大时旧权重下溢为 0（与一次性 numpy 参考计算
    ``alpha ** arange`` 的行为一致），分母恒 >= 1，不会除零。
    无需保留历史快拍。
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
        return RULE_FORGETTING

    def _batch_weights(self, k: int) -> tuple[np.ndarray, float, float]:
        """返回批内每条权重（最旧->最新）、批内权重和、alpha^K。"""
        # 最新一条 1，向前逐条乘 alpha
        exponents = np.arange(k - 1, -1, -1, dtype=np.float64)
        w = self.alpha ** exponents
        return w, float(w.sum()), float(self.alpha ** k)

    def add_batch(self, X: np.ndarray) -> int:
        X = check_snapshots(X, self.m)
        k = X.shape[1]
        weights, w_sum, alpha_k = self._batch_weights(k)
        # 批内逐快拍加权；旧状态整体再乘 alpha^K
        new_sum = (alpha_k * self.n_eff) * self.C \
            + (X * weights) @ X.conj().T
        self.n_eff = alpha_k * self.n_eff + w_sum
        # 最新一条权重恒为 1，分母必 >= 1
        self.C = hermitianize(new_sum / self.n_eff)
        self.total_snapshots += k
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


class LegacyForgettingAccumulator(ForgettingAccumulator):
    """升级前的遗忘因子实现（``forgetting_batch_v1_legacy`` 旧口径）。

    旧递推把一整批 K 条等权、仅在批次之间乘一次 alpha：

    ```
    C <- (alpha * n_eff * C + X X^H) / (alpha * n_eff + K),
    n_eff <- alpha * n_eff + K
    ```

    同一批数据整块推、每包 16 条推、逐条推会得到不同的协方差与有效快拍
    数（分包相关）。升级后不再允许向这种会话追加（状态冻结为只读，需要
    继续采集请另开新会话），避免把两种口径悄悄拼在同一条时间线上。
    保留该类仅为：恢复旧状态、按原样读出历史状态并继续出估计。
    """

    @property
    def rule(self) -> str:
        return RULE_FORGETTING_LEGACY

    @property
    def append_frozen(self) -> bool:
        return True

    def add_batch(self, X: np.ndarray) -> int:
        raise CovarianceError(
            "该会话按升级前的旧遗忘因子口径（按批次等权 forgetting_"
            "batch_v1_legacy）累积，服务从未保存过该口径下的原始快拍，"
            "无法按新口径重建；为避免把两种权重口径混在同一条时间线上，"
            "该会话已冻结为只读（历史结果与状态可继续查询/出估计）。"
            "如需继续采集，请新建 forgetting 会话。"
        )

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
    kind = state["kind"]
    # 升级前落盘的 state.npz 不带 rule 字段：sliding 口径从未变化，
    # forgetting 无标记则是按批次等权的旧口径（冻结只读）。
    rule = state.get("rule")
    if kind == "sliding":
        return SlidingWindowAccumulator.from_state(state)
    if kind == "forgetting":
        if rule in (None, RULE_FORGETTING_LEGACY):
            return LegacyForgettingAccumulator.from_state(state)
        return ForgettingAccumulator.from_state(state)
    raise CovarianceError(f"未知的增量更新状态：{kind!r}。")
