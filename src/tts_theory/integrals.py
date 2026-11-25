"""双精度数值积分引擎。

提供基于概率积分变换的高精度 Best-of-N 顺序统计量积分、
极值分布精确数值寻优以及校准后验 BoN 的精确分布积分。
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

if TYPE_CHECKING:
    from tts_theory.theorems import ContinuousVerifierSpec


# 左半区 [0, 0.5] 的对数加密子区间边界（变量为 v）
_LEFT_EDGES = (
    0.0,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.05,
    0.15,
    0.30,
    0.50,
)

# 右半区 [0.5, 1.0] 的对数加密子区间边界（变量为互补坐标 w = 1 - v）
_RIGHT_W_EDGES = (
    0.0,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.05,
    0.15,
    0.30,
    0.50,
)


@lru_cache(maxsize=32)
def _composite_gl_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建并缓存分段 Gauss-Legendre 求积节点与权重。

    为了在 N in [1, 10^5] 全域内避免 v -> 1 时 1 - v^(1/N) 发生相消误差，
    返回：
        log_v: 所有节点对应的 ln(v) 数组（右侧用 log1p(-w) 精确计算）
        weights: 对应求积权重数组（总和严格等于 1.0）
        is_valid: 节点占位标识（保留便于扩展）
    """
    n_intervals = len(_LEFT_EDGES) - 1 + len(_RIGHT_W_EDGES) - 1
    order_per_interval = max(8, int(np.ceil(n_nodes / 16)))
    xi, wi = leggauss(order_per_interval)

    log_v_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    # 左半区 v in [0, 0.5]
    for a, b in zip(_LEFT_EDGES[:-1], _LEFT_EDGES[1:]):
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        v_pts = mid + half * xi
        w_pts = half * wi
        log_v_parts.append(np.log(v_pts))
        weight_parts.append(w_pts)

    # 右半区 w = 1 - v in [0, 0.5]，直接用 log1p(-w) 保持极限精度
    for a, b in zip(_RIGHT_W_EDGES[:-1], _RIGHT_W_EDGES[1:]):
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        w_coord = mid + half * xi
        w_pts = half * wi
        log_v_parts.append(np.log1p(-w_coord))
        weight_parts.append(w_pts)

    log_v = np.concatenate(log_v_parts)
    weights = np.concatenate(weight_parts)
    # 保证总权重和严格为 1.0
    weights = weights / np.sum(weights)
    return log_v, weights, np.array([n_intervals, order_per_interval], dtype=np.int64)


def _validate_positive_int(n: int, name: str = "n") -> int:
    """校验标量正整数参数。"""
    if isinstance(n, (bool, np.bool_)):
        raise ValueError(f"{name} 必须为正整数，不能为布尔值")
    n_int = int(n)
    if n_int != n or n_int < 1:
        raise ValueError(f"{name} 必须为 >= 1 的整数，当前收到: {n}")
    return n_int


def exact_bon_accuracy(
    spec: ContinuousVerifierSpec,
    n: int,
    method: str = "gauss_legendre",
    n_nodes: int = 256,
) -> float:
    """精确计算连续打分 Verifier 在 Best-of-N 下的正确率。

    利用概率积分变换 v = [F_mix(x)]^N in (0, 1)，记上尾概率
    q_sf = 1 - v^(1/N) = -expm1(ln(v) / N)，计算：
        P_BoN(N) = int_0^1 posterior(mix_isf(q_sf)) dv

    Args:
        spec: 连续打分 Verifier 规格对象。
        n: 采样预算 N (>= 1)。
        method: 积分方法，支持 "gauss_legendre" 与 "quad"。
        n_nodes: 分段 Gauss-Legendre 总参考节点规模（默认 256）。

    Returns:
        Best-of-N 精确理论正确率，范围 [0.0, 1.0]。
    """
    n_int = _validate_positive_int(n, "n")
    if n_int == 1:
        return float(np.clip(spec.p, 0.0, 1.0))

    if method == "gauss_legendre":
        if n_nodes < 16:
            raise ValueError(f"n_nodes 必须 >= 16，当前收到: {n_nodes}")
        log_v, weights, _ = _composite_gl_nodes(int(n_nodes))
        # q_sf = 1 - v^(1/N) = -expm1(ln(v) / N)
        q_sf = -np.expm1(log_v / float(n_int))
        # 防止极左/极右边界数值触碰 0 或 1 引起奇异
        q_sf = np.clip(q_sf, 1e-300, 1.0 - 1e-15)
        x = np.asarray(spec.mix_isf(q_sf), dtype=np.float64)
        post = np.asarray(spec.posterior(x), dtype=np.float64)
        acc = float(np.dot(weights, post))
        return float(np.clip(acc, 0.0, 1.0))

    if method == "quad":
        # 分别在 v in [0, 0.5] 与 w = 1 - v in [0, 0.5] 上做自适应高精度求积
        def integrand_left(v: float) -> float:
            q_sf = float(-np.expm1(np.log(max(v, 1e-300)) / float(n_int)))
            q_sf = min(max(q_sf, 1e-300), 1.0 - 1e-15)
            x_val = float(spec.mix_isf(q_sf))
            return float(spec.posterior(x_val))

        def integrand_right(w: float) -> float:
            q_sf = float(-np.expm1(np.log1p(-min(max(w, 1e-300), 0.5)) / float(n_int)))
            q_sf = min(max(q_sf, 1e-300), 1.0 - 1e-15)
            x_val = float(spec.mix_isf(q_sf))
            return float(spec.posterior(x_val))

        pts = [1e-10, 1e-7, 1e-4, 1e-2, 0.1]
        val_left, _ = integrate.quad(
            integrand_left,
            0.0,
            0.5,
            points=pts,
            epsabs=1e-11,
            epsrel=1e-10,
            limit=150,
        )
        val_right, _ = integrate.quad(
            integrand_right,
            0.0,
            0.5,
            points=pts,
            epsabs=1e-11,
            epsrel=1e-10,
            limit=150,
        )
        return float(np.clip(val_left + val_right, 0.0, 1.0))

    raise ValueError(f"不支持的积分方法 method={method!r}，仅支持 'gauss_legendre' 或 'quad'")


def exact_bon_curve(
    spec: ContinuousVerifierSpec,
    n_values: Sequence[int] | np.ndarray,
    method: str = "gauss_legendre",
    n_nodes: int = 256,
) -> np.ndarray:
    """批量计算给定 N 序列上的 Best-of-N 精确理论正确率曲线。"""
    n_seq = list(n_values)
    if len(n_seq) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(len(n_seq), dtype=np.float64)
    for idx, n_val in enumerate(n_seq):
        out[idx] = exact_bon_accuracy(spec, int(n_val), method=method, n_nodes=n_nodes)
    return out


def numerical_optimal_n(
    spec: ContinuousVerifierSpec,
    n_max: int = 100_000,
    n_nodes: int = 256,
) -> dict[str, int | float]:
    """通过密集对数采样与局部整数单峰搜索找到精确数值最优预算 N*。

    Returns:
        包含以下键的字典：
        - "n_star": 最优整数预算 N* in [1, n_max]
        - "max_accuracy": N* 对应的精确正确率
        - "p_one": N=1 基准正确率
    """
    n_max_int = _validate_positive_int(n_max, "n_max")
    p_one = float(np.clip(spec.p, 0.0, 1.0))

    if n_max_int == 1:
        return {"n_star": 1, "max_accuracy": p_one, "p_one": p_one}

    # 构造覆盖小整数与对数全域的采样点
    small_ns = np.arange(1, min(16, n_max_int) + 1, dtype=np.int64)
    if n_max_int > 16:
        geom_ns = np.unique(np.round(np.geomspace(1, n_max_int, 64)).astype(np.int64))
        grid_ns = np.unique(np.concatenate([small_ns, geom_ns]))
    else:
        grid_ns = small_ns

    grid_accs = exact_bon_curve(spec, grid_ns, method="gauss_legendre", n_nodes=n_nodes)
    best_idx = int(np.argmax(grid_accs))
    best_n = int(grid_ns[best_idx])
    best_acc = float(grid_accs[best_idx])

    # 在最佳网格点左右扩展 2 个网格步长形成包围区间，进行离散黄金分割搜索
    left_idx = max(0, best_idx - 2)
    right_idx = min(len(grid_ns) - 1, best_idx + 2)
    lo = int(grid_ns[left_idx])
    hi = int(grid_ns[right_idx])

    cache: dict[int, float] = {int(n_val): float(acc_val) for n_val, acc_val in zip(grid_ns, grid_accs)}

    def eval_n(n_val: int) -> float:
        if n_val not in cache:
            cache[n_val] = exact_bon_accuracy(spec, n_val, method="gauss_legendre", n_nodes=n_nodes)
        return cache[n_val]

    inv_phi = (np.sqrt(5.0) - 1.0) * 0.5
    while hi - lo > 16:
        span = hi - lo
        m1 = int(round(hi - inv_phi * span))
        m2 = int(round(lo + inv_phi * span))
        m1 = max(lo + 1, min(m1, hi - 1))
        m2 = max(lo + 1, min(m2, hi - 1))
        if m1 >= m2:
            m1 = lo + span // 3
            m2 = hi - span // 3
        if m1 == m2:
            break
        f1 = eval_n(m1)
        f2 = eval_n(m2)
        if f1 < f2:
            lo = m1
        else:
            hi = m2

    # 在最终窄区间内逐整数精确比较，确保找到唯一真实全局最优点
    for cand_n in range(lo, hi + 1):
        cand_acc = eval_n(cand_n)
        if cand_acc > best_acc or (cand_acc == best_acc and cand_n < best_n):
            best_acc = cand_acc
            best_n = cand_n

    return {
        "n_star": int(best_n),
        "max_accuracy": float(best_acc),
        "p_one": float(p_one),
    }


def _build_calibrated_bins(
    spec: ContinuousVerifierSpec,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造校准后验分布的高精度分位桶与局部峰加密网格，返回 (q_sorted, m_sorted)。"""
    # 1. 基础网格覆盖混合分布全域及极左/极右双尾
    n_base = max(64, n_grid // 2)
    n_tail = max(16, n_base // 6)
    n_mid = max(32, n_base - 2 * n_tail)

    q_sf_right = np.geomspace(1e-11, 0.05, n_tail)
    u_left = np.geomspace(1e-11, 0.05, n_tail)
    u_mid = np.linspace(0.05, 0.95, n_mid)

    x_left = np.asarray(spec.mix_isf(1.0 - u_left), dtype=np.float64)
    x_mid = np.asarray(spec.mix_isf(1.0 - u_mid), dtype=np.float64)
    x_right = np.asarray(spec.mix_isf(q_sf_right), dtype=np.float64)

    x_base = np.unique(np.concatenate([x_left, x_mid, x_right]))
    x_base = x_base[np.isfinite(x_base)]

    # 2. 对校准后验函数 q(x) 的高值区域进行局部高密度加密
    q_base = np.asarray(spec.posterior(x_base), dtype=np.float64)
    n_refine_budget = max(32, n_grid - len(x_base))
    extra_pts: list[np.ndarray] = []

    if len(x_base) >= 3 and n_refine_budget > 0:
        # 找到最高分位及邻近局部峰中心（最多前 3 个突出区间）
        top_indices = np.argsort(q_base)[-min(8, len(q_base)) :]
        selected_centers: list[int] = []
        min_sep = max(3, len(x_base) // 40)
        for idx in top_indices[::-1]:
            if all(abs(int(idx) - c) >= min_sep for c in selected_centers):
                selected_centers.append(int(idx))
            if len(selected_centers) >= 3:
                break

        per_center = max(8, n_refine_budget // max(1, len(selected_centers)))
        for c_idx in selected_centers:
            lo_idx = max(0, c_idx - 6)
            hi_idx = min(len(x_base) - 1, c_idx + 6)
            x_lo = x_base[lo_idx]
            x_hi = x_base[hi_idx]
            if x_hi > x_lo:
                extra_pts.append(np.linspace(x_lo, x_hi, per_center))

    if extra_pts:
        x_all = np.sort(np.unique(np.concatenate([x_base] + extra_pts)))
    else:
        x_all = x_base

    # 清理间距过小的重合节点以防除零
    if len(x_all) > 2:
        keep_mask = np.ones(len(x_all), dtype=bool)
        keep_mask[1:] = np.diff(x_all) > 1e-12
        x_all = x_all[keep_mask]

    # 3. 利用左侧 CDF 差、右侧 SF 差精确计算各桶质量 m_j 与条件后验均值 q_j
    cdf_mix = np.asarray(spec.mix_cdf(x_all), dtype=np.float64)
    sf_mix = np.asarray(spec.mix_sf(x_all), dtype=np.float64)
    cdf1 = np.asarray(spec.cdf1(x_all), dtype=np.float64)
    sf1 = np.asarray(spec.sf1(x_all), dtype=np.float64)
    p = float(spec.p)

    # 中间区间 [x_j, x_{j+1}]
    use_sf = cdf_mix[:-1] > 0.5
    m_mid = np.where(use_sf, sf_mix[:-1] - sf_mix[1:], cdf_mix[1:] - cdf_mix[:-1])
    m1_mid = np.where(use_sf, sf1[:-1] - sf1[1:], cdf1[1:] - cdf1[:-1])

    # 包含左右两个无界尾部桶 (-inf, x_0] 与 [x_K, +inf)
    m_bins = np.concatenate([[cdf_mix[0]], m_mid, [sf_mix[-1]]])
    m1_bins = np.concatenate([[cdf1[0]], m1_mid, [sf1[-1]]])

    valid = m_bins > 1e-18
    m_bins = m_bins[valid]
    m1_bins = np.maximum(0.0, m1_bins[valid])
    q_bins = np.clip((p * m1_bins) / m_bins, 0.0, 1.0)

    # 归一化保证总测度精确为 1.0
    m_bins = m_bins / np.sum(m_bins)

    # 按 q_bins 升序排列
    order = np.argsort(q_bins, kind="mergesort")
    return q_bins[order], m_bins[order]


def exact_calibrated_bon_accuracy(
    spec: ContinuousVerifierSpec,
    n: int,
    n_grid: int = 4096,
) -> float:
    """精确计算以校准后验分数 Q = q(S) 做最大化选择时的 Best-of-N 正确率。

    通过高精度自适应混合分位桶离散化 Q 的分布 F_Q，
    并基于右尾互补累积概率无相消计算：
        P_Cal-BoN(N) = E[max_{i=1..N} Q_i] = sum_j q_{(j)} * (F_Q(q_{(j)})^N - F_Q(q_{(j-1)})^N)

    Args:
        spec: 连续打分 Verifier 规格对象。
        n: 采样预算 N (>= 1)。
        n_grid: 自适应离散化网格基准点数（默认 4096）。

    Returns:
        Calibrated-BoN 精确理论正确率，范围 [0.0, 1.0]。
    """
    n_int = _validate_positive_int(n, "n")
    if n_int == 1:
        return float(np.clip(spec.p, 0.0, 1.0))
    if n_grid < 64:
        raise ValueError(f"n_grid 必须 >= 64，当前收到: {n_grid}")

    q_sorted, m_sorted = _build_calibrated_bins(spec, int(n_grid))

    # 计算右尾剩余概率 S_tail[j] = sum_{k=j}^K m_{(k)}
    s_tail = np.cumsum(m_sorted[::-1])[::-1]
    s_after = np.empty_like(s_tail)
    s_after[:-1] = s_tail[1:]
    s_after[-1] = 0.0
    s_before = s_tail

    # L = N * log1p(-S)，注意 S >= 1 时 F = 0 -> L = -inf
    n_f = float(n_int)
    l_after = np.where(s_after >= 1.0, -np.inf, n_f * np.log1p(-np.minimum(s_after, 1.0 - 1e-16)))
    l_before = np.where(s_before >= 1.0, -np.inf, n_f * np.log1p(-np.minimum(s_before, 1.0 - 1e-16)))

    # prob_j = exp(L_after) - exp(L_before) = -exp(L_after) * expm1(L_before - L_after)
    prob = np.empty_like(q_sorted)
    finite_after = np.isfinite(l_after)
    finite_both = finite_after & np.isfinite(l_before)
    only_after = finite_after & (~np.isfinite(l_before))

    prob[~finite_after] = 0.0
    prob[only_after] = np.exp(l_after[only_after])
    diff_l = np.minimum(0.0, l_before[finite_both] - l_after[finite_both])
    prob[finite_both] = -np.exp(l_after[finite_both]) * np.expm1(diff_l)

    acc = float(np.dot(q_sorted, np.maximum(0.0, prob)))
    return float(np.clip(acc, 0.0, 1.0))
