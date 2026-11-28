"""有限校准样本 M 与推理预算 N 的二次坍塌定律及悲观 KL-LCB 悲观保底模块（定理 6）。

本模块实现：
1. 精确分桶统计构建（真实边际概率 pi_k 与真实桶内正确率 q_k）；
2. 定理 6A：给定有限校准集估值向量（含任意平局组）时的闭式精确条件 Best-of-N
   正确率公式及无相消对数实现；
3. 朴素 MLE、Laplace 平滑、Wilson-LCB、KL-LCB 与 Empirical-Bernstein LCB 五类校准得分；
4. Minimax 最优分桶标度律 K*(M) ~ (M / ln M)^{1 / (2*beta + 1)}（beta=2 时为 1/5）
   及联合 (M, N) 信息论下界 Omega((ln M/M)^{beta/(2*beta+1)} + N^{-beta}) 评估；
5. 有限校准集重复采样 + 精确条件推理评估引擎。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special

from tts_theory.theorems import ContinuousVerifierSpec


def build_score_bins(
    spec: ContinuousVerifierSpec,
    n_bins: int,
    binning: str = "quantile",
    tail_clip: float = 5e-4,
) -> dict[str, np.ndarray]:
    """根据真实 Verifier 分布构建覆盖全实轴的 K 个互斥分桶及其真实参数 (pi_k, q_k)。

    Args:
        spec: 连续打分 Verifier 规格。
        n_bins: 目标分桶数 K (>= 2)。
        binning: 分桶方式，"quantile"（等频分位切分）或 "uniform"（主区间等宽+两端无限尾桶）。
        tail_clip: 等宽分桶时主支撑截断尾概率（仅 binning="uniform" 时生效）。

    Returns:
        包含 "edges", "bin_probs", "true_bin_q", "n_bins" 的字典，满足 sum(bin_probs) == 1.0。
    """
    k_int = int(n_bins)
    if k_int < 2:
        raise ValueError(f"n_bins 必须 >= 2，当前收到: {n_bins}")
    if binning not in ("quantile", "uniform"):
        raise ValueError(f"不支持的 binning={binning!r}，仅支持 'quantile' 或 'uniform'")

    if binning == "quantile":
        probs_inner = np.linspace(0.0, 1.0, k_int + 1)[1:-1]
        inner_edges = np.asarray(spec.mix_ppf(probs_inner), dtype=np.float64)
        inner_edges = np.unique(inner_edges)
    else:
        if not (0.0 < tail_clip < 0.25):
            raise ValueError("tail_clip 必须位于 (0, 0.25) 内")
        lo_x = float(spec.mix_ppf(tail_clip))
        hi_x = float(spec.mix_isf(tail_clip))
        if not np.isfinite(lo_x) or not np.isfinite(hi_x) or hi_x <= lo_x:
            raise ValueError("等宽分桶主区间边界退化")
        # K 个桶需要 K - 1 个内部边界：若把 [lo_x, hi_x] 均分为 K-2 段，加上 (-inf, lo_x] 和 (hi_x, +inf)
        if k_int == 2:
            inner_edges = np.array([0.5 * (lo_x + hi_x)], dtype=np.float64)
        else:
            inner_edges = np.linspace(lo_x, hi_x, k_int - 1)

    # 精确计算各桶边际测度 pi_k 与正类测度 p*m1_k（左半用 cdf 差分，右半用 sf 差分以防相消）
    cdf_mix = np.asarray(spec.mix_cdf(inner_edges), dtype=np.float64)
    sf_mix = np.asarray(spec.mix_sf(inner_edges), dtype=np.float64)
    cdf1 = np.asarray(spec.cdf1(inner_edges), dtype=np.float64)
    sf1 = np.asarray(spec.sf1(inner_edges), dtype=np.float64)

    if inner_edges.size >= 2:
        use_sf = cdf_mix[:-1] > 0.5
        m_mid = np.where(use_sf, sf_mix[:-1] - sf_mix[1:], cdf_mix[1:] - cdf_mix[:-1])
        m1_mid = np.where(use_sf, sf1[:-1] - sf1[1:], cdf1[1:] - cdf1[:-1])
        bin_probs = np.concatenate([[cdf_mix[0]], m_mid, [sf_mix[-1]]])
        m1_probs = np.concatenate([[cdf1[0]], m1_mid, [sf1[-1]]])
    else:
        bin_probs = np.array([cdf_mix[0], sf_mix[0]], dtype=np.float64)
        m1_probs = np.array([cdf1[0], sf1[0]], dtype=np.float64)

    bin_probs = np.maximum(0.0, bin_probs)
    m1_probs = np.maximum(0.0, m1_probs)

    # 剔除数值为零的空测度桶并严格归一化总概率
    valid = bin_probs > 1e-18
    bin_probs = bin_probs[valid]
    m1_probs = m1_probs[valid]
    total = float(np.sum(bin_probs))
    bin_probs = bin_probs / total
    true_bin_q = np.clip((float(spec.p) * m1_probs / total) / bin_probs, 0.0, 1.0)

    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    return {
        "edges": edges,
        "bin_probs": bin_probs,
        "true_bin_q": true_bin_q,
        "n_bins": np.array(len(bin_probs), dtype=np.int64),
    }


def exact_conditional_bin_bon_accuracy(
    true_bin_q: np.ndarray,
    bin_probs: np.ndarray,
    est_scores: np.ndarray,
    n_values: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """基于定理 6A 精确闭式平局分组公式计算给定校准估值下的条件 BoN 准确率。

    对按 est_scores 降序排列后的等价组 G_1, ..., G_L，令组边缘概率为 P_l，
    组内真实正确率为 q_bar_l，后续剩余概率 R_l = sum_{j > l} P_j，则：
        P(N | D_M) = sum_{l=1}^L q_bar_l * [(R_l + P_l)^N - R_l^N]
    全程使用 log1p / expm1 避免 R_l -> 1 或大 N 下的有效数字相消。
    """
    q_arr = np.asarray(true_bin_q, dtype=np.float64)
    pi_arr = np.asarray(bin_probs, dtype=np.float64)
    s_arr = np.asarray(est_scores, dtype=np.float64)
    n_arr = np.asarray(list(n_values), dtype=np.float64)

    if q_arr.shape != pi_arr.shape or q_arr.shape != s_arr.shape or q_arr.ndim != 1:
        raise ValueError("true_bin_q, bin_probs, est_scores 必须为等长一维数组")
    if n_arr.ndim != 1 or n_arr.size == 0 or np.any(n_arr < 1):
        raise ValueError("n_values 必须为非空正整数序列")

    # 仅保留正概率桶
    pos = pi_arr > 0.0
    q_arr = q_arr[pos]
    pi_arr = pi_arr[pos]
    pi_arr = pi_arr / np.sum(pi_arr)
    s_arr = s_arr[pos]

    # 按估值降序排序并聚合同分平局组（浮点容差内精确相等聚合）
    order = np.argsort(-s_arr, kind="mergesort")
    s_sorted = s_arr[order]
    pi_sorted = pi_arr[order]
    q_sorted = q_arr[order]

    # 检测平局边界
    if s_sorted.size == 1:
        group_probs = pi_sorted.copy()
        group_q = q_sorted.copy()
    else:
        is_new = np.empty(s_sorted.size, dtype=bool)
        is_new[0] = True
        is_new[1:] = np.abs(s_sorted[1:] - s_sorted[:-1]) > 1e-13
        group_starts = np.flatnonzero(is_new)
        group_probs = np.add.reduceat(pi_sorted, group_starts)
        group_pos_mass = np.add.reduceat(pi_sorted * q_sorted, group_starts)
        group_q = np.where(group_probs > 0.0, group_pos_mass / group_probs, 0.0)

    # 计算剩余尾概率 R_l = sum_{j > l} P_j，U_l = R_l + P_l
    # 从后向前累加，保证 U_1 == 1.0 且 R_L == 0.0
    rev_cumsum = np.cumsum(group_probs[::-1])[::-1]
    u_probs = np.clip(rev_cumsum, 0.0, 1.0)
    r_probs = np.empty_like(u_probs)
    r_probs[:-1] = u_probs[1:]
    r_probs[-1] = 0.0

    # 用无相消方法算 log_U = log1p(-(1 - U)) 与 log_R = log(R)
    # 事实上 sum_{j <= l} P_j = 1 - U_l，若前部累加 S_prefix 较小则 log_U = log1p(-S_prefix) 更稳
    prefix_before = np.empty_like(group_probs)
    prefix_before[0] = 0.0
    if group_probs.size > 1:
        prefix_before[1:] = np.cumsum(group_probs[:-1])
    prefix_incl = np.minimum(1.0, np.cumsum(group_probs))

    log_u = np.where(
        prefix_before < 0.5,
        np.log1p(-np.minimum(prefix_before, 1.0 - 1e-16)),
        np.log(np.maximum(u_probs, 1e-300)),
    )
    log_r = np.where(
        r_probs <= 0.0,
        -np.inf,
        np.where(
            prefix_incl < 0.5,
            np.log1p(-np.minimum(prefix_incl, 1.0 - 1e-16)),
            np.log(np.maximum(r_probs, 1e-300)),
        ),
    )

    # 对所有 N 批量矩阵化求解 (len(N), L)
    # P(l 组胜出) = exp(N * log_U) * (-expm1(N * (log_R - log_U)))
    n_col = n_arr[:, None]
    term_u = np.exp(np.clip(n_col * log_u[None, :], -745.0, 0.0))
    diff_log = np.minimum(0.0, log_r[None, :] - log_u[None, :])
    win_prob = np.where(
        np.isinf(log_r[None, :]),
        term_u,
        term_u * (-np.expm1(np.maximum(-745.0, n_col * diff_log))),
    )
    acc = win_prob @ group_q
    return np.clip(acc, 0.0, 1.0)


def _kl_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """计算二元相对熵 kl(p, q) = p*ln(p/q) + (1-p)*ln((1-p)/(1-q))，边界处安全处理。"""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    term1 = np.where(p_arr > 0.0, p_arr * np.log(np.maximum(p_arr, 1e-300) / np.maximum(q_arr, 1e-300)), 0.0)
    term0 = np.where(
        p_arr < 1.0,
        (1.0 - p_arr) * np.log(np.maximum(1.0 - p_arr, 1e-300) / np.maximum(1.0 - q_arr, 1e-300)),
        0.0,
    )
    return np.maximum(0.0, term1 + term0)


def _kl_binomial_lcb(
    positives: np.ndarray,
    counts: np.ndarray,
    rhs: float,
    n_bisect: int = 55,
) -> np.ndarray:
    """求解精确非渐近二项 KL-LCB：inf { q in [0, p_hat] : m * kl(p_hat, q) <= rhs }。"""
    m = np.asarray(counts, dtype=np.float64)
    s = np.asarray(positives, dtype=np.float64)
    out = np.zeros_like(m, dtype=np.float64)
    valid = (m > 0.0) & (s > 0.0)
    if not np.any(valid):
        return out

    m_v = m[valid]
    p_v = np.clip(s[valid] / m_v, 0.0, 1.0)
    res_v = np.zeros_like(m_v, dtype=np.float64)

    # p_hat == 1 时精确解析解为 exp(-rhs / m)
    full_mask = p_v >= 1.0 - 1e-15
    if np.any(full_mask):
        res_v[full_mask] = np.exp(-float(rhs) / m_v[full_mask])

    mid_mask = ~full_mask
    if np.any(mid_mask):
        pm = p_v[mid_mask]
        mm = m_v[mid_mask]
        # 检查 q = 0 处是否有 mm * (-ln(1 - pm)) <= rhs
        kl_at_zero = -mm * np.log1p(-pm)
        active = kl_at_zero > float(rhs)
        sol = np.zeros_like(pm, dtype=np.float64)
        if np.any(active):
            lo = np.zeros(int(np.count_nonzero(active)), dtype=np.float64)
            hi = pm[active].copy()
            pa = pm[active]
            ma = mm[active]
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                too_low = ma * _kl_bernoulli(pa, mid) > float(rhs)
                lo = np.where(too_low, mid, lo)
                hi = np.where(too_low, hi, mid)
            sol[active] = 0.5 * (lo + hi)
        res_v[mid_mask] = sol

    out[valid] = np.clip(res_v, 0.0, 1.0)
    return out


def bernstein_zero_collapse_threshold(
    n_bins: int,
    q_peak: float,
    delta: float = 0.05,
) -> float:
    """计算 Empirical-Bernstein LCB 发生全零截断崩溃的临界桶样本阈值（定理 6B）。

    确定性形式针对经验桶均值 q_hat_k：当 m_k < 1 + (7 / (3 * q_hat_k)) * ln(2K / delta)
    时线性惩罚项已超过 q_hat_k，LCB 被截断至 0。传入真均值或峰值上界 q_peak 时给出
    对应的高概率或大样本参考阈值（在 1 - delta/K 单侧上偏事件外一致生效）。
    """
    k_int = int(n_bins)
    qp = float(q_peak)
    if k_int < 1 or not (0.0 < qp <= 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("n_bins >= 1, q_peak in (0, 1], delta in (0, 1)")
    return float(1.0 + (7.0 / (3.0 * qp)) * math.log(2.0 * float(k_int) / float(delta)))


def compute_bin_calibrator_scores(
    counts: np.ndarray,
    positives: np.ndarray,
    method: str,
    prior_p: float,
    delta: float = 0.05,
    prior_strength: float = 2.0,
) -> np.ndarray:
    """给定每个分桶的离线样本计数 m_k 与正例数 S_k，计算校准排序分数。

    支持方法：
    - "mle": 朴素频率 S_k / m_k（m_k=0 时置 -1.0，全 0 时退化为 0.0）
    - "laplace": Beta 先验平滑 (S_k + s*p) / (m_k + s)（m_k=0 时置 -1.0）
    - "wilson_lcb": 联合置信度 1 - delta 的 Wilson 得分区间下界（m_k=0 时置 0.0）
    - "kl_lcb": 严格非渐近 Chernoff-Sanov 二项 KL 下置信界（m_k=0 时置 0.0）
    - "empirical_bernstein_lcb": 联合置信度 1 - delta 的经验 Bernstein LCB（m_k=0 时置 0.0）
    """
    m = np.asarray(counts, dtype=np.float64)
    s = np.asarray(positives, dtype=np.float64)
    k = int(m.size)
    if k < 1 or s.shape != m.shape:
        raise ValueError("counts 与 positives 必须为非空等长一维数组")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta 必须位于 (0, 1) 内")

    nonempty = m > 0.0
    p_hat = np.zeros(k, dtype=np.float64)
    p_hat[nonempty] = np.clip(s[nonempty] / m[nonempty], 0.0, 1.0)

    if method == "mle":
        out = np.where(nonempty, p_hat, -1.0)
        return out if np.any(nonempty) else np.zeros(k, dtype=np.float64)

    if method == "laplace":
        if prior_strength < 0.0:
            raise ValueError("prior_strength 不能为负")
        sm = np.where(
            nonempty,
            (s + prior_strength * float(prior_p)) / np.maximum(1.0, m + prior_strength),
            -1.0,
        )
        return sm if np.any(nonempty) else np.full(k, float(prior_p), dtype=np.float64)

    if method == "wilson_lcb":
        # 联合 Bonferroni 临界值 z = Phi^{-1}(1 - delta / (2K))
        z = float(-special.ndtri(float(delta) / (2.0 * float(k))))
        z2 = z * z
        out = np.zeros(k, dtype=np.float64)
        if np.any(nonempty):
            m_e = m[nonempty]
            q_e = p_hat[nonempty]
            denom = 1.0 + z2 / m_e
            center = q_e + z2 / (2.0 * m_e)
            rad = z * np.sqrt(np.maximum(0.0, q_e * (1.0 - q_e) / m_e + z2 / (4.0 * m_e * m_e)))
            out[nonempty] = np.clip((center - rad) / denom, 0.0, 1.0)
        return out

    if method == "kl_lcb":
        # 联合非渐近 Sanov 单侧边界 m * kl(p_hat, q) <= ln(K / delta)
        rhs = math.log(float(k) / float(delta))
        return _kl_binomial_lcb(s, m, rhs)

    if method == "empirical_bernstein_lcb":
        log_term = math.log(2.0 * float(k) / float(delta))
        out = np.zeros(k, dtype=np.float64)
        if np.any(nonempty):
            m_e = m[nonempty]
            q_e = p_hat[nonempty]
            # 无偏二项样本方差 v_hat = (m / max(1, m-1)) * q_hat * (1 - q_hat)
            v_hat = (m_e / np.maximum(1.0, m_e - 1.0)) * q_e * (1.0 - q_e)
            rad = np.sqrt(2.0 * v_hat * log_term / m_e) + (7.0 * log_term) / (
                3.0 * np.maximum(1.0, m_e - 1.0)
            )
            out[nonempty] = np.clip(q_e - rad, 0.0, 1.0)
        return out

    raise ValueError(f"不支持的校准方法 method={method!r}")


def calibration_minimax_scaling(
    m_total: int,
    curvature_c: float = 1.0,
    q_peak: float = 0.5,
    delta: float = 0.05,
    beta: float = 2.0,
) -> dict[str, float | int]:
    """计算等频分桶下的精确理论 Minimax 最优分桶数与大 N 遗憾标度律（定理 6A 修正版）。

    在 beta-Holder 光滑峰 q* - q(x) ~ c*|x - x*|^beta 下，离散化偏差 ~ c / K^beta，
    LCB 极值随机遗憾 ~ sqrt(K * ln(K/delta) / M)，精确平衡得：
        K*(M) ~ (M / ln M)^{1 / (2*beta + 1)}   （beta=2 时为 1/5）
        Regret(M) ~ (ln M / M)^{beta / (2*beta + 1)}   （beta=2 时为 2/5）
    """
    m_int = int(m_total)
    beta_f = float(beta)
    if m_int < 10:
        raise ValueError("m_total 必须 >= 10")
    if beta_f <= 0.0:
        raise ValueError("beta 必须 > 0")
    log_m = max(1.0, math.log(float(m_int)))
    var_scale = max(1e-4, float(q_peak) * (1.0 - float(q_peak)))
    k_exp = 1.0 / (2.0 * beta_f + 1.0)
    r_exp = beta_f / (2.0 * beta_f + 1.0)

    # 经验自适应系数 c_0：由局部曲率与方差尺度给出（理论速率 (M / ln M)^{1/(2*beta+1)} 中的常数）
    prefactor = max(1.5, (float(curvature_c) ** 2 / var_scale) ** k_exp)
    k_opt_float = prefactor * ((float(m_int) / log_m) ** k_exp)
    k_opt = int(max(3, min(m_int // 5, round(k_opt_float))))
    regret_rate = float((log_m / float(m_int)) ** r_exp)
    lcb_penalty = float(math.sqrt(float(k_opt) * math.log(2.0 * k_opt / delta) / float(m_int)))

    return {
        "m_total": m_int,
        "beta": beta_f,
        "k_exponent": float(k_exp),
        "regret_exponent": float(r_exp),
        "k_opt_theory": k_opt,
        "k_opt_float": float(k_opt_float),
        "regret_rate_theory": regret_rate,
        "lcb_penalty_scale": lcb_penalty,
    }


def joint_calibration_compute_scaling(
    m_total: int,
    n_budget: int,
    beta: float = 2.0,
    curvature_c: float = 1.0,
    q_peak: float = 0.5,
    delta: float = 0.05,
) -> dict[str, float | int | str | bool]:
    """计算有限校准量 M 与有限推理预算 N 的联合分桶与率指标（定理 6C，适用于一致正则类 F_beta^unif）。"""
    m_int = int(m_total)
    n_int = int(n_budget)
    beta_f = float(beta)
    if m_int < 10 or n_int < 1 or beta_f <= 0.0:
        raise ValueError("要求 m_total >= 10, n_budget >= 1 且 beta > 0")

    cal_info = calibration_minimax_scaling(
        m_int, curvature_c=curvature_c, q_peak=q_peak, delta=delta, beta=beta_f
    )
    k_cal = int(cal_info["k_opt_theory"])
    # 旧式分桶截断基线：1/K >= beta * ln(K) / N => K_comp ~ N / (beta * max(1, ln N))
    log_n = max(1.0, math.log(float(max(2, n_int))))
    k_comp_float = max(2.0, float(n_int) / (beta_f * log_n))
    k_comp = max(2, int(math.floor(k_comp_float)))
    k_joint = max(2, min(k_cal, k_comp))

    n_match = int(math.ceil(float(k_cal) * beta_f * max(1.0, math.log(float(max(2, k_cal))))))
    oracle_n_rate = float(float(n_int) ** (-beta_f))
    cal_m_rate = float(cal_info["regret_rate_theory"])
    delta_opt = 1.0 / float(m_int)
    minimax_lower_rate = float(cal_m_rate + oracle_n_rate)
    kl_lcb_compute_rate = float(((beta_f * log_n) / float(n_int)) ** beta_f)
    kl_lcb_upper_rate = float(cal_m_rate + kl_lcb_compute_rate)
    compute_log_gap_factor = float((beta_f * log_n) ** beta_f)
    regime = "compute_limited" if k_comp < k_cal else "calibration_limited"

    return {
        "m_total": m_int,
        "n_budget": n_int,
        "beta": beta_f,
        "k_cal_star": k_cal,
        "k_comp_limit": k_comp,
        "k_joint_star": k_joint,
        "n_match": n_match,
        "oracle_n_rate": oracle_n_rate,
        "cal_m_rate": cal_m_rate,
        "failure_prob_delta_opt": delta_opt,
        "minimax_lower_rate": minimax_lower_rate,
        "kl_lcb_compute_rate": kl_lcb_compute_rate,
        "kl_lcb_upper_rate": kl_lcb_upper_rate,
        "is_minimax_tight": bool(regime == "calibration_limited"),
        "compute_log_gap_factor": compute_log_gap_factor,
        "regime": regime,
        "n_agnostic_k_opt": k_cal,
        "n_agnostic_upper_rate": minimax_lower_rate,
    }


def empirical_quantile_spacings_bounds(
    m_total: int,
    n_bins: int,
    delta: float = 0.05,
) -> dict[str, float | int | bool]:
    """评估样本分割经验等频分桶的一致桶质量乘性浓缩界（定理 6C 命题 6.1，覆盖一切 beta > 0）。

    将容量 M 的校准集平分为边界子集 D_{M_1}^{edge}（M_1 = floor(M/2)）与估计子集 D_{M_2}^{est}
    （M_2 = M - M_1）。取顺序统计量下标 r_k = floor(k * (M_1 + 1) / K)，由连续 CDF 变换与
    Rényi-Pyke 指数间距表示，真实桶质量满足：
        pi_hat_k = U_{(r_k)} - U_{(r_{k-1})} =^d G_k / G_tot,
    其中 G_k ~ Gamma(d_k, 1)（d_k = r_k - r_{k-1}），G_tot ~ Gamma(M_1 + 1, 1)。
    当 M_1 + 1 >= 12*K 时，11/12 <= d_k * K / (M_1 + 1) <= 13/12；对 Gamma(d, 1) 应用精确 Cramér 界：
        P(G_k < 0.60 * d_k) <= exp(-psi_-(0.60) * d_k),   psi_-(0.60) = -0.40 - ln(0.60) > 1/9.1,
        P(G_k > 1.65 * d_k) <= exp(-psi_+(1.65) * d_k),   psi_+(1.65) =  0.65 - ln(1.65) > 1/9.1,
        P(G_tot notin [0.90*(M_1+1), 1.10*(M_1+1)]) <= exp(-psi_-(0.90)*(M_1+1)) + exp(-psi_+(1.10)*(M_1+1)).
    当 M_1 + 1 >= max(12*K, 214*ln(4/delta)) 且 d_min >= 9.1 * ln(4*K / delta) 时（即 M = Omega(K * ln(K / delta))），
    以至少 1 - delta 的概率对全体 k = 1, ..., K 同时成立：
        1 / (2*K) <= pi_hat_k <= 143 / (72*K) < 2 / K.
    此时估计集桶样本下界由总体等频的 M/(2K) 变为 M/(8K)，故 Pinsker/Hoeffding 统计惩罚项常数放大因子精确为
    sqrt((M/(2K)) / (M/(8K))) = 2.0（而非 sqrt(2)）。
    """
    m_int = int(m_total)
    k_int = int(n_bins)
    delta_f = float(delta)
    if m_int < 4 or k_int < 2 or not (0.0 < delta_f < 1.0):
        raise ValueError("要求 m_total >= 4, n_bins >= 2 且 delta in (0, 1)")

    m1 = m_int // 2
    m2 = m_int - m1
    r_idx = np.floor(np.arange(k_int + 1, dtype=np.float64) * float(m1 + 1) / float(k_int)).astype(np.int64)
    d_spacings = np.diff(r_idx)
    d_min = int(np.min(d_spacings))
    d_max = int(np.max(d_spacings))

    psi_lo = -0.40 - math.log(0.60)   # 0.11082562... > 1 / 9.1
    psi_hi = 0.65 - math.log(1.65)    # 0.14922468... > 1 / 9.1
    psi_tot_lo = -0.10 - math.log(0.90)  # 0.00536051... > 1 / 187
    psi_tot_hi = 0.10 - math.log(1.10)   # 0.00468982... > 1 / 214

    fail_prob_spacings = float(
        np.sum(np.exp(-psi_lo * d_spacings) + np.exp(-psi_hi * d_spacings))
        + math.exp(-psi_tot_lo * float(m1 + 1))
        + math.exp(-psi_tot_hi * float(m1 + 1))
    )
    d_min_required = float(9.1 * math.log(4.0 * float(k_int) / delta_f))
    m1_tot_required = float(max(12.0 * float(k_int), 214.0 * math.log(4.0 / delta_f)))
    spacings_condition_holds = bool(
        (float(m1 + 1) >= m1_tot_required) and (float(d_min) >= d_min_required)
    )

    # 对比加性 DKW 相对误差上界：sup_k |pi_hat_k - 1/K| * K <= 2 * K * sqrt(ln(2/delta) / (2 * M_1))
    dkw_rel_err = float(2.0 * float(k_int) * math.sqrt(math.log(2.0 / delta_f) / max(1.0, 2.0 * float(m1))))

    return {
        "m_total": m_int,
        "m_edge": m1,
        "m_est": m2,
        "n_bins": k_int,
        "d_min": d_min,
        "d_max": d_max,
        "d_min_required": d_min_required,
        "m_edge_plus_one_required": m1_tot_required,
        "spacings_failure_prob_upper": min(1.0, fail_prob_spacings),
        "spacings_condition_holds": spacings_condition_holds,
        "guaranteed_pi_lower_factor": 0.5,
        "guaranteed_pi_upper_factor": 143.0 / 72.0,
        "lcb_penalty_inflation_factor": 2.0,
        "dkw_relative_error_bound": dkw_rel_err,
        "dkw_non_vacuous": bool(dkw_rel_err < 0.5),
    }


def tight_joint_minimax_scaling(
    m_total: int,
    n_budget: int,
    beta: float = 2.0,
    curvature_c: float = 1.0,
    f_min: float = 0.25,
    eta_radius: float = 0.5,
) -> dict[str, float | int | bool]:
    """在一致正则分布类 F_beta^unif(C_1, f_min, eta_0) 上评估等频 KL-LCB 的非渐近上界（定理 6C）。

    针对公共类参数 (curvature_c=C_1, f_min, eta_radius) 下的全体分布，使用 N-无关最优分桶
    K*(M) 与 delta = 1/(3M) 三事件离线并界（上越界 + 桶计数不足 + Hoeffding 下侧偏差）：
    - 采用通用幂次不等式 (a+b)^beta <= c_beta (a^beta + b^beta)，其中 c_beta = max(1, 2^{beta-1})
      对一切 beta > 0 统一成立；
    - 同时通过样本分割顺序统计量 Gamma 间距界（empirical_quantile_spacings_bounds）检验未知总体分位数下的
      经验等频分桶条件 M_1 >= C * K * ln(K*M)（该条件对一切 beta > 0 均随 M 增大自动满足）。
    """
    m_int = int(m_total)
    n_int = int(n_budget)
    beta_f = float(beta)
    c_curv = float(curvature_c)
    f_m = float(f_min)
    eta_r = float(eta_radius)
    if m_int < 10 or n_int < 1 or beta_f <= 0.0 or c_curv <= 0.0 or f_m <= 0.0 or eta_r <= 0.0:
        raise ValueError("tight_joint_minimax_scaling 输入参数无效")

    delta_opt = 1.0 / float(m_int)
    cal_info = calibration_minimax_scaling(
        m_int, curvature_c=c_curv, q_peak=0.5, delta=delta_opt, beta=beta_f
    )
    k_opt = int(cal_info["k_opt_theory"])

    c_beta = max(1.0, 2.0 ** (beta_f - 1.0))
    failure_term = delta_opt
    miss_neighborhood_term = float(math.exp(-eta_r * f_m * float(n_int)))
    bin_bias_term = float(c_curv * c_beta / ((float(k_opt) * f_m) ** beta_f))
    pinsker_lcb_term = float(2.0 * math.sqrt(float(k_opt) * math.log(3.0 * float(k_opt) * float(m_int)) / float(m_int)))
    online_nearest_term = float(
        c_curv * c_beta * math.gamma(beta_f + 1.0) / ((2.0 * f_m) ** beta_f) * (float(n_int) ** (-beta_f))
    )
    total_upper_bound = (
        failure_term
        + miss_neighborhood_term
        + bin_bias_term
        + pinsker_lcb_term
        + online_nearest_term
    )
    minimax_rate_lower = float(
        (math.log(float(m_int)) / float(m_int)) ** (beta_f / (2.0 * beta_f + 1.0))
        + (float(n_int) ** (-beta_f))
    )
    uniform_bin_count_condition_holds = bool(
        float(m_int) >= 8.0 * float(k_opt) * math.log(3.0 * float(k_opt) * float(m_int))
        and float(k_opt) > 2.0 / (eta_r * f_m)
    )
    spacings_info = empirical_quantile_spacings_bounds(
        m_total=m_int, n_bins=max(2, k_opt), delta=min(0.25, 1.0 / float(4 * m_int))
    )
    upper_to_lower_rate_ratio = float(total_upper_bound / max(minimax_rate_lower, 1e-30))
    balance_n = float((float(m_int) / max(1.0, math.log(float(m_int)))) ** (1.0 / (2.0 * beta_f + 1.0)))
    bottleneck_regime = (
        "offline_calibration_dominated" if float(n_int) > balance_n else "online_compute_dominated"
    )

    return {
        "m_total": m_int,
        "n_budget": n_int,
        "beta": beta_f,
        "c_beta_factor": float(c_beta),
        "n_agnostic_k_opt": k_opt,
        "calibration_compute_balance_n": balance_n,
        "bottleneck_regime": bottleneck_regime,
        "failure_term": failure_term,
        "miss_neighborhood_term": miss_neighborhood_term,
        "bin_bias_term": bin_bias_term,
        "pinsker_lcb_term": pinsker_lcb_term,
        "online_nearest_term": online_nearest_term,
        "total_upper_bound": float(total_upper_bound),
        "minimax_rate_lower": minimax_rate_lower,
        "upper_to_lower_rate_ratio": upper_to_lower_rate_ratio,
        "uniform_bin_count_condition_holds": uniform_bin_count_condition_holds,
        "spacings_sample_split_condition_holds": bool(spacings_info["spacings_condition_holds"]),
        "dkw_relative_error_bound": float(spacings_info["dkw_relative_error_bound"]),
    }


def simulate_finite_calibration_scaling(
    spec: ContinuousVerifierSpec,
    m_cal: int,
    n_bins: int,
    n_values: Sequence[int] | np.ndarray,
    binning: str = "uniform",
    methods: Sequence[str] = (
        "mle",
        "laplace",
        "wilson_lcb",
        "empirical_bernstein_lcb",
    ),
    n_repeats: int = 1_000,
    delta: float = 0.05,
    prior_strength: float = 2.0,
    tail_clip: float = 5e-4,
    seed: int = 42,
) -> dict:
    """仿真有限校准样本 M 下的期望 BoN 正确率曲线（外层有限校准集采样 + 内层精确条件积分）。

    利用 multinomial(M, pi_k) + binomial(m_k, q_k) 与真实 i.i.d. 抽取 M 条校准样本
    在统计分布上 100% 等价的性质，并结合定理 6A 精确闭式公式评估任意 N（含 N=10^4）。
    """
    if m_cal < 5 or n_repeats < 1:
        raise ValueError("m_cal 必须 >= 5 且 n_repeats 必须 >= 1")

    n_arr = np.asarray(list(n_values), dtype=np.int64)
    bins = build_score_bins(spec, n_bins=n_bins, binning=binning, tail_clip=tail_clip)
    pi_k = bins["bin_probs"]
    q_k = bins["true_bin_q"]
    k_actual = int(bins["n_bins"])

    # 理论 Oracle 桶排序（按真实 q_k 排序）的精确条件曲线
    oracle_acc = exact_conditional_bin_bon_accuracy(q_k, pi_k, q_k, n_arr)

    rng = np.random.default_rng(seed)
    sum_acc = {m_name: np.zeros(len(n_arr), dtype=np.float64) for m_name in methods}
    sum_sq = {m_name: np.zeros(len(n_arr), dtype=np.float64) for m_name in methods}
    asymp_limit_sum = {m_name: 0.0 for m_name in methods}

    for _ in range(n_repeats):
        counts = rng.multinomial(int(m_cal), pi_k)
        positives = rng.binomial(counts, q_k)
        for m_name in methods:
            est = compute_bin_calibrator_scores(
                counts,
                positives,
                method=m_name,
                prior_p=float(spec.p),
                delta=delta,
                prior_strength=prior_strength,
            )
            cond_acc = exact_conditional_bin_bon_accuracy(q_k, pi_k, est, n_arr)
            sum_acc[m_name] += cond_acc
            sum_sq[m_name] += cond_acc * cond_acc

            # 统计 N -> inf 渐近极限：并列最高估值组内按 pi_k 加权平均真实正确率
            max_est = np.max(est)
            top_mask = (np.abs(est - max_est) <= 1e-13) & (pi_k > 0.0)
            asymp_q = float(np.sum(pi_k[top_mask] * q_k[top_mask]) / np.sum(pi_k[top_mask]))
            asymp_limit_sum[m_name] += asymp_q

    method_results: dict[str, dict[str, np.ndarray | float]] = {}
    for m_name in methods:
        mean_acc = sum_acc[m_name] / float(n_repeats)
        var_acc = np.maximum(0.0, sum_sq[m_name] / float(n_repeats) - mean_acc * mean_acc)
        se_acc = np.sqrt(var_acc / float(n_repeats))
        method_results[m_name] = {
            "accuracy": mean_acc,
            "se": se_acc,
            "ci_low": np.clip(mean_acc - 1.96 * se_acc, 0.0, 1.0),
            "ci_high": np.clip(mean_acc + 1.96 * se_acc, 0.0, 1.0),
            "expected_asymptotic_limit": float(asymp_limit_sum[m_name] / float(n_repeats)),
        }

    return {
        "n_values": n_arr,
        "m_cal": int(m_cal),
        "n_bins_actual": k_actual,
        "binning": binning,
        "oracle_bin_accuracy": oracle_acc,
        "oracle_asymptotic_limit": float(np.max(q_k)),
        "methods": method_results,
    }


__all__ = [
    "bernstein_zero_collapse_threshold",
    "build_score_bins",
    "calibration_minimax_scaling",
    "compute_bin_calibrator_scores",
    "empirical_quantile_spacings_bounds",
    "exact_conditional_bin_bon_accuracy",
    "joint_calibration_compute_scaling",
    "simulate_finite_calibration_scaling",
    "tight_joint_minimax_scaling",
]
