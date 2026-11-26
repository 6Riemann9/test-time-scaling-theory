"""真实大模型候选池与开源 Reward Model 实证评估引擎（V3）。

提供完全离线的评估工具：
1. 数学答案提取（嵌套 \\boxed{...}）与等价规范化判分；
2. 真实有限候选池精确超几何无放回 Best-of-N 正确率（exact_pool_bon_accuracy）；
3. 真实 Verifier 分布诊断（异方差 kappa、超额峰度、尾部后验倒挂检验）；
4. 真实多裁判相关性矩阵 (R_1, R_0) 与定理 5 相变判据映射；
5. 真实候选池上的有限校准二次坍塌评估与离散答案抗坍塌算法对比。
"""

from __future__ import annotations

from fractions import Fraction
import math
import re
from typing import Any, Callable, Sequence

import numpy as np
from scipy import optimize, special, stats

from tts_theory.algorithms import (
    CalibratedWeightedBoN,
    EmpiricalScoreCalibrator,
    MajorityVoting,
    StandardBoN,
    TwoStageVerifiedMajority,
    _argmax_random_tie_2d,
)
from tts_theory.calibration_theory import (
    calibration_minimax_scaling,
    compute_bin_calibrator_scores,
    joint_calibration_compute_scaling,
)
from tts_theory.ensemble_theory import (
    GeneralGaussianMultiVerifierSpec,
    MultiVerifierGaussianSpec,
    ensemble_phase_reversal_criterion,
    general_min_ensemble_qp_criterion,
    optimal_linear_ensemble_criterion,
    solve_positive_quadrant_qp,
)


def extract_boxed_answer(text: str) -> str:
    """从模型推理补全文本中提取最后一个 \\boxed{...}（支持任意嵌套花括号）。"""
    if not isinstance(text, str) or not text:
        return ""
    for token in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(token)
        if idx != -1:
            start = idx + len(token)
            depth = 1
            pos = start
            while pos < len(text) and depth > 0:
                ch = text[pos]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:pos].strip()
                pos += 1
            return text[start:].strip()

    # 无 \\boxed 时回退匹配常见最终答案表述
    m = re.search(
        r"(?:final answer is|the answer is|answer:)\s*[:=]?\s*\$?([^\n.$]+)\$?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def _strip_tex_wrapper(expr: str, macro: str) -> str:
    """去除如 \\text{...}、\\mathrm{...}、\\mathbf{...} 等单参数 LaTeX 壳。"""
    pattern = "\\" + macro + "{"
    out = expr
    while pattern in out:
        idx = out.find(pattern)
        start = idx + len(pattern)
        depth = 1
        pos = start
        while pos < len(out) and depth > 0:
            if out[pos] == "{":
                depth += 1
            elif out[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            inner = out[start : pos - 1]
            out = out[:idx] + inner + out[pos:]
        else:
            break
    return out


def _try_parse_numeric(expr: str) -> Fraction | float | None:
    """尝试将规范化后的纯数值/LaTeX 分数表达式解析为精确有理数或浮点数。"""
    s = expr.strip()
    if not s:
        return None
    # 处理如 \\frac{a}{b} 或 \\fracab（单字符简写）
    m_frac = re.fullmatch(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if m_frac:
        sign, num_s, den_s = m_frac.groups()
        num = _try_parse_numeric(num_s)
        den = _try_parse_numeric(den_s)
        if isinstance(num, Fraction) and isinstance(den, Fraction) and den != 0:
            val = num / den
            return -val if sign == "-" else val
        if num is not None and den is not None and float(den) != 0.0:
            val_f = float(num) / float(den)
            return -val_f if sign == "-" else val_f
        return None

    m_short_frac = re.fullmatch(r"([+-]?)\\frac(\d)(\d)", s)
    if m_short_frac:
        sign, n_ch, d_ch = m_short_frac.groups()
        if d_ch != "0":
            val = Fraction(int(n_ch), int(d_ch))
            return -val if sign == "-" else val

    # 处理带逗号数字（如 1,000）或普通整数/小数/a/b
    s_clean = s.replace(",", "")
    if re.fullmatch(r"[+-]?\d+", s_clean):
        return Fraction(int(s_clean), 1)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", s_clean):
        try:
            return Fraction(s_clean)
        except Exception:
            return float(s_clean)
    if re.fullmatch(r"[+-]?\d+/[1-9]\d*", s_clean):
        try:
            return Fraction(s_clean)
        except Exception:
            return None
    return None


def normalize_math_answer(ans: str) -> str:
    """将数学答案字符串规范化以便进行等价归并与判分。"""
    if not isinstance(ans, str):
        return ""
    s = ans.strip()
    if not s:
        return ""
    s = s.replace("$", "")
    # 先剥离如 864\mbox{ inches}^2、15\text{ cm} 等前置值后的纯字母单位后缀，再剥壳
    s = re.sub(r"(?<=\S)\s*\\(?:text|mbox|mathrm)\s*\{\s*[a-zA-Z]+\s*\}(?:\^\d+)?", "", s)
    for macro in ("text", "mathrm", "mathbf", "textbf", "mbox"):
        s = _strip_tex_wrapper(s, macro)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    s = s.replace(" ", "").replace("\n", "").replace("\t", "")
    # 将简写 \sqrt2 或 \frac12 规范化为 \sqrt{2} 与 \frac{1}{2}
    s = re.sub(r"\\sqrt([0-9a-zA-Z])", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    # 剥离单变量方程前缀（如 x=5 -> 5）
    m_eq = re.fullmatch(r"[a-zA-Z]=(.+)", s)
    if m_eq:
        s = m_eq.group(1)
    # 去除外层成对多余花括号
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        matched_outer = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    matched_outer = False
                    break
        if matched_outer and depth == 0:
            s = s[1:-1]
        else:
            break
    # 尝试规范化纯有理数表示（如 0.5 -> 1/2，\frac{14}{3} -> 14/3）
    num = _try_parse_numeric(s)
    if isinstance(num, Fraction):
        if num.denominator == 1:
            return str(num.numerator)
        return f"{num.numerator}/{num.denominator}"
    return s


def _latex_to_sympy_expr(s: str) -> Any:
    """将常见初等数学 LaTeX 字符串转为 SymPy 表达式（纯离线零外部解析器依赖）。"""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    if not s or len(s) > 120 or any(tok in s for tok in ("pmatrix", "matrix", "cup", "infty")):
        return None
    expr = s
    # 交替递归展开 \sqrt 与 \frac，支持 \frac{\sqrt{8}}{2} 与 \sqrt{\frac{1}{2}} 任意嵌套
    for _ in range(6):
        new_expr = re.sub(r"\\sqrt\[([^{}]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", expr)
        new_expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", new_expr)
        new_expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", new_expr)
        if new_expr == expr:
            break
        expr = new_expr
    expr = expr.replace("\\pi", "pi").replace("^", "**")
    if "\\" in expr:
        return None
    trans = standard_transformations + (implicit_multiplication_application,)
    local_dict = {"pi": sp.pi, "i": sp.I, "sqrt": sp.sqrt, "e": sp.E}
    return parse_expr(expr, local_dict=local_dict, transformations=trans)


def is_math_answer_correct(pred: str, gold: str) -> bool:
    """判断预测数学答案与标准答案是否等价（含分数与 SymPy 符号化简回退）。"""
    np_pred = normalize_math_answer(pred)
    np_gold = normalize_math_answer(gold)
    if not np_pred or not np_gold:
        return False
    if np_pred == np_gold:
        return True
    v1 = _try_parse_numeric(np_pred)
    v2 = _try_parse_numeric(np_gold)
    if v1 is not None and v2 is not None:
        if isinstance(v1, Fraction) and isinstance(v2, Fraction):
            return v1 == v2
        return abs(float(v1) - float(v2)) <= 1e-6 * max(1.0, abs(float(v2)))
    try:
        import sympy as sp

        e1 = _latex_to_sympy_expr(np_pred)
        e2 = _latex_to_sympy_expr(np_gold)
        if e1 is not None and e2 is not None:
            diff = sp.simplify(e1 - e2)
            if diff == 0:
                return True
    except Exception:
        pass
    return False


def _log_binom(n: np.ndarray | int, k: int) -> np.ndarray | float:
    """对数域安全计算 ln C(n, k)，当 n < k 时返回 -inf。"""
    n_arr = np.asarray(n, dtype=np.float64)
    out = np.full(n_arr.shape, -np.inf, dtype=np.float64)
    valid = n_arr >= float(k)
    if np.any(valid):
        nv = n_arr[valid]
        out[valid] = (
            special.gammaln(nv + 1.0)
            - special.gammaln(float(k) + 1.0)
            - special.gammaln(nv - float(k) + 1.0)
        )
    return float(out) if np.ndim(n) == 0 else out


def exact_pool_bon_accuracy(
    scores: np.ndarray,
    is_correct: np.ndarray,
    n_values: Sequence[int],
) -> np.ndarray:
    """精确计算大小为 L 的有限真实候选池在不放回 Best-of-N 下的零方差 Pass@1。

    按分数升序合并相同分数的平局组 H_1, ..., H_G，累计元素数分别为 C_1, ..., C_G
    （C_0 = 0, C_G = L），组内真实平均正确率为 q_bar_g。则不放回抽 N 个样本时，
    组 g 恰好成为最大组并被均匀随机选中的精确概率为：
        P(max_group = g) = (C(C_g, N) - C(C_{g-1}, N)) / C(L, N)
    全程在对数域通过 gammaln 计算二项式系数，严格避免大 L 组合数溢出。
    """
    s_arr = np.asarray(scores, dtype=np.float64).ravel()
    y_arr = np.asarray(is_correct, dtype=np.float64).ravel()
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    pool_size = int(s_arr.size)
    if pool_size == 0 or y_arr.shape != s_arr.shape:
        raise ValueError("scores 与 is_correct 必须非空且长度一致")
    if n_arr.ndim != 1 or n_arr.size == 0 or np.any(n_arr < 1) or np.any(n_arr > pool_size):
        raise ValueError(f"n_values 必须全部位于 [1, {pool_size}] 内")

    order = np.argsort(s_arr, kind="mergesort")
    s_sorted = s_arr[order]
    y_sorted = y_arr[order]

    # 识别同分平局组边界
    if pool_size == 1:
        starts = np.array([0], dtype=np.int64)
    else:
        is_new = np.empty(pool_size, dtype=bool)
        is_new[0] = True
        is_new[1:] = np.abs(s_sorted[1:] - s_sorted[:-1]) > 1e-12
        starts = np.flatnonzero(is_new)

    counts = np.diff(np.append(starts, pool_size))
    cum_hi = np.cumsum(counts)
    cum_lo = cum_hi - counts
    group_q = np.add.reduceat(y_sorted, starts) / counts.astype(np.float64)

    out = np.empty(n_arr.size, dtype=np.float64)
    for idx, n_val in enumerate(n_arr):
        nv = int(n_val)
        if nv == 1:
            out[idx] = float(np.mean(y_sorted))
            continue
        log_denom = float(_log_binom(pool_size, nv))
        log_hi = _log_binom(cum_hi, nv) - log_denom
        log_lo = _log_binom(cum_lo, nv) - log_denom

        # prob = exp(log_hi) * (-expm1(log_lo - log_hi))
        prob = np.zeros(len(counts), dtype=np.float64)
        valid_hi = np.isfinite(log_hi)
        both = valid_hi & np.isfinite(log_lo)
        only_hi = valid_hi & (~np.isfinite(log_lo))
        prob[only_hi] = np.exp(log_hi[only_hi])
        if np.any(both):
            diff = np.minimum(0.0, log_lo[both] - log_hi[both])
            prob[both] = -np.exp(log_hi[both]) * np.expm1(diff)
        out[idx] = float(np.clip(np.dot(prob, group_q), 0.0, 1.0))
    return out


def diagnose_real_verifier_distribution(
    scores_list: Sequence[np.ndarray],
    correct_list: Sequence[np.ndarray],
) -> dict[str, Any]:
    """诊断真实大模型候选池的边际分布统计量、异方差比 kappa 与右尾倒挂现象。"""
    if len(scores_list) == 0 or len(scores_list) != len(correct_list):
        raise ValueError("scores_list 与 correct_list 必须非空且长度相等")
    s_all = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in scores_list])
    y_all = np.concatenate([np.asarray(y, dtype=bool).ravel() for y in correct_list])

    pos_s = s_all[y_all]
    neg_s = s_all[~y_all]
    p_hat = float(np.mean(y_all))
    mu1 = float(np.mean(pos_s)) if pos_s.size > 0 else 0.0
    sigma1 = float(np.std(pos_s, ddof=1)) if pos_s.size > 1 else 1e-6
    mu0 = float(np.mean(neg_s)) if neg_s.size > 0 else 0.0
    sigma0 = float(np.std(neg_s, ddof=1)) if neg_s.size > 1 else 1e-6
    kurt1 = float(stats.kurtosis(pos_s, fisher=True)) if pos_s.size > 3 else 0.0
    kurt0 = float(stats.kurtosis(neg_s, fisher=True)) if neg_s.size > 3 else 0.0

    # 分位后验曲线检验尾部是否出现非单调倒挂
    q_edges_pct = [0.0, 20.0, 40.0, 60.0, 80.0, 90.0, 95.0, 99.0, 100.0]
    edges = np.percentile(s_all, q_edges_pct)
    bin_stats = []
    for i in range(len(q_edges_pct) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(q_edges_pct) - 2:
            mask = (s_all >= lo) & (s_all <= hi)
        else:
            mask = (s_all >= lo) & (s_all < hi)
        cnt = int(np.count_nonzero(mask))
        post = float(np.mean(y_all[mask])) if cnt > 0 else 0.0
        bin_stats.append({
            "quantile_range": f"{int(q_edges_pct[i])}%-{int(q_edges_pct[i+1])}%",
            "count": cnt,
            "posterior": post,
        })
    posteriors = [b["posterior"] for b in bin_stats]
    tail_turn_down = bool(posteriors[-1] < max(posteriors[:-1]) - 1e-4)

    return {
        "n_problems": len(scores_list),
        "n_total_candidates": int(s_all.size),
        "p_hat": p_hat,
        "mu1": mu1,
        "sigma1": sigma1,
        "mu0": mu0,
        "sigma0": sigma0,
        "delta_mu": mu1 - mu0,
        "kappa": float(sigma0 / max(1e-12, sigma1)),
        "excess_kurtosis_pos": kurt1,
        "excess_kurtosis_neg": kurt0,
        "quantile_posterior_profile": bin_stats,
        "tail_turn_down": tail_turn_down,
    }


def diagnose_multi_verifier_correlations(
    verifier_scores: dict[str, np.ndarray],
    is_correct: np.ndarray,
) -> dict[str, Any]:
    """计算多个真实 Verifier 在正确/错误集合上的相关性矩阵与定理 5A 相变诊断。"""
    names = list(verifier_scores.keys())
    if len(names) < 2:
        raise ValueError("至少需要 2 个 Verifier 才可计算成对相关性")
    y = np.asarray(is_correct, dtype=bool).ravel()
    mat = np.vstack([np.asarray(verifier_scores[n], dtype=np.float64).ravel() for n in names])
    if mat.shape[1] != y.size:
        raise ValueError("各 Verifier 分数向量长度必须与 is_correct 一致")

    mat_pos = mat[:, y]
    mat_neg = mat[:, ~y]
    corr_pos = np.corrcoef(mat_pos) if mat_pos.shape[1] > 2 else np.eye(len(names))
    corr_neg = np.corrcoef(mat_neg) if mat_neg.shape[1] > 2 else np.eye(len(names))

    k = len(names)
    off_diag = ~np.eye(k, dtype=bool)
    mean_rho1 = float(np.mean(corr_pos[off_diag]))
    mean_rho0 = float(np.mean(corr_neg[off_diag]))

    per_kappa: dict[str, float] = {}
    per_mu1: list[float] = []
    per_mu0: list[float] = []
    per_s1: list[float] = []
    per_s0: list[float] = []
    for idx, name in enumerate(names):
        s_pos = mat_pos[idx]
        s_neg = mat_neg[idx]
        s1 = float(np.std(s_pos, ddof=1))
        s0 = float(np.std(s_neg, ddof=1))
        per_kappa[name] = float(s0 / max(1e-12, s1))
        per_mu1.append(float(np.mean(s_pos)))
        per_mu0.append(float(np.mean(s_neg)))
        per_s1.append(s1)
        per_s0.append(s0)

    mean_kappa = float(np.mean(list(per_kappa.values())))
    mean_s1 = float(np.mean(per_s1))
    mean_s0 = float(np.mean(per_s0))
    # 映射到对称相关高斯模型检验定理 5A 判据
    rho1_clip = float(np.clip(mean_rho1, 0.0, 0.999))
    rho0_clip = float(np.clip(mean_rho0, 0.0, 0.999))
    p_hat = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    spec_proxy = MultiVerifierGaussianSpec(
        p=p_hat,
        mu1=float(np.mean(per_mu1)),
        sigma1=max(1e-6, mean_s1),
        mu0=float(np.mean(per_mu0)),
        sigma0=max(1e-6, mean_s0),
        num_verifiers=k,
        rho1=rho1_clip,
        rho0=rho0_clip,
    )
    crit = ensemble_phase_reversal_criterion(spec_proxy)

    return {
        "names": names,
        "corr_pos": corr_pos,
        "corr_neg": corr_neg,
        "mean_rho1": mean_rho1,
        "mean_rho0": mean_rho0,
        "per_verifier_kappa": per_kappa,
        "mean_kappa": mean_kappa,
        "theorem5a_diagnosis": crit,
    }


def _build_hypergeom_table(pool_size: int, n_values: Sequence[int]) -> np.ndarray:
    """预计算表 table[c, j] = C(c, N_j) / C(L, N_j)，形状 (L+1, len(N))。"""
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    c_arr = np.arange(pool_size + 1, dtype=np.float64)[:, None]
    table = np.zeros((pool_size + 1, n_arr.size), dtype=np.float64)
    for j, nv in enumerate(n_arr):
        nv_int = int(nv)
        if nv_int == 1:
            table[:, j] = c_arr[:, 0] / float(pool_size)
        else:
            log_den = float(_log_binom(pool_size, nv_int))
            log_num = _log_binom(c_arr[:, 0], nv_int)
            valid = np.isfinite(log_num)
            table[valid, j] = np.exp(log_num[valid] - log_den)
    return table


def _eval_mapped_pools_fast(
    test_groups_by_len: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    est: np.ndarray,
    n_values: Sequence[int],
    n_tests_total: int,
) -> np.ndarray:
    """给定桶估值向量 est (K,)，全向量化精确计算测试池平均 BoN 曲线（等价于逐题 exact_pool_bon_accuracy）。"""
    order = np.argsort(est, kind="mergesort")
    est_sorted = est[order]
    if est_sorted.size == 1:
        starts = np.array([0], dtype=np.int64)
    else:
        is_new = np.empty(est_sorted.size, dtype=bool)
        is_new[0] = True
        is_new[1:] = np.abs(est_sorted[1:] - est_sorted[:-1]) > 1e-12
        starts = np.flatnonzero(is_new)

    tot_curve = np.zeros(len(n_values), dtype=np.float64)
    for _, (cnt_mat, pos_mat, hg_table) in test_groups_by_len.items():
        # 按升序排列并合并同分平局组 -> 形状 (P_L, G)
        c_ord = cnt_mat[:, order]
        y_ord = pos_mat[:, order]
        if starts.size == est_sorted.size:
            g_cnt, g_pos = c_ord, y_ord
        else:
            g_cnt = np.add.reduceat(c_ord, starts, axis=1)
            g_pos = np.add.reduceat(y_ord, starts, axis=1)

        cum_hi = np.cumsum(g_cnt, axis=1)
        cum_lo = cum_hi - g_cnt
        # 查表直接计算每组选中概率: hg_table[cum_hi] - hg_table[cum_lo] -> (P_L, G, len(N))
        prob = hg_table[cum_hi] - hg_table[cum_lo]
        # 组平均正确率 q_bar = g_pos / g_cnt（空组贡献为 0）
        with np.errstate(divide="ignore", invalid="ignore"):
            q_bar = np.where(g_cnt > 0, g_pos / np.maximum(1, g_cnt), 0.0)
        tot_curve += np.einsum("pgn,pg->n", prob, q_bar)

    return np.clip(tot_curve / float(n_tests_total), 0.0, 1.0)


def evaluate_finite_calibration_on_real_pools(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    test_pools: Sequence[tuple[np.ndarray, np.ndarray]],
    m_cal: int,
    n_bins: int,
    n_values: Sequence[int],
    binning: str = "uniform",
    methods: Sequence[str] = (
        "mle",
        "laplace",
        "wilson_lcb",
        "empirical_bernstein_lcb",
    ),
    n_repeats: int = 200,
    delta: float = 0.05,
    prior_strength: float = 2.0,
    seed: int = 42,
    quantile_protocol: str = "semi_supervised",
) -> dict[str, Any]:
    """在真实大模型候选池上评估有限校准集 M 对线上 BoN 的二次坍塌与 LCB 效应。

    当 binning='quantile' 时，支持三种严格区分样本预算边界的分桶协议（对应命题 6.1）：
    - 'semi_supervised'（默认）：使用全校准池无标签分数估计分位数边界（U 条无标签 + M 条有标签）；
    - 'sample_split'（命题 6.1 严格总预算 M 算法）：每次仅抽取 M 条样本，前 M_1=floor(M/2) 条用顺序统计量
      r_k = floor(k*(M_1+1)/K) 构造分位边界，后 M_2 = M - M_1 条独立估计桶内正例率（零池外无标签信息）；
    - 'in_sample'：每次仅抽取 M 条样本，在同一批 M 条样本上同时估计分位数边界与桶内正例率。
    """
    c_s = np.asarray(cal_scores, dtype=np.float64).ravel()
    c_y = np.asarray(cal_correct, dtype=np.int64).ravel()
    if c_s.size == 0 or c_s.shape != c_y.shape:
        raise ValueError("cal_scores 与 cal_correct 必须非空且长度一致")
    if binning not in ("uniform", "quantile"):
        raise ValueError("binning 必须为 'uniform' 或 'quantile'")
    if quantile_protocol not in ("semi_supervised", "sample_split", "in_sample"):
        raise ValueError("quantile_protocol 必须为 'semi_supervised', 'sample_split' 或 'in_sample'")

    n_tests = len(test_pools)
    raw_curves = np.empty((n_tests, len(n_values)), dtype=np.float64)
    pools_by_len: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for idx_t, (ts, ty) in enumerate(test_pools):
        ts_arr = np.asarray(ts, dtype=np.float64).ravel()
        ty_arr = np.asarray(ty, dtype=np.int64).ravel()
        raw_curves[idx_t] = exact_pool_bon_accuracy(ts_arr, ty_arr, n_values)
        l_len = int(ts_arr.size)
        if l_len not in pools_by_len:
            pools_by_len[l_len] = ([], [])
        pools_by_len[l_len][0].append(ts_arr)
        pools_by_len[l_len][1].append(ty_arr)
    standard_bon_curve = np.mean(raw_curves, axis=0)

    rng = np.random.default_rng(seed)
    replace_draw = int(m_cal) > c_s.size
    acc_runs = {m_name: np.zeros((n_repeats, len(n_values)), dtype=np.float64) for m_name in methods}

    if binning == "uniform" or quantile_protocol == "semi_supervised":
        if binning == "quantile":
            edges = np.quantile(c_s, np.linspace(0.0, 1.0, int(n_bins) + 1))
            edges = np.unique(edges)
            inner_edges = edges[1:-1] if edges.size >= 3 else np.array([float(np.median(c_s))])
        else:
            lo_x, hi_x = np.quantile(c_s, [5e-4, 1.0 - 5e-4])
            inner_edges = (
                np.linspace(float(lo_x), float(hi_x), int(n_bins) - 1)
                if int(n_bins) > 2
                else np.array([0.5 * float(lo_x + hi_x)])
            )
        k_actual = int(inner_edges.size + 1)
        cal_bin_idx = np.searchsorted(inner_edges, c_s, side="right")

        test_groups_by_len: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for l_len, (s_list, y_list) in pools_by_len.items():
            s_mat = np.asarray(s_list, dtype=np.float64)
            y_mat = np.asarray(y_list, dtype=np.int64)
            p_l = s_mat.shape[0]
            b_mat = np.searchsorted(inner_edges, s_mat, side="right")
            flat_idx = (np.arange(p_l, dtype=np.int64)[:, None] * k_actual + b_mat).ravel()
            c_mat = np.bincount(flat_idx, minlength=p_l * k_actual).reshape(p_l, k_actual)
            pos_mat = np.bincount(
                flat_idx, weights=y_mat.ravel().astype(np.float64), minlength=p_l * k_actual
            ).astype(np.int64).reshape(p_l, k_actual)
            test_groups_by_len[l_len] = (c_mat, pos_mat, _build_hypergeom_table(l_len, n_values))

        prior_p = float(np.mean(c_y))
        for r_idx in range(n_repeats):
            pick = rng.choice(c_s.size, size=int(m_cal), replace=replace_draw)
            b_pick = cal_bin_idx[pick]
            y_pick = c_y[pick]
            counts = np.bincount(b_pick, minlength=k_actual).astype(np.float64)
            positives = np.bincount(b_pick, weights=y_pick, minlength=k_actual).astype(np.float64)
            for m_name in methods:
                est = compute_bin_calibrator_scores(
                    counts, positives, method=m_name, prior_p=prior_p, delta=delta, prior_strength=prior_strength
                )
                acc_runs[m_name][r_idx] = _eval_mapped_pools_fast(
                    test_groups_by_len, est, n_values, n_tests
                )
    else:
        # 'sample_split'（命题 6.1 严格样本分割）或 'in_sample'：绝不访问 M 之外任何无标签分数
        pre_tables = {
            l_len: (
                np.asarray(s_list, dtype=np.float64),
                np.asarray(y_list, dtype=np.int64),
                _build_hypergeom_table(l_len, n_values),
            )
            for l_len, (s_list, y_list) in pools_by_len.items()
        }
        k_req = max(2, int(n_bins))
        k_sum = 0
        for r_idx in range(n_repeats):
            pick = rng.choice(c_s.size, size=int(m_cal), replace=replace_draw)
            s_draw = c_s[pick]
            y_draw = c_y[pick]
            if quantile_protocol == "sample_split" and int(m_cal) >= 4:
                m_edge = int(m_cal) // 2
                s_edge = np.sort(s_draw[:m_edge])
                r_ord = np.floor(
                    np.arange(1, k_req, dtype=np.float64) * float(m_edge + 1) / float(k_req)
                ).astype(np.int64)
                r_ord = np.clip(r_ord - 1, 0, m_edge - 1)
                cand_edges = np.unique(s_edge[r_ord])
                inner_edges = cand_edges if cand_edges.size >= 1 else np.array([float(np.median(s_edge))])
                s_est = s_draw[m_edge:]
                y_est = y_draw[m_edge:]
            else:
                edges = np.unique(np.quantile(s_draw, np.linspace(0.0, 1.0, k_req + 1)))
                inner_edges = edges[1:-1] if edges.size >= 3 else np.array([float(np.median(s_draw))])
                s_est = s_draw
                y_est = y_draw

            k_cur = int(inner_edges.size + 1)
            k_sum += k_cur
            b_est = np.searchsorted(inner_edges, s_est, side="right")
            counts = np.bincount(b_est, minlength=k_cur).astype(np.float64)
            positives = np.bincount(b_est, weights=y_est.astype(np.float64), minlength=k_cur)
            prior_p = float(np.mean(y_est)) if y_est.size > 0 else 0.5

            dyn_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for l_len, (s_mat, y_mat, hg_tbl) in pre_tables.items():
                p_l = s_mat.shape[0]
                b_mat = np.searchsorted(inner_edges, s_mat, side="right")
                flat_idx = (np.arange(p_l, dtype=np.int64)[:, None] * k_cur + b_mat).ravel()
                c_mat = np.bincount(flat_idx, minlength=p_l * k_cur).reshape(p_l, k_cur)
                pos_mat = np.bincount(
                    flat_idx, weights=y_mat.ravel().astype(np.float64), minlength=p_l * k_cur
                ).astype(np.int64).reshape(p_l, k_cur)
                dyn_groups[l_len] = (c_mat, pos_mat, hg_tbl)

            for m_name in methods:
                est = compute_bin_calibrator_scores(
                    counts, positives, method=m_name, prior_p=prior_p, delta=delta, prior_strength=prior_strength
                )
                acc_runs[m_name][r_idx] = _eval_mapped_pools_fast(dyn_groups, est, n_values, n_tests)
        k_actual = int(round(k_sum / float(max(1, n_repeats))))

    method_out = {}
    for m_name in methods:
        mean_c = np.mean(acc_runs[m_name], axis=0)
        if n_repeats > 1:
            se_c = np.std(acc_runs[m_name], axis=0, ddof=1) / math.sqrt(n_repeats)
        else:
            se_c = np.zeros_like(mean_c)
        method_out[m_name] = {
            "accuracy": mean_c,
            "se": se_c,
            "ci_low": np.clip(mean_c - 1.96 * se_c, 0.0, 1.0),
            "ci_high": np.clip(mean_c + 1.96 * se_c, 0.0, 1.0),
        }

    return {
        "n_values": np.asarray(list(n_values), dtype=np.int64),
        "m_cal": int(m_cal),
        "n_bins_actual": k_actual,
        "binning": binning,
        "quantile_protocol": quantile_protocol if binning == "quantile" else "uniform_fixed_range",
        "standard_bon_accuracy": standard_bon_curve,
        "methods": method_out,
    }


def evaluate_discrete_algorithms_on_real_pools(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    test_pools_with_answers: Sequence[tuple[np.ndarray, np.ndarray, Sequence[str]]],
    n_values: Sequence[int],
    n_trials_per_problem: int = 50,
    n_bins: int = 25,
    lcb_delta: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """在留出真实测试题库（含真实文本提取答案）上横向评测 5 类推理选择算法。

    算法包括：
    - Standard-BoN
    - Majority-Voting
    - Calibrated-Weighted-BoN (quantile_lcb)
    - Two-Stage-Verified-Majority (quantile_lcb)
    - Calibrated-Weighted-BoN (quantile_hist)
    """
    c_s = np.asarray(cal_scores, dtype=np.float64).ravel()
    c_y = np.asarray(cal_correct, dtype=np.float64).ravel()
    cal_lcb = EmpiricalScoreCalibrator(
        n_bins=n_bins, method="quantile_lcb", lcb_delta=lcb_delta
    ).fit(c_s, c_y)
    cal_hist = EmpiricalScoreCalibrator(
        n_bins=n_bins, prior_strength=2.0, method="quantile_hist"
    ).fit(c_s, c_y)

    selectors = {
        "Standard-BoN": StandardBoN(),
        "Majority-Voting": MajorityVoting(),
        "Calibrated-Weighted-BoN (Wilson-LCB)": CalibratedWeightedBoN(calibrator=cal_lcb),
        "Two-Stage-Verified-Majority (Wilson-LCB)": TwoStageVerifiedMajority(calibrator=cal_lcb),
        "Calibrated-Weighted-BoN (Hist)": CalibratedWeightedBoN(calibrator=cal_hist),
    }

    # 预先将每道题的答案字符串编码为局部整数 ID，并预计算每个答案 ID 是否正确
    encoded_problems = []
    for scores, is_corr, ans_strs in test_pools_with_answers:
        s_arr = np.asarray(scores, dtype=np.float64)
        y_arr = np.asarray(is_corr, dtype=bool)
        if len(ans_strs) != s_arr.size:
            raise ValueError("ans_strs 长度必须与 scores 一致")
        str_to_id: dict[str, int] = {}
        ans_ids = np.empty(s_arr.size, dtype=np.int64)
        id_correct_map: dict[int, bool] = {}
        for idx, a_raw in enumerate(ans_strs):
            # 空提取答案赋予唯一负 ID 避免空串之间相互投票聚簇
            norm = normalize_math_answer(a_raw)
            key = norm if norm else f"__EMPTY_{idx}__"
            if key not in str_to_id:
                aid = len(str_to_id)
                str_to_id[key] = aid
                id_correct_map[aid] = bool(y_arr[idx])
            else:
                aid = str_to_id[key]
                # 若同一规范答案有多条记录，以真实正确性标签多数/或逻辑为准
                if y_arr[idx]:
                    id_correct_map[aid] = True
            ans_ids[idx] = aid
        lookup = np.array([id_correct_map[i] for i in range(len(str_to_id))], dtype=bool)
        encoded_problems.append((s_arr, ans_ids, lookup))

    n_arr = np.asarray(list(n_values), dtype=np.int64)
    n_probs = len(encoded_problems)
    rng = np.random.default_rng(seed)
    algo_names = list(selectors.keys())
    prompt_acc = {name: np.zeros((n_probs, len(n_arr)), dtype=np.float64) for name in algo_names}

    for p_idx, (s_arr, ans_ids, lookup) in enumerate(encoded_problems):
        pool_len = s_arr.size
        max_n = int(np.max(n_arr))
        if max_n > pool_len:
            raise ValueError(f"采样数 N={max_n} 超过候选池大小 {pool_len}")

        # 构造配对无放回随机排列矩阵 (n_trials_per_problem, max_n)
        perms = np.array(
            [rng.permutation(pool_len)[:max_n] for _ in range(n_trials_per_problem)],
            dtype=np.int64,
        )
        s_mat = s_arr[perms]
        a_mat = ans_ids[perms]

        for k_idx, nv in enumerate(n_arr):
            nv_int = int(nv)
            s_sub = s_mat[:, :nv_int]
            a_sub = a_mat[:, :nv_int]
            for name, sel in selectors.items():
                chosen_aids = sel.select_batch(s_sub, a_sub, rng=rng)
                prompt_acc[name][p_idx, k_idx] = float(np.count_nonzero(lookup[chosen_aids])) / float(
                    n_trials_per_problem
                )

    results = {}
    for name in algo_names:
        mat = prompt_acc[name]
        acc = np.mean(mat, axis=0)
        if n_probs > 1:
            se = np.std(mat, axis=0, ddof=1) / math.sqrt(float(n_probs))
        else:
            se = np.zeros_like(acc)
        results[name] = {
            "accuracy": acc,
            "se": se,
            "ci_low": np.clip(acc - 1.96 * se, 0.0, 1.0),
            "ci_high": np.clip(acc + 1.96 * se, 0.0, 1.0),
        }

    return {
        "n_values": n_arr,
        "n_problems": n_probs,
        "n_trials_per_problem": int(n_trials_per_problem),
        "se_method": "prompt_level_cluster_se",
        "algorithms": results,
    }


def is_math_answer_correct_any(pred: str, gold_list: Sequence[str] | str) -> bool:
    """判断预测数学答案是否与标准答案列表中的任意一个等价（适配 OlympiadBench 等多解字段）。"""
    if isinstance(gold_list, str):
        return is_math_answer_correct(pred, gold_list)
    for gold in gold_list:
        if is_math_answer_correct(pred, str(gold)):
            return True
    return False


_SOTA_BIN_METHOD_MAP = {
    "histogram_mle": "mle",
    "laplace_smoothed": "laplace",
    "empirical_bernstein_lcb": "empirical_bernstein_lcb",
    "wilson_lcb": "wilson_lcb",
    "kl_lcb": "kl_lcb",
}

_SUPPORTED_SOTA_CALIBRATORS = (
    "standard_bon",
    "platt_scaling",
    "isotonic_pava",
    "histogram_mle",
    "laplace_smoothed",
    "empirical_bernstein_lcb",
    "wilson_lcb",
    "kl_lcb",
)


def _fit_logistic_1d(x: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4) -> tuple[float, float, float, float]:
    """纯 NumPy/SciPy 拟合一维带微小 L2 正则的 Platt Logistic 回归 y ~ sigma(a*z + b)。"""
    mu = float(np.mean(x))
    sd = max(1e-8, float(np.std(x)))
    z = (x - mu) / sd

    def _loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = float(params[0]), float(params[1])
        logits = a * z + b
        # log(1 + exp(logits)) - y * logits 的数值稳定形式
        loss_vec = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.mean(loss_vec) + 0.5 * l2_reg * (a * a + b * b))
        p_hat = special.expit(logits)
        diff = (p_hat - y) / float(z.size)
        ga = float(np.dot(diff, z) + l2_reg * a)
        gb = float(np.sum(diff) + l2_reg * b)
        return loss, np.array([ga, gb], dtype=np.float64)

    res = optimize.minimize(_loss_and_grad, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1]), mu, sd


def _fit_logistic_multivariate(z_mat: np.ndarray, y: np.ndarray, l2_reg: float = 1e-3) -> np.ndarray:
    """拟合多元 L2 正则化 Logistic 堆叠权重 w（不含截距仅用于排序）。"""
    n_samples, k_dim = z_mat.shape

    def _loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:-1]
        b = float(params[-1])
        logits = z_mat @ w + b
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * l2_reg * float(np.dot(w, w)))
        diff = (special.expit(logits) - y) / float(n_samples)
        gw = z_mat.T @ diff + l2_reg * w
        gb = float(np.sum(diff))
        return loss, np.concatenate([gw, [gb]])

    init = np.zeros(k_dim + 1, dtype=np.float64)
    init[:-1] = 1.0 / float(k_dim)
    res = optimize.minimize(_loss_and_grad, init, jac=True, method="L-BFGS-B")
    w_out = np.asarray(res.x[:-1], dtype=np.float64)
    norm = float(np.linalg.norm(w_out))
    return w_out / max(1e-12, norm)


def _build_empirical_edges(scores: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """从校准分数构造严格递增的内部桶边界 inner_edges（长度 K_actual - 1）。"""
    k_int = max(2, int(n_bins))
    if binning == "quantile":
        edges = np.quantile(scores, np.linspace(0.0, 1.0, k_int + 1))
        edges = np.unique(edges)
        if edges.size >= 3:
            return edges[1:-1]
        return np.array([float(np.median(scores))], dtype=np.float64)
    if binning == "uniform":
        lo_x, hi_x = np.quantile(scores, [5e-4, 1.0 - 5e-4])
        if hi_x <= lo_x + 1e-12:
            return np.array([float(lo_x)], dtype=np.float64)
        if k_int == 2:
            return np.array([0.5 * float(lo_x + hi_x)], dtype=np.float64)
        return np.linspace(float(lo_x), float(hi_x), k_int - 1)
    raise ValueError(f"binning 必须为 'quantile' 或 'uniform'，当前收到: {binning!r}")


def _fit_sota_calibrator_predictor(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    method: str = "kl_lcb",
    n_bins: int | None = None,
    n_budget: int = 64,
    delta: float = 0.05,
    prior_strength: float = 2.0,
    binning: str = "quantile",
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """拟合 8 类校准/选择映射器，返回向量化预测函数及元信息。"""
    if method not in _SUPPORTED_SOTA_CALIBRATORS:
        raise ValueError(f"不支持的校准方法 {method!r}，可选: {_SUPPORTED_SOTA_CALIBRATORS}")
    c_s = np.asarray(cal_scores, dtype=np.float64).ravel()
    c_y = np.asarray(cal_correct, dtype=np.float64).ravel()
    if c_s.size == 0 or c_s.shape != c_y.shape:
        raise ValueError("cal_scores 与 cal_correct 必须非空且长度一致")

    if method == "standard_bon":
        return (lambda x: np.asarray(x, dtype=np.float64)), {"method": method, "n_bins_actual": 0}

    if method == "platt_scaling":
        a, b, mu, sd = _fit_logistic_1d(c_s, c_y, l2_reg=1e-4)
        return (
            lambda x, _a=a, _b=b, _mu=mu, _sd=sd: special.expit(
                _a * ((np.asarray(x, dtype=np.float64) - _mu) / _sd) + _b
            )
        ), {"method": method, "a": a, "b": b, "n_bins_actual": 0}

    if method == "isotonic_pava":
        iso = EmpiricalScoreCalibrator(method="isotonic").fit(c_s, c_y)
        return (lambda x, _iso=iso: np.asarray(_iso.predict_proba(x), dtype=np.float64)), {
            "method": method,
            "n_knots": int(iso.knots_x_.size) if iso.knots_x_ is not None else 0,
        }

    # 直方图分桶系列（MLE / Laplace / Empirical-Bernstein / Wilson-LCB / KL-LCB）
    if n_bins is None:
        m_eff = max(10, int(c_s.size))
        k_target = int(
            joint_calibration_compute_scaling(
                m_total=m_eff, n_budget=max(1, int(n_budget)), delta=delta
            )["k_joint_star"]
        )
        k_target = max(2, min(k_target, max(2, c_s.size // 4)))
    else:
        k_target = max(2, int(n_bins))

    inner_edges = _build_empirical_edges(c_s, k_target, binning=binning)
    k_actual = int(inner_edges.size + 1)
    b_idx = np.searchsorted(inner_edges, c_s, side="right")
    counts = np.bincount(b_idx, minlength=k_actual).astype(np.float64)
    positives = np.bincount(b_idx, weights=c_y, minlength=k_actual).astype(np.float64)
    prior_p = float(np.clip(np.mean(c_y), 1e-6, 1.0 - 1e-6))

    est = compute_bin_calibrator_scores(
        counts,
        positives,
        method=_SOTA_BIN_METHOD_MAP[method],
        prior_p=prior_p,
        delta=delta,
        prior_strength=prior_strength,
    )
    return (
        lambda x, _edges=inner_edges, _est=est: _est[
            np.searchsorted(_edges, np.asarray(x, dtype=np.float64), side="right")
        ]
    ), {"method": method, "n_bins_actual": k_actual, "inner_edges": inner_edges, "bin_scores": est}


def fit_and_predict_sota_calibrator(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    test_scores: np.ndarray,
    method: str = "kl_lcb",
    n_bins: int | None = None,
    n_budget: int = 64,
    delta: float = 0.05,
    prior_strength: float = 2.0,
    binning: str = "quantile",
) -> np.ndarray:
    """在离线校准集上拟合 8 类 SOTA 校准器之一，并预测测试分数上的校准得分。"""
    pred_fn, _ = _fit_sota_calibrator_predictor(
        cal_scores,
        cal_correct,
        method=method,
        n_bins=n_bins,
        n_budget=n_budget,
        delta=delta,
        prior_strength=prior_strength,
        binning=binning,
    )
    return pred_fn(np.asarray(test_scores, dtype=np.float64))


def _ensure_spd(mat: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """对样本协方差或相关矩阵做对称化与最小特征值保护，确保 Cholesky/QP 数值正定。"""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if float(np.min(eigvals)) >= min_eig:
        return sym
    clipped = np.maximum(eigvals, min_eig)
    fixed = (eigvecs * clipped[None, :]) @ eigvecs.T
    return 0.5 * (fixed + fixed.T)


def _eval_bon_variable_pools(
    score_list: Sequence[np.ndarray],
    correct_list: Sequence[np.ndarray],
    n_values: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """在允许变长候选池（如 RB2 Ties 11-38 vs 普通 4 候选）上精确计算均值 BoN 曲线。"""
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    sums = np.zeros(n_arr.size, dtype=np.float64)
    counts = np.zeros(n_arr.size, dtype=np.int64)
    for sc, yc in zip(score_list, correct_list):
        pool_len = int(sc.size)
        valid_mask = n_arr <= pool_len
        if not np.any(valid_mask):
            continue
        sub_n = n_arr[valid_mask]
        sums[valid_mask] += exact_pool_bon_accuracy(sc, yc, sub_n)
        counts[valid_mask] += 1
    if np.any(counts == 0):
        raise ValueError("存在没有任何题目满足 pool_size >= n 的预算值")
    return sums / counts.astype(np.float64), counts


def evaluate_heterogeneous_multi_rm_on_real_pools(
    verifier_names: Sequence[str],
    score_matrices: Sequence[np.ndarray],
    correct_list: Sequence[np.ndarray],
    n_values: Sequence[int],
    n_folds: int = 5,
    n_bins_cal: int = 20,
    delta: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """在真实多裁判评分矩阵上执行定理 5 诊断与严格 5-Fold 无泄露对比评估。

    Args:
        verifier_names: K 个 Verifier 名称列表。
        score_matrices: P 道题的评分矩阵列表，每题形状为 (L_i, K)。
        correct_list: P 道题的布尔正确性数组列表，每题形状为 (L_i,)。
        n_values: 待评估的 Best-of-N 预算序列。
    """
    names = list(verifier_names)
    k_ver = len(names)
    n_probs = len(score_matrices)
    if k_ver < 2 or n_probs == 0 or len(correct_list) != n_probs:
        raise ValueError("需要至少 2 个 Verifier 且 score_matrices 与 correct_list 非空等长")

    mats = [np.asarray(m, dtype=np.float64) for m in score_matrices]
    ys = [np.asarray(y, dtype=bool).ravel() for y in correct_list]
    for idx, (m, y) in enumerate(zip(mats, ys)):
        if m.ndim != 2 or m.shape[1] != k_ver or m.shape[0] != y.size or m.shape[0] == 0:
            raise ValueError(f"第 {idx} 题矩阵形状 {m.shape} 与标签长度 {y.size} 或裁判数 {k_ver} 不匹配")

    # 1. 全量条件统计量与定理 5A/5B 理论判据诊断
    all_mat = np.vstack(mats)
    all_y = np.concatenate(ys)
    z_all = (all_mat - np.mean(all_mat, axis=0, keepdims=True)) / np.maximum(
        1e-8, np.std(all_mat, axis=0, keepdims=True)
    )
    pos_z = z_all[all_y]
    neg_z = z_all[~all_y]

    p_hat = float(np.clip(np.mean(all_y), 1e-4, 1.0 - 1e-4))
    mu1_z = np.mean(pos_z, axis=0)
    mu0_z = np.mean(neg_z, axis=0)
    cov1_z = _ensure_spd(np.cov(pos_z, rowvar=False))
    cov0_z = _ensure_spd(np.cov(neg_z, rowvar=False))

    corr_pos = np.corrcoef(pos_z, rowvar=False)
    corr_neg = np.corrcoef(neg_z, rowvar=False)
    corr_pos_spd = _ensure_spd(corr_pos)
    corr_neg_spd = _ensure_spd(corr_neg)
    np.fill_diagonal(corr_pos_spd, 1.0)
    np.fill_diagonal(corr_neg_spd, 1.0)

    gen_spec = GeneralGaussianMultiVerifierSpec(p=p_hat, mu1=mu1_z, cov1=cov1_z, mu0=mu0_z, cov0=cov0_z)
    opt_lin_diag = optimal_linear_ensemble_criterion(gen_spec)
    min_qp_diag = general_min_ensemble_qp_criterion(gen_spec)
    qp_r1 = solve_positive_quadrant_qp(corr_pos_spd)
    qp_r0 = solve_positive_quadrant_qp(corr_neg_spd)
    eta1 = 1.0 / float(qp_r1["gamma"])
    eta0 = 1.0 / float(qp_r0["gamma"])

    sym_diag = diagnose_multi_verifier_correlations(
        {n: z_all[:, i] for i, n in enumerate(names)},
        all_y,
    )
    per_diag = {
        n: diagnose_real_verifier_distribution([m[:, i] for m in mats], ys)
        for i, n in enumerate(names)
    }

    # 2. 严格按题 K-Fold 交叉验证（所有均值/方差/协方差/权重/校准表仅从训练折估计）
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    eff_folds = min(int(n_folds), n_probs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_probs)
    folds = np.array_split(perm, eff_folds)

    ens_names = [
        "Mean-Ensemble",
        "Min-Ensemble",
        "Nonneg-OptLinear",
        "GenEig-OptLinear",
        "Logit-Stack",
        "Calibrated-Min (KL-LCB)",
    ]
    all_method_names = names + ens_names
    per_problem_curves: dict[str, list[tuple[int, np.ndarray]]] = {m: [] for m in all_method_names}

    for f_idx in range(eff_folds):
        test_idx = folds[f_idx]
        train_idx = np.concatenate([folds[j] for j in range(eff_folds) if j != f_idx])

        tr_mat = np.vstack([mats[i] for i in train_idx])
        tr_y = np.concatenate([ys[i] for i in train_idx])
        tr_mu = np.mean(tr_mat, axis=0, keepdims=True)
        tr_sd = np.maximum(1e-8, np.std(tr_mat, axis=0, keepdims=True))
        tr_z = (tr_mat - tr_mu) / tr_sd

        tr_pos = tr_z[tr_y]
        tr_neg = tr_z[~tr_y]
        tr_p = float(np.clip(np.mean(tr_y), 1e-4, 1.0 - 1e-4))
        tr_spec = GeneralGaussianMultiVerifierSpec(
            p=tr_p,
            mu1=np.mean(tr_pos, axis=0),
            cov1=_ensure_spd(np.cov(tr_pos, rowvar=False)),
            mu0=np.mean(tr_neg, axis=0),
            cov0=_ensure_spd(np.cov(tr_neg, rowvar=False)),
        )
        tr_opt = optimal_linear_ensemble_criterion(tr_spec)
        w_star = np.asarray(tr_opt["w_star"], dtype=np.float64)
        w_nonneg = np.asarray(tr_opt["w_nonneg_star"], dtype=np.float64)
        w_logit = _fit_logistic_multivariate(tr_z, tr_y.astype(np.float64), l2_reg=1e-3)

        # 为每个 Verifier 拟合单维 KL-LCB 映射（用于 Calibrated-Min）
        k_cal = min(int(n_bins_cal), max(4, tr_z.shape[0] // 25))
        cal_predictors = [
            _fit_sota_calibrator_predictor(
                tr_mat[:, col], tr_y.astype(np.float64), method="kl_lcb", n_bins=k_cal, delta=delta
            )[0]
            for col in range(k_ver)
        ]

        for p_idx in test_idx:
            raw_p = mats[p_idx]
            y_p = ys[p_idx]
            pool_len = int(y_p.size)
            valid_mask = n_arr <= pool_len
            if not np.any(valid_mask):
                continue
            sub_n = n_arr[valid_mask]
            z_p = (raw_p - tr_mu) / tr_sd

            score_map: dict[str, np.ndarray] = {n: raw_p[:, c] for c, n in enumerate(names)}
            score_map["Mean-Ensemble"] = np.mean(z_p, axis=1)
            score_map["Min-Ensemble"] = np.min(z_p, axis=1)
            score_map["Nonneg-OptLinear"] = z_p @ w_nonneg
            score_map["GenEig-OptLinear"] = z_p @ w_star
            score_map["Logit-Stack"] = z_p @ w_logit
            cal_stack = np.vstack([cal_predictors[c](raw_p[:, c]) for c in range(k_ver)])
            score_map["Calibrated-Min (KL-LCB)"] = np.min(cal_stack, axis=0)

            for m_name, sc_vec in score_map.items():
                curve_sub = exact_pool_bon_accuracy(sc_vec, y_p, sub_n)
                full_row = np.full(n_arr.size, np.nan, dtype=np.float64)
                full_row[valid_mask] = curve_sub
                per_problem_curves[m_name].append((int(p_idx), full_row))

    # 汇总全题平均准确率与标准误
    cv_results: dict[str, dict[str, np.ndarray]] = {}
    counts_by_n = np.zeros(n_arr.size, dtype=np.int64)
    for m_name in all_method_names:
        mat_curves = np.vstack([row for _, row in per_problem_curves[m_name]])
        valid_counts = np.sum(~np.isnan(mat_curves), axis=0)
        if m_name == all_method_names[0]:
            counts_by_n = valid_counts
        mean_c = np.nanmean(mat_curves, axis=0)
        std_c = np.nanstd(mat_curves, axis=0, ddof=1)
        se_c = np.where(valid_counts > 1, std_c / np.sqrt(np.maximum(1, valid_counts)), 0.0)
        cv_results[m_name] = {
            "accuracy": mean_c,
            "se": se_c,
            "ci_low": np.clip(mean_c - 1.96 * se_c, 0.0, 1.0),
            "ci_high": np.clip(mean_c + 1.96 * se_c, 0.0, 1.0),
        }

    return {
        "verifier_names": names,
        "n_problems": n_probs,
        "n_total_candidates": int(all_y.size),
        "p_hat": p_hat,
        "n_values": n_arr,
        "problem_counts_by_n": counts_by_n,
        "per_verifier_diag": per_diag,
        "symmetric_diag": sym_diag,
        "general_linear_diag": opt_lin_diag,
        "general_min_qp_diag": min_qp_diag,
        "copula_eta1": float(eta1),
        "copula_eta0": float(eta0),
        "copula_active_set0": qp_r0["active_set"],
        "is_nonneg_corr0": bool(np.all(corr_neg_spd >= -1e-12)),
        "cv_results": cv_results,
    }


def evaluate_joint_m_n_calibration_on_real_pools(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    test_pools: Sequence[tuple[np.ndarray, np.ndarray]],
    m_grid: Sequence[int] = (100, 200, 400, 800, 2000, 5000, 15000, 40000),
    n_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    sota_m_cal: int = 400,
    sota_n_bins: int = 20,
    sota_repeats: int = 60,
    grid_repeats: int = 80,
    delta: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """在真实候选池上对比 8 类 SOTA 校准器及定理 6C 联合 (M, N) 2D 网格与 1/5 vs 2/5 标度。"""
    c_s = np.asarray(cal_scores, dtype=np.float64).ravel()
    c_y = np.asarray(cal_correct, dtype=np.int64).ravel()
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    m_arr = np.asarray([int(m) for m in m_grid if int(m) <= c_s.size], dtype=np.int64)
    if m_arr.size == 0:
        m_arr = np.array([min(200, int(c_s.size))], dtype=np.int64)

    # 1. SOTA 8 方法同场对比（固定小样本 M=sota_m_cal）
    rng = np.random.default_rng(seed)
    m_sota = min(int(sota_m_cal), int(c_s.size))
    sota_runs: dict[str, np.ndarray] = {
        m_name: np.zeros((int(sota_repeats), n_arr.size), dtype=np.float64)
        for m_name in _SUPPORTED_SOTA_CALIBRATORS
    }

    # 先对测试池按长度分组加速所有方法
    parsed_pools = [
        (np.asarray(ts, dtype=np.float64).ravel(), np.asarray(ty, dtype=np.int64).ravel())
        for ts, ty in test_pools
    ]
    raw_curves = np.array([exact_pool_bon_accuracy(ts, ty, n_arr) for ts, ty in parsed_pools])
    standard_bon_curve = np.mean(raw_curves, axis=0)

    # 分桶方法加速查表；连续方法（Platt / Isotonic）直接逐题精确池评估
    bucket_methods = tuple(m for m in _SUPPORTED_SOTA_CALIBRATORS if m in _SOTA_BIN_METHOD_MAP)
    for r_idx in range(int(sota_repeats)):
        pick = rng.choice(c_s.size, size=m_sota, replace=False)
        s_sub = c_s[pick]
        y_sub = c_y[pick].astype(np.float64)

        sota_runs["standard_bon"][r_idx] = standard_bon_curve

        # Platt & Isotonic 逐题精确评估
        for cont_m in ("platt_scaling", "isotonic_pava"):
            pred_fn, _ = _fit_sota_calibrator_predictor(s_sub, y_sub, method=cont_m)
            # 对 Platt：若斜率 a > 0 且无异常浮点塌陷，等价于 Standard-BoN；仍精确调用确保普适性
            curves = [exact_pool_bon_accuracy(pred_fn(ts), ty, n_arr) for ts, ty in parsed_pools]
            sota_runs[cont_m][r_idx] = np.mean(curves, axis=0)

        # 分桶 5 法共用同一个统一参考分位边界，保证公平对比
        edges_ref = _build_empirical_edges(c_s, int(sota_n_bins), binning="quantile")
        k_ref = int(edges_ref.size + 1)
        b_pick = np.searchsorted(edges_ref, s_sub, side="right")
        counts = np.bincount(b_pick, minlength=k_ref).astype(np.float64)
        pos = np.bincount(b_pick, weights=y_sub, minlength=k_ref).astype(np.float64)
        prior_p = float(np.mean(y_sub))

        if r_idx == 0:
            # 惰性构造测试池超几何查表结构
            buckets_by_len: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
            for ts, ty in parsed_pools:
                b_t = np.searchsorted(edges_ref, ts, side="right")
                c_row = np.bincount(b_t, minlength=k_ref)
                y_row = np.bincount(b_t, weights=ty, minlength=k_ref).astype(np.int64)
                l_len = int(ts.size)
                buckets_by_len.setdefault(l_len, ([], []))
                buckets_by_len[l_len][0].append(c_row)
                buckets_by_len[l_len][1].append(y_row)
            sota_groups = {
                l_len: (
                    np.asarray(cl, dtype=np.int64),
                    np.asarray(yl, dtype=np.int64),
                    _build_hypergeom_table(l_len, n_arr),
                )
                for l_len, (cl, yl) in buckets_by_len.items()
            }

        for bm in bucket_methods:
            est = compute_bin_calibrator_scores(
                counts,
                pos,
                method=_SOTA_BIN_METHOD_MAP[bm],
                prior_p=prior_p,
                delta=delta,
                prior_strength=2.0,
            )
            sota_runs[bm][r_idx] = _eval_mapped_pools_fast(sota_groups, est, n_arr, len(parsed_pools))

    sota_summary = {}
    for m_name, mat in sota_runs.items():
        mean_c = np.mean(mat, axis=0)
        se_c = np.std(mat, axis=0, ddof=1) / math.sqrt(max(1, int(sota_repeats))) if sota_repeats > 1 else np.zeros_like(mean_c)
        sota_summary[m_name] = {
            "accuracy": mean_c,
            "se": se_c,
            "ci_low": np.clip(mean_c - 1.96 * se_c, 0.0, 1.0),
            "ci_high": np.clip(mean_c + 1.96 * se_c, 0.0, 1.0),
        }

    # 2. 2D (M, N) 网格评估：1/5 最优标度 vs 联合 K*(M,N) vs 错标 2/5 vs 固定过细 K=60+MLE
    grid_shape = (m_arr.size, n_arr.size)
    mat_opt_1_5_kl = np.zeros(grid_shape, dtype=np.float64)
    mat_opt_1_5_se = np.zeros(grid_shape, dtype=np.float64)
    mat_joint_kl = np.zeros(grid_shape, dtype=np.float64)
    mat_legacy_2_5_kl = np.zeros(grid_shape, dtype=np.float64)
    mat_fixed60_mle = np.zeros(grid_shape, dtype=np.float64)

    k_opt_1_5_list: list[int] = []
    k_legacy_2_5_list: list[int] = []
    n_match_list: list[int] = []

    for i_m, m_val in enumerate(m_arr):
        mv = int(m_val)
        cal_info = calibration_minimax_scaling(mv, delta=delta, beta=2.0)
        k_1_5 = max(3, int(cal_info["k_opt_theory"]))
        k_2_5 = max(5, min(mv // 3, int(round(2.0 * (mv / max(1.0, math.log(mv))) ** 0.4))))
        k_opt_1_5_list.append(k_1_5)
        k_legacy_2_5_list.append(k_2_5)

        joint_ref = joint_calibration_compute_scaling(mv, int(n_arr[-1]), beta=2.0, delta=delta)
        n_match_list.append(int(joint_ref["n_match"]))

        r_opt = evaluate_finite_calibration_on_real_pools(
            c_s, c_y, parsed_pools, m_cal=mv, n_bins=k_1_5, n_values=n_arr,
            binning="quantile", methods=("kl_lcb",), n_repeats=grid_repeats, delta=delta, seed=seed + 100 + i_m,
        )
        mat_opt_1_5_kl[i_m] = r_opt["methods"]["kl_lcb"]["accuracy"]
        mat_opt_1_5_se[i_m] = r_opt["methods"]["kl_lcb"]["se"]

        r_leg = evaluate_finite_calibration_on_real_pools(
            c_s, c_y, parsed_pools, m_cal=mv, n_bins=k_2_5, n_values=n_arr,
            binning="quantile", methods=("kl_lcb",), n_repeats=grid_repeats, delta=delta, seed=seed + 200 + i_m,
        )
        mat_legacy_2_5_kl[i_m] = r_leg["methods"]["kl_lcb"]["accuracy"]

        r_fix = evaluate_finite_calibration_on_real_pools(
            c_s, c_y, parsed_pools, m_cal=mv, n_bins=60, n_values=n_arr,
            binning="quantile", methods=("mle",), n_repeats=grid_repeats, delta=delta, seed=seed + 300 + i_m,
        )
        mat_fixed60_mle[i_m] = r_fix["methods"]["mle"]["accuracy"]

        # Joint K*(M, N) 可能随 N 变化：按不同 k_joint 去重批量评估
        k_joint_by_n = [
            max(3, int(joint_calibration_compute_scaling(mv, int(nv), beta=2.0, delta=delta)["k_joint_star"]))
            for nv in n_arr
        ]
        if len(set(k_joint_by_n)) == 1 and k_joint_by_n[0] == k_1_5:
            mat_joint_kl[i_m] = mat_opt_1_5_kl[i_m]
        else:
            for k_u in sorted(set(k_joint_by_n)):
                n_sub_idx = [j for j, kv in enumerate(k_joint_by_n) if kv == k_u]
                if k_u == k_1_5:
                    mat_joint_kl[i_m, n_sub_idx] = mat_opt_1_5_kl[i_m, n_sub_idx]
                else:
                    r_j = evaluate_finite_calibration_on_real_pools(
                        c_s, c_y, parsed_pools, m_cal=mv, n_bins=k_u, n_values=n_arr[n_sub_idx],
                        binning="quantile", methods=("kl_lcb",), n_repeats=grid_repeats, delta=delta, seed=seed + 400 + i_m,
                    )
                    mat_joint_kl[i_m, n_sub_idx] = r_j["methods"]["kl_lcb"]["accuracy"]

    return {
        "n_values": n_arr,
        "m_grid": m_arr,
        "sota_m_cal": m_sota,
        "sota_n_bins": int(sota_n_bins),
        "sota_comparison": sota_summary,
        "k_opt_1_5_by_m": k_opt_1_5_list,
        "k_legacy_2_5_by_m": k_legacy_2_5_list,
        "n_match_theory_by_m": n_match_list,
        "standard_bon_curve": standard_bon_curve,
        "grid_opt_1_5_kl": mat_opt_1_5_kl,
        "grid_opt_1_5_se": mat_opt_1_5_se,
        "grid_joint_kl": mat_joint_kl,
        "grid_legacy_2_5_kl": mat_legacy_2_5_kl,
        "grid_fixed60_mle": mat_fixed60_mle,
    }


def _within_prompt_zscore_2d(scores: np.ndarray) -> np.ndarray:
    """对每行（单次试验抽出的 N 个候选分数）严格在线计算 Z-Score，零池外泄露。"""
    if scores.shape[1] <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    mean = np.mean(scores, axis=1, keepdims=True)
    std = np.std(scores, axis=1, ddof=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-6)


def evaluate_real_prompt_heterogeneity(
    pools_with_strata: Sequence[tuple[str, np.ndarray, np.ndarray]],
    n_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    n_folds: int = 5,
    n_trials_per_problem: int = 25,
    n_bins: int = 15,
    n_simpson_bins: int = 10,
    delta: float = 0.05,
    seed: int = 42,
    n_bootstrap: int = 2_000,
    n_permutations: int = 10_000,
) -> dict[str, Any]:
    """在真实跨领域/多难度题目池上验证定理 7A 辛普森混杂遮蔽与定理 7B/7C 三分支自适应切换。

    统计推断说明：
    - 门控选择（Hybrid-Adaptive-KL-LCB）：在每折训练集的严格不相交内部留出集 inner_val_idx 上，
      从三元候选规则族 {Standard-BoN, Pooled-KL-LCB, Within-Prompt-Z-KL-LCB} 中按经验验证准确率择优
      （N=2 时因题内 Z-Score 纯秩退化，在 {Standard-BoN, Pooled-KL-LCB} 上使用精确超几何无方差评估），
      随后在全量训练折上重拟合校准器；
    - 置信区间与标准误：对折外（out-of-fold）独立题目级得分向量执行 n_bootstrap 次 Paired Cluster Bootstrap；
    - 假设检验 p 值：对配对题目级差异 d_j 执行 n_permutations 次 Rademacher 对称符号翻转置换检验，
      采用 Phipson & Smyth (2010) 精确无偏修正 p = (1 + #{|d_bar^(b)| >= |d_bar_obs|}) / (1 + B_perm)，
      保证 p >= 1 / (1 + B_perm) > 0；注意折外成簇推断条件于已拟合的交叉验证折模型（不重拟合每折分桶器）。

    Args:
        pools_with_strata: 题目序列，每项为 (stratum_name, scores_1d, is_correct_1d)。
    """
    if len(pools_with_strata) == 0:
        raise ValueError("pools_with_strata 不能为空")
    parsed: list[tuple[str, np.ndarray, np.ndarray]] = []
    for st_name, sc, yc in pools_with_strata:
        s_arr = np.asarray(sc, dtype=np.float64).ravel()
        y_arr = np.asarray(yc, dtype=bool).ravel()
        if s_arr.size == 0 or s_arr.shape != y_arr.shape:
            raise ValueError(f"域 {st_name!r} 存在空池或长度不匹配")
        parsed.append((str(st_name), s_arr, y_arr))

    # 1. 定理 7A 真实分桶辛普森分解与方差诊断
    strata_names = sorted({item[0] for item in parsed})
    all_s = np.concatenate([item[1] for item in parsed])
    all_y = np.concatenate([item[2] for item in parsed])

    # 按题内归一化方差估计各域仿射偏移强度 vs 先验异质性
    strata_summary: dict[str, dict[str, float | int]] = {}
    st_weights = []
    st_means = []
    st_ps = []
    st_within_vars = []
    for st in strata_names:
        sub = [item for item in parsed if item[0] == st]
        s_st = np.concatenate([x[1] for x in sub])
        y_st = np.concatenate([x[2] for x in sub])
        pos_s = s_st[y_st]
        neg_s = s_st[~y_st]
        mu_s = float(np.mean(s_st))
        sd_s = float(np.std(s_st, ddof=1)) if s_st.size > 1 else 1e-6
        s1 = float(np.std(pos_s, ddof=1)) if pos_s.size > 1 else 1e-6
        s0 = float(np.std(neg_s, ddof=1)) if neg_s.size > 1 else 1e-6
        p_st = float(np.mean(y_st))
        strata_summary[st] = {
            "n_problems": len(sub),
            "n_candidates": int(s_st.size),
            "p_hat": p_st,
            "mean_score": mu_s,
            "std_score": sd_s,
            "mu1": float(np.mean(pos_s)) if pos_s.size > 0 else 0.0,
            "sigma1": s1,
            "mu0": float(np.mean(neg_s)) if neg_s.size > 0 else 0.0,
            "sigma0": s0,
            "kappa": float(s0 / max(1e-12, s1)),
        }
        st_weights.append(float(s_st.size))
        st_means.append(mu_s)
        st_ps.append(p_st)
        st_within_vars.append(sd_s * sd_s)

    w_st = np.asarray(st_weights, dtype=np.float64)
    w_st /= float(np.sum(w_st))
    mean_arr = np.asarray(st_means, dtype=np.float64)
    p_arr = np.asarray(st_ps, dtype=np.float64)
    within_sd_pooled = math.sqrt(max(1e-12, float(np.sum(w_st * np.asarray(st_within_vars)))))
    norm_means = mean_arr / within_sd_pooled
    affine_shift_var = float(np.sum(w_st * (norm_means - float(np.sum(w_st * norm_means))) ** 2))
    prior_induced_shift_var = float(np.sum(w_st * (p_arr - float(np.sum(w_st * p_arr))) ** 2))

    # 离散分位桶辛普森斜率分解
    inner_edges = _build_empirical_edges(all_s, int(n_simpson_bins), binning="quantile")
    k_simp = int(inner_edges.size + 1)
    bin_all = np.searchsorted(inner_edges, all_s, side="right")
    cnt_pool = np.bincount(bin_all, minlength=k_simp).astype(np.float64)
    pos_pool = np.bincount(bin_all, weights=all_y.astype(np.float64), minlength=k_simp)
    q_pool_bins = np.where(cnt_pool > 0, pos_pool / np.maximum(1.0, cnt_pool), np.nan)

    g_cnt = len(strata_names)
    cnt_g = np.zeros((g_cnt, k_simp), dtype=np.float64)
    q_g = np.full((g_cnt, k_simp), np.nan, dtype=np.float64)
    for g_idx, st in enumerate(strata_names):
        sub = [item for item in parsed if item[0] == st]
        s_st = np.concatenate([x[1] for x in sub])
        y_st = np.concatenate([x[2] for x in sub]).astype(np.float64)
        b_st = np.searchsorted(inner_edges, s_st, side="right")
        c_row = np.bincount(b_st, minlength=k_simp).astype(np.float64)
        p_row = np.bincount(b_st, weights=y_st, minlength=k_simp)
        cnt_g[g_idx] = c_row
        valid_g = c_row > 0
        q_g[g_idx, valid_g] = p_row[valid_g] / c_row[valid_g]

    w_post = cnt_g / np.maximum(1.0, cnt_pool[None, :])
    delta_pool = []
    within_slopes = []
    between_terms = []
    masked_transitions = 0
    for b in range(k_simp - 1):
        dq_p = float(q_pool_bins[b + 1] - q_pool_bins[b])
        both = (cnt_g[:, b] > 0) & (cnt_g[:, b + 1] > 0)
        if np.any(both):
            w_avg = 0.5 * (w_post[both, b] + w_post[both, b + 1])
            w_sum = max(1e-12, float(np.sum(w_avg)))
            w_norm = w_avg / w_sum
            within_s = float(np.sum(w_norm * (q_g[both, b + 1] - q_g[both, b])))
        else:
            within_s = dq_p
        between_s = dq_p - within_s
        delta_pool.append(dq_p)
        within_slopes.append(within_s)
        between_terms.append(between_s)
        if within_s < -1e-4 and dq_p > 1e-4:
            masked_transitions += 1

    # 2. 定理 7B/7C 严格 5-Fold 按题 CV + 三分支内部留出门控 + 严格在线 N-样本 Z-Score 评估
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    n_probs = len(parsed)
    eff_folds = min(int(n_folds), n_probs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_probs)
    folds = np.array_split(perm, eff_folds)

    algo_names = [
        "Standard-BoN",
        "Pooled-KL-LCB",
        "Within-Prompt-Z-KL-LCB",
        "Hybrid-Adaptive-KL-LCB",
    ]
    base_branches = ("Standard-BoN", "Pooled-KL-LCB", "Within-Prompt-Z-KL-LCB")
    hybrid_branch_counts: dict[str, dict[str, int]] = {
        str(int(nv)): {b_name: 0 for b_name in base_branches} for nv in n_arr
    }
    prompt_acc = {a: np.full((n_probs, n_arr.size), np.nan, dtype=np.float64) for a in algo_names}
    eval_counts_by_n = np.zeros(n_arr.size, dtype=np.int64)
    n_t = float(n_trials_per_problem)
    n_val_trials = max(12, min(int(n_trials_per_problem), 20))

    for f_idx in range(eff_folds):
        test_idx = folds[f_idx]
        train_idx = np.concatenate([folds[j] for j in range(eff_folds) if j != f_idx])

        # 严格互不相交的内部留出分割（inner_fit vs inner_val）：仅在 inner_fit 上拟合门控对比校准器，
        # 在完全未参与拟合的 inner_val 上从 {Standard-BoN, Pooled-KL-LCB, Within-Prompt-Z-KL-LCB} 择优。
        # 使用独立子随机数流 val_rng，确保门控评估不扰动外部测试折抽样排列流 rng。
        val_rng = np.random.default_rng(int(seed) + 50_003 + f_idx)
        hybrid_branch_by_n: dict[int, str] = {}
        if len(train_idx) >= 2:
            if eff_folds >= 3:
                # 标准嵌套交叉验证：轮换选取下一折作为不相交内部验证折，避免多折重复复用同一子集
                inner_val_idx = folds[(f_idx + 1) % eff_folds]
                inner_fit_idx = np.concatenate(
                    [folds[j] for j in range(eff_folds) if j != f_idx and j != (f_idx + 1) % eff_folds]
                )
            else:
                n_val = max(1, int(round(0.2 * len(train_idx))))
                if n_val >= len(train_idx):
                    n_val = len(train_idx) - 1
                inner_val_idx = train_idx[:n_val]
                inner_fit_idx = train_idx[n_val:]

            in_s = np.concatenate([parsed[i][1] for i in inner_fit_idx])
            in_y = np.concatenate([parsed[i][2] for i in inner_fit_idx]).astype(np.float64)
            in_z = np.concatenate([
                _within_prompt_zscore_2d(parsed[i][1][None, :]).ravel() for i in inner_fit_idx
            ])
            pred_pooled_in, _ = _fit_sota_calibrator_predictor(
                in_s, in_y, method="kl_lcb", n_bins=n_bins, delta=delta
            )
            pred_within_in, _ = _fit_sota_calibrator_predictor(
                in_z, in_y, method="kl_lcb", n_bins=n_bins, delta=delta
            )
            val_row_i = np.arange(n_val_trials)
            for nv in n_arr:
                nv_int = int(nv)
                if nv_int <= 1:
                    hybrid_branch_by_n[nv_int] = "Standard-BoN"
                    continue
                # 对一切 N >= 2（含 N = 2，允许负均值偏移下的 Anti-BoN Z 分支参与竞争）执行统一三分支验证风险最小化
                std_hits, pool_hits, within_hits, cal_cnt = 0.0, 0.0, 0.0, 0
                for ci in inner_val_idx:
                    _, cs, cy = parsed[ci]
                    if cs.size < nv_int:
                        continue
                    std_hits += float(exact_pool_bon_accuracy(cs, cy, [nv_int])[0]) * float(n_val_trials)
                    pool_hits += float(exact_pool_bon_accuracy(pred_pooled_in(cs), cy, [nv_int])[0]) * float(n_val_trials)
                    c_perms = np.array(
                        [val_rng.permutation(cs.size)[:nv_int] for _ in range(n_val_trials)],
                        dtype=np.int64,
                    )
                    cs_mat = cs[c_perms]
                    cy_mat = cy[c_perms]
                    idx_w = _argmax_random_tie_2d(pred_within_in(_within_prompt_zscore_2d(cs_mat)), val_rng)
                    within_hits += float(np.count_nonzero(cy_mat[val_row_i, idx_w]))
                    cal_cnt += n_val_trials
                if cal_cnt == 0:
                    hybrid_branch_by_n[nv_int] = "Pooled-KL-LCB"
                elif within_hits > max(pool_hits, std_hits) + 1e-12:
                    hybrid_branch_by_n[nv_int] = "Within-Prompt-Z-KL-LCB"
                elif std_hits > pool_hits + 1e-12:
                    hybrid_branch_by_n[nv_int] = "Standard-BoN"
                else:
                    hybrid_branch_by_n[nv_int] = "Pooled-KL-LCB"
        else:
            hybrid_branch_by_n = {int(nv): ("Standard-BoN" if int(nv) <= 1 else "Pooled-KL-LCB") for nv in n_arr}

        for nv in n_arr:
            nv_int = int(nv)
            hybrid_branch_counts[str(nv_int)][hybrid_branch_by_n[nv_int]] += 1

        cal_s = np.concatenate([parsed[i][1] for i in train_idx])
        cal_y = np.concatenate([parsed[i][2] for i in train_idx]).astype(np.float64)
        cal_z = np.concatenate([
            _within_prompt_zscore_2d(parsed[i][1][None, :]).ravel() for i in train_idx
        ])

        pred_pooled, _ = _fit_sota_calibrator_predictor(
            cal_s, cal_y, method="kl_lcb", n_bins=n_bins, delta=delta
        )
        pred_within, _ = _fit_sota_calibrator_predictor(
            cal_z, cal_y, method="kl_lcb", n_bins=n_bins, delta=delta
        )

        row_idx = np.arange(int(n_trials_per_problem))
        for p_idx in test_idx:
            _, s_arr, y_arr = parsed[p_idx]
            pool_len = int(s_arr.size)
            valid_n = n_arr[n_arr <= pool_len]
            if valid_n.size == 0:
                continue
            max_n = int(np.max(valid_n))
            perms = np.array(
                [rng.permutation(pool_len)[:max_n] for _ in range(int(n_trials_per_problem))],
                dtype=np.int64,
            )
            s_full = s_arr[perms]
            y_full = y_arr[perms]
            p_full = pred_pooled(s_full)

            for k_idx, nv in enumerate(n_arr):
                nv_int = int(nv)
                if nv_int > pool_len:
                    continue
                eval_counts_by_n[k_idx] += int(n_trials_per_problem)
                s_sub = s_full[:, :nv_int]
                y_sub = y_full[:, :nv_int]

                idx_std = _argmax_random_tie_2d(s_sub, rng)
                acc_std = float(np.count_nonzero(y_sub[row_idx, idx_std])) / n_t
                prompt_acc["Standard-BoN"][p_idx, k_idx] = acc_std

                idx_pool = _argmax_random_tie_2d(p_full[:, :nv_int], rng)
                acc_pool = float(np.count_nonzero(y_sub[row_idx, idx_pool])) / n_t
                prompt_acc["Pooled-KL-LCB"][p_idx, k_idx] = acc_pool

                if nv_int == 1:
                    h_one = float(np.count_nonzero(y_sub[:, 0])) / n_t
                    prompt_acc["Within-Prompt-Z-KL-LCB"][p_idx, k_idx] = h_one
                    prompt_acc["Hybrid-Adaptive-KL-LCB"][p_idx, k_idx] = h_one
                else:
                    z_sub = _within_prompt_zscore_2d(s_sub)
                    w_sub = pred_within(z_sub)
                    idx_within = _argmax_random_tie_2d(w_sub, rng)
                    acc_within = float(np.count_nonzero(y_sub[row_idx, idx_within])) / n_t
                    prompt_acc["Within-Prompt-Z-KL-LCB"][p_idx, k_idx] = acc_within
                    chosen_branch = hybrid_branch_by_n[nv_int]
                    if chosen_branch == "Within-Prompt-Z-KL-LCB":
                        prompt_acc["Hybrid-Adaptive-KL-LCB"][p_idx, k_idx] = acc_within
                    elif chosen_branch == "Standard-BoN":
                        prompt_acc["Hybrid-Adaptive-KL-LCB"][p_idx, k_idx] = acc_std
                    else:
                        prompt_acc["Hybrid-Adaptive-KL-LCB"][p_idx, k_idx] = acc_pool

    if np.any(eval_counts_by_n == 0):
        raise ValueError("存在没有任何题目满足 pool_size >= n 的预算值")

    # 题目级 Paired Cluster Bootstrap (SE / 95% CI) + Rademacher 符号翻转置换检验 (Phipson-Smyth p 值)
    n_boot = int(n_bootstrap)
    n_perm = max(1, int(n_permutations))
    n_evaluated_problems_by_n = np.zeros(n_arr.size, dtype=np.int64)
    algo_results: dict[str, dict[str, np.ndarray]] = {
        a_name: {
            "accuracy": np.zeros(n_arr.size, dtype=np.float64),
            "se": np.zeros(n_arr.size, dtype=np.float64),
            "ci_low": np.zeros(n_arr.size, dtype=np.float64),
            "ci_high": np.zeros(n_arr.size, dtype=np.float64),
        }
        for a_name in algo_names
    }
    comp_vs_pooled = ("Within-Prompt-Z-KL-LCB", "Hybrid-Adaptive-KL-LCB", "Standard-BoN")
    comp_vs_std = ("Pooled-KL-LCB", "Within-Prompt-Z-KL-LCB", "Hybrid-Adaptive-KL-LCB")

    def _init_paired_dict(keys: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
        return {
            k_name: {
                "mean_diff": np.zeros(n_arr.size, dtype=np.float64),
                "se_diff": np.zeros(n_arr.size, dtype=np.float64),
                "ci_low": np.zeros(n_arr.size, dtype=np.float64),
                "ci_high": np.zeros(n_arr.size, dtype=np.float64),
                "p_value": np.ones(n_arr.size, dtype=np.float64),
            }
            for k_name in keys
        }

    paired_vs_pooled = _init_paired_dict(comp_vs_pooled)
    paired_vs_standard_bon = _init_paired_dict(comp_vs_std)

    def _sign_flip_perm_pvalue(diff_vec: np.ndarray, perm_rng: np.random.Generator) -> float:
        obs_abs = abs(float(np.mean(diff_vec)))
        if obs_abs <= 1e-15 or float(np.max(np.abs(diff_vec))) <= 1e-15:
            return 1.0
        n_v = diff_vec.size
        exceed = 0
        chunk = 2_000
        done = 0
        while done < n_perm:
            cur = min(chunk, n_perm - done)
            signs = perm_rng.integers(0, 2, size=(cur, n_v), dtype=np.int8) * 2 - 1
            perm_means = (signs.astype(np.float64) @ diff_vec) / float(n_v)
            exceed += int(np.count_nonzero(np.abs(perm_means) >= obs_abs - 1e-15))
            done += cur
        return float((1 + exceed) / (1 + n_perm))

    for k_idx in range(n_arr.size):
        valid_mask = ~np.isnan(prompt_acc["Standard-BoN"][:, k_idx])
        n_valid = int(np.count_nonzero(valid_mask))
        n_evaluated_problems_by_n[k_idx] = n_valid
        vec_valid = {a_name: prompt_acc[a_name][valid_mask, k_idx] for a_name in algo_names}

        for a_name in algo_names:
            algo_results[a_name]["accuracy"][k_idx] = float(np.mean(vec_valid[a_name]))
        for comp in comp_vs_pooled:
            paired_vs_pooled[comp]["mean_diff"][k_idx] = float(
                np.mean(vec_valid[comp] - vec_valid["Pooled-KL-LCB"])
            )
        for comp in comp_vs_std:
            paired_vs_standard_bon[comp]["mean_diff"][k_idx] = float(
                np.mean(vec_valid[comp] - vec_valid["Standard-BoN"])
            )

        if n_valid <= 1 or n_boot < 2:
            for a_name in algo_names:
                v = algo_results[a_name]["accuracy"][k_idx]
                algo_results[a_name]["ci_low"][k_idx] = float(np.clip(v, 0.0, 1.0))
                algo_results[a_name]["ci_high"][k_idx] = float(np.clip(v, 0.0, 1.0))
            for comp in comp_vs_pooled:
                d = paired_vs_pooled[comp]["mean_diff"][k_idx]
                paired_vs_pooled[comp]["ci_low"][k_idx] = d
                paired_vs_pooled[comp]["ci_high"][k_idx] = d
            for comp in comp_vs_std:
                d = paired_vs_standard_bon[comp]["mean_diff"][k_idx]
                paired_vs_standard_bon[comp]["ci_low"][k_idx] = d
                paired_vs_standard_bon[comp]["ci_high"][k_idx] = d
            continue

        boot_rng = np.random.default_rng(int(seed) + 100_003 + k_idx)
        boot_means = {a_name: np.empty(n_boot, dtype=np.float64) for a_name in algo_names}
        b_chunk = 500
        b_done = 0
        while b_done < n_boot:
            cur_b = min(b_chunk, n_boot - b_done)
            idx_mat = boot_rng.integers(0, n_valid, size=(cur_b, n_valid))
            for a_name in algo_names:
                boot_means[a_name][b_done : b_done + cur_b] = np.mean(vec_valid[a_name][idx_mat], axis=1)
            b_done += cur_b

        for a_name in algo_names:
            bm = boot_means[a_name]
            algo_results[a_name]["se"][k_idx] = float(np.std(bm, ddof=1))
            q_lo, q_hi = np.percentile(bm, [2.5, 97.5])
            algo_results[a_name]["ci_low"][k_idx] = float(np.clip(q_lo, 0.0, 1.0))
            algo_results[a_name]["ci_high"][k_idx] = float(np.clip(q_hi, 0.0, 1.0))

        perm_rng = np.random.default_rng(int(seed) + 200_003 + k_idx)
        bm_pool = boot_means["Pooled-KL-LCB"]
        for comp in comp_vs_pooled:
            bdiff = boot_means[comp] - bm_pool
            paired_vs_pooled[comp]["se_diff"][k_idx] = float(np.std(bdiff, ddof=1))
            d_lo, d_hi = np.percentile(bdiff, [2.5, 97.5])
            paired_vs_pooled[comp]["ci_low"][k_idx] = float(d_lo)
            paired_vs_pooled[comp]["ci_high"][k_idx] = float(d_hi)
            paired_vs_pooled[comp]["p_value"][k_idx] = _sign_flip_perm_pvalue(
                vec_valid[comp] - vec_valid["Pooled-KL-LCB"], perm_rng
            )

        bm_std = boot_means["Standard-BoN"]
        for comp in comp_vs_std:
            bdiff_s = boot_means[comp] - bm_std
            paired_vs_standard_bon[comp]["se_diff"][k_idx] = float(np.std(bdiff_s, ddof=1))
            ds_lo, ds_hi = np.percentile(bdiff_s, [2.5, 97.5])
            paired_vs_standard_bon[comp]["ci_low"][k_idx] = float(ds_lo)
            paired_vs_standard_bon[comp]["ci_high"][k_idx] = float(ds_hi)
            paired_vs_standard_bon[comp]["p_value"][k_idx] = _sign_flip_perm_pvalue(
                vec_valid[comp] - vec_valid["Standard-BoN"], perm_rng
            )

    return {
        "n_problems": n_probs,
        "n_total_candidates": int(all_s.size),
        "strata_names": strata_names,
        "strata_summary": strata_summary,
        "affine_shift_var": affine_shift_var,
        "prior_induced_shift_var": prior_induced_shift_var,
        "simpson_decomposition": {
            "n_bins": k_simp,
            "q_pool_bins": q_pool_bins,
            "q_strata_bins": q_g,
            "strata_post_weights": w_post,
            "delta_q_pool": np.asarray(delta_pool, dtype=np.float64),
            "within_slope_term": np.asarray(within_slopes, dtype=np.float64),
            "between_confounding_term": np.asarray(between_terms, dtype=np.float64),
            "masked_transitions": int(masked_transitions),
            "pooled_appears_monotone": bool(all(d >= -1e-4 for d in delta_pool)),
        },
        "n_values": n_arr,
        "eval_trials_by_n": eval_counts_by_n,
        "n_evaluated_problems_by_n": n_evaluated_problems_by_n,
        "hybrid_candidate_family": list(base_branches),
        "hybrid_branch_selection_counts": hybrid_branch_counts,
        "se_method": "prompt_cluster_bootstrap",
        "p_value_method": "paired_sign_flip_permutation_phipson_smyth",
        "inference_scope_note": (
            "Out-of-fold prompt-cluster bootstrap and Rademacher sign-flip permutation "
            "condition on fitted fold calibrators (capturing test-prompt sampling and "
            "finite-trial candidate draw variance, without refitting fold calibrators per replicate)."
        ),
        "n_bootstrap": n_boot,
        "n_permutations": n_perm,
        "paired_bootstrap_vs_pooled": paired_vs_pooled,
        "paired_bootstrap_vs_standard_bon": paired_vs_standard_bon,
        "algorithms": algo_results,
    }


def evaluate_single_split_prompt_heterogeneity(
    pools_with_strata: Sequence[tuple[str, np.ndarray, np.ndarray]],
    n_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    split_ratios: tuple[float, float, float] = (0.50, 0.20, 0.30),
    n_trials_per_problem: int = 25,
    n_bins: int = 15,
    delta: float = 0.05,
    seed: int = 42,
    n_bootstrap: int = 2_000,
) -> dict[str, Any]:
    """采用互不相交单次划分（Train / Val / Test）评估三分支门控与独立测试集显著性（修复 P1-03 & P1-04）。

    设计与统计保证：
    1. 将全部 P 道独立题目一次性划分为互不相交的 Train（默认 50%）、Val（20%）与 Held-Out Test（30%）；
    2. 仅在 Train 上拟合 Pooled-KL-LCB 与 Within-Prompt-Z-KL-LCB，固定这三个候选规则（含 Standard-BoN）；
    3. 在独立 Val 上对一切 N >= 2 执行统一三分支经验风险最小化选出 m_hat_N，**不重拟合训练模型**
       （严格满足命题 7.1 固定规则条件，重拟合稳定性误差 epsilon_refit == 0；同时另行记录重拟合对照差值）；
    4. 在完全独立的 Held-Out Test 题目上评估各规则：条件于已固定的单次训练/验证模型，各测试题配对差值
       D_j（j = 1, ..., P_test）严格相互独立（完全消除交叉验证折间训练集重叠相关性），并采用无需对称分布假设的
       双侧配对学生化 t 检验（Studentized paired t-test）作为主显著性 p 值，辅以 2000x 独立测试题 Cluster Bootstrap CI。
    """
    if len(pools_with_strata) < 6:
        raise ValueError("evaluate_single_split_prompt_heterogeneity 至少需要 6 道题目以完成三段划分")
    r_tr, r_val, r_te = (float(split_ratios[0]), float(split_ratios[1]), float(split_ratios[2]))
    if r_tr <= 0.0 or r_val <= 0.0 or r_te <= 0.0 or abs((r_tr + r_val + r_te) - 1.0) > 1e-6:
        raise ValueError("split_ratios 三分量必须为正且和等于 1")

    parsed: list[tuple[str, np.ndarray, np.ndarray]] = []
    for st_name, sc, yc in pools_with_strata:
        s_arr = np.asarray(sc, dtype=np.float64).ravel()
        y_arr = np.asarray(yc, dtype=bool).ravel()
        if s_arr.size == 0 or s_arr.shape != y_arr.shape:
            raise ValueError(f"域 {st_name!r} 存在空池或长度不匹配")
        parsed.append((str(st_name), s_arr, y_arr))

    n_probs = len(parsed)
    rng = np.random.default_rng(seed)
    # 按域分层随机打乱以保持 Train / Val / Test 域比例一致且无泄露
    by_stratum: dict[str, list[int]] = {}
    for idx, (st_name, _, _) in enumerate(parsed):
        by_stratum.setdefault(st_name, []).append(idx)

    train_idx_l, val_idx_l, test_idx_l = [], [], []
    for st_name in sorted(by_stratum.keys()):
        idxs = np.asarray(by_stratum[st_name], dtype=np.int64)
        rng.shuffle(idxs)
        n_s = idxs.size
        n_tr_s = max(1, int(round(r_tr * n_s)))
        n_val_s = max(1, int(round(r_val * n_s)))
        if n_tr_s + n_val_s >= n_s:
            n_tr_s = max(1, n_s - 2)
            n_val_s = max(1, n_s - 1 - n_tr_s)
        train_idx_l.append(idxs[:n_tr_s])
        val_idx_l.append(idxs[n_tr_s : n_tr_s + n_val_s])
        test_idx_l.append(idxs[n_tr_s + n_val_s :])

    train_idx = np.concatenate(train_idx_l)
    val_idx = np.concatenate(val_idx_l)
    test_idx = np.concatenate(test_idx_l)

    tr_s = np.concatenate([parsed[i][1] for i in train_idx])
    tr_y = np.concatenate([parsed[i][2] for i in train_idx]).astype(np.float64)
    tr_z = np.concatenate([_within_prompt_zscore_2d(parsed[i][1][None, :]).ravel() for i in train_idx])

    pred_pooled_fixed, _ = _fit_sota_calibrator_predictor(tr_s, tr_y, method="kl_lcb", n_bins=n_bins, delta=delta)
    pred_within_fixed, _ = _fit_sota_calibrator_predictor(tr_z, tr_y, method="kl_lcb", n_bins=n_bins, delta=delta)

    # 同时拟合一份合并 Train+Val 的重拟合模型，仅用于定量测算 epsilon_refit 稳定性上界
    trv_idx = np.concatenate([train_idx, val_idx])
    trv_s = np.concatenate([parsed[i][1] for i in trv_idx])
    trv_y = np.concatenate([parsed[i][2] for i in trv_idx]).astype(np.float64)
    trv_z = np.concatenate([_within_prompt_zscore_2d(parsed[i][1][None, :]).ravel() for i in trv_idx])
    pred_pooled_refit, _ = _fit_sota_calibrator_predictor(trv_s, trv_y, method="kl_lcb", n_bins=n_bins, delta=delta)
    pred_within_refit, _ = _fit_sota_calibrator_predictor(trv_z, trv_y, method="kl_lcb", n_bins=n_bins, delta=delta)

    n_arr = np.asarray(list(n_values), dtype=np.int64)
    n_t = int(n_trials_per_problem)
    val_rng = np.random.default_rng(int(seed) + 60_003)
    val_rows = np.arange(n_t)

    selected_branch_by_n: dict[str, str] = {}
    for nv in n_arr:
        nv_int = int(nv)
        if nv_int <= 1:
            selected_branch_by_n[str(nv_int)] = "Standard-BoN"
            continue
        std_h, pool_h, within_h, v_cnt = 0.0, 0.0, 0.0, 0
        for vi in val_idx:
            _, cs, cy = parsed[int(vi)]
            if cs.size < nv_int:
                continue
            std_h += float(exact_pool_bon_accuracy(cs, cy, [nv_int])[0]) * float(n_t)
            pool_h += float(exact_pool_bon_accuracy(pred_pooled_fixed(cs), cy, [nv_int])[0]) * float(n_t)
            c_perms = np.array([val_rng.permutation(cs.size)[:nv_int] for _ in range(n_t)], dtype=np.int64)
            cs_m = cs[c_perms]
            cy_m = cy[c_perms]
            idx_w = _argmax_random_tie_2d(pred_within_fixed(_within_prompt_zscore_2d(cs_m)), val_rng)
            within_h += float(np.count_nonzero(cy_m[val_rows, idx_w]))
            v_cnt += n_t
        if v_cnt == 0:
            selected_branch_by_n[str(nv_int)] = "Pooled-KL-LCB"
        elif within_h > max(pool_h, std_h) + 1e-12:
            selected_branch_by_n[str(nv_int)] = "Within-Prompt-Z-KL-LCB"
        elif std_h > pool_h + 1e-12:
            selected_branch_by_n[str(nv_int)] = "Standard-BoN"
        else:
            selected_branch_by_n[str(nv_int)] = "Pooled-KL-LCB"

    algo_names = (
        "Standard-BoN",
        "Pooled-KL-LCB",
        "Within-Prompt-Z-KL-LCB",
        "Hybrid-Fixed-KL-LCB",
        "Hybrid-Refit-KL-LCB",
    )
    n_test = int(test_idx.size)
    test_acc = {a: np.full((n_test, n_arr.size), np.nan, dtype=np.float64) for a in algo_names}
    refit_pooled_mat = np.full((n_test, n_arr.size), np.nan, dtype=np.float64)
    refit_within_mat = np.full((n_test, n_arr.size), np.nan, dtype=np.float64)

    test_rng = np.random.default_rng(int(seed) + 70_003)
    t_rows = np.arange(n_t)
    for local_i, p_idx in enumerate(test_idx):
        _, s_arr, y_arr = parsed[int(p_idx)]
        pool_len = int(s_arr.size)
        valid_n = n_arr[n_arr <= pool_len]
        if valid_n.size == 0:
            continue
        max_n = int(np.max(valid_n))
        perms = np.array([test_rng.permutation(pool_len)[:max_n] for _ in range(n_t)], dtype=np.int64)
        s_full = s_arr[perms]
        y_full = y_arr[perms]
        pf_full = pred_pooled_fixed(s_full)
        pr_full = pred_pooled_refit(s_full)

        for k_idx, nv in enumerate(n_arr):
            nv_int = int(nv)
            if nv_int > pool_len:
                continue
            s_sub = s_full[:, :nv_int]
            y_sub = y_full[:, :nv_int]

            idx_std = _argmax_random_tie_2d(s_sub, test_rng)
            a_std = float(np.count_nonzero(y_sub[t_rows, idx_std])) / float(n_t)
            idx_pf = _argmax_random_tie_2d(pf_full[:, :nv_int], test_rng)
            a_pf = float(np.count_nonzero(y_sub[t_rows, idx_pf])) / float(n_t)
            idx_pr = _argmax_random_tie_2d(pr_full[:, :nv_int], test_rng)
            a_pr = float(np.count_nonzero(y_sub[t_rows, idx_pr])) / float(n_t)
            refit_pooled_mat[local_i, k_idx] = a_pr

            test_acc["Standard-BoN"][local_i, k_idx] = a_std
            test_acc["Pooled-KL-LCB"][local_i, k_idx] = a_pf

            if nv_int == 1:
                h1 = float(np.count_nonzero(y_sub[:, 0])) / float(n_t)
                test_acc["Within-Prompt-Z-KL-LCB"][local_i, k_idx] = h1
                test_acc["Hybrid-Fixed-KL-LCB"][local_i, k_idx] = h1
                test_acc["Hybrid-Refit-KL-LCB"][local_i, k_idx] = h1
                refit_within_mat[local_i, k_idx] = h1
            else:
                z_sub = _within_prompt_zscore_2d(s_sub)
                idx_wf = _argmax_random_tie_2d(pred_within_fixed(z_sub), test_rng)
                a_wf = float(np.count_nonzero(y_sub[t_rows, idx_wf])) / float(n_t)
                idx_wr = _argmax_random_tie_2d(pred_within_refit(z_sub), test_rng)
                a_wr = float(np.count_nonzero(y_sub[t_rows, idx_wr])) / float(n_t)
                test_acc["Within-Prompt-Z-KL-LCB"][local_i, k_idx] = a_wf
                refit_within_mat[local_i, k_idx] = a_wr

                br = selected_branch_by_n[str(nv_int)]
                if br == "Within-Prompt-Z-KL-LCB":
                    test_acc["Hybrid-Fixed-KL-LCB"][local_i, k_idx] = a_wf
                    test_acc["Hybrid-Refit-KL-LCB"][local_i, k_idx] = a_wr
                elif br == "Standard-BoN":
                    test_acc["Hybrid-Fixed-KL-LCB"][local_i, k_idx] = a_std
                    test_acc["Hybrid-Refit-KL-LCB"][local_i, k_idx] = a_std
                else:
                    test_acc["Hybrid-Fixed-KL-LCB"][local_i, k_idx] = a_pf
                    test_acc["Hybrid-Refit-KL-LCB"][local_i, k_idx] = a_pr

    n_boot = int(n_bootstrap)
    algo_out: dict[str, dict[str, np.ndarray]] = {
        a: {
            "accuracy": np.zeros(n_arr.size, dtype=np.float64),
            "se": np.zeros(n_arr.size, dtype=np.float64),
            "ci_low": np.zeros(n_arr.size, dtype=np.float64),
            "ci_high": np.zeros(n_arr.size, dtype=np.float64),
        }
        for a in algo_names
    }
    refit_shift_by_n = np.zeros(n_arr.size, dtype=np.float64)

    def _paired_ttest_and_boot(
        v1: np.ndarray, v2: np.ndarray, b_rng: np.random.Generator
    ) -> tuple[float, float, float, float, float]:
        d_vec = v1 - v2
        m_d = float(np.mean(d_vec))
        n_v = d_vec.size
        if n_v <= 1:
            return m_d, 0.0, m_d, m_d, 1.0
        sd_d = float(np.std(d_vec, ddof=1))
        if sd_d <= 1e-15:
            p_t = 1.0 if abs(m_d) <= 1e-15 else 0.0
            return m_d, 0.0, m_d, m_d, p_t
        se_analytic = sd_d / math.sqrt(float(n_v))
        t_stat = abs(m_d) / se_analytic
        p_t = float(2.0 * special.stdtr(float(n_v - 1), -t_stat))
        b_idx = b_rng.integers(0, n_v, size=(max(2, n_boot), n_v))
        b_means = np.mean(d_vec[b_idx], axis=1)
        lo_d, hi_d = np.percentile(b_means, [2.5, 97.5])
        return m_d, float(np.std(b_means, ddof=1)), float(lo_d), float(hi_d), p_t

    comp_targets = ("Within-Prompt-Z-KL-LCB", "Hybrid-Fixed-KL-LCB", "Hybrid-Refit-KL-LCB", "Standard-BoN")
    vs_pooled = {
        c: {
            "mean_diff": np.zeros(n_arr.size),
            "se_diff": np.zeros(n_arr.size),
            "ci_low": np.zeros(n_arr.size),
            "ci_high": np.zeros(n_arr.size),
            "p_value_ttest": np.ones(n_arr.size),
        }
        for c in comp_targets
    }
    vs_std = {
        c: {
            "mean_diff": np.zeros(n_arr.size),
            "se_diff": np.zeros(n_arr.size),
            "ci_low": np.zeros(n_arr.size),
            "ci_high": np.zeros(n_arr.size),
            "p_value_ttest": np.ones(n_arr.size),
        }
        for c in ("Pooled-KL-LCB", "Within-Prompt-Z-KL-LCB", "Hybrid-Fixed-KL-LCB", "Hybrid-Refit-KL-LCB")
    }

    for k_idx in range(n_arr.size):
        valid = ~np.isnan(test_acc["Standard-BoN"][:, k_idx])
        n_v = int(np.count_nonzero(valid))
        vecs = {a: test_acc[a][valid, k_idx] for a in algo_names}
        shift_p = abs(float(np.mean(refit_pooled_mat[valid, k_idx] - vecs["Pooled-KL-LCB"])))
        shift_w = abs(float(np.mean(refit_within_mat[valid, k_idx] - vecs["Within-Prompt-Z-KL-LCB"])))
        refit_shift_by_n[k_idx] = max(shift_p, shift_w)

        b_rng = np.random.default_rng(int(seed) + 80_003 + k_idx)
        b_idx = b_rng.integers(0, n_v, size=(max(2, n_boot), n_v))
        for a in algo_names:
            algo_out[a]["accuracy"][k_idx] = float(np.mean(vecs[a]))
            bm = np.mean(vecs[a][b_idx], axis=1)
            algo_out[a]["se"][k_idx] = float(np.std(bm, ddof=1))
            ql, qh = np.percentile(bm, [2.5, 97.5])
            algo_out[a]["ci_low"][k_idx] = float(np.clip(ql, 0.0, 1.0))
            algo_out[a]["ci_high"][k_idx] = float(np.clip(qh, 0.0, 1.0))

        for c in comp_targets:
            md, sed, cl, ch, pt = _paired_ttest_and_boot(vecs[c], vecs["Pooled-KL-LCB"], b_rng)
            vs_pooled[c]["mean_diff"][k_idx] = md
            vs_pooled[c]["se_diff"][k_idx] = sed
            vs_pooled[c]["ci_low"][k_idx] = cl
            vs_pooled[c]["ci_high"][k_idx] = ch
            vs_pooled[c]["p_value_ttest"][k_idx] = pt

        for c in vs_std:
            md, sed, cl, ch, pt = _paired_ttest_and_boot(vecs[c], vecs["Standard-BoN"], b_rng)
            vs_std[c]["mean_diff"][k_idx] = md
            vs_std[c]["se_diff"][k_idx] = sed
            vs_std[c]["ci_low"][k_idx] = cl
            vs_std[c]["ci_high"][k_idx] = ch
            vs_std[c]["p_value_ttest"][k_idx] = pt

    return {
        "n_total_problems": n_probs,
        "n_train_problems": int(train_idx.size),
        "n_val_problems": int(val_idx.size),
        "n_test_problems": n_test,
        "n_values": n_arr,
        "selected_branch_by_n": selected_branch_by_n,
        "empirical_refit_shift_by_n": refit_shift_by_n,
        "holdout_selection_bound_95": float(
            2.0 * math.sqrt(math.log(2.0 * 3.0 / 0.05) / max(1.0, 2.0 * float(val_idx.size)))
        ),
        "primary_inference_method": "disjoint_single_split_paired_studentized_ttest",
        "paired_test_vs_pooled": vs_pooled,
        "paired_test_vs_standard_bon": vs_std,
        "algorithms": algo_out,
    }


__all__ = [
    "diagnose_multi_verifier_correlations",
    "diagnose_real_verifier_distribution",
    "evaluate_discrete_algorithms_on_real_pools",
    "evaluate_finite_calibration_on_real_pools",
    "evaluate_heterogeneous_multi_rm_on_real_pools",
    "evaluate_joint_m_n_calibration_on_real_pools",
    "evaluate_real_prompt_heterogeneity",
    "evaluate_single_split_prompt_heterogeneity",
    "exact_pool_bon_accuracy",
    "extract_boxed_answer",
    "fit_and_predict_sota_calibrator",
    "is_math_answer_correct",
    "is_math_answer_correct_any",
    "normalize_math_answer",
]
