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
from typing import Any, Sequence

import numpy as np
from scipy import special, stats

from tts_theory.algorithms import (
    CalibratedWeightedBoN,
    EmpiricalScoreCalibrator,
    MajorityVoting,
    StandardBoN,
    TwoStageVerifiedMajority,
)
from tts_theory.calibration_theory import compute_bin_calibrator_scores
from tts_theory.ensemble_theory import (
    MultiVerifierGaussianSpec,
    ensemble_phase_reversal_criterion,
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
    全程在对数域通过 gammaln 计算二项式系数，彻底避免大 L 组合数溢出。
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
) -> dict[str, Any]:
    """在真实大模型候选池上评估有限校准集 M 对线上 BoN 的二次坍塌与 LCB 效应。"""
    c_s = np.asarray(cal_scores, dtype=np.float64).ravel()
    c_y = np.asarray(cal_correct, dtype=np.int64).ravel()
    if c_s.size == 0 or c_s.shape != c_y.shape:
        raise ValueError("cal_scores 与 cal_correct 必须非空且长度一致")
    if binning not in ("uniform", "quantile"):
        raise ValueError("binning 必须为 'uniform' 或 'quantile'")

    # 由真实校准集总分布确定统一固定参考桶边界，保证可复现与无测试泄露
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

    # 预先把校准集和全部测试题按候选池长度分组并聚合到桶计数矩阵 (P_L, K)
    cal_bin_idx = np.searchsorted(inner_edges, c_s, side="right")
    buckets_by_len: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    raw_curves = np.empty((len(test_pools), len(n_values)), dtype=np.float64)
    for idx_t, (ts, ty) in enumerate(test_pools):
        ts_arr = np.asarray(ts, dtype=np.float64).ravel()
        ty_arr = np.asarray(ty, dtype=np.int64).ravel()
        raw_curves[idx_t] = exact_pool_bon_accuracy(ts_arr, ty_arr, n_values)
        b_idx = np.searchsorted(inner_edges, ts_arr, side="right")
        c_row = np.bincount(b_idx, minlength=k_actual)
        y_row = np.bincount(b_idx, weights=ty_arr, minlength=k_actual).astype(np.int64)
        l_len = int(ts_arr.size)
        if l_len not in buckets_by_len:
            buckets_by_len[l_len] = ([], [])
        buckets_by_len[l_len][0].append(c_row)
        buckets_by_len[l_len][1].append(y_row)

    test_groups_by_len: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        l_len: (
            np.asarray(c_list, dtype=np.int64),
            np.asarray(y_list, dtype=np.int64),
            _build_hypergeom_table(l_len, n_values),
        )
        for l_len, (c_list, y_list) in buckets_by_len.items()
    }
    standard_bon_curve = np.mean(raw_curves, axis=0)

    rng = np.random.default_rng(seed)
    replace_draw = int(m_cal) > c_s.size
    prior_p = float(np.mean(c_y))
    n_tests = len(test_pools)

    acc_runs = {m_name: np.zeros((n_repeats, len(n_values)), dtype=np.float64) for m_name in methods}
    for r_idx in range(n_repeats):
        pick = rng.choice(c_s.size, size=int(m_cal), replace=replace_draw)
        b_pick = cal_bin_idx[pick]
        y_pick = c_y[pick]
        counts = np.bincount(b_pick, minlength=k_actual).astype(np.float64)
        positives = np.bincount(b_pick, weights=y_pick, minlength=k_actual).astype(np.float64)

        for m_name in methods:
            est = compute_bin_calibrator_scores(
                counts,
                positives,
                method=m_name,
                prior_p=prior_p,
                delta=delta,
                prior_strength=prior_strength,
            )
            acc_runs[m_name][r_idx] = _eval_mapped_pools_fast(
                test_groups_by_len, est, n_values, n_tests
            )

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
    rng = np.random.default_rng(seed)
    algo_names = list(selectors.keys())
    hits = {name: np.zeros(len(n_arr), dtype=np.float64) for name in algo_names}
    total_evals = float(len(encoded_problems) * n_trials_per_problem)

    for s_arr, ans_ids, lookup in encoded_problems:
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
                hits[name][k_idx] += float(np.count_nonzero(lookup[chosen_aids]))

    results = {}
    for name in algo_names:
        acc = hits[name] / total_evals
        se = np.sqrt(acc * (1.0 - acc) / total_evals)
        results[name] = {
            "accuracy": acc,
            "se": se,
            "ci_low": np.clip(acc - 1.96 * se, 0.0, 1.0),
            "ci_high": np.clip(acc + 1.96 * se, 0.0, 1.0),
        }

    return {
        "n_values": n_arr,
        "n_problems": len(encoded_problems),
        "n_trials_per_problem": int(n_trials_per_problem),
        "algorithms": results,
    }


__all__ = [
    "diagnose_multi_verifier_correlations",
    "diagnose_real_verifier_distribution",
    "evaluate_discrete_algorithms_on_real_pools",
    "evaluate_finite_calibration_on_real_pools",
    "exact_pool_bon_accuracy",
    "extract_boxed_answer",
    "is_math_answer_correct",
    "normalize_math_answer",
]
