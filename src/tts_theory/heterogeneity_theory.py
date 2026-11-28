"""题目异质性、辛普森悖论与题内标准化理论模块（定理 7）。

本模块刻画真实推理基准中题目潜变量 X ~ P_X 引起的层次混合效应：
1. PromptHeterogeneitySpec：离散题目难度/域分层高斯模型；
2. diagnose_simpson_confounding（定理 7A）：精确计算池化后验导数的
   “题内条件导数期望 + 跨题混杂协方差”分解，检验假单调遮蔽现象；
3. evaluate_pooled_vs_within_prompt_calibration（定理 7B）：在层次抽样下定量对比
   全局池化校准与题内 Z-score 标准化校准在不同偏差与预算 N 下的权衡与交叉点。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from tts_theory.algorithms import EmpiricalScoreCalibrator
from tts_theory.calibration_theory import exact_conditional_bin_bon_accuracy
from tts_theory.integrals import exact_bon_curve, exact_calibrated_bon_accuracy
from tts_theory.theorems import (
    GaussianVerifierSpec,
    _LOG_SQRT_2PI,
    _logsumexp_components,
    _norm_logpdf,
    gaussian_peak_threshold,
)


@dataclass(frozen=True)
class PromptHeterogeneitySpec:
    """离散分层题目异质性生成—验证概率模型。

    共 G 类题目，权重 prompt_weights (pi_1, ..., pi_G)，每层 g 具有真实先验正确率 p_g，
    偏移 b_g 与尺度 a_g > 0：
        S | (Y=1, X=g) ~ N(b_g + a_g * mu1_base, (a_g * sigma1_base)^2)
        S | (Y=0, X=g) ~ N(b_g + a_g * mu0_base, (a_g * sigma0_base)^2)
    """

    prompt_weights: ArrayLike
    prompt_p: ArrayLike
    prompt_shift: ArrayLike
    prompt_scale: ArrayLike | None = None
    mu1_base: float = 1.2
    sigma1_base: float = 1.0
    mu0_base: float = 0.0
    sigma0_base: float = 1.5

    def __post_init__(self) -> None:
        w = np.asarray(self.prompt_weights, dtype=np.float64).ravel()
        p_arr = np.asarray(self.prompt_p, dtype=np.float64).ravel()
        b_arr = np.asarray(self.prompt_shift, dtype=np.float64).ravel()
        if w.size == 0 or p_arr.shape != w.shape or b_arr.shape != w.shape:
            raise ValueError("prompt_weights, prompt_p, prompt_shift 必须为非空等长一维数组")
        if np.any(w <= 0.0):
            raise ValueError("prompt_weights 必须严格为正")
        w = w / float(np.sum(w))
        if np.any((p_arr <= 0.0) | (p_arr >= 1.0)):
            raise ValueError("prompt_p 每个分量必须位于 (0, 1) 内")
        if self.prompt_scale is None:
            a_arr = np.ones_like(w, dtype=np.float64)
        else:
            a_arr = np.asarray(self.prompt_scale, dtype=np.float64).ravel()
            if a_arr.shape != w.shape or np.any(a_arr <= 0.0):
                raise ValueError("prompt_scale 必须与 prompt_weights 等长且严格大于 0")
        if self.sigma1_base <= 0.0 or self.sigma0_base <= 0.0:
            raise ValueError("sigma1_base 与 sigma0_base 必须严格大于 0")

        object.__setattr__(self, "prompt_weights", w)
        object.__setattr__(self, "prompt_p", p_arr)
        object.__setattr__(self, "prompt_shift", b_arr)
        object.__setattr__(self, "prompt_scale", a_arr)

    @property
    def num_strata(self) -> int:
        return int(np.asarray(self.prompt_weights).size)

    @property
    def weights_arr(self) -> np.ndarray:
        return np.asarray(self.prompt_weights, dtype=np.float64)

    @property
    def p_arr(self) -> np.ndarray:
        return np.asarray(self.prompt_p, dtype=np.float64)

    @property
    def shift_arr(self) -> np.ndarray:
        return np.asarray(self.prompt_shift, dtype=np.float64)

    @property
    def scale_arr(self) -> np.ndarray:
        return np.asarray(self.prompt_scale, dtype=np.float64)

    def stratum_spec(self, g: int) -> GaussianVerifierSpec:
        """返回第 g 层题目的精确单题 GaussianVerifierSpec。"""
        a = float(self.scale_arr[g])
        b = float(self.shift_arr[g])
        return GaussianVerifierSpec(
            p=float(self.p_arr[g]),
            mu1=b + a * float(self.mu1_base),
            sigma1=a * float(self.sigma1_base),
            mu0=b + a * float(self.mu0_base),
            sigma0=a * float(self.sigma0_base),
        )

    def stratum_stats(self, s: ArrayLike) -> dict[str, np.ndarray]:
        """批量计算全部 G 层在网格 s 上的混合密度、后验、题内导数与对数密度导数。"""
        s_arr = np.asarray(s, dtype=np.float64)
        flat = s_arr.ravel()
        g_cnt = self.num_strata

        f_mix_g = np.empty((g_cnt, flat.size), dtype=np.float64)
        q_g = np.empty((g_cnt, flat.size), dtype=np.float64)
        dq_g = np.empty((g_cnt, flat.size), dtype=np.float64)
        g_log_deriv = np.empty((g_cnt, flat.size), dtype=np.float64)

        for g in range(g_cnt):
            sp = self.stratum_spec(g)
            lpdf1 = _norm_logpdf(flat, sp.mu1, sp.sigma1)
            lpdf0 = _norm_logpdf(flat, sp.mu0, sp.sigma0)
            lmix = _logsumexp_components(
                [math.log(sp.p) + lpdf1, math.log1p(-sp.p) + lpdf0]
            )
            fmix = np.exp(lmix)
            qg = np.asarray(sp.posterior(flat), dtype=np.float64)
            # 精确高斯对数似然比导数 l'_g(s) = (s-mu0)/sig0^2 - (s-mu1)/sig1^2
            llr_d = (flat - sp.mu0) / (sp.sigma0**2) - (flat - sp.mu1) / (sp.sigma1**2)
            dqg = qg * (1.0 - qg) * llr_d
            # d/ds ln f_{mix,g}(s) = q_g * (- (s-mu1)/sig1^2) + (1-q_g) * (- (s-mu0)/sig0^2)
            log_d = -qg * ((flat - sp.mu1) / (sp.sigma1**2)) - (1.0 - qg) * (
                (flat - sp.mu0) / (sp.sigma0**2)
            )

            f_mix_g[g] = fmix
            q_g[g] = qg
            dq_g[g] = dqg
            g_log_deriv[g] = log_d

        w_prior = self.weights_arr[:, None]
        weighted_f = w_prior * f_mix_g
        f_pool = np.maximum(1e-300, np.sum(weighted_f, axis=0))
        w_post = weighted_f / f_pool[None, :]
        q_pool = np.sum(w_post * q_g, axis=0)

        return {
            "f_mix_g": f_mix_g,
            "q_g": q_g,
            "dq_g": dq_g,
            "g_log_deriv": g_log_deriv,
            "f_pool": f_pool,
            "w_post": w_post,
            "q_pool": q_pool,
        }

    def pooled_posterior(self, s: ArrayLike) -> np.ndarray:
        """计算全局池化后验 q_pool(s) = P(Y=1 | S=s)。"""
        s_arr = np.asarray(s, dtype=np.float64)
        st = self.stratum_stats(s_arr)
        out = st["q_pool"].reshape(s_arr.shape)
        return float(out) if s_arr.ndim == 0 else out

    def pooled_cdf(self, s: ArrayLike) -> np.ndarray:
        """计算全局池化混合分布函数 F_pool(s)。"""
        s_arr = np.asarray(s, dtype=np.float64)
        res = np.zeros_like(s_arr, dtype=np.float64)
        for g, w_g in enumerate(self.weights_arr):
            res += float(w_g) * np.asarray(self.stratum_spec(g).mix_cdf(s_arr), dtype=np.float64)
        return float(res) if s_arr.ndim == 0 else res

    def exact_within_prompt_raw_bon(self, n_values: Sequence[int] | np.ndarray) -> np.ndarray:
        """计算未校准 Raw-BoN 在每道题内部抽样后的总体期望准确率。"""
        n_arr = np.asarray(list(n_values), dtype=np.int64)
        acc = np.zeros(n_arr.size, dtype=np.float64)
        for g, w_g in enumerate(self.weights_arr):
            acc += float(w_g) * exact_bon_curve(self.stratum_spec(g), n_arr)
        return acc

    def exact_within_prompt_oracle_calibrated_bon(
        self, n_values: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """计算已知每题自身正确后验 q_g(s) 排序时的 Oracle 题内校准 BoN 准确率。"""
        n_arr = np.asarray(list(n_values), dtype=np.int64)
        acc = np.zeros(n_arr.size, dtype=np.float64)
        for g, w_g in enumerate(self.weights_arr):
            sp = self.stratum_spec(g)
            acc += float(w_g) * np.array(
                [exact_calibrated_bon_accuracy(sp, int(n)) for n in n_arr],
                dtype=np.float64,
            )
        return acc

    def exact_pooled_calibrated_bon(
        self, n_values: Sequence[int] | np.ndarray, n_grid: int = 800
    ) -> np.ndarray:
        """用高精细分层分位桶近似全局池化后验 q_pool(S) 排序下的单题 BoN 准确率。"""
        n_arr = np.asarray(list(n_values), dtype=np.int64)
        acc = np.zeros(n_arr.size, dtype=np.float64)
        probs = np.linspace(0.0, 1.0, n_grid + 1)
        mids = 0.5 * (probs[:-1] + probs[1:])
        pi_uniform = np.full(n_grid, 1.0 / float(n_grid), dtype=np.float64)

        for g, w_g in enumerate(self.weights_arr):
            sp = self.stratum_spec(g)
            edges = np.asarray(sp.mix_ppf(probs[1:-1]), dtype=np.float64)
            centers = np.asarray(sp.mix_ppf(mids), dtype=np.float64)
            # 精确求各分位桶的题内真后验 q_g 与池化排序分 q_pool(center)
            cdf1 = np.concatenate([[0.0], np.asarray(sp.cdf1(edges), dtype=np.float64), [1.0]])
            cdf0 = np.concatenate([[0.0], np.asarray(sp.cdf0(edges), dtype=np.float64), [1.0]])
            m1 = np.maximum(0.0, np.diff(cdf1))
            m0 = np.maximum(0.0, np.diff(cdf0))
            denom = np.maximum(1e-300, sp.p * m1 + (1.0 - sp.p) * m0)
            q_true_bins = np.clip((sp.p * m1) / denom, 0.0, 1.0)
            est_pooled = np.asarray(self.pooled_posterior(centers), dtype=np.float64)
            acc += float(w_g) * exact_conditional_bin_bon_accuracy(
                q_true_bins, pi_uniform, est_pooled, n_arr
            )
        return acc


def diagnose_simpson_confounding(
    spec: PromptHeterogeneitySpec,
    s_grid: ArrayLike | None = None,
    work_quantile_range: tuple[float, float] = (0.05, 0.97),
) -> dict:
    """计算定理 7A 辛普森混杂导数精确分解并诊断假单调遮蔽现象。

    恒等式分解：
        q'_pool(s) = E[partial_s q_X(s) | S=s] + Cov(Y, partial_s ln f_{mix,X}(s) | S=s)
    并检验是否存在“单题内部必坍塌（kappa > 1），但全局池化后验却呈伪单调递增”的假单调遮蔽。
    """
    if s_grid is None:
        # 自动覆盖全局 [1%, 99.5%] 工作分位区间
        mus_all = []
        sigs_all = []
        for g in range(spec.num_strata):
            sp = spec.stratum_spec(g)
            mus_all.extend([sp.mu1, sp.mu0])
            sigs_all.extend([sp.sigma1, sp.sigma0])
        lo = float(min(mus_all) - 1.5 * max(sigs_all))
        hi = float(max(mus_all) + 2.8 * max(sigs_all))
        grid = np.linspace(lo, hi, 240)
    else:
        grid = np.asarray(s_grid, dtype=np.float64).ravel()

    st = spec.stratum_stats(grid)
    w_post = st["w_post"]
    q_g = st["q_g"]
    dq_g = st["dq_g"]
    g_d = st["g_log_deriv"]
    q_pool = st["q_pool"]

    within_term = np.sum(w_post * dq_g, axis=0)
    mean_g_d = np.sum(w_post * g_d, axis=0)
    between_cov_term = np.sum(w_post * q_g * (g_d - mean_g_d[None, :]), axis=0)
    analytic_total = within_term + between_cov_term

    # 用中心有限差分交叉校验解析分解无误
    eps = 1e-6
    q_plus = np.asarray(spec.pooled_posterior(grid + eps), dtype=np.float64)
    q_minus = np.asarray(spec.pooled_posterior(grid - eps), dtype=np.float64)
    fd_deriv = (q_plus - q_minus) / (2.0 * eps)
    decomposition_max_err = float(np.max(np.abs(analytic_total - fd_deriv)))

    within_prompt_collapses = bool(spec.sigma0_base > spec.sigma1_base + 1e-12)
    f_cdf = np.asarray(spec.pooled_cdf(grid), dtype=np.float64)
    q_lo, q_hi = float(work_quantile_range[0]), float(work_quantile_range[1])
    work_mask = (f_cdf >= q_lo) & (f_cdf <= q_hi)
    if not np.any(work_mask):
        work_mask = np.ones_like(grid, dtype=bool)
    min_deriv_work = float(np.min(analytic_total[work_mask]))
    pooled_appears_monotone = bool(min_deriv_work > 0.0)
    # 存在题内期望导数为负却被跨题正协方差掩盖为正的点
    masked_points = int(np.count_nonzero((within_term < -1e-6) & (analytic_total > 1e-6) & work_mask))

    return {
        "s_grid": grid,
        "q_pool": q_pool,
        "q_pool_deriv": analytic_total,
        "within_slope_term": within_term,
        "between_cov_term": between_cov_term,
        "decomposition_max_abs_err": decomposition_max_err,
        "within_prompt_collapses": within_prompt_collapses,
        "min_pooled_deriv_working_range": min_deriv_work,
        "pooled_appears_monotone": pooled_appears_monotone,
        "false_monotonicity_masking": bool(within_prompt_collapses and pooled_appears_monotone),
        "masked_point_count": masked_points,
    }


def _sample_prompt_batch(
    spec: PromptHeterogeneitySpec,
    n_prompts: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """抽取层次推理池：返回形状 (n_prompts, n_candidates) 的 scores 与 labels。"""
    strata = rng.choice(spec.num_strata, size=n_prompts, p=spec.weights_arr)
    p_per_prompt = spec.p_arr[strata][:, None]
    a_per_prompt = spec.scale_arr[strata][:, None]
    b_per_prompt = spec.shift_arr[strata][:, None]

    labels = rng.random((n_prompts, n_candidates)) < p_per_prompt
    z = rng.normal(0.0, 1.0, size=(n_prompts, n_candidates))
    raw_base = np.where(
        labels,
        float(spec.mu1_base) + float(spec.sigma1_base) * z,
        float(spec.mu0_base) + float(spec.sigma0_base) * z,
    )
    scores = b_per_prompt + a_per_prompt * raw_base
    return scores, labels


def _within_prompt_zscore(scores: np.ndarray) -> np.ndarray:
    """对每行（单道题内 N 个样本）做 Z-score 标准化（零方差行精确退化为全 0）。"""
    mean = np.mean(scores, axis=1, keepdims=True)
    if scores.shape[1] <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    std = np.std(scores, axis=1, ddof=1, keepdims=True)
    return np.where(std > 1e-12, (scores - mean) / np.maximum(std, 1e-12), 0.0)


def evaluate_pooled_vs_within_prompt_calibration(
    spec: PromptHeterogeneitySpec,
    n_values: Sequence[int] | np.ndarray,
    m_cal_prompts: int = 600,
    n_cal_per_prompt: int = 32,
    n_eval_prompts: int = 4_000,
    n_bins: int = 15,
    delta: float = 0.05,
    seed: int = 42,
) -> dict:
    """在层次题目异质性生成模型下端到端对比全局校准与题内 Z-标准化校准（定理 7B）。"""
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    if n_arr.ndim != 1 or n_arr.size == 0 or np.any(n_arr < 1):
        raise ValueError("n_values 必须为非空正整数序列")

    rng = np.random.default_rng(seed)
    cal_scores, cal_labels = _sample_prompt_batch(spec, int(m_cal_prompts), int(n_cal_per_prompt), rng)
    cal_z = _within_prompt_zscore(cal_scores)

    cal_pooled = EmpiricalScoreCalibrator(
        n_bins=n_bins, method="quantile_lcb", lcb_delta=delta, lcb_type="kl"
    ).fit(cal_scores.ravel(), cal_labels.ravel())
    cal_within = EmpiricalScoreCalibrator(
        n_bins=n_bins, method="quantile_lcb", lcb_delta=delta, lcb_type="kl"
    ).fit(cal_z.ravel(), cal_labels.ravel())

    n_max = int(np.max(n_arr))
    eval_scores, eval_labels = _sample_prompt_batch(spec, int(n_eval_prompts), n_max, rng)
    row_idx = np.arange(int(n_eval_prompts))

    def _pick_acc(score_mat: np.ndarray, sub_lbl: np.ndarray) -> float:
        max_v = np.max(score_mat, axis=1, keepdims=True)
        is_m = np.abs(score_mat - max_v) <= 1e-13
        tie = np.where(is_m, rng.random(score_mat.shape), -1.0)
        idx = np.argmax(tie, axis=1)
        return float(np.mean(sub_lbl[row_idx, idx]))

    acc_raw = []
    acc_pooled_cal = []
    acc_within_z = []

    for nv in n_arr:
        n_int = int(nv)
        sub_s = eval_scores[:, :n_int]
        sub_y = eval_labels[:, :n_int]

        acc_raw.append(_pick_acc(sub_s, sub_y))
        pooled_q = np.asarray(cal_pooled.predict_proba(sub_s), dtype=np.float64)
        acc_pooled_cal.append(_pick_acc(pooled_q, sub_y))

        if n_int == 1:
            acc_within_z.append(float(np.mean(sub_y[:, 0])))
        else:
            z_sub = _within_prompt_zscore(sub_s)
            within_q = np.asarray(cal_within.predict_proba(z_sub), dtype=np.float64)
            acc_within_z.append(_pick_acc(within_q, sub_y))

    return {
        "n_values": n_arr,
        "raw_bon_mc": np.asarray(acc_raw, dtype=np.float64),
        "pooled_kl_lcb_mc": np.asarray(acc_pooled_cal, dtype=np.float64),
        "within_prompt_z_kl_lcb_mc": np.asarray(acc_within_z, dtype=np.float64),
        "oracle_within_prompt_exact": spec.exact_within_prompt_oracle_calibrated_bon(n_arr),
        "oracle_raw_bon_exact": spec.exact_within_prompt_raw_bon(n_arr),
    }


def diagnose_within_prompt_residual(spec: PromptHeterogeneitySpec) -> dict:
    """诊断题内 Z-Score 标准化的先验诱导残差偏移与 N=2 纯秩天花板（定理 7B）。

    明确区分两类情形（对一般基底分布，样本标准差 s_hat_N 的 O_p(N^{-1/2}) 收敛
    额外依赖有限四阶矩 E[S_tilde^4 | X] < inf；高斯族自动满足）：
    - 恒定先验 p(X) == p_0：消去仿射偏移 (b(X), a(X)) 后残差为零，后验跨题完全对齐；
    - 异质先验 p(X)：消去 (b(X), a(X)) 后仍保留由 p(X) 诱导的混合矩偏移与峰值 z_peak(X) 漂移。
    同时给出 N=2 时三种纯秩规则（Raw-BoN, Anti-BoN, Random）的理论天花板对比 Oracle 校准。
    """
    w = spec.weights_arr
    p_arr = spec.p_arr
    a_arr = spec.scale_arr
    b_arr = spec.shift_arr

    dmu_base = float(spec.mu1_base - spec.mu0_base)
    mu_tilde = float(spec.mu0_base) + p_arr * dmu_base
    var_tilde = (
        p_arr * (float(spec.sigma1_base) ** 2)
        + (1.0 - p_arr) * (float(spec.sigma0_base) ** 2)
        + p_arr * (1.0 - p_arr) * (dmu_base**2)
    )
    std_tilde = np.sqrt(var_tilde)

    if spec.sigma0_base > spec.sigma1_base:
        base_spec = GaussianVerifierSpec(
            p=float(p_arr[0]),
            mu1=float(spec.mu1_base),
            sigma1=float(spec.sigma1_base),
            mu0=float(spec.mu0_base),
            sigma0=float(spec.sigma0_base),
        )
        x_peak_base: float | None = gaussian_peak_threshold(base_spec)
        z_peak: np.ndarray | None = (x_peak_base - mu_tilde) / std_tilde
        peak_z_span = float(np.max(z_peak) - np.min(z_peak))
    else:
        x_peak_base = None
        z_peak = None
        peak_z_span = 0.0

    def _weighted_var(vals: np.ndarray) -> float:
        mean_v = float(np.sum(w * vals))
        return float(np.sum(w * (vals - mean_v) ** 2))

    b_norm = b_arr / a_arr
    affine_shift_var = _weighted_var(b_norm)
    prior_induced_shift_var = _weighted_var(mu_tilde)
    is_constant_prior = bool(np.ptp(p_arr) <= 1e-12)

    mean_prior_p = float(np.sum(w * p_arr))
    raw_bon_n2 = float(spec.exact_within_prompt_raw_bon([2])[0])
    anti_bon_n2 = float(2.0 * mean_prior_p - raw_bon_n2)
    rank_only_n2_ceiling = float(max(raw_bon_n2, anti_bon_n2, mean_prior_p))
    oracle_cal_n2 = float(spec.exact_within_prompt_oracle_calibrated_bon([2])[0])
    rank_vs_oracle_n2_gap = float(max(0.0, oracle_cal_n2 - rank_only_n2_ceiling))

    return {
        "is_constant_prior": is_constant_prior,
        "mu_tilde_by_stratum": mu_tilde,
        "std_tilde_by_stratum": std_tilde,
        "x_peak_base": x_peak_base,
        "z_peak_by_stratum": z_peak,
        "peak_z_span": peak_z_span,
        "affine_shift_var": affine_shift_var,
        "prior_induced_shift_var": prior_induced_shift_var,
        "mean_prior_p": mean_prior_p,
        "raw_bon_n2": raw_bon_n2,
        "anti_bon_n2": anti_bon_n2,
        "rank_only_n2_ceiling": rank_only_n2_ceiling,
        "oracle_cal_n2": oracle_cal_n2,
        "rank_vs_oracle_n2_gap": rank_vs_oracle_n2_gap,
        "strict_gap_over_rank_n2": bool(rank_vs_oracle_n2_gap > 1e-10),
    }


def diagnose_zscore_induced_collapse(
    prompt_weights: ArrayLike,
    prompt_p: ArrayLike,
    mu1_base: float = 1.5,
    sigma1_base: float = 1.0,
    mu0_base: float = 0.0,
    sigma0_base: float = 1.0,
    z_grid: ArrayLike | None = None,
    constrain_nonnegative_z: bool = False,
) -> dict:
    """诊断总体题内 Z-Score 标准化下的右尾竞争反转与渐近上限损失（定理 7C）。

    在支持边界原子层 p_g in {0.0, 1.0} 的同方差单调高斯基底上：
    1. 核心尾部反转效应（无条件成立）：边界原子方差缩放为 1，而所有内点可解题承受方差压缩
       v(p) < 1，因此在 Z*-空间精确导出右尾极限 pi_1 / (pi_0 + pi_1)（当 pi_0 > 0 = pi_1 时
       跌向 0 并导致 Oracle Z*-校准器在大 N 下的渐近上限低于 Raw-BoN）；
       当 pi_0 = 0, pi_1 > 0 且无右半轴约束（constrain_nonnegative_z=False）时，若多内点层
       触发高先验层左尾反超条件，亦会产生正的渐近上限损失。
    2. 全方差分解中的总体异方差比 kappa_Z：一般情形下 Var(Z*|Y=c) = E[v(p)|Y=c] + Var(m_c|Y=c)，
       多内点层且 Delta 足够大时可能出现 kappa_Z < 1（虽右尾仍趋于 0）；在单内点可解层 {0, p_1}
       （pi_0 > 0, Delta > 0）下因 Var(m_1 | Y=1) = 0 而严格满足 kappa_Z > 1。
    """
    w = np.asarray(prompt_weights, dtype=np.float64).ravel()
    p_arr = np.asarray(prompt_p, dtype=np.float64).ravel()
    if w.size == 0 or p_arr.shape != w.shape or np.any(w <= 0.0):
        raise ValueError("prompt_weights 与 prompt_p 必须为非空等长正权数组")
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("prompt_p 分量必须位于闭区间 [0, 1] 内")
    w = w / float(np.sum(w))
    p_bar = float(np.sum(w * p_arr))
    if not (0.0 < p_bar < 1.0):
        raise ValueError("混合先验均正确率必须严格位于 (0, 1) 内")
    if float(sigma1_base) <= 0.0 or float(sigma0_base) <= 0.0:
        raise ValueError("sigma1_base 与 sigma0_base 必须严格为正")

    if z_grid is None:
        grid = np.linspace(-4.0, 8.0, 1201, dtype=np.float64)
    else:
        grid = np.asarray(z_grid, dtype=np.float64).ravel()
        if grid.size < 5:
            raise ValueError("z_grid 至少需要 5 个网格点")

    s1 = float(sigma1_base)
    s0 = float(sigma0_base)
    dmu = float(mu1_base) - float(mu0_base)
    mu_tilde = float(mu0_base) + p_arr * dmu
    var_tilde = p_arr * (s1 * s1) + (1.0 - p_arr) * (s0 * s0) + p_arr * (1.0 - p_arr) * (dmu * dmu)
    std_tilde = np.sqrt(var_tilde)

    m1_g = (float(mu1_base) - mu_tilde) / std_tilde
    sd1_g = s1 / std_tilde
    m0_g = (float(mu0_base) - mu_tilde) / std_tilde
    sd0_g = s0 / std_tilde

    # 由全方差公式精确分解 Z*-空间条件方差（组内方差均值 + 组间均值方差）
    alpha1 = (w * p_arr) / p_bar
    alpha0 = (w * (1.0 - p_arr)) / (1.0 - p_bar)
    ez_pos = float(np.sum(alpha1 * m1_g))
    ez_neg = float(np.sum(alpha0 * m0_g))
    within_var_z_pos = float(np.sum(alpha1 * (sd1_g * sd1_g)))
    between_var_z_pos = float(np.sum(alpha1 * ((m1_g - ez_pos) ** 2)))
    within_var_z_neg = float(np.sum(alpha0 * (sd0_g * sd0_g)))
    between_var_z_neg = float(np.sum(alpha0 * ((m0_g - ez_neg) ** 2)))
    var_z_pos = within_var_z_pos + between_var_z_pos
    var_z_neg = within_var_z_neg + between_var_z_neg
    kappa_z = math.sqrt(max(var_z_neg, 1e-30) / max(var_z_pos, 1e-30))

    # 精确解析计算 Z*-空间池化密度与后验 q_{pool, Z}(z)
    pos_idx = np.flatnonzero(p_arr > 0.0)
    neg_idx = np.flatnonzero(p_arr < 1.0)
    log_w_pos = np.log(w[pos_idx] * p_arr[pos_idx])
    log_w_neg = np.log(w[neg_idx] * (1.0 - p_arr[neg_idx]))

    def _eval_z_stats(z_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (q_pool_z, logit_q_pool_z, dq_dz) 避免极端尾部饱和导致的假峰值。"""
        lp_comps = np.array(
            [float(log_w_pos[j]) + _norm_logpdf(z_pts, float(m1_g[g]), float(sd1_g[g])) for j, g in enumerate(pos_idx)]
        )
        ln_comps = np.array(
            [float(log_w_neg[j]) + _norm_logpdf(z_pts, float(m0_g[g]), float(sd0_g[g])) for j, g in enumerate(neg_idx)]
        )
        lp = _logsumexp_components(lp_comps)
        ln = _logsumexp_components(ln_comps)
        logit_q = lp - ln
        q_vals = np.asarray(special.expit(logit_q), dtype=np.float64)

        # 精确对数导数：d/dz ln f_pos - d/dz ln f_neg
        w1 = np.exp(lp_comps - lp[None, :])
        w0 = np.exp(ln_comps - ln[None, :])
        d_pos = np.sum(w1 * (-(z_pts[None, :] - m1_g[pos_idx, None]) / (sd1_g[pos_idx, None] ** 2)), axis=0)
        d_neg = np.sum(w0 * (-(z_pts[None, :] - m0_g[neg_idx, None]) / (sd0_g[neg_idx, None] ** 2)), axis=0)
        llr_deriv = d_pos - d_neg
        return q_vals, logit_q, q_vals * (1.0 - q_vals) * llr_deriv

    q_pool_z, logit_grid, deriv_grid = _eval_z_stats(grid)

    pi_0 = float(np.sum(w[p_arr <= 1e-15]))
    pi_1 = float(np.sum(w[p_arr >= 1.0 - 1e-15]))
    is_homoscedastic_boundary = bool((pi_0 + pi_1 > 0.0) and abs(s1 - s0) <= 1e-12)
    if is_homoscedastic_boundary:
        z_tail_limit = float(pi_1 / (pi_0 + pi_1))
    else:
        min_rate_pos = float(np.min(0.5 / (sd1_g[pos_idx] ** 2)))
        min_rate_neg = float(np.min(0.5 / (sd0_g[neg_idx] ** 2)))
        if abs(min_rate_pos - min_rate_neg) > 1e-12:
            z_tail_limit = 1.0 if min_rate_pos < min_rate_neg else 0.0
        else:
            z_tail_limit = float(q_pool_z[-1])

    # 实用窗口严格差分单调性检验：逐点检查 q_pool_z 在 grid 上是否全局单调不减，
    # 诚实捕获左端浅谷回撤（例如 Case 2 在 [-4, -3.68] 微降至 0.5233，仅在 [-3.6, 8] 严格单调）
    grid_peak_idx = int(np.argmax(logit_grid))
    practical_window_monotone = bool(np.all(np.diff(q_pool_z) >= -1e-12))
    right_prac_mask = grid >= (-3.6 - 1e-12)
    practical_right_window_monotone = (
        bool(np.all(np.diff(q_pool_z[right_prac_mask]) >= -1e-12))
        if int(np.sum(right_prac_mask)) >= 2
        else practical_window_monotone
    )
    prac_min_idx = int(np.argmin(q_pool_z))
    practical_window_min_z = float(grid[prac_min_idx])
    practical_window_min_q = float(q_pool_z[prac_min_idx])

    # 全局理论峰检测：当默认网格未覆盖真峰（如高 pi_1 把 K* 推向右尾 z > 8）时自动扩界搜索
    search_grid = grid
    search_q = q_pool_z
    search_logit = logit_grid
    search_deriv = deriv_grid
    need_adaptive = bool(
        (z_grid is None)
        or (
            is_homoscedastic_boundary
            and pi_0 > 0.0
            and dmu > 0.0
            and (grid_peak_idx in (0, grid.size - 1) or float(deriv_grid[-1]) > 0.0)
        )
    )
    if need_adaptive and not (is_homoscedastic_boundary and pi_0 == 0.0 and pi_1 > 0.0 and dmu > 0.0):
        hi_ext = max(30.0, float(grid[-1]))
        lo_ext = min(-30.0, float(grid[0]))
        for _ in range(6):
            cand_grid = np.linspace(lo_ext, hi_ext, 12001, dtype=np.float64)
            cand_q, cand_logit, cand_deriv = _eval_z_stats(cand_grid)
            c_idx = int(np.argmax(cand_logit))
            # 若仍在右端上升或未超越 L_tail（且理论保证 pi_0 > 0 下必超越），继续右扩界
            if is_homoscedastic_boundary and pi_0 > 0.0 and dmu > 0.0:
                if c_idx >= cand_grid.size - 2 or float(cand_deriv[-1]) > 0.0 or float(cand_q[c_idx]) <= z_tail_limit + 1e-12:
                    hi_ext *= 2.0
                    continue
            search_grid, search_q, search_logit, search_deriv = cand_grid, cand_q, cand_logit, cand_deriv
            break
        else:
            search_grid, search_q, search_logit, search_deriv = cand_grid, cand_q, cand_logit, cand_deriv

    if is_homoscedastic_boundary and pi_0 == 0.0 and pi_1 > 0.0 and dmu > 0.0:
        # Theorem 3(1)(a)：L_tail = 1，所有有限点 q_{pool,Z*}(z) < 1，无紧致全局极大值点
        peak_idx = int(np.argmax(q_pool_z))
        z_peak = float(grid[peak_idx])
        q_z_peak = float(q_pool_z[peak_idx])
        tail_turn_down = False
        peak_minus_tail_limit = float(q_z_peak - z_tail_limit)
        peak_exceeds_tail_limit = False
        compact_maximizer_exists = False
        boundary_dominated_regime = False
    else:
        s_peak_idx = int(np.argmax(search_logit))
        z_peak = float(search_grid[s_peak_idx])
        q_z_peak = float(search_q[s_peak_idx])
        # 若为严格内点峰，用有界一维寻优精确锁定 K* 坐标
        if 0 < s_peak_idx < search_grid.size - 1:
            from scipy import optimize

            opt_res = optimize.minimize_scalar(
                lambda zz: -float(_eval_z_stats(np.array([zz], dtype=np.float64))[1][0]),
                bounds=(float(search_grid[s_peak_idx - 1]), float(search_grid[s_peak_idx + 1])),
                method="bounded",
                options={"xatol": 1e-8},
            )
            if opt_res.success:
                z_opt = float(opt_res.x)
                q_opt, logit_opt, _ = _eval_z_stats(np.array([z_opt], dtype=np.float64))
                if float(logit_opt[0]) >= float(search_logit[s_peak_idx]):
                    z_peak = z_opt
                    q_z_peak = float(q_opt[0])

        tail_turn_down = bool(float(search_q[-1]) < q_z_peak - 1e-3)
        peak_minus_tail_limit = float(q_z_peak - z_tail_limit)
        peak_exceeds_tail_limit = bool(pi_0 > 0.0 and peak_minus_tail_limit > 1e-10)
        compact_maximizer_exists = bool(peak_exceeds_tail_limit and is_homoscedastic_boundary)
        boundary_dominated_regime = bool(pi_0 > 0.0 and not peak_exceeds_tail_limit)

    if s0 < s1 - 1e-12 or (abs(s0 - s1) <= 1e-12 and dmu > 0.0):
        raw_bon_asymptotic_acc = float(1.0 - pi_0)
    elif abs(s0 - s1) <= 1e-12 and abs(dmu) <= 1e-12:
        raw_bon_asymptotic_acc = p_bar
    else:
        raw_bon_asymptotic_acc = float(pi_1)

    # 总体标准化 Z* 空间 Oracle 校准器在极限处选择 K* 附近样本；
    # 仅当无紧致全局极大值且（启用 Z>=0 约束、至多单内点层、或未触发多内点左尾反超）时渐近无上限损失；
    # 若 pi_0 = 0, pi_1 > 0 且无非负约束并在多内点下触发左尾反超，则逐层判定反超损失。
    solvable_interior = (p_arr > 0.0) & (p_arr < 1.0)
    int_indices = np.flatnonzero(solvable_interior)
    overtaken_strata: list[int] = []
    left_tail_overtake = False
    if (not compact_maximizer_exists) and is_homoscedastic_boundary and (pi_0 <= 1e-15) and (pi_1 > 0.0) and dmu > 0.0:
        if (not constrain_nonnegative_z) and int_indices.size >= 2:
            v_int = sd1_g[int_indices] ** 2
            v_max = float(np.max(v_int))
            vmax_mask = v_int >= (v_max - 1e-12)
            abs_m0_vmax = np.abs(m0_g[int_indices[vmax_mask]])
            k_plus_abs_m0 = float(np.min(abs_m0_vmax))
            k_minus_abs_m0 = float(np.max(abs_m0_vmax))
            for g_val in int_indices:
                g = int(g_val)
                overtake_margin = (1.0 - v_max) * (abs(float(m0_g[g])) - float(m1_g[g])) - (
                    k_plus_abs_m0 + k_minus_abs_m0
                )
                if overtake_margin > 1e-12:
                    overtaken_strata.append(g)
                elif abs(overtake_margin) <= 1e-12:
                    # 精确边界平局时，使用极大渐近偏差 u=50 在对数几率域判定左右极值先后
                    u_probe = 50.0
                    z_pair = np.array(
                        [
                            float(m1_g[g]) + float(sd1_g[g]) * u_probe,
                            float(m0_g[g]) - float(sd0_g[g]) * u_probe,
                        ],
                        dtype=np.float64,
                    )
                    _, logit_pair, _ = _eval_z_stats(z_pair)
                    if float(logit_pair[1]) > float(logit_pair[0]) + 1e-9:
                        overtaken_strata.append(g)
        left_tail_overtake = len(overtaken_strata) > 0
        if not left_tail_overtake:
            oracle_z_acc = float(raw_bon_asymptotic_acc)
            ceiling_loss = 0.0
            has_ceiling_loss = False
        else:
            overtaken_set = set(overtaken_strata)
            oracle_z_acc = float(
                pi_1 + sum(float(w[int(g)]) for g in int_indices if int(g) not in overtaken_set)
            )
            ceiling_loss = float(raw_bon_asymptotic_acc - oracle_z_acc)
            has_ceiling_loss = bool(ceiling_loss > 1e-4)
    else:
        oracle_z_acc = float(pi_1)
        if np.any(solvable_interior):
            # 若存在多个离散全局极大值点 |K*| >= 2，此处取优化器定位的代表性极大值 z_peak ∈ K* 评估极限准确度
            for g in int_indices:
                lp_g = math.log(float(p_arr[g])) + _norm_logpdf(
                    np.array([z_peak]), float(m1_g[g]), float(sd1_g[g])
                )[0]
                ln_g = math.log(float(1.0 - p_arr[g])) + _norm_logpdf(
                    np.array([z_peak]), float(m0_g[g]), float(sd0_g[g])
                )[0]
                q_at_peak = float(special.expit(lp_g - ln_g))
                oracle_z_acc += float(w[g]) * q_at_peak

        ceiling_loss = float(raw_bon_asymptotic_acc - float(oracle_z_acc))
        has_ceiling_loss = bool(tail_turn_down and ceiling_loss > 1e-4)

    return {
        "z_grid": grid,
        "q_pool_z": q_pool_z,
        "pi_0_unsolvable": pi_0,
        "pi_1_trivial": pi_1,
        "p_bar": p_bar,
        "within_var_z_pos": within_var_z_pos,
        "between_var_z_pos": between_var_z_pos,
        "within_var_z_neg": within_var_z_neg,
        "between_var_z_neg": between_var_z_neg,
        "var_z_pos": var_z_pos,
        "var_z_neg": var_z_neg,
        "kappa_z": kappa_z,
        "z_peak": z_peak,
        "q_z_peak": q_z_peak,
        "z_tail_value": float(q_pool_z[-1]),
        "z_tail_posterior_limit": z_tail_limit,
        "peak_minus_tail_limit": peak_minus_tail_limit,
        "peak_exceeds_tail_limit": peak_exceeds_tail_limit,
        "compact_maximizer_exists": compact_maximizer_exists,
        "boundary_dominated_regime": boundary_dominated_regime,
        "practical_window_monotone": practical_window_monotone,
        "practical_right_window_monotone": practical_right_window_monotone,
        "practical_window_min_z": practical_window_min_z,
        "practical_window_min_q": practical_window_min_q,
        "constrain_nonnegative_z": bool(constrain_nonnegative_z),
        "left_tail_overtake_trap": left_tail_overtake,
        "left_tail_overtake_strata": overtaken_strata,
        "tail_turn_down": tail_turn_down,
        "raw_bon_asymptotic_acc": raw_bon_asymptotic_acc,
        "oracle_z_cal_asymptotic_acc": float(oracle_z_acc),
        "asymptotic_ceiling_loss": ceiling_loss,
        "has_asymptotic_ceiling_loss": has_ceiling_loss,
        "z_induces_collapse": has_ceiling_loss,
    }


def simulate_zscore_induced_collapse(
    prompt_weights: ArrayLike,
    prompt_p: ArrayLike,
    n_values: Sequence[int] | np.ndarray,
    mu1_base: float = 1.5,
    sigma1_base: float = 1.0,
    mu0_base: float = 0.0,
    sigma0_base: float = 1.0,
    m_cal_prompts: int = 1_500,
    n_cal_per_prompt: int = 32,
    n_eval_prompts: int = 1_500,
    n_bins: int = 20,
    delta: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """仿真支持边界原子 p_g in {0, 1} 的题内 Z-Score 诱导坍塌（定理 7C 端到端验证）。"""
    w = np.asarray(prompt_weights, dtype=np.float64).ravel()
    p_arr = np.asarray(prompt_p, dtype=np.float64).ravel()
    if w.size == 0 or p_arr.shape != w.shape or np.any(w <= 0.0):
        raise ValueError("prompt_weights 与 prompt_p 无效")
    w = w / float(np.sum(w))
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    if n_arr.ndim != 1 or n_arr.size == 0 or np.any(n_arr < 1):
        raise ValueError("n_values 必须为非空正整数序列")

    rng = np.random.default_rng(seed)

    def _draw(nprom: int, ncand: int) -> tuple[np.ndarray, np.ndarray]:
        st = rng.choice(w.size, size=nprom, p=w)
        pp = p_arr[st][:, None]
        lbl = rng.random((nprom, ncand)) < pp
        sc = np.where(
            lbl,
            float(mu1_base) + float(sigma1_base) * rng.normal(size=(nprom, ncand)),
            float(mu0_base) + float(sigma0_base) * rng.normal(size=(nprom, ncand)),
        )
        return sc, lbl

    cal_s, cal_y = _draw(int(m_cal_prompts), int(n_cal_per_prompt))
    cal_z = _within_prompt_zscore(cal_s)
    cal_pooled = EmpiricalScoreCalibrator(
        n_bins=int(n_bins), method="quantile_lcb", lcb_delta=float(delta), lcb_type="kl"
    ).fit(cal_s.ravel(), cal_y.ravel())
    cal_within = EmpiricalScoreCalibrator(
        n_bins=int(n_bins), method="quantile_lcb", lcb_delta=float(delta), lcb_type="kl"
    ).fit(cal_z.ravel(), cal_y.ravel())

    eval_s, eval_y = _draw(int(n_eval_prompts), int(np.max(n_arr)))
    rows = np.arange(int(n_eval_prompts))

    def _pick(mat: np.ndarray, lbl: np.ndarray) -> float:
        mx = np.max(mat, axis=1, keepdims=True)
        mask = np.abs(mat - mx) <= 1e-13
        noise = np.where(mask, rng.random(mat.shape), -1.0)
        return float(np.mean(lbl[rows, np.argmax(noise, axis=1)]))

    acc_raw, acc_pooled, acc_within = [], [], []
    for nv in n_arr:
        n_i = int(nv)
        sub_s = eval_s[:, :n_i]
        sub_y = eval_y[:, :n_i]
        acc_raw.append(_pick(sub_s, sub_y))
        acc_pooled.append(_pick(np.asarray(cal_pooled.predict_proba(sub_s), dtype=np.float64), sub_y))
        if n_i == 1:
            acc_within.append(float(np.mean(sub_y[:, 0])))
        else:
            z_sub = _within_prompt_zscore(sub_s)
            acc_within.append(_pick(np.asarray(cal_within.predict_proba(z_sub), dtype=np.float64), sub_y))

    return {
        "n_values": n_arr,
        "standard_bon_mc": np.asarray(acc_raw, dtype=np.float64),
        "pooled_kl_lcb_mc": np.asarray(acc_pooled, dtype=np.float64),
        "within_prompt_z_kl_lcb_mc": np.asarray(acc_within, dtype=np.float64),
    }


def verify_sample_zscore_extremal_convergence(
    prompt_weights: ArrayLike,
    prompt_p: ArrayLike,
    n_values: Sequence[int] | np.ndarray = (16, 64, 256, 1024),
    mu1_base: float = 1.5,
    sigma1_base: float = 1.0,
    mu0_base: float = 0.0,
    sigma0_base: float = 1.0,
    prompt_shift: ArrayLike | None = None,
    prompt_scale: ArrayLike | None = None,
    n_eval_prompts: int = 3_000,
    seed: int = 42,
    constrain_nonnegative_z: bool = False,
) -> dict:
    """定量验证在线样本题内标准化 Z_{i,N} 对总体 Z_i^*(X) 的一致极值收敛与仿射不变上限损失（TMLR Theorem 3）。

    对任意题间仿射变换 S_i = b(X) + a(X) * S_tilde_i（a(X) > 0）与预算 N >= 2：
    1. 验证极值偏差上确界 Delta_N(X) := max_{1<=i<=N} |Z_{i,N} - Z_i^*(X)| 满足 O_p(sqrt((ln N) / N))；
    2. 核验精确仿射坐标恒等式：
       q_X(S_i) = q_X(b(X) + a(X) * [mu_tilde(X) + sigma_tilde(X) * Z_i^*(X)]) == q_tilde_X(mu_tilde(X) + sigma_tilde(X) * Z_i^*(X))，
       消除漏写 (b(X), a(X)) 导致的大偏移不一致；
    3. 验证在线样本 Z_{i,N} Oracle 选择准确率 lim_{N->inf} P_{Sample-Z-Cal}^{oracle}(N) 对任意 (b(X), a(X))
       严格不变且收敛至理论渐近上限 (< 1 - pi_0)。若 constrain_nonnegative_z=True，则在非负半轴 Z >= 0 上做右尾约束校准，防止多内点左尾反超陷阱。
    """
    w = np.asarray(prompt_weights, dtype=np.float64).ravel()
    p_arr = np.asarray(prompt_p, dtype=np.float64).ravel()
    if w.size == 0 or p_arr.shape != w.shape or np.any(w <= 0.0):
        raise ValueError("prompt_weights 与 prompt_p 必须为非空等长正权数组")
    w = w / float(np.sum(w))
    b_arr = np.zeros_like(w) if prompt_shift is None else np.asarray(prompt_shift, dtype=np.float64).ravel()
    a_arr = np.ones_like(w) if prompt_scale is None else np.asarray(prompt_scale, dtype=np.float64).ravel()
    if b_arr.shape != w.shape or a_arr.shape != w.shape or np.any(a_arr <= 0.0):
        raise ValueError("prompt_shift 与 prompt_scale 必须与 prompt_weights 等长且 prompt_scale > 0")
    n_arr = np.asarray(list(n_values), dtype=np.int64)
    if n_arr.ndim != 1 or n_arr.size == 0 or np.any(n_arr < 2):
        raise ValueError("n_values 必须为 >= 2 的整数序列")

    diag = diagnose_zscore_induced_collapse(
        prompt_weights=w,
        prompt_p=p_arr,
        mu1_base=mu1_base,
        sigma1_base=sigma1_base,
        mu0_base=mu0_base,
        sigma0_base=sigma0_base,
        constrain_nonnegative_z=constrain_nonnegative_z,
    )
    s1 = float(sigma1_base)
    s0 = float(sigma0_base)
    dmu = float(mu1_base) - float(mu0_base)
    mu_tilde = float(mu0_base) + p_arr * dmu
    std_tilde = np.sqrt(
        p_arr * (s1 * s1) + (1.0 - p_arr) * (s0 * s0) + p_arr * (1.0 - p_arr) * (dmu * dmu)
    )
    m1_g = (float(mu1_base) - mu_tilde) / std_tilde
    sd1_g = s1 / std_tilde
    m0_g = (float(mu0_base) - mu_tilde) / std_tilde
    sd0_g = s0 / std_tilde
    pos_idx = np.flatnonzero(p_arr > 0.0)
    neg_idx = np.flatnonzero(p_arr < 1.0)

    def _eval_logit_q_pool_zstar(z_mat: np.ndarray) -> np.ndarray:
        """返回 logit(q_{pool,Z*}(z)) 以避免极端尾部 expit 饱和导致误平局。"""
        flat = z_mat.ravel()
        lp = _logsumexp_components(
            [math.log(float(w[g] * p_arr[g])) + _norm_logpdf(flat, float(m1_g[g]), float(sd1_g[g])) for g in pos_idx]
        )
        ln = _logsumexp_components(
            [
                math.log(float(w[g] * (1.0 - p_arr[g]))) + _norm_logpdf(flat, float(m0_g[g]), float(sd0_g[g]))
                for g in neg_idx
            ]
        )
        return (lp - ln).reshape(z_mat.shape)

    rng = np.random.default_rng(seed)
    n_max = int(np.max(n_arr))
    strata = rng.choice(w.size, size=int(n_eval_prompts), p=w)
    pp = p_arr[strata][:, None]
    b_row = b_arr[strata][:, None]
    a_row = a_arr[strata][:, None]
    labels = rng.random((int(n_eval_prompts), n_max)) < pp
    tilde_scores = np.where(
        labels,
        float(mu1_base) + s1 * rng.normal(size=(int(n_eval_prompts), n_max)),
        float(mu0_base) + s0 * rng.normal(size=(int(n_eval_prompts), n_max)),
    )
    # 原始观测打分 S_i = b(X) + a(X) * tilde_S_i
    raw_scores = b_row + a_row * tilde_scores
    mu_row = mu_tilde[strata][:, None]
    sd_row = std_tilde[strata][:, None]
    z_pop_full = (tilde_scores - mu_row) / sd_row

    # 同时由原始打分坐标 q_X(b(X) + a(X)*[mu_tilde + sigma_tilde*Z*]) 与基底坐标 q_tilde_X 计算条件后验并核对一致
    q_true_raw_coord = np.zeros_like(raw_scores, dtype=np.float64)
    q_true_base_coord = np.zeros_like(raw_scores, dtype=np.float64)
    for g in range(w.size):
        mask_g = strata == g
        if not np.any(mask_g):
            continue
        pg = float(p_arr[g])
        if pg <= 0.0:
            q_true_raw_coord[mask_g] = 0.0
            q_true_base_coord[mask_g] = 0.0
        elif pg >= 1.0:
            q_true_raw_coord[mask_g] = 1.0
            q_true_base_coord[mask_g] = 1.0
        else:
            bg = float(b_arr[g])
            ag = float(a_arr[g])
            s_raw_g = bg + ag * (float(mu_tilde[g]) + float(std_tilde[g]) * z_pop_full[mask_g])
            lp_r = math.log(pg) + _norm_logpdf(s_raw_g, bg + ag * float(mu1_base), ag * s1)
            ln_r = math.log1p(-pg) + _norm_logpdf(s_raw_g, bg + ag * float(mu0_base), ag * s0)
            q_true_raw_coord[mask_g] = special.expit(lp_r - ln_r)

            s_base_g = float(mu_tilde[g]) + float(std_tilde[g]) * z_pop_full[mask_g]
            lp_b = math.log(pg) + _norm_logpdf(s_base_g, float(mu1_base), s1)
            ln_b = math.log1p(-pg) + _norm_logpdf(s_base_g, float(mu0_base), s0)
            q_true_base_coord[mask_g] = special.expit(lp_b - ln_b)

    max_affine_discrepancy = float(np.max(np.abs(q_true_raw_coord - q_true_base_coord)))

    rows = np.arange(int(n_eval_prompts))
    mean_sup_err = np.zeros(n_arr.size, dtype=np.float64)
    scaled_sup_err = np.zeros(n_arr.size, dtype=np.float64)
    pop_z_oracle_acc = np.zeros(n_arr.size, dtype=np.float64)
    sample_z_oracle_acc = np.zeros(n_arr.size, dtype=np.float64)
    raw_bon_acc = np.zeros(n_arr.size, dtype=np.float64)

    for j, nv in enumerate(n_arr):
        n_i = int(nv)
        s_sub = raw_scores[:, :n_i]
        z_pop = z_pop_full[:, :n_i]
        # 在线样本标准化直接作用于含偏移/缩放的原始打分 S_i
        z_smp = _within_prompt_zscore(s_sub)
        q_sub = q_true_raw_coord[:, :n_i]

        sup_vec = np.max(np.abs(z_smp - z_pop), axis=1)
        mean_sup_err[j] = float(np.mean(sup_vec))
        scaled_sup_err[j] = mean_sup_err[j] / math.sqrt(math.log(float(n_i)) / float(n_i))

        idx_raw = np.argmax(s_sub, axis=1)
        logit_pop = _eval_logit_q_pool_zstar(z_pop)
        logit_smp = _eval_logit_q_pool_zstar(z_smp)
        if constrain_nonnegative_z:
            # 仅在非负半轴 Z >= 0 内极大化；若某行全为负值（仅总体 Z* 偶然全负时发生），回退至最大 Z 候选
            has_pos_pop = np.any(z_pop >= 0.0, axis=1)
            idx_pop = np.where(
                has_pos_pop,
                np.argmax(np.where(z_pop >= 0.0, logit_pop, -np.inf), axis=1),
                np.argmax(z_pop, axis=1),
            )
            has_pos_smp = np.any(z_smp >= 0.0, axis=1)
            idx_smp = np.where(
                has_pos_smp,
                np.argmax(np.where(z_smp >= 0.0, logit_smp, -np.inf), axis=1),
                np.argmax(z_smp, axis=1),
            )
        else:
            idx_pop = np.argmax(logit_pop, axis=1)
            idx_smp = np.argmax(logit_smp, axis=1)
        raw_bon_acc[j] = float(np.mean(q_sub[rows, idx_raw]))
        pop_z_oracle_acc[j] = float(np.mean(q_sub[rows, idx_pop]))
        sample_z_oracle_acc[j] = float(np.mean(q_sub[rows, idx_smp]))

    return {
        "n_values": n_arr,
        "mean_sup_abs_z_error": mean_sup_err,
        "scaled_sup_error_over_sqrt_lnN_N": scaled_sup_err,
        "raw_bon_acc": raw_bon_acc,
        "population_z_oracle_acc": pop_z_oracle_acc,
        "sample_z_oracle_acc": sample_z_oracle_acc,
        "max_affine_posterior_discrepancy": max_affine_discrepancy,
        "theoretical_raw_ceiling": float(diag["raw_bon_asymptotic_acc"]),
        "theoretical_z_oracle_ceiling": float(diag["oracle_z_cal_asymptotic_acc"]),
        "theoretical_ceiling_loss": float(diag["asymptotic_ceiling_loss"]),
        "pi_0_unsolvable": float(diag["pi_0_unsolvable"]),
        "pi_1_trivial": float(diag["pi_1_trivial"]),
        "z_tail_posterior_limit": float(diag["z_tail_posterior_limit"]),
        "z_peak": float(diag["z_peak"]),
        "q_z_peak": float(diag["q_z_peak"]),
        "peak_exceeds_tail_limit": bool(diag["peak_exceeds_tail_limit"]),
        "compact_maximizer_exists": bool(diag["compact_maximizer_exists"]),
        "practical_window_monotone": bool(diag["practical_window_monotone"]),
        "practical_right_window_monotone": bool(diag["practical_right_window_monotone"]),
        "left_tail_overtake_trap": bool(diag["left_tail_overtake_trap"]),
        "constrain_nonnegative_z": bool(constrain_nonnegative_z),
    }


__all__ = [
    "PromptHeterogeneitySpec",
    "diagnose_simpson_confounding",
    "diagnose_within_prompt_residual",
    "diagnose_zscore_induced_collapse",
    "evaluate_pooled_vs_within_prompt_calibration",
    "simulate_zscore_induced_collapse",
    "verify_sample_zscore_extremal_convergence",
]
