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
from tts_theory.theorems import GaussianVerifierSpec, _LOG_SQRT_2PI, _logsumexp_components, _norm_logpdf


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
    # 存在题内期望导数为负却被跨题正协方差彻底掩盖为正的点
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
    """对每行（单道题内 N 个样本）做 Z-score 标准化（std 极小时退化为中心化）。"""
    mean = np.mean(scores, axis=1, keepdims=True)
    if scores.shape[1] <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    std = np.std(scores, axis=1, ddof=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-6)


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


__all__ = [
    "PromptHeterogeneitySpec",
    "diagnose_simpson_confounding",
    "evaluate_pooled_vs_within_prompt_calibration",
]
