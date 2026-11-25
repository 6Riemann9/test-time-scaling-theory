"""多噪声 Verifier 集成的相变逆转与重尾分离理论模块（定理 5）。

本模块实现：
1. 定理 5A：对称相关高斯 K-Verifier 的均值融合（Mean-Ensemble）与最小值融合
   （Min-Ensemble）精确分布建模、临界相界与相变逆转条件、以及基于鞍点平移
   Gauss-Hermite 的双精度一维因子积分接口；
2. 定理 5B：幂律重尾（Student-t）Reward Hacking 下 Mean 与 Min 的三重分离
   及同源漏洞击穿机理、异质污染脆弱性分离；
3. 流式分块向量化多裁判蒙特卡洛仿真器（支持 100,000 次试验且严控内存峰值）。
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence, Union

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike
from scipy import optimize, special

from tts_theory.theorems import (
    ContinuousVerifierSpec,
    GaussianVerifierSpec,
    ScalarOrArray,
    _LOG_SQRT_2PI,
    _format_output,
    _logsumexp_components,
    _norm_logpdf,
    _norm_logsf,
    _prepare_array,
    _std_norm_inv_sf,
)

# 预计算 64 节点 Gauss-Hermite 求积点与对数权重（带 +u^2 因子以便与高斯核混合）
_GH_U, _GH_W = hermgauss(64)
_GH_LOG_W_PLUS_U2 = np.log(_GH_W) + _GH_U * _GH_U
_LOG_SQRT_PI = 0.5 * math.log(math.pi)


def _wilson_ci(
    successes: np.ndarray,
    n_trials: int,
    z: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算二项比例估计、标准误及 95% Wilson 得分区间。"""
    p_hat = successes.astype(np.float64) / float(n_trials)
    se = np.sqrt(p_hat * (1.0 - p_hat) / float(n_trials))
    z2 = z * z
    denom = 1.0 + z2 / float(n_trials)
    center = (p_hat + z2 / (2.0 * float(n_trials))) / denom
    half_width = (
        z
        * np.sqrt(
            (p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n_trials))) / float(n_trials)
        )
        / denom
    )
    return (
        p_hat,
        se,
        np.clip(center - half_width, 0.0, 1.0),
        np.clip(center + half_width, 0.0, 1.0),
    )


def _min_gaussian_component_log_stats(
    x: np.ndarray,
    mu: float,
    sigma: float,
    k: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单类（Y=c）下 K 个对称相关高斯最小值 M_min 的对数生存函数、分布函数与密度。

    对 rho in (0, 1) 且 K >= 2 采用鞍点平移缩放 64 节点 Gauss-Hermite 一维积分：
        z*(w) = max(0, K*sqrt(rho)/(1 + (K-1)*rho) * w)
        tau_z = sqrt((1 - rho)/(1 + (K-1)*rho))
        z = z* + tau_z * sqrt(2) * u
    彻底消除极右尾（w >> 1）下的积分节点漏采与下溢问题。

    Returns:
        (log_sf, log_cdf, log_pdf) 三个与 x 同形状的 float64 数组。
    """
    w = (x - mu) / sigma

    # 退化或零相关精确解析通道
    if k == 1 or rho == 0.0:
        log_sf1 = special.log_ndtr(-w)
        log_sf = float(k) * log_sf1
        log_pdf = (
            math.log(float(k))
            - math.log(sigma)
            - _LOG_SQRT_2PI
            - 0.5 * w * w
            + float(k - 1) * log_sf1
        )
        # 无相消计算 log_cdf = ln(1 - exp(log_sf))
        # 当 w < 0 时直接用 log1p(-(1 - Phi(w))^K) 仍稳定；若 log_sf 极近 0，改用 Phi(-w) 展开
        log_cdf = np.where(
            log_sf > -1e-12,
            np.log(np.maximum(1e-300, float(k) * special.ndtr(w))),
            np.log(-np.expm1(np.minimum(log_sf, -1e-15))),
        )
        # 对中等左尾 w < -0.5：1 - (1 - Phi(w))^K 精确用 binomial/expm1 补余
        left_mask = (w < -0.5) & (k > 1)
        if np.any(left_mask):
            # 1 - (1 - Phi(w))^K = -expm1(K * log1p(-Phi(w)))，在 Phi(w) 极小时无相消
            phi_left = special.ndtr(w[left_mask])
            log_sf_stable = float(k) * np.log1p(-phi_left)
            log_cdf[left_mask] = np.log(
                np.maximum(1e-300, -np.expm1(log_sf_stable))
            )
        return log_sf, log_cdf, log_pdf

    # 非负相关因子积分通道：S^{(j)} = mu + sigma * (sqrt(rho)*Z + sqrt(1-rho)*E_j)
    sqrt_rho = math.sqrt(rho)
    sqrt_1mrho = math.sqrt(1.0 - rho)
    tau_z = math.sqrt((1.0 - rho) / (1.0 + (k - 1.0) * rho))
    saddle_slope = (float(k) * sqrt_rho) / (1.0 + (k - 1.0) * rho)
    log_pref = math.log(tau_z) - _LOG_SQRT_PI

    w_flat = w.ravel()
    z_star = np.maximum(0.0, saddle_slope * w_flat)[:, None]  # (M, 1)
    z_pts = z_star + (tau_z * math.sqrt(2.0)) * _GH_U[None, :]  # (M, 64)
    a_pts = (w_flat[:, None] - sqrt_rho * z_pts) / sqrt_1mrho

    log_base = (
        log_pref
        + _GH_LOG_W_PLUS_U2[None, :]
        - 0.5 * z_pts * z_pts
    )
    log_ndtr_neg_a = special.log_ndtr(-a_pts)

    # 1. log_sf = ln E[sf(a)^K]
    log_sf_terms = log_base + float(k) * log_ndtr_neg_a
    log_sf_flat = special.logsumexp(log_sf_terms, axis=1)

    # 2. log_pdf = ln(K/(sigma*sqrt(1-rho)) * E[phi(a) * sf(a)^{K-1}])
    log_pdf_terms = (
        log_base
        - _LOG_SQRT_2PI
        - 0.5 * a_pts * a_pts
        + float(k - 1) * log_ndtr_neg_a
    )
    log_pdf_flat = (
        math.log(float(k))
        - math.log(sigma)
        - math.log(sqrt_1mrho)
        + special.logsumexp(log_pdf_terms, axis=1)
    )

    # 3. log_cdf = ln E[1 - sf(a)^K]（按掩码分支计算，消除无效区域 log1p(-1) 警告与大数相消）
    log_1m_sfk = np.empty_like(a_pts)
    left_m = a_pts < -0.5
    if np.any(left_m):
        phi_l = np.minimum(special.ndtr(a_pts[left_m]), 1.0 - 1e-16)
        log_1m_sfk[left_m] = np.log(
            np.maximum(1e-300, -np.expm1(float(k) * np.log1p(-phi_l)))
        )
    if np.any(~left_m):
        lsf_r = np.minimum(float(k) * log_ndtr_neg_a[~left_m], -1e-15)
        log_1m_sfk[~left_m] = np.log(np.maximum(1e-300, -np.expm1(lsf_r)))
    log_cdf_flat = special.logsumexp(log_base + log_1m_sfk, axis=1)

    return (
        log_sf_flat.reshape(x.shape),
        log_cdf_flat.reshape(x.shape),
        log_pdf_flat.reshape(x.shape),
    )


@dataclass(frozen=True)
class MultiVerifierGaussianSpec:
    """对称相关高斯 K-Verifier 联合分布规格（定理 5A）。

    给定 Y=c in {0, 1}，K 个 Verifier 打分服从联合正态分布：
        S | (Y=c) ~ N(mu_c * 1_K, sigma_c^2 * [(1 - rho_c) I_K + rho_c 1_K 1_K^T])
    其中 rho1, rho0 in [0, 1) 分别为各裁判在正确样本与错误样本上的成对相关系数。
    """

    p: float
    mu1: float
    sigma1: float
    mu0: float = 0.0
    sigma0: float = 1.0
    num_verifiers: int = 2
    rho1: float = 0.0
    rho0: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"先验正确率 p 必须位于 (0, 1) 内，当前为 {self.p}")
        if self.sigma1 <= 0.0 or self.sigma0 <= 0.0:
            raise ValueError("条件标准差 sigma1 与 sigma0 必须严格大于 0")
        if isinstance(self.num_verifiers, bool) or int(self.num_verifiers) < 1:
            raise ValueError(f"num_verifiers 必须为 >= 1 的整数，当前为 {self.num_verifiers}")
        if not (0.0 <= self.rho1 < 1.0) or not (0.0 <= self.rho0 < 1.0):
            raise ValueError("条件相关系数 rho1 与 rho0 必须位于 [0, 1) 内")

    @property
    def delta_mu(self) -> float:
        """正确与错误样本打分均值差 Delta_mu = mu1 - mu0。"""
        return float(self.mu1 - self.mu0)

    @property
    def kappa_single(self) -> float:
        """单裁判噪声比 kappa = sigma0 / sigma1。"""
        return float(self.sigma0 / self.sigma1)

    def mean_verifier_spec(self) -> GaussianVerifierSpec:
        """返回 Mean-Ensemble (1/K) sum S^{(k)} 的精确单变量高斯规格。"""
        k = float(self.num_verifiers)
        s1_eff = self.sigma1 * math.sqrt((1.0 + (k - 1.0) * self.rho1) / k)
        s0_eff = self.sigma0 * math.sqrt((1.0 + (k - 1.0) * self.rho0) / k)
        return GaussianVerifierSpec(
            p=float(self.p),
            mu1=float(self.mu1),
            sigma1=float(s1_eff),
            mu0=float(self.mu0),
            sigma0=float(s0_eff),
        )

    def min_verifier_spec(self) -> MinEnsembleVerifierSpec:
        """返回 Min-Ensemble min_k S^{(k)} 的精确一维因子积分 Verifier 规格。"""
        return MinEnsembleVerifierSpec(base=self)


@dataclass(frozen=True)
class MinEnsembleVerifierSpec:
    """K 个对称相关高斯 Verifier 取最小值 M_min = min_k S^{(k)} 的连续规格实现。

    完全兼容 `ContinuousVerifierSpec` 协议，可直接传入 `exact_bon_accuracy` 与
    `numerical_optimal_n` 进行双精度顺序统计量数值积分。
    """

    base: MultiVerifierGaussianSpec

    @property
    def p(self) -> float:
        return float(self.base.p)

    def _stats1(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _min_gaussian_component_log_stats(
            arr,
            self.base.mu1,
            self.base.sigma1,
            int(self.base.num_verifiers),
            self.base.rho1,
        )

    def _stats0(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _min_gaussian_component_log_stats(
            arr,
            self.base.mu0,
            self.base.sigma0,
            int(self.base.num_verifiers),
            self.base.rho0,
        )

    def pdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, _, lpdf = self._stats1(arr)
        return _format_output(np.exp(lpdf), is_scalar)

    def pdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, _, lpdf = self._stats0(arr)
        return _format_output(np.exp(lpdf), is_scalar)

    def cdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, lcdf, _ = self._stats1(arr)
        return _format_output(np.exp(lcdf), is_scalar)

    def cdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, lcdf, _ = self._stats0(arr)
        return _format_output(np.exp(lcdf), is_scalar)

    def sf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        lsf, _, _ = self._stats1(arr)
        return _format_output(np.exp(lsf), is_scalar)

    def sf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        lsf, _, _ = self._stats0(arr)
        return _format_output(np.exp(lsf), is_scalar)

    def mix_pdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, _, lpdf1 = self._stats1(arr)
        _, _, lpdf0 = self._stats0(arr)
        lpdf = _logsumexp_components(
            [math.log(self.p) + lpdf1, math.log1p(-self.p) + lpdf0]
        )
        return _format_output(np.exp(lpdf), is_scalar)

    def mix_cdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, lcdf1, _ = self._stats1(arr)
        _, lcdf0, _ = self._stats0(arr)
        val = self.p * np.exp(lcdf1) + (1.0 - self.p) * np.exp(lcdf0)
        return _format_output(np.clip(val, 0.0, 1.0), is_scalar)

    def mix_sf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        lsf1, _, _ = self._stats1(arr)
        lsf0, _, _ = self._stats0(arr)
        lsf = _logsumexp_components(
            [math.log(self.p) + lsf1, math.log1p(-self.p) + lsf0]
        )
        return _format_output(np.clip(np.exp(lsf), 0.0, 1.0), is_scalar)

    def _eval_log_mix_sf_and_grad(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 ln(mix_sf(x)) 及其精确导数 -mix_pdf(x)/mix_sf(x)。"""
        lsf1, _, lpdf1 = self._stats1(x)
        lsf0, _, lpdf0 = self._stats0(x)
        lp = math.log(self.p)
        lq = math.log1p(-self.p)
        lsf = _logsumexp_components([lp + lsf1, lq + lsf0])
        lpdf = _logsumexp_components([lp + lpdf1, lq + lpdf0])
        grad = -np.exp(np.clip(lpdf - lsf, -700.0, 700.0))
        return lsf, grad

    def mix_isf(self, q: ArrayLike) -> ScalarOrArray:
        """基于 Slepian 解析下界与单样本上界的对数生存域保护牛顿求逆。"""
        q_arr, is_scalar = _prepare_array(q)
        if np.any(np.isnan(q_arr)) or np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("分位数参数 q 必须全部位于 [0, 1] 区间内")

        out = np.empty_like(q_arr, dtype=np.float64)
        zero_mask = q_arr == 0.0
        one_mask = q_arr == 1.0
        interior = ~(zero_mask | one_mask)
        out[zero_mask] = np.inf
        out[one_mask] = -np.inf
        if not np.any(interior):
            return _format_output(out, is_scalar)

        qi = q_arr[interior]
        log_qi = np.log(qi)
        k = float(self.base.num_verifiers)
        weights = (self.p, 1.0 - self.p)
        mus = (self.base.mu1, self.base.mu0)
        sigmas = (self.base.sigma1, self.base.sigma0)

        # 上界：M_min <= S^{(1)} => mix_sf_min(x) <= mix_sf_single(x)
        z_single = _std_norm_inv_sf(qi)
        hi = np.full_like(qi, -np.inf)
        # 下界（Slepian）：对 rho >= 0，P(min_k S^{(k)} > x | Y=c) >= [sf_c(x)]^K
        lo_slep = np.full_like(qi, -np.inf)
        lo_min = np.full_like(qi, np.inf)

        for w_c, mu_c, sig_c in zip(weights, mus, sigmas):
            hi = np.maximum(hi, mu_c + sig_c * z_single)
            q_root = np.power(np.minimum(1.0, qi), 1.0 / k)
            z_root = _std_norm_inv_sf(q_root)
            lo_min = np.minimum(lo_min, mu_c + sig_c * z_root)
            sub = qi < w_c
            if np.any(sub):
                q_sub_root = np.power(qi[sub] / w_c, 1.0 / k)
                z_sub = _std_norm_inv_sf(q_sub_root)
                lo_slep[sub] = np.maximum(lo_slep[sub], mu_c + sig_c * z_sub)

        lo = np.maximum(lo_min, lo_slep)
        pad = 1e-8 * np.maximum(1.0, np.abs(hi - lo))
        lo = lo - pad
        hi = hi + pad

        # 若边界略微不够宽则指数扩界保障区间套严格成立
        for _ in range(8):
            lsf_lo, _ = self._eval_log_mix_sf_and_grad(lo)
            need_lo = lsf_lo < log_qi
            if not np.any(need_lo):
                break
            lo = np.where(need_lo, lo - np.maximum(1.0, np.abs(hi - lo)), lo)
        for _ in range(8):
            lsf_hi, _ = self._eval_log_mix_sf_and_grad(hi)
            need_hi = lsf_hi > log_qi
            if not np.any(need_hi):
                break
            hi = np.where(need_hi, hi + np.maximum(1.0, np.abs(hi - lo)), hi)

        x = 0.5 * (lo + hi)
        for it in range(42):
            lsf, grad = self._eval_log_mix_sf_and_grad(x)
            fval = lsf - log_qi
            lo = np.where(fval > 0.0, x, lo)
            hi = np.where(fval <= 0.0, x, hi)
            if np.all(np.abs(hi - lo) <= 1e-13 * np.maximum(1.0, np.abs(x))):
                break
            if it >= 4:
                safe_grad = np.where(np.abs(grad) > 1e-300, grad, -1e-300)
                x_new = x - fval / safe_grad
                inside = (x_new > lo) & (x_new < hi)
                x = np.where(inside, x_new, 0.5 * (lo + hi))
            else:
                x = 0.5 * (lo + hi)

        out[interior] = x
        return _format_output(out, is_scalar)

    def mix_ppf(self, u: ArrayLike) -> ScalarOrArray:
        u_arr, is_scalar = _prepare_array(u)
        if np.any(np.isnan(u_arr)) or np.any((u_arr < 0.0) | (u_arr > 1.0)):
            raise ValueError("分位数参数 u 必须全部位于 [0, 1] 区间内")
        return self.mix_isf(1.0 - u_arr) if not is_scalar else float(self.mix_isf(1.0 - float(u_arr)))

    def log_likelihood_ratio(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        _, _, lpdf1 = self._stats1(arr)
        _, _, lpdf0 = self._stats0(arr)
        return _format_output(lpdf1 - lpdf0, is_scalar)

    def posterior(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        llr = np.asarray(self.log_likelihood_ratio(arr), dtype=np.float64)
        logit_p = math.log(self.p / (1.0 - self.p))
        return _format_output(special.expit(logit_p + llr), is_scalar)


def ensemble_phase_reversal_criterion(spec: MultiVerifierGaussianSpec) -> dict:
    """计算对称相关高斯 K-Verifier 融合的精确相变逆转判据（定理 5A）。

    解析临界相界：
        kappa <= kappa_critical(K) = sqrt((1 + (K-1)*rho1) / (1 + (K-1)*rho0))
    其中 kappa = sigma0 / sigma1。当 rho0 < rho1 / kappa^2 且 kappa > 1 时，
    存在有限临界裁判数 K_c = ceil(1 + (kappa^2 - 1) / (rho1 - kappa^2 * rho0))
    使融合器跨越相界逆转为永不坍塌。
    """
    k = int(spec.num_verifiers)
    kappa_single = spec.kappa_single
    rho1 = float(spec.rho1)
    rho0 = float(spec.rho0)
    dmu = spec.delta_mu

    factor1 = (1.0 + (k - 1.0) * rho1) / float(k)
    factor0 = (1.0 + (k - 1.0) * rho0) / float(k)
    kappa_eff = kappa_single * math.sqrt(factor0 / factor1)
    kappa_critical = math.sqrt((1.0 + (k - 1.0) * rho1) / (1.0 + (k - 1.0) * rho0))

    if rho0 > 0.0:
        max_reversible_kappa = math.sqrt(rho1 / rho0) if rho1 > 0.0 else 1.0
    else:
        max_reversible_kappa = math.inf if rho1 > 0.0 else 1.0

    # 是否可通过增加 K 逆转
    denom = rho1 - (kappa_single**2) * rho0
    if kappa_single <= 1.0:
        reversible_by_scaling_k = dmu > 0.0
        critical_k: int | None = 1 if dmu > 0.0 else None
    elif denom > 1e-15 and dmu > 0.0:
        reversible_by_scaling_k = True
        raw_k = 1.0 + (kappa_single**2 - 1.0) / denom
        critical_k = int(math.ceil(raw_k - 1e-12))
        if critical_k < 2:
            critical_k = 2
    else:
        reversible_by_scaling_k = False
        critical_k = None

    # 判定当前 K 下的渐近相区
    if kappa_eff > 1.0 + 1e-12:
        will_collapse = True
        regime = "collapse" if reversible_by_scaling_k or kappa_single <= 1.0 else "irreversible_collapse"
    elif abs(kappa_eff - 1.0) <= 1e-12:
        if dmu > 0.0:
            will_collapse = False
            regime = "critical_reversed_consistent" if kappa_single > 1.0 else "critical_consistent"
        elif dmu == 0.0:
            will_collapse = False
            regime = "uninformative_constant"
        else:
            will_collapse = True
            regime = "monotone_collapse"
    else:
        will_collapse = False
        if kappa_single > 1.0 and dmu > 0.0:
            regime = "reversed_consistent"
        elif dmu > 0.0:
            regime = "consistent_monotone"
        else:
            regime = "non_monotone_recovery"

    return {
        "kappa_single": float(kappa_single),
        "kappa_eff": float(kappa_eff),
        "kappa_critical": float(kappa_critical),
        "max_reversible_kappa": float(max_reversible_kappa),
        "reversible_by_scaling_k": bool(reversible_by_scaling_k),
        "critical_k": critical_k,
        "will_collapse": bool(will_collapse),
        "regime": regime,
    }


@dataclass(frozen=True)
class HeavyTailMultiVerifierSpec:
    """幂律重尾（Student-t）Reward Hacking 竞争下的多裁判联合模型（定理 5B）。

    - 正类（Y=1）：S^{(k)} = mu1 + sigma1 * T1 + eta1 * eps_k，
      共享重尾信号 T1 ~ t_{alpha1}（lambda_{U,1} > 0），独立微扰 eps_k ~ N(0, 1)；
    - 错类（Y=0）：以概率 1 - epsilon_shared 各裁判遭受独立私有重尾欺骗
      S^{(k)} = mu0 + sigma0 * T_{0,k}（T_{0,k} i.i.d. ~ t_{alpha0}, alpha0 < alpha1, lambda_{U,0}=0）；
      以概率 epsilon_shared 所有裁判同时命中同源重尾漏洞
      S^{(k)} = mu0 + sigma0 * T_shared + eta0 * eps_k（T_shared ~ t_{alpha0}, lambda_{U,0} > 0）。
    """

    p: float
    mu1: float = 1.2
    sigma1: float = 1.0
    mu0: float = 0.0
    sigma0: float = 1.0
    num_verifiers: int = 4
    alpha1: float = 5.0
    alpha0: float = 2.2
    eta1: float = 0.15
    eta0: float = 0.15
    epsilon_shared: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError("p 必须位于 (0, 1) 内")
        if self.sigma1 <= 0.0 or self.sigma0 <= 0.0:
            raise ValueError("sigma1 与 sigma0 必须 > 0")
        if self.num_verifiers < 1:
            raise ValueError("num_verifiers 必须 >= 1")
        if self.alpha1 <= 0.0 or self.alpha0 <= 0.0:
            raise ValueError("Student-t 自由度 alpha1 与 alpha0 必须 > 0")
        if self.eta1 < 0.0 or self.eta0 < 0.0:
            raise ValueError("微扰尺度 eta1 与 eta0 不能为负")
        if not (0.0 <= self.epsilon_shared <= 1.0):
            raise ValueError("epsilon_shared 必须位于 [0, 1] 内")


def heavy_tail_separation_criterion(spec: HeavyTailMultiVerifierSpec) -> dict:
    """判定幂律重尾模型下 Mean-Ensemble 与 Min-Ensemble 的渐近相变分离（定理 5B）。"""
    k = int(spec.num_verifiers)
    a1 = float(spec.alpha1)
    a0 = float(spec.alpha0)
    eps = float(spec.epsilon_shared)

    # 临界独立重尾逆转人数：K * alpha0 > alpha1
    critical_k_heavy = int(math.floor(a1 / a0) + 1) if a0 < a1 else 1
    mean_collapses = a0 < a1
    if eps > 0.0:
        min_collapses = a0 < a1
        regime = "shared_vulnerability_breakdown"
    elif k * a0 > a1:
        min_collapses = False
        regime = "min_reversed_mean_collapses"
    elif k * a0 < a1:
        min_collapses = True
        regime = "both_collapse_insufficient_k"
    else:
        min_collapses = False
        regime = "boundary_equal_tail_index"

    return {
        "num_verifiers": k,
        "alpha1": a1,
        "alpha0": a0,
        "effective_min_wrong_alpha": float(a0 if eps > 0.0 else k * a0),
        "critical_k_heavy": critical_k_heavy,
        "epsilon_shared": eps,
        "mean_will_collapse": bool(mean_collapses),
        "min_will_collapse": bool(min_collapses),
        "regime": regime,
    }


@dataclass(frozen=True)
class HeterogeneousVulnerabilitySpec:
    """异质漏洞分离模型：K - n_vulnerable 个健康裁判 + n_vulnerable 个受重尾污染裁判。

    - 正类（Y=1）：全部 K 个裁判 S^{(k)} = mu1 + sigma1 * T1 + eta * eps_k（T1 ~ t_{alpha1}）；
    - 错类（Y=0）：前 (K - n_vulnerable) 个健康裁判仅受高斯轻尾打分 N(mu0, sigma0^2)，
      后 n_vulnerable 个脆弱裁判以概率 1 遭受重尾错分 S^{(k)} = mu0 + sigma_vuln * T_0（T_0 ~ t_{alpha0}）。
    只要存在至少 1 个健康裁判（n_vulnerable < K），Min-Ensemble 即可完全屏蔽污染裁判而永不坍塌，
    而 Mean-Ensemble 会被哪怕 1 个重尾污染裁判彻底拖垮。
    """

    p: float = 0.25
    mu1: float = 1.2
    sigma1: float = 1.0
    mu0: float = 0.0
    sigma0: float = 0.8
    num_verifiers: int = 4
    n_vulnerable: int = 1
    alpha1: float = 6.0
    alpha0: float = 2.0
    sigma_vuln: float = 1.2
    eta1: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError("p 必须位于 (0, 1) 内")
        if self.num_verifiers < 1 or not (0 <= self.n_vulnerable <= self.num_verifiers):
            raise ValueError("n_vulnerable 必须位于 [0, num_verifiers] 内")


def _validate_spd_matrix(mat: ArrayLike, name: str, expected_dim: int | None = None) -> np.ndarray:
    """校验矩阵为对称正定 float64 方阵，并返回对称化副本。"""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"{name} 必须为非空二维方阵，当前形状为 {arr.shape}")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(f"{name} 维度 {arr.shape[0]} 与预期 {expected_dim} 不符")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 不能包含 NaN 或 Inf")
    if float(np.max(np.abs(arr - arr.T))) > 1e-8 * max(1.0, float(np.max(np.abs(arr)))):
        raise ValueError(f"{name} 必须为对称矩阵")
    sym = 0.5 * (arr + arr.T)
    try:
        np.linalg.cholesky(sym)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} 必须严格正定（Cholesky 分解失败）") from exc
    return sym


def solve_positive_quadrant_qp(cov: ArrayLike) -> dict:
    """精确求解一般正定协方差矩阵上的正交象限二次规划（Hashorva-Hüsler 2003）。

    求解严格凸二次规划：
        gamma(Sigma) := min_{u >= 1_K} u^T Sigma^{-1} u
    及其 KKT 唯一最优解 u* 与乘子 w* = Sigma^{-1} u* >= 0，激活集 I = {k : u*_k = 1}。
    """
    sig = _validate_spd_matrix(cov, "cov")
    k = sig.shape[0]
    ones = np.ones(k, dtype=np.float64)

    if k == 1:
        w0 = np.array([1.0 / sig[0, 0]], dtype=np.float64)
        return {
            "gamma": float(w0[0]),
            "u_star": np.array([1.0], dtype=np.float64),
            "active_set": [0],
            "multipliers": w0,
            "strict_complementarity": True,
        }

    def _check_active_subset(active: list[int]) -> tuple[float, np.ndarray, np.ndarray] | None:
        if not active:
            return None
        idx = np.asarray(active, dtype=np.int64)
        sub = sig[np.ix_(idx, idx)]
        try:
            w_act = np.linalg.solve(sub, np.ones(len(active), dtype=np.float64))
        except np.linalg.LinAlgError:
            return None
        if np.any(w_act < -1e-11):
            return None
        w_full = np.zeros(k, dtype=np.float64)
        w_full[idx] = np.maximum(w_act, 0.0)
        u_full = sig @ w_full
        if np.any(u_full < 1.0 - 1e-10):
            return None
        u_full[idx] = 1.0
        gamma_val = float(np.sum(w_full))
        return gamma_val, u_full, w_full

    best: tuple[float, list[int], np.ndarray, np.ndarray] | None = None

    # 优先检验全激活快捷路径
    all_idx = list(range(k))
    full_res = _check_active_subset(all_idx)
    if full_res is not None:
        best = (full_res[0], all_idx, full_res[1], full_res[2])
    elif k <= 12:
        for size in range(k - 1, 0, -1):
            for comb in itertools.combinations(range(k), size):
                act = list(comb)
                cand = _check_active_subset(act)
                if cand is not None:
                    if best is None or cand[0] < best[0] - 1e-12:
                        best = (cand[0], act, cand[1], cand[2])

    # 对大维度 K > 12（或兜底）采用数值 QP 初猜 + primal-dual active-set 精化
    if best is None:
        inv_sig = np.linalg.inv(sig)
        opt_res = optimize.minimize(
            lambda u: float(u @ inv_sig @ u),
            ones.copy(),
            jac=lambda u: 2.0 * (inv_sig @ u),
            bounds=[(1.0, None)] * k,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-13},
        )
        u_init = np.maximum(1.0, opt_res.x)
        active_set = sorted(np.flatnonzero(u_init <= 1.0 + 1e-5).tolist())
        if not active_set:
            active_set = [int(np.argmin(u_init))]
        for _ in range(max(20, 4 * k)):
            idx = np.asarray(active_set, dtype=np.int64)
            w_act = np.linalg.solve(sig[np.ix_(idx, idx)], np.ones(len(active_set), dtype=np.float64))
            w_full = np.zeros(k, dtype=np.float64)
            w_full[idx] = w_act
            u_full = sig @ w_full
            neg_w = np.flatnonzero(w_full < -1e-11)
            viol_u = np.setdiff1d(np.flatnonzero(u_full < 1.0 - 1e-11), idx)
            if neg_w.size == 0 and viol_u.size == 0:
                u_full[idx] = 1.0
                best = (float(np.sum(w_full)), active_set, u_full, np.maximum(w_full, 0.0))
                break
            if viol_u.size > 0:
                add_k = int(viol_u[np.argmin(u_full[viol_u])])
                active_set = sorted(set(active_set) | {add_k})
            elif neg_w.size > 0:
                drop_k = int(neg_w[np.argmin(w_full[neg_w])])
                active_set = [a for a in active_set if a != drop_k]
                if not active_set:
                    active_set = [int(np.argmin(np.diag(sig)))]
        if best is None:
            inv_sig = np.linalg.inv(sig)
            w_num = inv_sig @ u_init
            best = (float(u_init @ w_num), active_set, u_init, np.maximum(w_num, 0.0))

    gamma_opt, act_opt, u_opt, w_opt = best
    act_arr = np.asarray(act_opt, dtype=np.int64)
    inact_arr = np.asarray([j for j in range(k) if j not in act_opt], dtype=np.int64)
    strict_comp = bool(
        np.all(w_opt[act_arr] > 1e-9)
        and (inact_arr.size == 0 or np.all(u_opt[inact_arr] > 1.0 + 1e-9))
    )
    return {
        "gamma": float(gamma_opt),
        "u_star": u_opt,
        "active_set": act_opt,
        "multipliers": w_opt,
        "strict_complementarity": strict_comp,
    }


@dataclass(frozen=True)
class GeneralGaussianMultiVerifierSpec:
    """一般非对称相关高斯 K-Verifier 联合分布规格（定理 5A 推广版）。

    给定 Y=c in {0, 1}，K 个不同能力与相关结构的裁判联合分布：
        S | (Y=c) ~ N(mu_c, Sigma_c)
    """

    p: float
    mu1: ArrayLike
    cov1: ArrayLike
    mu0: ArrayLike
    cov0: ArrayLike

    def __post_init__(self) -> None:
        if not (0.0 < float(self.p) < 1.0):
            raise ValueError(f"p 必须位于 (0, 1) 内，当前为 {self.p}")
        m1 = np.asarray(self.mu1, dtype=np.float64).ravel()
        m0 = np.asarray(self.mu0, dtype=np.float64).ravel()
        if m1.size == 0 or m1.shape != m0.shape:
            raise ValueError("mu1 与 mu0 必须为等长非空一维向量")
        c1 = _validate_spd_matrix(self.cov1, "cov1", expected_dim=m1.size)
        c0 = _validate_spd_matrix(self.cov0, "cov0", expected_dim=m1.size)
        object.__setattr__(self, "mu1", m1)
        object.__setattr__(self, "mu0", m0)
        object.__setattr__(self, "cov1", c1)
        object.__setattr__(self, "cov0", c0)

    @property
    def num_verifiers(self) -> int:
        return int(self.mu1_arr.size)

    @property
    def mu1_arr(self) -> np.ndarray:
        return np.asarray(self.mu1, dtype=np.float64)

    @property
    def mu0_arr(self) -> np.ndarray:
        return np.asarray(self.mu0, dtype=np.float64)

    @property
    def cov1_arr(self) -> np.ndarray:
        return np.asarray(self.cov1, dtype=np.float64)

    @property
    def cov0_arr(self) -> np.ndarray:
        return np.asarray(self.cov0, dtype=np.float64)

    def linear_verifier_spec(self, weights: ArrayLike) -> GaussianVerifierSpec:
        """返回任意非零线性组合 w^T S 的精确一维 GaussianVerifierSpec。"""
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != self.num_verifiers:
            raise ValueError(f"weights 长度 ({w.size}) 必须等于 {self.num_verifiers}")
        v1 = float(w @ self.cov1_arr @ w)
        v0 = float(w @ self.cov0_arr @ w)
        if v1 <= 0.0 or v0 <= 0.0:
            raise ValueError("weights 必须为非零向量以确保组合方差为正")
        return GaussianVerifierSpec(
            p=float(self.p),
            mu1=float(w @ self.mu1_arr),
            sigma1=math.sqrt(v1),
            mu0=float(w @ self.mu0_arr),
            sigma0=math.sqrt(v0),
        )

    def mean_verifier_spec(self) -> GaussianVerifierSpec:
        """返回等权 Mean-Ensemble 的精确一维 GaussianVerifierSpec。"""
        w = np.full(self.num_verifiers, 1.0 / float(self.num_verifiers), dtype=np.float64)
        return self.linear_verifier_spec(w)


def optimal_linear_ensemble_criterion(spec: GeneralGaussianMultiVerifierSpec) -> dict:
    """基于广义特征值计算一般高斯 K-Verifier 的最优线性融合权重与相变界（定理 5A）。"""
    k = spec.num_verifiers
    c1 = spec.cov1_arr
    c0 = spec.cov0_arr
    dmu_vec = spec.mu1_arr - spec.mu0_arr

    # 广义特征值: Sigma_0 w = lambda Sigma_1 w
    l1 = np.linalg.cholesky(c1)
    l1_inv = np.linalg.inv(l1)
    mat_sym = 0.5 * ((l1_inv @ c0 @ l1_inv.T) + (l1_inv @ c0 @ l1_inv.T).T)
    eigvals, eigvecs = np.linalg.eigh(mat_sym)
    lam_min = float(max(0.0, eigvals[0]))
    w_raw = l1_inv.T @ eigvecs[:, 0]

    # 定向使投影均值差非负，并按 L2 单位化（兼容零和差分权重）
    proj_diff = float(w_raw @ dmu_vec)
    if proj_diff < 0.0 or (abs(proj_diff) <= 1e-12 and float(np.sum(w_raw)) < 0.0):
        w_raw = -w_raw
    w_norm = float(np.linalg.norm(w_raw))
    w_star = w_raw / max(w_norm, 1e-15)
    w_delta_mu = float(w_star @ dmu_vec)
    kappa_linear_min = math.sqrt(lam_min)

    # 等权 Mean 融合检验
    kappa_mean = math.sqrt(float(np.sum(c0)) / float(np.sum(c1)))
    mean_dmu = float(np.mean(dmu_vec))
    mean_collapses = bool((kappa_mean > 1.0 + 1e-12) or (abs(kappa_mean - 1.0) <= 1e-12 and mean_dmu <= 0.0))

    # 非负单纯形最优比值 min_{w>=0, 1^T w=1} (w^T C0 w) / (w^T C1 w)
    def _neg_obj(w: np.ndarray) -> float:
        denom = float(w @ c1 @ w)
        return float(w @ c0 @ w) / max(denom, 1e-18)

    starts = [np.full(k, 1.0 / float(k), dtype=np.float64)]
    for i in range(k):
        ei = np.zeros(k, dtype=np.float64)
        ei[i] = 1.0
        starts.append(ei)
    w_pos = np.maximum(0.0, w_star)
    if float(np.sum(w_pos)) > 1e-12:
        starts.append(w_pos / float(np.sum(w_pos)))

    best_nn_ratio = float(kappa_mean**2)
    best_nn_w = starts[0]
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    bounds = [(0.0, 1.0)] * k
    for x0 in starts:
        res = optimize.minimize(_neg_obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success or res.fun < best_nn_ratio:
            val = float(_neg_obj(res.x))
            if val < best_nn_ratio:
                best_nn_ratio = val
                best_nn_w = res.x / max(1e-15, float(np.sum(res.x)))

    kappa_nonneg_min = math.sqrt(max(0.0, best_nn_ratio))
    linear_reversible = bool(
        (kappa_linear_min < 1.0 - 1e-12)
        or (abs(kappa_linear_min - 1.0) <= 1e-12 and w_delta_mu > 1e-12)
    )
    negative_weight_rescues_mean = bool(
        mean_collapses and linear_reversible and np.any(w_star < -1e-9)
    )

    return {
        "lambda_min": lam_min,
        "kappa_linear_min": kappa_linear_min,
        "w_star": w_star,
        "w_star_delta_mu": w_delta_mu,
        "kappa_mean": kappa_mean,
        "mean_collapses": mean_collapses,
        "kappa_nonneg_min": kappa_nonneg_min,
        "w_nonneg_star": best_nn_w,
        "linear_reversible": linear_reversible,
        "negative_weight_rescues_mean": negative_weight_rescues_mean,
    }


def general_min_ensemble_qp_criterion(spec: GeneralGaussianMultiVerifierSpec) -> dict:
    """一般非对称相关高斯 K-Verifier 上 Min-Ensemble 的 Hashorva-Hüsler QP 相变判据（定理 5A）。"""
    k = spec.num_verifiers
    qp1 = solve_positive_quadrant_qp(spec.cov1_arr)
    qp0 = solve_positive_quadrant_qp(spec.cov0_arr)

    g1 = float(qp1["gamma"])
    g0 = float(qp0["gamma"])
    beta1 = float(qp1["multipliers"] @ spec.mu1_arr)
    beta0 = float(qp0["multipliers"] @ spec.mu0_arr)
    act1_len = len(qp1["active_set"])
    act0_len = len(qp0["active_set"])

    kappa_min_eff = math.sqrt(g1 / g0)
    kappa_mean = math.sqrt(float(np.sum(spec.cov0_arr)) / float(np.sum(spec.cov1_arr)))
    mean_dmu = float(np.mean(spec.mu1_arr - spec.mu0_arr))
    mean_collapses = bool((kappa_mean > 1.0 + 1e-12) or (abs(kappa_mean - 1.0) <= 1e-12 and mean_dmu <= 0.0))

    gamma0_cauchy_lb = float((k * k) / float(np.sum(spec.cov0_arr)))

    if abs(g0 - g1) > 1e-11:
        if g0 > g1:
            min_will_collapse = False
            regime = "min_consistent_quadratic"
        else:
            min_will_collapse = True
            regime = "min_collapse_quadratic"
    elif abs(beta1 - beta0) > 1e-11:
        if beta1 > beta0:
            min_will_collapse = False
            regime = "min_consistent_linear_tiebreak"
        else:
            min_will_collapse = True
            regime = "min_collapse_linear_tiebreak"
    else:
        if act0_len > act1_len:
            min_will_collapse = False
            regime = "min_consistent_log_tiebreak"
        elif act0_len < act1_len:
            min_will_collapse = True
            regime = "min_collapse_log_tiebreak"
        else:
            min_will_collapse = False
            regime = "min_boundary_identical_tail"

    return {
        "gamma1": g1,
        "gamma0": g0,
        "beta1": beta1,
        "beta0": beta0,
        "active_set1": qp1["active_set"],
        "active_set0": qp0["active_set"],
        "weights1": qp1["multipliers"],
        "weights0": qp0["multipliers"],
        "kappa_min_eff": float(kappa_min_eff),
        "kappa_mean": float(kappa_mean),
        "gamma0_cauchy_lb": float(gamma0_cauchy_lb),
        "mean_collapses": mean_collapses,
        "min_will_collapse": bool(min_will_collapse),
        "min_rescues_mean_in_gaussian": bool(mean_collapses and (not min_will_collapse)),
        "regime": regime,
    }


@dataclass(frozen=True)
class GaussianCopulaHeavyTailSpec:
    """一般相关矩阵下的 Gaussian-Copula + Student-t 统一重尾模型（定理 5B 统一版）。

    给定 Y=c in {0, 1}，先生成相关潜变量 Z ~ N(0, R_c)，再经逐维边际分位映射：
        S^{(k)} = mu_c + sigma_c * F_{t_{alpha_c}}^{-1}(Phi(Z_k))
    其中当 epsilon_shared > 0 时，错类样本以概率 epsilon_shared 落入全共享同源分支。
    """

    p: float
    alpha1: float
    alpha0: float
    corr1: ArrayLike
    corr0: ArrayLike
    mu1: float = 1.2
    sigma1: float = 1.0
    mu0: float = 0.0
    sigma0: float = 1.0
    epsilon_shared: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.p) < 1.0):
            raise ValueError("p 必须位于 (0, 1) 内")
        if self.alpha1 <= 0.0 or self.alpha0 <= 0.0:
            raise ValueError("alpha1 与 alpha0 必须 > 0")
        if self.sigma1 <= 0.0 or self.sigma0 <= 0.0:
            raise ValueError("sigma1 与 sigma0 必须 > 0")
        if not (0.0 <= self.epsilon_shared <= 1.0):
            raise ValueError("epsilon_shared 必须位于 [0, 1] 内")
        r1 = _validate_spd_matrix(self.corr1, "corr1")
        r0 = _validate_spd_matrix(self.corr0, "corr0", expected_dim=r1.shape[0])
        if not np.allclose(np.diag(r1), 1.0, atol=1e-6) or not np.allclose(np.diag(r0), 1.0, atol=1e-6):
            raise ValueError("corr1 与 corr0 必须为对角元全等于 1 的相关矩阵")
        object.__setattr__(self, "corr1", r1)
        object.__setattr__(self, "corr0", r0)

    @property
    def num_verifiers(self) -> int:
        return int(self.corr1_arr.shape[0])

    @property
    def corr1_arr(self) -> np.ndarray:
        return np.asarray(self.corr1, dtype=np.float64)

    @property
    def corr0_arr(self) -> np.ndarray:
        return np.asarray(self.corr0, dtype=np.float64)


def gaussian_copula_heavy_tail_criterion(spec: GaussianCopulaHeavyTailSpec) -> dict:
    """判定 Gaussian-Copula + Student-t 重尾模型下剩余尾相依系数与统一相界（定理 5B）。"""
    qp1 = solve_positive_quadrant_qp(spec.corr1_arr)
    qp0 = solve_positive_quadrant_qp(spec.corr0_arr)

    eta1 = 1.0 / float(qp1["gamma"])
    eta0_indep = 1.0 / float(qp0["gamma"])
    eta0_eff = 1.0 if spec.epsilon_shared > 0.0 else eta0_indep

    eff_alpha1 = float(spec.alpha1) / eta1
    eff_alpha0_min = float(spec.alpha0) / eta0_eff

    mean_will_collapse = bool(spec.alpha0 < spec.alpha1)
    if eff_alpha0_min > eff_alpha1 + 1e-12:
        min_will_collapse = False
        regime = "min_consistent_copula"
    elif eff_alpha0_min < eff_alpha1 - 1e-12:
        min_will_collapse = True
        regime = (
            "shared_upper_tail_breakdown"
            if spec.epsilon_shared > 0.0 and (spec.alpha0 / eta0_indep > eff_alpha1)
            else "min_collapse_copula"
        )
    else:
        min_will_collapse = False
        regime = "boundary_equal_effective_alpha"

    return {
        "eta1": float(eta1),
        "eta0_copula": float(eta0_indep),
        "eta0_eff": float(eta0_eff),
        "effective_alpha1": float(eff_alpha1),
        "effective_alpha0_min": float(eff_alpha0_min),
        "mean_will_collapse": mean_will_collapse,
        "min_will_collapse": bool(min_will_collapse),
        "regime": regime,
    }


def _sample_multi_verifier_scores(
    spec: Union[
        MultiVerifierGaussianSpec,
        HeavyTailMultiVerifierSpec,
        HeterogeneousVulnerabilitySpec,
    ],
    b: int,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """抽取形状为 (b, n, K) 的多裁判打分张量与 (b, n) 的正确性标签。"""
    k = int(spec.num_verifiers)
    labels = rng.random((b, n)) < float(spec.p)
    scores = np.empty((b, n, k), dtype=np.float64)

    n_pos = int(np.count_nonzero(labels))
    n_neg = labels.size - n_pos

    if isinstance(spec, MultiVerifierGaussianSpec):
        if n_pos > 0:
            if k == 1 or spec.rho1 == 0.0:
                scores[labels] = rng.normal(spec.mu1, spec.sigma1, size=(n_pos, k))
            else:
                z = rng.normal(0.0, 1.0, size=(n_pos, 1))
                e = rng.normal(0.0, 1.0, size=(n_pos, k))
                scores[labels] = spec.mu1 + spec.sigma1 * (
                    math.sqrt(spec.rho1) * z + math.sqrt(1.0 - spec.rho1) * e
                )
        if n_neg > 0:
            if k == 1 or spec.rho0 == 0.0:
                scores[~labels] = rng.normal(spec.mu0, spec.sigma0, size=(n_neg, k))
            else:
                z = rng.normal(0.0, 1.0, size=(n_neg, 1))
                e = rng.normal(0.0, 1.0, size=(n_neg, k))
                scores[~labels] = spec.mu0 + spec.sigma0 * (
                    math.sqrt(spec.rho0) * z + math.sqrt(1.0 - spec.rho0) * e
                )
        return scores, labels

    if isinstance(spec, HeavyTailMultiVerifierSpec):
        if n_pos > 0:
            t1 = rng.standard_t(spec.alpha1, size=(n_pos, 1))
            eps1 = rng.normal(0.0, 1.0, size=(n_pos, k)) if spec.eta1 > 0 else 0.0
            scores[labels] = spec.mu1 + spec.sigma1 * t1 + spec.eta1 * eps1
        if n_neg > 0:
            neg_scores = spec.mu0 + spec.sigma0 * rng.standard_t(
                spec.alpha0, size=(n_neg, k)
            )
            if spec.epsilon_shared > 0.0:
                shared_mask = rng.random(n_neg) < spec.epsilon_shared
                n_shared = int(np.count_nonzero(shared_mask))
                if n_shared > 0:
                    t_s = rng.standard_t(spec.alpha0, size=(n_shared, 1))
                    eps0 = (
                        rng.normal(0.0, 1.0, size=(n_shared, k))
                        if spec.eta0 > 0
                        else 0.0
                    )
                    neg_scores[shared_mask] = (
                        spec.mu0 + spec.sigma0 * t_s + spec.eta0 * eps0
                    )
            scores[~labels] = neg_scores
        return scores, labels

    if isinstance(spec, HeterogeneousVulnerabilitySpec):
        if n_pos > 0:
            t1 = rng.standard_t(spec.alpha1, size=(n_pos, 1))
            eps1 = rng.normal(0.0, 1.0, size=(n_pos, k)) if spec.eta1 > 0 else 0.0
            scores[labels] = spec.mu1 + spec.sigma1 * t1 + spec.eta1 * eps1
        if n_neg > 0:
            n_healthy = k - int(spec.n_vulnerable)
            neg_scores = np.empty((n_neg, k), dtype=np.float64)
            if n_healthy > 0:
                neg_scores[:, :n_healthy] = rng.normal(
                    spec.mu0, spec.sigma0, size=(n_neg, n_healthy)
                )
            if spec.n_vulnerable > 0:
                neg_scores[:, n_healthy:] = spec.mu0 + spec.sigma_vuln * rng.standard_t(
                    spec.alpha0, size=(n_neg, int(spec.n_vulnerable))
                )
            scores[~labels] = neg_scores
        return scores, labels

    if isinstance(spec, GeneralGaussianMultiVerifierSpec):
        l1 = np.linalg.cholesky(spec.cov1_arr)
        l0 = np.linalg.cholesky(spec.cov0_arr)
        if n_pos > 0:
            z1 = rng.normal(0.0, 1.0, size=(n_pos, k))
            scores[labels] = spec.mu1_arr[None, :] + z1 @ l1.T
        if n_neg > 0:
            z0 = rng.normal(0.0, 1.0, size=(n_neg, k))
            scores[~labels] = spec.mu0_arr[None, :] + z0 @ l0.T
        return scores, labels

    if isinstance(spec, GaussianCopulaHeavyTailSpec):
        l1 = np.linalg.cholesky(spec.corr1_arr)
        l0 = np.linalg.cholesky(spec.corr0_arr)
        eps_clip = 1e-12
        if n_pos > 0:
            z1 = rng.normal(0.0, 1.0, size=(n_pos, k)) @ l1.T
            u1 = np.clip(special.ndtr(z1), eps_clip, 1.0 - eps_clip)
            t1 = special.stdtrit(spec.alpha1, u1)
            scores[labels] = spec.mu1 + spec.sigma1 * t1
        if n_neg > 0:
            z0 = rng.normal(0.0, 1.0, size=(n_neg, k)) @ l0.T
            u0 = np.clip(special.ndtr(z0), eps_clip, 1.0 - eps_clip)
            t0 = special.stdtrit(spec.alpha0, u0)
            if spec.epsilon_shared > 0.0:
                shared_mask = rng.random(n_neg) < spec.epsilon_shared
                n_shared = int(np.count_nonzero(shared_mask))
                if n_shared > 0:
                    # 同源漏洞事件：全体裁判共受同一极值重尾因子驱动（lambda_{U,0} = 1）
                    t_s = rng.standard_t(spec.alpha0, size=(n_shared, 1))
                    t0[shared_mask] = t_s
            scores[~labels] = spec.mu0 + spec.sigma0 * t0
        return scores, labels

    raise TypeError(f"不支持的多裁判规格类型: {type(spec)!r}")


def simulate_multi_verifier_bon(
    spec: Union[
        MultiVerifierGaussianSpec,
        HeavyTailMultiVerifierSpec,
        HeterogeneousVulnerabilitySpec,
        GeneralGaussianMultiVerifierSpec,
        GaussianCopulaHeavyTailSpec,
    ],
    n_values: Sequence[int] | np.ndarray,
    aggregation: str = "min",
    n_trials: int = 100_000,
    seed: int = 42,
    batch_size: int = 5_000,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """流式分块向量化仿真多裁判融合（Min / Mean / Linear-Ensemble）的 Best-of-N 正确率。

    采用按采样预算增量滚动更新最大值，单块三维张量内存严格控制在 ~32 MB 以内。
    """
    if aggregation not in ("min", "mean", "linear"):
        raise ValueError(f"aggregation 必须为 'min', 'mean' 或 'linear'，当前收到: {aggregation!r}")
    w_arr: np.ndarray | None = None
    if aggregation == "linear":
        if weights is None:
            raise ValueError("aggregation='linear' 时必须提供一维权重向量 weights")
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if w_arr.size != int(spec.num_verifiers):
            raise ValueError(f"weights 长度 ({w_arr.size}) 必须等于 num_verifiers ({spec.num_verifiers})")
    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials 与 batch_size 必须 >= 1")

    orig_n = np.asarray(list(n_values), dtype=np.int64)
    if orig_n.ndim != 1 or orig_n.size == 0 or np.any(orig_n < 1):
        raise ValueError("n_values 必须为非空一维正整数序列")
    unique_n, inv_idx = np.unique(orig_n, return_inverse=True)

    k_ver = int(spec.num_verifiers)
    rng = np.random.default_rng(seed)
    correct_counts = np.zeros(len(unique_n), dtype=np.int64)

    # 控制三维分块 (b, step_n, K) 总元素不超过 4,000,000 (~32 MB)
    max_elements = 4_000_000
    done = 0
    while done < n_trials:
        b = min(int(batch_size), n_trials - done)
        best_score = np.full(b, -np.inf, dtype=np.float64)
        best_label = np.zeros(b, dtype=bool)
        tie_count = np.ones(b, dtype=np.int64)

        prev_n = 0
        for idx_n, target_n in enumerate(unique_n):
            remain = int(target_n) - prev_n
            max_step = max(1, max_elements // max(1, b * k_ver))
            while remain > 0:
                step = min(remain, max_step)
                raw_scores, chunk_labels = _sample_multi_verifier_scores(
                    spec, b, step, rng
                )
                if aggregation == "min":
                    agg_scores = np.min(raw_scores, axis=2)
                elif aggregation == "mean":
                    agg_scores = np.mean(raw_scores, axis=2)
                else:
                    assert w_arr is not None
                    agg_scores = raw_scores @ w_arr

                # 滚动更新批内最大值及水塘平局随机打破
                chunk_max = np.max(agg_scores, axis=1)
                is_cmax = agg_scores == chunk_max[:, None]
                if np.any(np.sum(is_cmax, axis=1) > 1):
                    noise = np.where(is_cmax, rng.random(agg_scores.shape), -1.0)
                    w_idx = np.argmax(noise, axis=1)
                    c_mult = np.sum(is_cmax, axis=1)
                else:
                    w_idx = np.argmax(agg_scores, axis=1)
                    c_mult = np.ones(b, dtype=np.int64)

                w_lbl = chunk_labels[np.arange(b), w_idx]
                better = chunk_max > best_score
                if np.any(better):
                    best_score[better] = chunk_max[better]
                    best_label[better] = w_lbl[better]
                    tie_count[better] = c_mult[better]

                tie = chunk_max == best_score
                if np.any(tie):
                    new_tot = tie_count[tie] + c_mult[tie]
                    prob = c_mult[tie] / new_tot
                    rep = rng.random(int(np.count_nonzero(tie))) < prob
                    t_rows = np.flatnonzero(tie)[rep]
                    best_label[t_rows] = w_lbl[t_rows]
                    tie_count[tie] = new_tot

                remain -= step
            correct_counts[idx_n] += int(np.count_nonzero(best_label))
            prev_n = int(target_n)

        done += b

    acc, se, ci_low, ci_high = _wilson_ci(correct_counts, n_trials)
    return {
        "n_values": orig_n,
        "accuracy": acc[inv_idx],
        "se": se[inv_idx],
        "ci_low": ci_low[inv_idx],
        "ci_high": ci_high[inv_idx],
    }


__all__ = [
    "GaussianCopulaHeavyTailSpec",
    "GeneralGaussianMultiVerifierSpec",
    "HeavyTailMultiVerifierSpec",
    "HeterogeneousVulnerabilitySpec",
    "MinEnsembleVerifierSpec",
    "MultiVerifierGaussianSpec",
    "ensemble_phase_reversal_criterion",
    "gaussian_copula_heavy_tail_criterion",
    "general_min_ensemble_qp_criterion",
    "heavy_tail_separation_criterion",
    "optimal_linear_ensemble_criterion",
    "simulate_multi_verifier_bon",
    "solve_positive_quadrant_qp",
]
