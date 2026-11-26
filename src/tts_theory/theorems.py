"""测试时计算缩放（Test-Time Scaling）相变理论与闭式解析模块。

本模块严格实现论文稿 `PROOF.md` 中定理 1–4 所推导的分布模型、相变判据、
闭式最优预算公式、二元 Verifier 缩放律与校准加权多数决收敛边界。
全程仅依赖 Python 标准库、NumPy 与 SciPy，无任何外部网络或 LLM API 调用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence, Union, overload, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special

ScalarOrArray = Union[float, np.ndarray]

_EULER_GAMMA: float = 0.5772156649015329
_ZETA_2: float = math.pi * math.pi / 6.0
_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


def _prepare_array(x: ArrayLike) -> tuple[np.ndarray, bool]:
    """将标量或数组输入统一转换为 float64 ndarray，并记录原始输入是否为标量。"""
    arr = np.asarray(x, dtype=np.float64)
    is_scalar = arr.ndim == 0
    if is_scalar:
        return arr.reshape(1), True
    return arr, False


def _format_output(arr: np.ndarray, is_scalar: bool) -> ScalarOrArray:
    """根据原始输入形态还原标量 float 或 ndarray 输出。"""
    if is_scalar:
        return float(arr.reshape(()))
    return arr


def _norm_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """数值稳定的正态对数密度函数。"""
    z = (x - mu) / sigma
    return -_LOG_SQRT_2PI - math.log(sigma) - 0.5 * z * z


def _norm_logsf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """数值稳定的正态对数生存函数 ln(1 - Phi((x-mu)/sigma))。"""
    z = (x - mu) / sigma
    return special.log_ndtr(-z)


def _norm_logcdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """数值稳定的正态对数分布函数 ln(Phi((x-mu)/sigma))。"""
    z = (x - mu) / sigma
    return special.log_ndtr(z)


def _std_norm_inv_sf(q: np.ndarray) -> np.ndarray:
    """计算标准正态上分位数 Phi^{-1}(1 - q)，在极小 q < 1e-300 时无下溢退化。"""
    res = np.empty_like(q, dtype=np.float64)
    zero_mask = q <= 0.0
    one_mask = q >= 1.0
    mid_mask = (~zero_mask) & (~one_mask) & (q >= 1e-300)
    ultra_tail = (~zero_mask) & (~one_mask) & (q < 1e-300)

    res[zero_mask] = np.inf
    res[one_mask] = -np.inf
    if np.any(mid_mask):
        res[mid_mask] = -special.ndtri(q[mid_mask])
    if np.any(ultra_tail):
        # 对 q < 1e-300 使用两阶渐近展开与 3 步 log_ndtr 牛顿精化
        lq = np.log(q[ultra_tail])
        u = -2.0 * lq - math.log(2.0 * math.pi)
        z = np.sqrt(np.maximum(u - np.log(np.maximum(u, 1.0)), 1.0))
        for _ in range(4):
            err = special.log_ndtr(-z) - lq
            slope = -np.exp(-_LOG_SQRT_2PI - 0.5 * z * z - special.log_ndtr(-z))
            z = z - err / slope
        res[ultra_tail] = z
    return res


def _std_norm_hazard_and_deriv(z: float) -> tuple[float, float]:
    """计算标准正态危险率 h(z)=phi(z)/sf(z) 及其一阶导 h'(z)=h(z)(h(z)-z)。

    当 z > 20 时采用高阶渐近展开计算 h(z) - z，避免大数相消损失有效数字。
    """
    if z > 20.0:
        invz = 1.0 / z
        invz2 = invz * invz
        diff = invz * (
            1.0
            - 2.0 * invz2
            + 10.0 * invz2 * invz2
            - 74.0 * (invz2**3)
            + 706.0 * (invz2**4)
        )
        h = z + diff
        return h, h * diff
    log_phi = -_LOG_SQRT_2PI - 0.5 * z * z
    log_sf = float(special.log_ndtr(-z))
    h = math.exp(log_phi - log_sf)
    return h, h * (h - z)


def _logsumexp_components(log_terms: list[np.ndarray]) -> np.ndarray:
    """对多个同形状对数项列表执行数值稳定的 logsumexp。"""
    stacked = np.stack(log_terms, axis=0)
    return special.logsumexp(stacked, axis=0)


def _gaussian_mixture_isf(
    q: ArrayLike,
    weights: Sequence[float],
    mus: Sequence[float],
    sigmas: Sequence[float],
) -> ScalarOrArray:
    """高斯混合分布上分位数 F_mix^{-1}(1 - q) 的双精度安全求根器。

    在对数生存函数空间求解 log_mix_sf(x) = ln(q)，结合解析单成分上下界与
    带二分兜底的保护牛顿法（Safeguarded Newton-Raphson），确保在 q 极小时
    （如 q = 1e-15 乃至 1e-100）仍保持双精度且支持全向量化运算。
    """
    q_arr, is_scalar = _prepare_array(q)
    if np.any(np.isnan(q_arr)) or np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("分位数参数 q 必须全部位于 [0, 1] 区间内。")

    active = [
        (float(w), float(m), float(s))
        for w, m, s in zip(weights, mus, sigmas)
        if w > 0.0
    ]
    log_w = [math.log(w) for w, _, _ in active]

    out = np.empty_like(q_arr, dtype=np.float64)
    zero_mask = q_arr == 0.0
    one_mask = q_arr == 1.0
    interior_mask = ~(zero_mask | one_mask)

    out[zero_mask] = np.inf
    out[one_mask] = -np.inf
    if not np.any(interior_mask):
        return _format_output(out, is_scalar)

    qi = q_arr[interior_mask]
    log_qi = np.log(qi)

    # 构造严格且紧凑的解析上下界 [lo, hi]：
    # hi 取各成分单独满足 sf_k(x) = q 的最大值 => 所有成分 sf_k(hi) <= q => mix_sf(hi) <= q
    # lo 取各成分单独满足 sf_k(x) = q 的最小值，并利用 w_k * sf_k(x) <= mix_sf(x) 收紧下界
    z_q = _std_norm_inv_sf(qi)
    hi = np.full_like(qi, -np.inf)
    lo_min = np.full_like(qi, np.inf)
    lo_tight = np.full_like(qi, -np.inf)
    for w, mu, sigma in active:
        x_k = mu + sigma * z_q
        hi = np.maximum(hi, x_k)
        lo_min = np.minimum(lo_min, x_k)
        sub_mask = qi < w
        if np.any(sub_mask):
            z_sub = _std_norm_inv_sf(qi[sub_mask] / w)
            lo_tight[sub_mask] = np.maximum(
                lo_tight[sub_mask], mu + sigma * z_sub
            )
    lo = np.maximum(lo_min, lo_tight)
    # 防止浮点舍入导致极限情况边界略微越界
    pad = 1e-12 * np.maximum(1.0, np.abs(hi - lo))
    lo = lo - pad
    hi = hi + pad

    def _eval_log_sf_and_grad(x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_sf_terms = [
            lw + _norm_logsf(x_val, mu, sigma)
            for lw, (_, mu, sigma) in zip(log_w, active)
        ]
        log_pdf_terms = [
            lw + _norm_logpdf(x_val, mu, sigma)
            for lw, (_, mu, sigma) in zip(log_w, active)
        ]
        lsf = _logsumexp_components(log_sf_terms)
        lpdf = _logsumexp_components(log_pdf_terms)
        # d/dx [log_mix_sf(x)] = -mix_pdf(x) / mix_sf(x) = -exp(lpdf - lsf)
        grad = -np.exp(np.clip(lpdf - lsf, -700.0, 700.0))
        return lsf, grad

    # 先对宽区间进行少量二分预处理，随后执行保护牛顿迭代至机器精度
    x = 0.5 * (lo + hi)
    for it in range(40):
        lsf, grad = _eval_log_sf_and_grad(x)
        fval = lsf - log_qi
        # log_mix_sf 严格单调递减：fval > 0 说明 x 偏小，应右移更新 lo
        lo = np.where(fval > 0.0, x, lo)
        hi = np.where(fval <= 0.0, x, hi)

        if np.all(np.abs(hi - lo) <= 1e-14 * np.maximum(1.0, np.abs(x))):
            break

        if it >= 4:
            # 牛顿步：x_new = x - fval / grad
            safe_grad = np.where(np.abs(grad) > 1e-300, grad, -1e-300)
            x_newton = x - fval / safe_grad
            inside = (x_newton > lo) & (x_newton < hi)
            x = np.where(inside, x_newton, 0.5 * (lo + hi))
        else:
            x = 0.5 * (lo + hi)

    out[interior_mask] = x
    return _format_output(out, is_scalar)


def _gaussian_mixture_ppf(
    u: ArrayLike,
    weights: Sequence[float],
    mus: Sequence[float],
    sigmas: Sequence[float],
) -> ScalarOrArray:
    """高斯混合分布下分位数 F_mix^{-1}(u) 的双精度安全求根器。"""
    u_arr, is_scalar = _prepare_array(u)
    if np.any(np.isnan(u_arr)) or np.any((u_arr < 0.0) | (u_arr > 1.0)):
        raise ValueError("分位数参数 u 必须全部位于 [0, 1] 区间内。")

    # 对于 u > 0.5 直接映射到上分位数 mix_isf(1 - u) 仅在 u < 0.5 时对下尾对数 CDF 求解
    out = np.empty_like(u_arr, dtype=np.float64)
    upper_mask = u_arr > 0.5
    lower_mask = ~upper_mask

    if np.any(upper_mask):
        out[upper_mask] = np.asarray(
            _gaussian_mixture_isf(1.0 - u_arr[upper_mask], weights, mus, sigmas),
            dtype=np.float64,
        )
    if np.any(lower_mask):
        # 对称翻转为 (-mu, sigma) 的上分位数即可复用同一高精引擎：
        # P(X <= x) = u  <=>  P(-X >= -x) = u  <=>  x = -isf(-X; u)
        neg_mus = [-float(m) for m in mus]
        out[lower_mask] = -np.asarray(
            _gaussian_mixture_isf(u_arr[lower_mask], weights, neg_mus, sigmas),
            dtype=np.float64,
        )
    return _format_output(out, is_scalar)


@runtime_checkable
class ContinuousVerifierSpec(Protocol):
    """连续打分 Verifier 条件分布与后验分布的统一协议接口。"""

    p: float

    def pdf1(self, x: ArrayLike) -> ScalarOrArray:
        """正确样本条件概率密度 f_1(x)。"""
        ...

    def pdf0(self, x: ArrayLike) -> ScalarOrArray:
        """错误样本条件概率密度 f_0(x)。"""
        ...

    def cdf1(self, x: ArrayLike) -> ScalarOrArray:
        """正确样本条件累积分布函数 F_1(x)。"""
        ...

    def cdf0(self, x: ArrayLike) -> ScalarOrArray:
        """错误样本条件累积分布函数 F_0(x)。"""
        ...

    def sf1(self, x: ArrayLike) -> ScalarOrArray:
        """正确样本条件生存函数 1 - F_1(x)。"""
        ...

    def sf0(self, x: ArrayLike) -> ScalarOrArray:
        """错误样本条件生存函数 1 - F_0(x)。"""
        ...

    def mix_pdf(self, x: ArrayLike) -> ScalarOrArray:
        """边际混合密度 p*f_1(x) + (1-p)*f_0(x)。"""
        ...

    def mix_cdf(self, x: ArrayLike) -> ScalarOrArray:
        """边际混合分布函数 p*F_1(x) + (1-p)*F_0(x)。"""
        ...

    def mix_sf(self, x: ArrayLike) -> ScalarOrArray:
        """边际混合生存函数 p*sf_1(x) + (1-p)*sf_0(x)。"""
        ...

    def mix_isf(self, q: ArrayLike) -> ScalarOrArray:
        """混合分布上分位数 F_mix^{-1}(1 - q)，对极小 q 保证双精度稳定。"""
        ...

    def mix_ppf(self, u: ArrayLike) -> ScalarOrArray:
        """混合分布下分位数 F_mix^{-1}(u)。"""
        ...

    def posterior(self, x: ArrayLike) -> ScalarOrArray:
        """给定打分 S=x 时样本正确的贝叶斯后验概率 P(Y=1 | S=x)。"""
        ...

    def log_likelihood_ratio(self, x: ArrayLike) -> ScalarOrArray:
        """对数似然比 ln(f_1(x) / f_0(x))。"""
        ...


@dataclass(frozen=True)
class GaussianVerifierSpec:
    """高斯条件打分 Verifier 模型：S|Y=1 ~ N(mu1, sigma1^2), S|Y=0 ~ N(mu0, sigma0^2)。"""

    p: float
    mu1: float
    sigma1: float
    mu0: float = 0.0
    sigma0: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"先验正确率 p 必须严格位于 (0, 1) 内，当前为 {self.p}。")
        if self.sigma1 <= 0.0 or self.sigma0 <= 0.0:
            raise ValueError("条件标准差 sigma1 与 sigma0 必须严格大于 0。")

    @property
    def delta_mu(self) -> float:
        """正确样本与错误样本打分均值差 Delta_mu = mu1 - mu0。"""
        return float(self.mu1 - self.mu0)

    @property
    def kappa(self) -> float:
        """错对样本标准差比 kappa = sigma0 / sigma1（相变临界点为 kappa_c = 1）。"""
        return float(self.sigma0 / self.sigma1)

    def _weights(self) -> tuple[float, float]:
        return (float(self.p), float(1.0 - self.p))

    def pdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = np.exp(_norm_logpdf(arr, self.mu1, self.sigma1))
        return _format_output(val, is_scalar)

    def pdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = np.exp(_norm_logpdf(arr, self.mu0, self.sigma0))
        return _format_output(val, is_scalar)

    def cdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr((arr - self.mu1) / self.sigma1)
        return _format_output(val, is_scalar)

    def cdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr((arr - self.mu0) / self.sigma0)
        return _format_output(val, is_scalar)

    def sf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr(-(arr - self.mu1) / self.sigma1)
        return _format_output(val, is_scalar)

    def sf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr(-(arr - self.mu0) / self.sigma0)
        return _format_output(val, is_scalar)

    def mix_pdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        l1 = math.log(self.p) + _norm_logpdf(arr, self.mu1, self.sigma1)
        l0 = math.log1p(-self.p) + _norm_logpdf(arr, self.mu0, self.sigma0)
        val = np.exp(_logsumexp_components([l1, l0]))
        return _format_output(val, is_scalar)

    def mix_cdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = self.p * special.ndtr((arr - self.mu1) / self.sigma1) + (
            1.0 - self.p
        ) * special.ndtr((arr - self.mu0) / self.sigma0)
        return _format_output(val, is_scalar)

    def mix_sf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = self.p * special.ndtr(-(arr - self.mu1) / self.sigma1) + (
            1.0 - self.p
        ) * special.ndtr(-(arr - self.mu0) / self.sigma0)
        return _format_output(val, is_scalar)

    def mix_isf(self, q: ArrayLike) -> ScalarOrArray:
        return _gaussian_mixture_isf(
            q,
            self._weights(),
            (self.mu1, self.mu0),
            (self.sigma1, self.sigma0),
        )

    def mix_ppf(self, u: ArrayLike) -> ScalarOrArray:
        return _gaussian_mixture_ppf(
            u,
            self._weights(),
            (self.mu1, self.mu0),
            (self.sigma1, self.sigma0),
        )

    def log_likelihood_ratio(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        z1 = (arr - self.mu1) / self.sigma1
        z0 = (arr - self.mu0) / self.sigma0
        llr = math.log(self.sigma0 / self.sigma1) - 0.5 * z1 * z1 + 0.5 * z0 * z0
        return _format_output(llr, is_scalar)

    def posterior(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        llr = np.asarray(self.log_likelihood_ratio(arr), dtype=np.float64)
        logit_prior = math.log(self.p / (1.0 - self.p))
        val = special.expit(logit_prior + llr)
        return _format_output(val, is_scalar)


@dataclass(frozen=True)
class ContaminatedVerifierSpec:
    """Reward Hacking 混合污染尾部 Verifier 模型。

    S|Y=1 ~ N(mu1, sigma1^2),
    S|Y=0 ~ (1 - epsilon) * N(mu0, sigma0^2) + epsilon * N(mu_hack, sigma_hack^2).
    """

    p: float
    mu1: float
    sigma1: float
    mu0: float = 0.0
    sigma0: float = 1.0
    epsilon: float = 0.05
    mu_hack: float = 3.5
    sigma_hack: float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"先验正确率 p 必须严格位于 (0, 1) 内，当前为 {self.p}。")
        if self.sigma1 <= 0.0 or self.sigma0 <= 0.0 or self.sigma_hack <= 0.0:
            raise ValueError("所有标准差参数 sigma1, sigma0, sigma_hack 必须严格大于 0。")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"污染比例 epsilon 必须位于 [0, 1] 内，当前为 {self.epsilon}。")

    def _mix_components(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        w1 = float(self.p)
        w0 = float((1.0 - self.p) * (1.0 - self.epsilon))
        wh = float((1.0 - self.p) * self.epsilon)
        return (
            (w1, w0, wh),
            (self.mu1, self.mu0, self.mu_hack),
            (self.sigma1, self.sigma0, self.sigma_hack),
        )

    def _log_pdf0_array(self, arr: np.ndarray) -> np.ndarray:
        if self.epsilon == 0.0:
            return _norm_logpdf(arr, self.mu0, self.sigma0)
        if self.epsilon == 1.0:
            return _norm_logpdf(arr, self.mu_hack, self.sigma_hack)
        l_base = math.log1p(-self.epsilon) + _norm_logpdf(arr, self.mu0, self.sigma0)
        l_hack = math.log(self.epsilon) + _norm_logpdf(
            arr, self.mu_hack, self.sigma_hack
        )
        return _logsumexp_components([l_base, l_hack])

    def pdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = np.exp(_norm_logpdf(arr, self.mu1, self.sigma1))
        return _format_output(val, is_scalar)

    def pdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = np.exp(self._log_pdf0_array(arr))
        return _format_output(val, is_scalar)

    def cdf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr((arr - self.mu1) / self.sigma1)
        return _format_output(val, is_scalar)

    def cdf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = (1.0 - self.epsilon) * special.ndtr(
            (arr - self.mu0) / self.sigma0
        ) + self.epsilon * special.ndtr((arr - self.mu_hack) / self.sigma_hack)
        return _format_output(val, is_scalar)

    def sf1(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = special.ndtr(-(arr - self.mu1) / self.sigma1)
        return _format_output(val, is_scalar)

    def sf0(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        val = (1.0 - self.epsilon) * special.ndtr(
            -(arr - self.mu0) / self.sigma0
        ) + self.epsilon * special.ndtr(-(arr - self.mu_hack) / self.sigma_hack)
        return _format_output(val, is_scalar)

    def mix_pdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        l1 = math.log(self.p) + _norm_logpdf(arr, self.mu1, self.sigma1)
        l0 = math.log1p(-self.p) + self._log_pdf0_array(arr)
        val = np.exp(_logsumexp_components([l1, l0]))
        return _format_output(val, is_scalar)

    def mix_cdf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        c1 = np.asarray(self.cdf1(arr), dtype=np.float64)
        c0 = np.asarray(self.cdf0(arr), dtype=np.float64)
        val = self.p * c1 + (1.0 - self.p) * c0
        return _format_output(val, is_scalar)

    def mix_sf(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        s1 = np.asarray(self.sf1(arr), dtype=np.float64)
        s0 = np.asarray(self.sf0(arr), dtype=np.float64)
        val = self.p * s1 + (1.0 - self.p) * s0
        return _format_output(val, is_scalar)

    def mix_isf(self, q: ArrayLike) -> ScalarOrArray:
        weights, mus, sigmas = self._mix_components()
        return _gaussian_mixture_isf(q, weights, mus, sigmas)

    def mix_ppf(self, u: ArrayLike) -> ScalarOrArray:
        weights, mus, sigmas = self._mix_components()
        return _gaussian_mixture_ppf(u, weights, mus, sigmas)

    def log_likelihood_ratio(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        l1 = _norm_logpdf(arr, self.mu1, self.sigma1)
        l0 = self._log_pdf0_array(arr)
        return _format_output(l1 - l0, is_scalar)

    def posterior(self, x: ArrayLike) -> ScalarOrArray:
        arr, is_scalar = _prepare_array(x)
        llr = np.asarray(self.log_likelihood_ratio(arr), dtype=np.float64)
        logit_prior = math.log(self.p / (1.0 - self.p))
        val = special.expit(logit_prior + llr)
        return _format_output(val, is_scalar)


@dataclass(frozen=True)
class BinaryVerifierSpec:
    """离散二元判别器模型：TPR = P(S=1|Y=1), FPR = P(S=1|Y=0)。"""

    p: float
    tpr: float
    fpr: float

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"先验正确率 p 必须严格位于 (0, 1) 内，当前为 {self.p}。")
        if not (0.0 <= self.tpr <= 1.0) or not (0.0 <= self.fpr <= 1.0):
            raise ValueError("TPR 与 FPR 必须位于 [0, 1] 区间内。")

    @property
    def q1(self) -> float:
        """单次独立采样判正概率 q1 = p*TPR + (1-p)*FPR。"""
        return float(self.p * self.tpr + (1.0 - self.p) * self.fpr)

    @property
    def ppv(self) -> float:
        """阳性预测值 PPV = P(Y=1 | S=1) = p*TPR / q1（若 q1=0 则回退至先验 p）。"""
        q1 = self.q1
        if q1 == 0.0:
            return float(self.p)
        return float((self.p * self.tpr) / q1)


def gaussian_peak_threshold(spec: GaussianVerifierSpec) -> float:
    """计算高斯异方差（sigma0 > sigma1）下的唯一对数似然比峰值位置 x_peak。"""
    if spec.sigma0 <= spec.sigma1:
        raise ValueError(
            "似然比极大值点 x_peak 仅在错误样本方差更重（sigma0 > sigma1）时存在。"
        )
    s0_sq = spec.sigma0 * spec.sigma0
    s1_sq = spec.sigma1 * spec.sigma1
    return float((spec.mu1 * s0_sq - spec.mu0 * s1_sq) / (s0_sq - s1_sq))


def check_collapse_criterion(spec: GaussianVerifierSpec) -> dict:
    """判定高斯 Verifier 的相变状态、单调性与渐近极限（定理 2）。

    在 sigma0 < sigma1（kappa < 1）且渐近一致（lim P_BoN(N) = 1）相区下，细分三种初值行为：
    - delta_mu > 0：对一切 N >= 1 全局严格递增（initial_step_sign = +1）；
    - delta_mu = 0：初值平局 P_BoN(1) = P_BoN(2) = p，随后对一切 N >= 2 严格递增趋向 1（initial_step_sign = 0）；
    - delta_mu < 0：严格先降后升（U 型谷底，P_BoN(2) < P_BoN(1) = p，跨过有限谷底后升向 1，initial_step_sign = -1）。

    返回字段:
        - regime: 相变区域标识
        - will_collapse: 是否存在 N -> inf 时准确率为 0 的坍塌
        - is_monotone_increasing: P_BoN(N) 是否对所有 N >= 1 全局严格递增
        - initial_step_sign: sgn(P_BoN(2) - P_BoN(1)) in {-1, 0, +1}
        - asymptotic_limit: lim_{N -> inf} P_BoN(N)
        - x_peak: sigma0 > sigma1 时的似然比峰值点，否则为 None
    """
    dmu = spec.delta_mu
    same_sigma = math.isclose(spec.sigma0, spec.sigma1, rel_tol=1e-12, abs_tol=1e-12)
    init_sign = 0 if math.isclose(dmu, 0.0, rel_tol=1e-12, abs_tol=1e-12) else (1 if dmu > 0.0 else -1)

    if not same_sigma and spec.sigma0 > spec.sigma1:
        return {
            "regime": "collapse",
            "will_collapse": True,
            "is_monotone_increasing": False,
            "initial_step_sign": init_sign,
            "asymptotic_limit": 0.0,
            "x_peak": gaussian_peak_threshold(spec),
        }

    if same_sigma:
        if init_sign == 0:
            return {
                "regime": "uninformative_constant",
                "will_collapse": False,
                "is_monotone_increasing": False,
                "initial_step_sign": 0,
                "asymptotic_limit": float(spec.p),
                "x_peak": None,
            }
        if dmu > 0.0:
            return {
                "regime": "monotone_consistent",
                "will_collapse": False,
                "is_monotone_increasing": True,
                "initial_step_sign": 1,
                "asymptotic_limit": 1.0,
                "x_peak": None,
            }
        return {
            "regime": "monotone_collapse",
            "will_collapse": True,
            "is_monotone_increasing": False,
            "initial_step_sign": -1,
            "asymptotic_limit": 0.0,
            "x_peak": None,
        }

    # sigma0 < sigma1：右尾似然比趋向 +inf，必然渐近一致；
    # 全局从 N=1 起严格递增当且仅当 P_BoN(2) > P_BoN(1)，即 delta_mu > 0；
    # delta_mu = 0 时 P_BoN(1) = P_BoN(2) = p，随后对一切 N >= 2 严格递增；delta_mu < 0 时呈严格 U 型。
    if init_sign > 0:
        return {
            "regime": "consistent_monotone",
            "will_collapse": False,
            "is_monotone_increasing": True,
            "initial_step_sign": 1,
            "asymptotic_limit": 1.0,
            "x_peak": None,
        }
    return {
        "regime": "non_monotone_recovery",
        "will_collapse": False,
        "is_monotone_increasing": False,
        "initial_step_sign": init_sign,
        "asymptotic_limit": 1.0,
        "x_peak": None,
    }


def _gaussian_mix_log_sf_scalar(spec: GaussianVerifierSpec, x: float) -> float:
    """标量版本的混合对数生存函数 ln(bar_F_mix(x))，极端右尾无下溢。"""
    ls1 = math.log(spec.p) + float(special.log_ndtr(-(x - spec.mu1) / spec.sigma1))
    ls0 = math.log1p(-spec.p) + float(special.log_ndtr(-(x - spec.mu0) / spec.sigma0))
    hi = max(ls1, ls0)
    return hi + math.log1p(math.exp(min(ls1, ls0) - hi))


def closed_form_optimal_n(
    spec: GaussianVerifierSpec,
    method: str = "saddle_refined",
) -> dict:
    """计算高斯异方差情形下的闭式最优采样预算 N*（定理 2）。

    参数:
        spec: 高斯 Verifier 规格
        method:
            - "tail_quantile": 零阶极值分位闭式解 N*_0 = 1 / bar_F_mix(x_peak)
            - "saddle_refined": 基于 Gumbel 极值矩与危险率曲率 Padé 修正的一阶闭式解

    返回字段:
        n_star_float, n_star, x_peak, peak_posterior, will_collapse, regime
    """
    if method not in ("tail_quantile", "saddle_refined"):
        raise ValueError(
            f"未知的闭式近似方法 '{method}'，仅支持 'tail_quantile' 或 'saddle_refined'。"
        )

    criterion = check_collapse_criterion(spec)
    regime = str(criterion["regime"])
    will_collapse = bool(criterion["will_collapse"])

    if not will_collapse:
        return {
            "n_star_float": None,
            "n_star": None,
            "x_peak": None,
            "peak_posterior": float(criterion["asymptotic_limit"]),
            "will_collapse": False,
            "regime": regime,
        }

    # 同方差但均值倒置（sigma0 == sigma1 且 mu1 < mu0）：单调递减，N=1 即最优
    if criterion["x_peak"] is None:
        return {
            "n_star_float": 1.0,
            "n_star": 1,
            "x_peak": None,
            "peak_posterior": 1.0,
            "will_collapse": True,
            "regime": regime,
        }

    x_peak = float(criterion["x_peak"])
    peak_posterior = float(spec.posterior(x_peak))

    # 由精确二样本解析恒等式 P(2)-P(1) = p(1-p)(2*Phi(dmu/sqrt(s1^2+s0^2)) - 1)
    # 结合 sigma0 > sigma1 下的 Karlin 严格单峰性：当 delta_mu <= 0 时 N*=1 严格最优（dmu=0 时与 N=2 并列）
    if spec.delta_mu <= 0.0:
        return {
            "n_star_float": 1.0,
            "n_star": 1,
            "x_peak": x_peak,
            "peak_posterior": peak_posterior,
            "will_collapse": True,
            "regime": regime,
        }

    if method == "tail_quantile":
        log_t_pk = _gaussian_mix_log_sf_scalar(spec, x_peak)
        # 在浮点数可表达范围内返回 exp(-log_t_pk)
        n_star_float = float(math.exp(-min(log_t_pk, 0.0)))
        n_star = max(2, int(round(n_star_float)))
        return {
            "n_star_float": n_star_float,
            "n_star": n_star,
            "x_peak": x_peak,
            "peak_posterior": peak_posterior,
            "will_collapse": True,
            "regime": regime,
        }

    # method == "saddle_refined":
    # 在 x_peak 处计算混合危险率 h 与局部曲率参数 r = h'(x_peak) / h(x_peak)^2
    z1 = (x_peak - spec.mu1) / spec.sigma1
    z0 = (x_peak - spec.mu0) / spec.sigma0
    ls1 = math.log(spec.p) + float(special.log_ndtr(-z1))
    ls0 = math.log1p(-spec.p) + float(special.log_ndtr(-z0))
    log_t_pk = _gaussian_mix_log_sf_scalar(spec, x_peak)

    w1 = math.exp(ls1 - log_t_pk)
    w0 = math.exp(ls0 - log_t_pk)

    hz1, hpz1 = _std_norm_hazard_and_deriv(z1)
    hz0, hpz0 = _std_norm_hazard_and_deriv(z0)
    h1 = hz1 / spec.sigma1
    hp1 = hpz1 / (spec.sigma1 * spec.sigma1)
    h0 = hz0 / spec.sigma0
    hp0 = hpz0 / (spec.sigma0 * spec.sigma0)

    h = w1 * h1 + w0 * h0
    hp = w1 * hp1 + w0 * hp0 - w1 * w0 * (h1 - h0) ** 2
    r = max(-0.25, hp / (h * h))

    # Gumbel 极值均值偏移 (-gamma) + 危险率方差收缩一阶 Padé 修正
    hy = -_EULER_GAMMA + (_ZETA_2 * r) / (1.0 + _ZETA_2 * r)
    u_star = x_peak + hy / h

    log_t_u = _gaussian_mix_log_sf_scalar(spec, u_star)
    if log_t_u < -20.0:
        n_star_float = float(math.exp(-log_t_u))
    else:
        t_u = math.exp(log_t_u)
        n_star_float = max(1.0, float(-1.0 / math.log1p(-t_u)))

    n_star = max(2, int(round(n_star_float)))
    return {
        "n_star_float": n_star_float,
        "n_star": n_star,
        "x_peak": x_peak,
        "peak_posterior": peak_posterior,
        "will_collapse": True,
        "regime": regime,
    }


def binary_bon_accuracy(
    spec: BinaryVerifierSpec,
    n: Union[int, ArrayLike],
) -> ScalarOrArray:
    """计算二元 Verifier（均匀打破平局与全负随机回退）的精确 BoN 正确率（定理 3）。"""
    n_arr, is_scalar = _prepare_array(n)
    if np.any(n_arr < 1.0):
        raise ValueError("采样预算 n 必须全部大于等于 1。")

    q1 = spec.q1
    if math.isclose(q1, 0.0, abs_tol=1e-15) or math.isclose(q1, 1.0, abs_tol=1e-15):
        out = np.full_like(n_arr, spec.p, dtype=np.float64)
        return _format_output(out, is_scalar)

    ppv = (spec.p * spec.tpr) / q1
    coeff = (
        spec.p * (1.0 - spec.p) * (spec.tpr - spec.fpr) / (q1 * (1.0 - q1))
    )
    one_minus_q1_pow = np.exp(n_arr * math.log1p(-q1))
    out = ppv - coeff * one_minus_q1_pow
    return _format_output(out, is_scalar)


def binary_asymptotic_ceiling(spec: BinaryVerifierSpec) -> float:
    """返回二元 Verifier 在 N -> inf 时的渐近正确率上限 PPV（定理 3）。"""
    return spec.ppv


def binary_bon_divergence_bounds(spec: BinaryVerifierSpec, n: int) -> dict:
    """按 beta=0 (PPV=1) 与 beta>0 边界完备计算二元 Verifier 的 KL / chi^2 / TV 散度界（定理 3）。"""
    if isinstance(n, (bool, np.bool_)) or int(n) != n or int(n) < 1:
        raise ValueError(f"采样预算 n 必须为 >= 1 的整数，当前为 {n}。")
    n_int = int(n)
    q1 = spec.q1
    ppv = spec.ppv
    p_n = float(binary_bon_accuracy(spec, n_int))
    tv_distance = float(abs(ppv - p_n))

    # 退化或无分辨能力情况：P_N 恒等于 PPV == p
    if (
        math.isclose(q1, 0.0, abs_tol=1e-15)
        or math.isclose(q1, 1.0, abs_tol=1e-15)
        or math.isclose(spec.tpr, spec.fpr, rel_tol=0.0, abs_tol=1e-15)
    ):
        return {
            "n": n_int,
            "p_n": p_n,
            "ppv": ppv,
            "tv_distance": 0.0,
            "kl_forward": 0.0,
            "chi2_upper_bound": 0.0,
            "kl_reverse": 0.0,
            "is_ppv_one_boundary": False,
        }

    # 无假阳性边界：FPR=0 且 TPR>0 => PPV=1，前向散度发散但 TV 与反向 KL 精确有限
    if math.isclose(ppv, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return {
            "n": n_int,
            "p_n": p_n,
            "ppv": 1.0,
            "tv_distance": tv_distance,
            "kl_forward": math.inf,
            "chi2_upper_bound": math.inf,
            "kl_reverse": float(-math.log(max(p_n, 1e-300))),
            "is_ppv_one_boundary": True,
        }

    # 无真阳性零界：TPR=0 且 FPR>0 => PPV=0
    if math.isclose(ppv, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return {
            "n": n_int,
            "p_n": p_n,
            "ppv": 0.0,
            "tv_distance": tv_distance,
            "kl_forward": math.inf,
            "chi2_upper_bound": math.inf,
            "kl_reverse": float(-math.log(max(1.0 - p_n, 1e-300))),
            "is_ppv_one_boundary": False,
        }

    # 一般内点情况 0 < PPV < 1
    t1 = p_n * math.log(p_n / ppv) if p_n > 0.0 else 0.0
    t0 = (1.0 - p_n) * math.log((1.0 - p_n) / (1.0 - ppv)) if p_n < 1.0 else 0.0
    kl_forward = float(max(0.0, t1 + t0))

    coeff = spec.p * (1.0 - spec.p) * (spec.tpr - spec.fpr) / (q1 * (1.0 - q1))
    chi2_upper = float(
        (coeff * coeff) / (ppv * (1.0 - ppv)) * math.exp(2.0 * float(n_int) * math.log1p(-q1))
    )

    r1 = ppv * math.log(ppv / max(p_n, 1e-300))
    r0 = (1.0 - ppv) * math.log((1.0 - ppv) / max(1.0 - p_n, 1e-300))
    kl_reverse = float(max(0.0, r1 + r0))

    return {
        "n": n_int,
        "p_n": p_n,
        "ppv": ppv,
        "tv_distance": tv_distance,
        "kl_forward": kl_forward,
        "chi2_upper_bound": chi2_upper,
        "kl_reverse": kl_reverse,
        "is_ppv_one_boundary": False,
    }


def binary_marginal_gain(
    spec: BinaryVerifierSpec,
    n: Union[int, ArrayLike],
) -> ScalarOrArray:
    """计算第 N 步增至 N+1 步的离散边际信息增益 P_bin(N+1) - P_bin(N)（定理 3）。"""
    n_arr, is_scalar = _prepare_array(n)
    if np.any(n_arr < 1.0):
        raise ValueError("采样预算 n 必须全部大于等于 1。")

    q1 = spec.q1
    if math.isclose(q1, 0.0, abs_tol=1e-15) or math.isclose(q1, 1.0, abs_tol=1e-15):
        out = np.zeros_like(n_arr, dtype=np.float64)
        return _format_output(out, is_scalar)

    gain_1 = spec.p * (1.0 - spec.p) * (spec.tpr - spec.fpr)
    out = (gain_1 / (1.0 - q1)) * np.exp(n_arr * math.log1p(-q1))
    return _format_output(out, is_scalar)


def binary_optimal_budget(
    spec: BinaryVerifierSpec,
    eps: float | None = None,
    cost_per_sample: float | None = None,
) -> int:
    """求解二元 Verifier 给定逼近容差 eps 或单样本成本 lambda 下的闭式最优采样规模 N*。

    参数需恰好指定 eps 或 cost_per_sample 二者之一。
    """
    if (eps is None) == (cost_per_sample is None):
        raise ValueError("必须且只能提供 eps 与 cost_per_sample 两个参数之一。")

    q1 = spec.q1
    if (
        spec.tpr <= spec.fpr
        or math.isclose(q1, 0.0, abs_tol=1e-15)
        or math.isclose(q1, 1.0, abs_tol=1e-15)
    ):
        return 1

    neg_log_one_minus_q1 = -math.log1p(-q1)

    if eps is not None:
        if eps <= 0.0:
            raise ValueError(f"逼近容差 eps 必须严格大于 0，当前为 {eps}。")
        gap_coeff = (
            spec.p * (1.0 - spec.p) * (spec.tpr - spec.fpr) / (q1 * (1.0 - q1))
        )
        if gap_coeff <= eps:
            return 1
        n_float = math.log(gap_coeff / eps) / neg_log_one_minus_q1
        return max(1, int(math.ceil(n_float - 1e-12)))

    assert cost_per_sample is not None
    if cost_per_sample <= 0.0:
        raise ValueError(
            f"单样本计算成本 cost_per_sample 必须严格大于 0，当前为 {cost_per_sample}。"
        )
    # 净收益最大化条件：P(N+1) - P(N) <= cost_per_sample
    gain_at_1 = spec.p * (1.0 - spec.p) * (spec.tpr - spec.fpr)
    if gain_at_1 <= cost_per_sample:
        return 1
    n_float = 1.0 + math.log(gain_at_1 / cost_per_sample) / neg_log_one_minus_q1
    return max(1, int(math.ceil(n_float - 1e-12)))


def calibrated_bon_asymptotic_limit(spec: ContinuousVerifierSpec) -> float:
    """计算校准后验得分 BoN（Calibrated-BoN）的精确渐近极限 sup_x q(x)（定理 4）。"""
    if isinstance(spec, GaussianVerifierSpec):
        crit = check_collapse_criterion(spec)
        if crit["x_peak"] is not None:
            return float(spec.posterior(float(crit["x_peak"])))
        if crit["regime"] == "uninformative_constant":
            return float(spec.p)
        return 1.0

    if isinstance(spec, ContaminatedVerifierSpec):
        # 检验左右无穷远尾部：当正确分布标准差严格大于负类全部成分时，尾部后验趋于 1
        if spec.epsilon == 0.0:
            equiv = GaussianVerifierSpec(
                p=spec.p,
                mu1=spec.mu1,
                sigma1=spec.sigma1,
                mu0=spec.mu0,
                sigma0=spec.sigma0,
            )
            return calibrated_bon_asymptotic_limit(equiv)
        if spec.epsilon == 1.0:
            equiv = GaussianVerifierSpec(
                p=spec.p,
                mu1=spec.mu1,
                sigma1=spec.sigma1,
                mu0=spec.mu_hack,
                sigma0=spec.sigma_hack,
            )
            return calibrated_bon_asymptotic_limit(equiv)

        max_neg_sigma = max(spec.sigma0, spec.sigma_hack)
        if spec.sigma1 > max_neg_sigma:
            return 1.0

        # 检查是否存在某一侧由同方差成分主导且正确均值占据最外侧：
        # 仅当在某一极值方向上 sigma1 等于该方向所有最宽负成分的方差且 mu1 占优时极限为 1
        tol = 1e-12
        if math.isclose(spec.sigma1, max_neg_sigma, rel_tol=tol, abs_tol=tol):
            right_ok = True
            left_ok = True
            for w, mu_n, s_n in (
                (1.0 - spec.epsilon, spec.mu0, spec.sigma0),
                (spec.epsilon, spec.mu_hack, spec.sigma_hack),
            ):
                if w > 0.0 and math.isclose(s_n, spec.sigma1, rel_tol=tol, abs_tol=tol):
                    if spec.mu1 <= mu_n:
                        right_ok = False
                    if spec.mu1 >= mu_n:
                        left_ok = False
            if right_ok or left_ok:
                return 1.0

    # 通用数值全局最大值搜索（结合宽域网格采样、分位点网格与局部 Brent 精化）
    q_grid = np.concatenate(
        [
            np.asarray(
                spec.mix_ppf(np.linspace(1e-6, 1.0 - 1e-6, 400)), dtype=np.float64
            ),
            np.linspace(-25.0, 25.0, 1000),
        ]
    )
    if isinstance(spec, ContaminatedVerifierSpec):
        centers = [spec.mu1, spec.mu0, spec.mu_hack]
        spread = max(spec.sigma1, spec.sigma0, spec.sigma_hack)
        extra = np.linspace(
            min(centers) - 12.0 * spread, max(centers) + 12.0 * spread, 1500
        )
        q_grid = np.unique(np.concatenate([q_grid, extra]))

    llrs = np.asarray(spec.log_likelihood_ratio(q_grid), dtype=np.float64)
    best_idx = int(np.argmax(llrs))
    best_llr = float(llrs[best_idx])

    # 在前几个局部极大值邻域做 Brent 精确求极大
    candidates = [best_idx]
    if len(llrs) >= 3:
        local_peaks = np.where((llrs[1:-1] >= llrs[:-2]) & (llrs[1:-1] >= llrs[2:]))[0] + 1
        candidates.extend(local_peaks.tolist())

    for idx in set(candidates):
        left = float(q_grid[max(0, idx - 2)])
        right = float(q_grid[min(len(q_grid) - 1, idx + 2)])
        if right > left:
            res = optimize.minimize_scalar(
                lambda z: -float(spec.log_likelihood_ratio(float(z))),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if -float(res.fun) > best_llr:
                best_llr = -float(res.fun)

    return float(special.expit(math.log(spec.p / (1.0 - spec.p)) + best_llr))


def weighted_majority_convergence_bound(
    p: float,
    p_wrong_max: float,
    mu_w_pos: float,
    mu_w_neg: float,
    w_max: float,
    wrong_answer_probs: Sequence[float] | None = None,
    wrong_mean_weights: Sequence[float] | None = None,
) -> dict:
    """计算校准加权多数决（定理 4B）的期望优势间隙与 Hoeffding/Bernstein 指数收敛常数。

    参数:
        p: 单次采样命中正确答案 a* 的概率
        p_wrong_max: 当未提供异质错类列表时，单个最强错误诱饵答案的生成概率 max_j P(A=a_j)
        mu_w_pos: 正确样本条件权重期望 E[w(S) | A=a*]
        mu_w_neg: 当未提供异质错类列表时，错误样本条件权重期望 E[w(S) | Y=0]
        w_max: 有界权重上界 max w(S) > 0
        wrong_answer_probs: 可选错误答案概率分布 [p_1, ..., p_J]（需与 wrong_mean_weights 成对提供）
        wrong_mean_weights: 可选每个错误答案各自条件期望权重 [mu_{w,1}, ..., mu_{w,J}]
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须位于 (0, 1) 内，当前为 {p}。")
    if (wrong_answer_probs is None) != (wrong_mean_weights is None):
        raise ValueError("wrong_answer_probs 与 wrong_mean_weights 必须同时提供或同时省略。")
    if mu_w_pos < 0.0 or mu_w_neg < 0.0 or w_max <= 0.0:
        raise ValueError("条件权重期望必须非负且 w_max 必须严格大于 0。")

    expected_pos = float(p * mu_w_pos)
    tail_bucket_prob_threshold = float(expected_pos / (2.0 * w_max))

    if wrong_answer_probs is not None and wrong_mean_weights is not None:
        pj = np.asarray(list(wrong_answer_probs), dtype=np.float64)
        mwj = np.asarray(list(wrong_mean_weights), dtype=np.float64)
        if pj.ndim != 1 or pj.size == 0 or pj.shape != mwj.shape:
            raise ValueError("wrong_answer_probs 与 wrong_mean_weights 必须为非空等长一维序列。")
        if np.any(pj < 0.0) or float(np.sum(pj)) > 1.0 - p + 1e-9:
            raise ValueError("错误答案概率必须非负且总和不超过 1 - p。")
        if np.any(mwj < 0.0) or np.any(mwj > w_max + 1e-12):
            raise ValueError("各错误答案条件期望权重必须位于 [0, w_max] 内。")
        wrong_scores = pj * mwj
        max_idx = int(np.argmax(wrong_scores))
        expected_wrong_max = float(wrong_scores[max_idx])
        eff_p_wrong = float(pj[max_idx])
    else:
        if not (0.0 <= p_wrong_max <= 1.0 - p + 1e-12):
            raise ValueError(
                f"p_wrong_max 必须位于 [0, 1-p] 内，当前 p={p}, p_wrong_max={p_wrong_max}。"
            )
        expected_wrong_max = float(p_wrong_max * mu_w_neg)
        eff_p_wrong = float(p_wrong_max)

    margin = float(expected_pos - expected_wrong_max)
    is_consistent = margin > 0.0

    if not is_consistent:
        return {
            "margin": margin,
            "is_consistent": False,
            "hoeffding_exponent": 0.0,
            "bernstein_exponent": 0.0,
            "expected_pos_score": expected_pos,
            "expected_wrong_max_score": expected_wrong_max,
            "effective_max_wrong_prob": eff_p_wrong,
            "tail_bucket_prob_threshold": tail_bucket_prob_threshold,
            "convergence_rate": "non_convergent (margin <= 0)",
            "samples_for_error_1e_3": None,
        }

    # Z_{i,j} = W_i*(1{A_i=a*} - 1{A_i=a_j}) in [-w_max, w_max]，区间宽度 2*w_max
    hoeffding_exp = float((margin * margin) / (2.0 * w_max * w_max))

    # Bernstein 方差上界：E[Z_{i,j}^2] <= (p + p_j) * w_max^2，
    # 且有中心化随机变量上界 |Z - E[Z]| <= w_max + margin <= 2*w_max
    var_upper = max(
        1e-15, (p + eff_p_wrong) * (w_max * w_max) - margin * margin
    )
    b_bound = w_max + margin
    bernstein_exp = float(
        0.5 * margin * margin / (var_upper + (1.0 / 3.0) * b_bound * margin)
    )
    samples_1e3 = int(math.ceil(math.log(1000.0) / hoeffding_exp))

    return {
        "margin": margin,
        "is_consistent": True,
        "hoeffding_exponent": hoeffding_exp,
        "bernstein_exponent": bernstein_exp,
        "expected_pos_score": expected_pos,
        "expected_wrong_max_score": expected_wrong_max,
        "effective_max_wrong_prob": eff_p_wrong,
        "tail_bucket_prob_threshold": tail_bucket_prob_threshold,
        "convergence_rate": f"exponential O(exp(-{hoeffding_exp:.6g}*N))",
        "samples_for_error_1e_3": samples_1e3,
    }


__all__ = [
    "BinaryVerifierSpec",
    "ContaminatedVerifierSpec",
    "ContinuousVerifierSpec",
    "GaussianVerifierSpec",
    "binary_asymptotic_ceiling",
    "binary_bon_accuracy",
    "binary_bon_divergence_bounds",
    "binary_marginal_gain",
    "binary_optimal_budget",
    "calibrated_bon_asymptotic_limit",
    "check_collapse_criterion",
    "closed_form_optimal_n",
    "gaussian_peak_threshold",
    "weighted_majority_convergence_bound",
]
