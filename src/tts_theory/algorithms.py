"""Test-Time Scaling 四大核心选择算法与经验校准器实现。

包含：
- BaseSelector: 统一选择器基类（支持单样本 select 与全向量化批量 select_batch）
- StandardBoN: 标准 Best-of-N
- OptimalNStarBoN: 理论/数值最优预算截断与早停 Optimal-N*
- EmpiricalScoreCalibrator: 支持非单调分布的经验分数后验校准器
- CalibratedWeightedBoN: 有界校准后验加权 BoN
- TwoStageVerifiedMajority: 自适应两阶段筛选与加权多数决
- MajorityVoting: 无权普通多数决基线
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tts_theory.theorems import ContinuousVerifierSpec


def _argmax_random_tie_2d(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """沿二维数组第 1 轴找最大值索引，严格均匀随机打破平局。"""
    max_vals = np.max(values, axis=1, keepdims=True)
    is_max = values == max_vals
    # 快路径：若完全没有平局直接用 argmax
    if np.all(np.sum(is_max, axis=1) == 1):
        return np.argmax(values, axis=1)
    tie_noise = np.where(is_max, rng.random(values.shape), -1.0)
    return np.argmax(tie_noise, axis=1)


def _aggregate_weighted_votes_2d(
    answers: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """向量化按离散答案聚合样本权重并选出最高总权答案（平局随机）。"""
    batch_size, _ = answers.shape
    min_a = int(np.min(answers))
    max_a = int(np.max(answers))
    span = max_a - min_a + 1

    # 当离散答案编码跨度紧凑时使用展平 bincount 快速向量化通道
    if span <= 10_000:
        shifted = answers - min_a
        row_offsets = np.arange(batch_size, dtype=np.int64)[:, None] * span
        flat_idx = (row_offsets + shifted).ravel()
        total_len = batch_size * span

        counts = np.bincount(flat_idx, minlength=total_len).reshape(batch_size, span)
        sums = np.bincount(flat_idx, weights=weights.ravel(), minlength=total_len).reshape(
            batch_size, span
        )
        masked_sums = np.where(counts > 0, sums, -np.inf)
        winners_shifted = _argmax_random_tie_2d(masked_sums, rng)
        return winners_shifted.astype(answers.dtype) + min_a

    # 大跨度稀疏离散答案编码回退通道
    out = np.empty(batch_size, dtype=answers.dtype)
    for b in range(batch_size):
        uniq, inv = np.unique(answers[b], return_inverse=True)
        totals = np.bincount(inv, weights=weights[b])
        max_w = np.max(totals)
        cands = uniq[totals == max_w]
        out[b] = cands[int(rng.integers(0, len(cands)))]
    return out


def _validate_single_inputs(
    scores: np.ndarray,
    answers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """校验单样本集输入形状并返回规范化一维数组。"""
    s_arr = np.asarray(scores, dtype=np.float64)
    if s_arr.ndim != 1 or s_arr.size == 0:
        raise ValueError("scores 必须为非空一维数组")
    a_arr: np.ndarray | None = None
    if answers is not None:
        a_arr = np.asarray(answers)
        if a_arr.shape != s_arr.shape:
            raise ValueError(f"answers 形状 {a_arr.shape} 必须与 scores {s_arr.shape} 一致")
    return s_arr, a_arr


def _validate_batch_inputs(
    scores: np.ndarray,
    answers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """校验批量输入形状并返回规范化二维数组。"""
    s_arr = np.asarray(scores, dtype=np.float64)
    if s_arr.ndim != 2 or s_arr.shape[0] == 0 or s_arr.shape[1] == 0:
        raise ValueError("select_batch 要求 scores 为非空二维数组 (B, N)")
    a_arr: np.ndarray | None = None
    if answers is not None:
        a_arr = np.asarray(answers)
        if a_arr.shape != s_arr.shape:
            raise ValueError(f"answers 形状 {a_arr.shape} 必须与 scores {s_arr.shape} 一致")
    return s_arr, a_arr


class BaseSelector(ABC):
    """Test-Time Scaling 样本/答案选择器统一抽象基类。"""

    @abstractmethod
    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """从单次试验的 N 个候选分数（及可选离散答案）中做出选择。

        Args:
            scores: 一维打分数组，形状 (N,)。
            answers: 可选一维整数离散答案数组，形状 (N,)。
            rng: 随机数生成器（用于平局随机打破）。

        Returns:
            当 answers 为 None 时返回选中样本的下标 int；
            当 answers 不为 None 时返回选中答案的整数 ID。
        """

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """对一批试验 (B, N) 执行选择，返回长度为 B 的一维结果数组。"""
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        b_size = s_arr.shape[0]
        out = np.empty(b_size, dtype=np.int64)
        for i in range(b_size):
            row_ans = a_arr[i] if a_arr is not None else None
            out[i] = self.select(s_arr[i], row_ans, rng=rng)
        return out


class StandardBoN(BaseSelector):
    """标准 Best-of-N：直接选择原始 Verifier 分数最高者（平局随机打破）。"""

    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        s_arr, a_arr = _validate_single_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        chosen_idx = int(_argmax_random_tie_2d(s_arr[None, :], rng)[0])
        if a_arr is None:
            return chosen_idx
        return int(a_arr[chosen_idx])

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        chosen_idx = _argmax_random_tie_2d(s_arr, rng)
        if a_arr is None:
            return chosen_idx.astype(np.int64)
        row_idx = np.arange(s_arr.shape[0])
        return a_arr[row_idx, chosen_idx].astype(np.int64)


class MajorityVoting(BaseSelector):
    """无权普通多数决基线：仅统计离散答案出现频次，忽略 Verifier 打分。"""

    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        s_arr, a_arr = _validate_single_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        if a_arr is None:
            return int(rng.integers(0, s_arr.size))
        ones = np.ones((1, s_arr.size), dtype=np.float64)
        return int(_aggregate_weighted_votes_2d(a_arr[None, :], ones, rng)[0])

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        if a_arr is None:
            return rng.integers(0, s_arr.shape[1], size=s_arr.shape[0], dtype=np.int64)
        ones = np.ones_like(s_arr, dtype=np.float64)
        return _aggregate_weighted_votes_2d(a_arr, ones, rng).astype(np.int64)


class OptimalNStarBoN(BaseSelector):
    """理论/数值最优预算截断与早停 Optimal-N* 算法。

    仅在理论判定存在极值相变坍塌（will_collapse=True）时，将采样预算截断在
    min(N, N*)；当处于安全区（不坍塌）时保持完整预算不截断。
    亦可配置 early_stop_posterior 阈值在预算内实现高置信度早停。
    """

    def __init__(
        self,
        spec: ContinuousVerifierSpec | None = None,
        n_star: int | None = None,
        budget_method: str = "saddle_refined",
        max_search_n: int = 20_000,
        early_stop_posterior: float | None = None,
    ) -> None:
        self.spec = spec
        self.budget_method = budget_method
        self.early_stop_posterior = early_stop_posterior
        self.will_collapse = False
        self.n_star: int | None = None

        if n_star is not None:
            if int(n_star) < 1:
                raise ValueError(f"n_star 必须 >= 1，当前收到: {n_star}")
            self.n_star = int(n_star)
            self.will_collapse = True
        elif spec is not None:
            self._resolve_n_star_from_spec(spec, budget_method, max_search_n)

    def _resolve_n_star_from_spec(
        self,
        spec: ContinuousVerifierSpec,
        budget_method: str,
        max_search_n: int,
    ) -> None:
        """根据解析模型或数值寻优自动推断最优截断预算 N*。"""
        from tts_theory.theorems import GaussianVerifierSpec, closed_form_optimal_n

        if isinstance(spec, GaussianVerifierSpec):
            cf = closed_form_optimal_n(spec, method=budget_method)
            self.will_collapse = bool(cf["will_collapse"])
            self.n_star = int(cf["n_star"]) if (self.will_collapse and cf["n_star"] is not None) else None
            return

        # 对污染混合尾部或其他一般 ContinuousVerifierSpec，先试算数值最优点
        from tts_theory.integrals import numerical_optimal_n

        num_res = numerical_optimal_n(spec, n_max=int(max_search_n))
        cand_n = int(num_res["n_star"])
        if cand_n < int(max_search_n):
            self.will_collapse = True
            self.n_star = cand_n
        else:
            self.will_collapse = False
            self.n_star = None

    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        s_arr, a_arr = _validate_single_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        res = self.select_batch(
            s_arr[None, :],
            a_arr[None, :] if a_arr is not None else None,
            rng=rng,
        )
        return int(res[0])

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()

        eff_n = s_arr.shape[1]
        if self.n_star is not None:
            eff_n = min(eff_n, max(1, int(self.n_star)))

        s_sub = s_arr[:, :eff_n]
        chosen_idx = _argmax_random_tie_2d(s_sub, rng)

        if self.early_stop_posterior is not None and self.spec is not None:
            q_sub = np.asarray(self.spec.posterior(s_sub), dtype=np.float64)
            hit_mask = q_sub >= float(self.early_stop_posterior)
            has_hit = np.any(hit_mask, axis=1)
            if np.any(has_hit):
                first_hit = np.argmax(hit_mask, axis=1)
                chosen_idx = np.where(has_hit, first_hit, chosen_idx)

        if a_arr is None:
            return chosen_idx.astype(np.int64)
        row_idx = np.arange(s_arr.shape[0])
        return a_arr[row_idx, chosen_idx].astype(np.int64)


def _pava_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """一维加权 Pool Adjacent Violators 算法（单调非降等渗回归）。"""
    blocks_y: list[float] = []
    blocks_w: list[float] = []
    blocks_len: list[int] = []

    for yi, wi in zip(y, w):
        blocks_y.append(float(yi))
        blocks_w.append(float(wi))
        blocks_len.append(1)
        while len(blocks_y) >= 2 and blocks_y[-2] > blocks_y[-1]:
            w_new = blocks_w[-2] + blocks_w[-1]
            y_new = (blocks_y[-2] * blocks_w[-2] + blocks_y[-1] * blocks_w[-1]) / w_new
            len_new = blocks_len[-2] + blocks_len[-1]
            blocks_y[-2:] = [y_new]
            blocks_w[-2:] = [w_new]
            blocks_len[-2:] = [len_new]

    out = np.empty(len(y), dtype=np.float64)
    pos = 0
    for val, length in zip(blocks_y, blocks_len):
        out[pos : pos + length] = val
        pos += length
    return out


class EmpiricalScoreCalibrator:
    """经验打分后验概率校准器。

    默认采用带 Beta 先验平滑的自适应分位数直方图插值 ("quantile_hist")，
    能够自然捕获异方差与 Reward Hacking 混合尾部引起的**非单调**后验跌落；
    亦支持带联合置信下界惩罚的悲观分位数校准 ("quantile_lcb"，消除有限校准集
    下的二次坍塌)，以及 "isotonic" 单调等渗校准用于已知单调似然比场景。
    """

    def __init__(
        self,
        n_bins: int = 50,
        prior_strength: float = 2.0,
        method: str = "quantile_hist",
        lcb_delta: float = 0.05,
        lcb_type: str = "wilson",
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins 必须 >= 2")
        if prior_strength < 0.0:
            raise ValueError("prior_strength 不能为负")
        if method not in ("quantile_hist", "quantile_lcb", "isotonic"):
            raise ValueError(f"不支持的校准方法 method={method!r}")
        if not (0.0 < lcb_delta < 1.0):
            raise ValueError("lcb_delta 必须位于 (0, 1) 内")
        if lcb_type not in ("wilson", "kl", "bernstein"):
            raise ValueError(f"不支持的 lcb_type={lcb_type!r}，仅支持 'wilson', 'kl', 'bernstein'")

        self.n_bins = int(n_bins)
        self.prior_strength = float(prior_strength)
        self.method = method
        self.lcb_delta = float(lcb_delta)
        self.lcb_type = lcb_type

        self.is_fitted_: bool = False
        self.prior_p_: float = 0.5
        self.knots_x_: np.ndarray | None = None
        self.knots_q_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> EmpiricalScoreCalibrator:
        """利用历史标注样本 (scores, labels) 拟合分数到后验概率的映射。"""
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if s.size == 0 or s.shape != y.shape:
            raise ValueError("scores 与 labels 必须非空且长度一致")

        self.prior_p_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        y_sorted = y[order]

        if self.method == "isotonic":
            # 合并完全相同的 x 值后再运行 PAVA
            uniq_x, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
            sum_y = np.bincount(inv, weights=y_sorted)
            mean_y = sum_y / counts
            fitted_y = _pava_isotonic(mean_y, counts.astype(np.float64))
            self.knots_x_ = uniq_x
            self.knots_q_ = np.clip(fitted_y, 1e-6, 1.0 - 1e-6)
            self.is_fitted_ = True
            return self

        # 自适应分位数直方图校准（天然支持单峰及非单调右尾跌落）
        eff_bins = min(self.n_bins, max(2, s_sorted.size // 20))
        quantiles = np.linspace(0.0, 1.0, eff_bins + 1)
        edges = np.quantile(s_sorted, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            self.knots_x_ = np.array([s_sorted[0]], dtype=np.float64)
            self.knots_q_ = np.array([self.prior_p_], dtype=np.float64)
            self.is_fitted_ = True
            return self

        # 使用 searchsorted 划分区间，右端封闭到最后一桶
        bin_idx = np.searchsorted(edges[1:-1], s_sorted, side="right")
        n_actual = edges.size - 1
        totals = np.bincount(bin_idx, minlength=n_actual).astype(np.float64)
        positives = np.bincount(bin_idx, weights=y_sorted, minlength=n_actual).astype(np.float64)
        sum_x = np.bincount(bin_idx, weights=s_sorted, minlength=n_actual).astype(np.float64)

        valid = totals > 0
        centers = sum_x[valid] / totals[valid]
        if self.method == "quantile_lcb":
            from tts_theory.calibration_theory import compute_bin_calibrator_scores

            lcb_method_map = {
                "wilson": "wilson_lcb",
                "kl": "kl_lcb",
                "bernstein": "empirical_bernstein_lcb",
            }
            q_vals = compute_bin_calibrator_scores(
                totals[valid],
                positives[valid],
                method=lcb_method_map[self.lcb_type],
                prior_p=self.prior_p_,
                delta=self.lcb_delta,
            )
        else:
            q_vals = (positives[valid] + self.prior_strength * self.prior_p_) / (
                totals[valid] + self.prior_strength
            )

        self.knots_x_ = centers
        self.knots_q_ = np.clip(q_vals, 1e-6, 1.0 - 1e-6)
        self.is_fitted_ = True
        return self

    def predict_proba(self, scores: np.ndarray | float) -> np.ndarray | float:
        """预测校准后验概率 q(s) in (0, 1)。"""
        if not self.is_fitted_ or self.knots_x_ is None or self.knots_q_ is None:
            raise RuntimeError("EmpiricalScoreCalibrator 尚未拟合，请先调用 fit()")

        is_scalar = np.ndim(scores) == 0
        s_arr = np.asarray(scores, dtype=np.float64)
        q_pred = np.interp(s_arr, self.knots_x_, self.knots_q_)
        q_pred = np.clip(q_pred, 1e-6, 1.0 - 1e-6)
        return float(q_pred) if is_scalar else q_pred


class _CalibratedBaseSelector(BaseSelector):
    """封装解析后验/经验校准器共用的后验与有界权重计算逻辑。"""

    def __init__(
        self,
        spec: ContinuousVerifierSpec | None = None,
        calibrator: EmpiricalScoreCalibrator | None = None,
        w_max: float = 20.0,
        eta: float = 1e-6,
    ) -> None:
        if spec is None and calibrator is None:
            raise ValueError("必须至少提供 spec 或 calibrator 之一")
        if calibrator is not None and not calibrator.is_fitted_ and spec is None:
            raise ValueError("未提供解析 spec 时，传入的 calibrator 必须已完成 fit()")
        if w_max <= 0.0:
            raise ValueError(f"w_max 必须 > 0，当前收到: {w_max}")
        if eta <= 0.0:
            raise ValueError(f"eta 必须 > 0，当前收到: {eta}")

        self.spec = spec
        self.calibrator = calibrator
        self.w_max = float(w_max)
        self.eta = float(eta)

    def _prior_p(self) -> float:
        if self.spec is not None:
            return float(self.spec.p)
        assert self.calibrator is not None
        return float(self.calibrator.prior_p_)

    def _compute_posteriors(self, scores: np.ndarray) -> np.ndarray:
        if self.spec is not None:
            q = np.asarray(self.spec.posterior(scores), dtype=np.float64)
        else:
            assert self.calibrator is not None
            q = np.asarray(self.calibrator.predict_proba(scores), dtype=np.float64)
        return np.clip(q, 1e-9, 1.0 - 1e-9)

    def _compute_bounded_weights(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(self.w_max, q / np.maximum(1.0 - q + self.eta, self.eta))


class CalibratedWeightedBoN(_CalibratedBaseSelector):
    """校准后验加权 Best-of-N。

    - 无离散答案时：选择校准后验最高者 argmax_i q(S_i)（即 Calibrated-BoN，彻底消除右尾坍塌）；
    - 有离散答案时：按有界校准后验权重 w(S_i) = min(w_max, q / (1 - q + eta))
      聚合各个候选答案总得分，选总分最高答案。
    """

    def __init__(
        self,
        spec: ContinuousVerifierSpec | None = None,
        calibrator: EmpiricalScoreCalibrator | None = None,
        w_max: float = 20.0,
        eta: float = 1e-6,
        weight_mode: str = "odds",
    ) -> None:
        super().__init__(spec=spec, calibrator=calibrator, w_max=w_max, eta=eta)
        if weight_mode not in ("odds", "posterior"):
            raise ValueError(f"不支持的 weight_mode={weight_mode!r}")
        self.weight_mode = weight_mode

    def _weights_from_q(self, q: np.ndarray) -> np.ndarray:
        if self.weight_mode == "posterior":
            return q
        return self._compute_bounded_weights(q)

    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        s_arr, a_arr = _validate_single_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        res = self.select_batch(
            s_arr[None, :],
            a_arr[None, :] if a_arr is not None else None,
            rng=rng,
        )
        return int(res[0])

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        q = self._compute_posteriors(s_arr)

        if a_arr is None:
            return _argmax_random_tie_2d(q, rng).astype(np.int64)

        w = self._weights_from_q(q)
        return _aggregate_weighted_votes_2d(a_arr, w, rng).astype(np.int64)


class TwoStageVerifiedMajority(_CalibratedBaseSelector):
    """自适应两阶段筛选与加权多数决（Two-Stage Verified Majority）。

    第一阶段（高质量门控）：
        使用校准后验阈值 q(S_i) >= tau 过滤低质量样本与极右尾 Reward Hacking 假高分样本。
        - 若显式给定 threshold，使用 tau = threshold，全被过滤时回退至最高后验样本集合；
        - 若 threshold 为 None（自适应模式），默认目标 tau_base = max(p, 0.5)；
          只要有至少 1 条样本满足 q >= tau_base 即保留所有 >= tau_base 样本；
          仅当全批均低于 tau_base 时，自动启用保底阈值 max(p, q_{1 - min_keep_fraction})
          （若仍全低于 p 则回退至最大 q 样本集合），避免高难度任务被死阈值锁死。
    第二阶段（加权投票）：
        - 有离散答案时：在幸存高质量集合内按有界校准权重 w(S_i) 做加权多数决；
        - 无离散答案时：在幸存集中选择校准后验最高者。
    """

    def __init__(
        self,
        spec: ContinuousVerifierSpec | None = None,
        calibrator: EmpiricalScoreCalibrator | None = None,
        threshold: float | None = None,
        min_keep_fraction: float = 0.15,
        w_max: float = 20.0,
        eta: float = 1e-6,
    ) -> None:
        super().__init__(spec=spec, calibrator=calibrator, w_max=w_max, eta=eta)
        if threshold is not None and not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold 必须在 (0, 1) 内，当前收到: {threshold}")
        if not (0.0 < min_keep_fraction <= 1.0):
            raise ValueError("min_keep_fraction 必须在 (0, 1] 内")
        self.threshold = threshold
        self.min_keep_fraction = float(min_keep_fraction)

    def _stage1_survivor_mask(self, q: np.ndarray) -> np.ndarray:
        """计算第一阶段幸存候选掩码 (B, N)，保证每行至少有 1 个幸存者。"""
        q_max_row = np.max(q, axis=1, keepdims=True)

        if self.threshold is not None:
            keep = q >= float(self.threshold)
            empty_rows = ~np.any(keep, axis=1)
            if np.any(empty_rows):
                keep[empty_rows] = q[empty_rows] == q_max_row[empty_rows]
            return keep

        prior_p = self._prior_p()
        tau_base = max(prior_p, 0.5)
        keep = q >= tau_base
        empty_rows = ~np.any(keep, axis=1)
        if np.any(empty_rows):
            q_empty = q[empty_rows]
            # 保底阈值：在高于先验 p 的高分位样本中至少保留前 min_keep_fraction
            q_quant = np.quantile(q_empty, 1.0 - self.min_keep_fraction, axis=1, keepdims=True)
            tau_adapt = np.maximum(prior_p, q_quant)
            keep_adapt = q_empty >= tau_adapt
            still_empty = ~np.any(keep_adapt, axis=1)
            if np.any(still_empty):
                keep_adapt[still_empty] = q_empty[still_empty] == q_max_row[empty_rows][still_empty]
            keep[empty_rows] = keep_adapt
        return keep

    def select(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        s_arr, a_arr = _validate_single_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        res = self.select_batch(
            s_arr[None, :],
            a_arr[None, :] if a_arr is not None else None,
            rng=rng,
        )
        return int(res[0])

    def select_batch(
        self,
        scores: np.ndarray,
        answers: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        s_arr, a_arr = _validate_batch_inputs(scores, answers)
        rng = rng or np.random.default_rng()
        q = self._compute_posteriors(s_arr)
        keep = self._stage1_survivor_mask(q)

        if a_arr is None:
            masked_q = np.where(keep, q, -np.inf)
            return _argmax_random_tie_2d(masked_q, rng).astype(np.int64)

        w = np.where(keep, self._compute_bounded_weights(q), 0.0)
        return _aggregate_weighted_votes_2d(a_arr, w, rng).astype(np.int64)
