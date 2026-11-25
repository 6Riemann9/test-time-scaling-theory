"""向量化大规模蒙特卡洛仿真引擎。

支持流式/分块生成与在线更新，确保在 n_trials = 100,000、
大采样预算 N 下维持极低内存占用与高速并行计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from tts_theory.algorithms import BaseSelector
    from tts_theory.theorems import BinaryVerifierSpec, ContinuousVerifierSpec


_Z_95 = 1.959963984540054


def _wilson_ci(
    successes: np.ndarray,
    n_trials: int,
    z: float = _Z_95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算二项比例估计、标准误及 95% Wilson 得分区间。"""
    p_hat = successes.astype(np.float64) / float(n_trials)
    se = np.sqrt(p_hat * (1.0 - p_hat) / float(n_trials))
    z2 = z * z
    denom = 1.0 + z2 / float(n_trials)
    center = (p_hat + z2 / (2.0 * float(n_trials))) / denom
    half_width = (
        z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n_trials))) / float(n_trials)) / denom
    )
    ci_low = np.clip(center - half_width, 0.0, 1.0)
    ci_high = np.clip(center + half_width, 0.0, 1.0)
    return p_hat, se, ci_low, ci_high


def _normalize_n_values(n_values: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """校验并排序 n_values，返回 (原序数组, 升序唯一值, 原序到唯一值的逆映射索引)。"""
    arr = np.asarray(list(n_values), dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("n_values 必须为非空一维正整数序列")
    if np.any(arr < 1):
        raise ValueError("n_values 中的所有采样数 N 必须 >= 1")
    unique_sorted, inv_idx = np.unique(arr, return_inverse=True)
    return arr, unique_sorted, inv_idx


def _invert_cdf_bisect(cdf_fn: Any, u: np.ndarray, lo_init: float = -25.0, hi_init: float = 25.0) -> np.ndarray:
    """对任意未知分布的向量化二分求逆，用于通用 ContinuousVerifierSpec 抽样回退。"""
    lo = np.full(u.shape, lo_init, dtype=np.float64)
    hi = np.full(u.shape, hi_init, dtype=np.float64)

    # 若边界不够宽，指数扩界至涵盖所有分位点
    for _ in range(12):
        need_lower = np.asarray(cdf_fn(lo), dtype=np.float64) > u
        if not np.any(need_lower):
            break
        lo = np.where(need_lower, lo * 2.0, lo)

    for _ in range(12):
        need_upper = np.asarray(cdf_fn(hi), dtype=np.float64) < u
        if not np.any(need_upper):
            break
        hi = np.where(need_upper, hi * 2.0, hi)

    for _ in range(54):
        mid = 0.5 * (lo + hi)
        val = np.asarray(cdf_fn(mid), dtype=np.float64)
        go_right = val < u
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)

    return 0.5 * (lo + hi)


def _sample_scores_and_labels(
    spec: ContinuousVerifierSpec,
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """从连续打分 Verifier 规格向量化抽取 (scores, is_correct_labels)。"""
    labels = rng.random(shape) < float(spec.p)
    scores = np.empty(shape, dtype=np.float64)

    n_pos = int(np.count_nonzero(labels))
    n_neg = labels.size - n_pos

    # 快速通道 1：污染混合高斯 Verifier
    if (
        hasattr(spec, "epsilon")
        and hasattr(spec, "mu_hack")
        and hasattr(spec, "sigma_hack")
        and hasattr(spec, "mu1")
        and hasattr(spec, "sigma1")
        and hasattr(spec, "mu0")
        and hasattr(spec, "sigma0")
    ):
        if n_pos > 0:
            scores[labels] = rng.normal(float(spec.mu1), float(spec.sigma1), size=n_pos)
        if n_neg > 0:
            is_hack = rng.random(n_neg) < float(spec.epsilon)
            neg_scores = rng.normal(float(spec.mu0), float(spec.sigma0), size=n_neg)
            n_hack = int(np.count_nonzero(is_hack))
            if n_hack > 0:
                neg_scores[is_hack] = rng.normal(
                    float(spec.mu_hack), float(spec.sigma_hack), size=n_hack
                )
            scores[~labels] = neg_scores
        return scores, labels

    # 快速通道 2：标准高斯 Verifier
    if (
        hasattr(spec, "mu1")
        and hasattr(spec, "sigma1")
        and hasattr(spec, "mu0")
        and hasattr(spec, "sigma0")
    ):
        if n_pos > 0:
            scores[labels] = rng.normal(float(spec.mu1), float(spec.sigma1), size=n_pos)
        if n_neg > 0:
            scores[~labels] = rng.normal(float(spec.mu0), float(spec.sigma0), size=n_neg)
        return scores, labels

    # 通用回退通道：条件逆变换抽样
    if n_pos > 0:
        u1 = np.clip(rng.random(n_pos), 1e-12, 1.0 - 1e-12)
        scores[labels] = _invert_cdf_bisect(spec.cdf1, u1)
    if n_neg > 0:
        u0 = np.clip(rng.random(n_neg), 1e-12, 1.0 - 1e-12)
        scores[~labels] = _invert_cdf_bisect(spec.cdf0, u0)
    return scores, labels


def _update_running_bon_max(
    best_score: np.ndarray,
    best_label: np.ndarray,
    tie_count: np.ndarray,
    chunk_scores: np.ndarray,
    chunk_labels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """使用批内最大值及水塘抽样平局规则滚动更新最优状态（原地修改）。"""
    # 先在 chunk 内部求最大值，并对内部平局随机打破
    chunk_max = np.max(chunk_scores, axis=1)
    is_chunk_max = chunk_scores == chunk_max[:, None]
    if np.any(np.sum(is_chunk_max, axis=1) > 1):
        tie_noise = np.where(is_chunk_max, rng.random(chunk_scores.shape), -1.0)
        winner_idx = np.argmax(tie_noise, axis=1)
        chunk_tie_mult = np.sum(is_chunk_max, axis=1)
    else:
        winner_idx = np.argmax(chunk_scores, axis=1)
        chunk_tie_mult = np.ones(chunk_scores.shape[0], dtype=np.int64)

    row_idx = np.arange(chunk_scores.shape[0])
    winner_label = chunk_labels[row_idx, winner_idx]

    strict_better = chunk_max > best_score
    if np.any(strict_better):
        best_score[strict_better] = chunk_max[strict_better]
        best_label[strict_better] = winner_label[strict_better]
        tie_count[strict_better] = chunk_tie_mult[strict_better]

    exact_tie = chunk_max == best_score
    if np.any(exact_tie):
        new_total_ties = tie_count[exact_tie] + chunk_tie_mult[exact_tie]
        replace_prob = chunk_tie_mult[exact_tie] / new_total_ties
        do_replace = rng.random(int(np.count_nonzero(exact_tie))) < replace_prob
        tie_indices = np.flatnonzero(exact_tie)
        chosen_rows = tie_indices[do_replace]
        best_label[chosen_rows] = winner_label[chosen_rows]
        tie_count[exact_tie] = new_total_ties


def simulate_bon(
    spec: ContinuousVerifierSpec,
    n_values: Sequence[int] | np.ndarray,
    n_trials: int = 100_000,
    seed: int = 42,
    batch_size: int = 10_000,
) -> dict[str, np.ndarray]:
    """流式分块向量化蒙特卡洛仿真连续打分 Verifier 的 Best-of-N 正确率。

    绝不一次性分配 n_trials x N_max 巨型矩阵，而是按试验批次及采样增量分段滚动更新最大值。

    Returns:
        包含 "n_values", "accuracy", "se", "ci_low", "ci_high" 的字典。
    """
    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials 与 batch_size 必须 >= 1")

    orig_n, unique_n, inv_idx = _normalize_n_values(n_values)
    rng = np.random.default_rng(seed)

    # 限制每个分块临时张量不超过约 2,000,000 个元素 (~16 MB)
    max_chunk_elements = 2_000_000
    correct_counts = np.zeros(len(unique_n), dtype=np.int64)

    done_trials = 0
    while done_trials < n_trials:
        b = min(batch_size, n_trials - done_trials)
        best_score = np.full(b, -np.inf, dtype=np.float64)
        best_label = np.zeros(b, dtype=bool)
        tie_count = np.zeros(b, dtype=np.int64)

        max_cols_per_chunk = max(1, max_chunk_elements // b)
        prev_n = 0

        for k, target_n in enumerate(unique_n):
            delta = int(target_n) - prev_n
            while delta > 0:
                step = min(delta, max_cols_per_chunk)
                chunk_scores, chunk_labels = _sample_scores_and_labels(spec, (b, step), rng)
                _update_running_bon_max(
                    best_score, best_label, tie_count, chunk_scores, chunk_labels, rng
                )
                delta -= step
            correct_counts[k] += int(np.count_nonzero(best_label))
            prev_n = int(target_n)

        done_trials += b

    acc, se, ci_low, ci_high = _wilson_ci(correct_counts, n_trials)
    return {
        "n_values": orig_n,
        "accuracy": acc[inv_idx],
        "se": se[inv_idx],
        "ci_low": ci_low[inv_idx],
        "ci_high": ci_high[inv_idx],
    }


def simulate_binary_bon(
    spec: BinaryVerifierSpec,
    n_values: Sequence[int] | np.ndarray,
    n_trials: int = 100_000,
    seed: int = 42,
    batch_size: int = 20_000,
) -> dict[str, np.ndarray]:
    """基于多项分布的高速向量化二元 Verifier Best-of-N 蒙特卡洛仿真。

    直接模拟单次试验内 (Y=1,S=1), (Y=0,S=1), (Y=1,S=0), (Y=0,S=0) 的联合频数，
    并在判正池中均匀打破平局、全负时随机回退。复杂度 O(n_trials)，与 N 大小无关。
    """
    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials 与 batch_size 必须 >= 1")

    orig_n, unique_n, inv_idx = _normalize_n_values(n_values)
    rng = np.random.default_rng(seed)

    p = float(spec.p)
    tpr = float(spec.tpr)
    fpr = float(spec.fpr)
    probs = np.array(
        [
            p * tpr,
            (1.0 - p) * fpr,
            p * (1.0 - tpr),
            (1.0 - p) * (1.0 - fpr),
        ],
        dtype=np.float64,
    )
    probs = probs / np.sum(probs)

    correct_counts = np.zeros(len(unique_n), dtype=np.int64)

    for k, target_n in enumerate(unique_n):
        n_int = int(target_n)
        done = 0
        hits = 0
        while done < n_trials:
            b = min(batch_size, n_trials - done)
            counts = rng.multinomial(n_int, probs, size=b)
            n_pos_pred = counts[:, 0] + counts[:, 1]
            has_pos = n_pos_pred > 0

            # 若有判正样本，均匀从中抽选；否则从全部 N 个样本中均匀抽选
            prob_correct = np.where(
                has_pos,
                counts[:, 0] / np.maximum(n_pos_pred, 1),
                counts[:, 2] / float(n_int),
            )
            hits += int(np.count_nonzero(rng.random(b) < prob_correct))
            done += b
        correct_counts[k] = hits

    acc, se, ci_low, ci_high = _wilson_ci(correct_counts, n_trials)
    return {
        "n_values": orig_n,
        "accuracy": acc[inv_idx],
        "se": se[inv_idx],
        "ci_low": ci_low[inv_idx],
        "ci_high": ci_high[inv_idx],
    }


@dataclass(frozen=True)
class DiscreteAnswerTaskSpec:
    """离散多选题/离散答案空间 Test-Time Scaling 仿真任务定义。

    约定正确答案编码为整数 0，错误答案编码为 1 .. num_wrong_answers。
    正确样本 (Y=1) 总是生成答案 0 且打分服从 F_1；
    错误样本 (Y=0) 按 wrong_answer_probs 生成某个错误答案且打分服从 F_0。
    """

    verifier_spec: ContinuousVerifierSpec
    num_wrong_answers: int = 5
    wrong_answer_probs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_wrong_answers < 1:
            raise ValueError(f"num_wrong_answers 必须 >= 1，当前收到: {self.num_wrong_answers}")
        if self.wrong_answer_probs is not None:
            if len(self.wrong_answer_probs) != self.num_wrong_answers:
                raise ValueError(
                    f"wrong_answer_probs 长度 ({len(self.wrong_answer_probs)}) "
                    f"必须等于 num_wrong_answers ({self.num_wrong_answers})"
                )
            arr = np.asarray(self.wrong_answer_probs, dtype=np.float64)
            if np.any(arr < 0.0):
                raise ValueError("wrong_answer_probs 不能包含负概率")
            total = float(np.sum(arr))
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"wrong_answer_probs 概率之和必须为 1.0，当前为 {total}")

    @property
    def wrong_probs_array(self) -> np.ndarray:
        """返回归一化后的错误答案概率分布向量。"""
        if self.wrong_answer_probs is None:
            return np.full(self.num_wrong_answers, 1.0 / self.num_wrong_answers, dtype=np.float64)
        arr = np.asarray(self.wrong_answer_probs, dtype=np.float64)
        return arr / np.sum(arr)


def simulate_algorithms_comparison(
    task_spec: DiscreteAnswerTaskSpec,
    n_values: Sequence[int] | np.ndarray,
    algorithms: Mapping[str, BaseSelector] | None = None,
    n_trials: int = 20_000,
    seed: int = 42,
    batch_size: int = 2_000,
) -> dict[str, Any]:
    """在统一离散答案任务上对多个选择算法进行配对流式蒙特卡洛对比。

    Args:
        task_spec: 离散答案与打分任务定义。
        n_values: 待评测采样预算 N 序列。
        algorithms: 算法名称到 BaseSelector 实例的映射；为 None 时构建默认五大对比算法。
        n_trials: 总试验次数（支持高达 100,000）。
        seed: 随机数种子。
        batch_size: 每批试验数（若 B * N_max 过大将自动缩放以防内存溢出）。

    Returns:
        包含 "n_values" 与 "algorithms" 字典（各算法含 accuracy, se, ci_low, ci_high）。
    """
    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials 与 batch_size 必须 >= 1")

    orig_n, unique_n, inv_idx = _normalize_n_values(n_values)
    n_max = int(unique_n[-1])

    if algorithms is None:
        from tts_theory.algorithms import (
            CalibratedWeightedBoN,
            MajorityVoting,
            OptimalNStarBoN,
            StandardBoN,
            TwoStageVerifiedMajority,
        )

        algorithms = {
            "Standard-BoN": StandardBoN(),
            "Optimal-N*": OptimalNStarBoN(spec=task_spec.verifier_spec),
            "Majority-Voting": MajorityVoting(),
            "Calibrated-Weighted-BoN": CalibratedWeightedBoN(spec=task_spec.verifier_spec),
            "Two-Stage-Verified-Majority": TwoStageVerifiedMajority(spec=task_spec.verifier_spec),
        }

    if len(algorithms) == 0:
        raise ValueError("algorithms 映射不能为空")

    algo_names = list(algorithms.keys())
    correct_counts = {name: np.zeros(len(unique_n), dtype=np.int64) for name in algo_names}

    # 为每个算法分配独立的子随机流，确保可复现且在平局随机时互不污染
    master_rng = np.random.default_rng(seed)
    algo_rngs = {
        name: np.random.default_rng(int(master_rng.integers(0, 2**63 - 1)))
        for name in algo_names
    }

    wrong_cdf = np.cumsum(task_spec.wrong_probs_array)
    wrong_cdf[-1] = 1.0

    # 自适应控制单批次矩阵大小不超过 ~4,000,000 个元素 (~32 MB)
    eff_batch = max(1, min(int(batch_size), 4_000_000 // max(1, n_max)))

    done_trials = 0
    while done_trials < n_trials:
        b = min(eff_batch, n_trials - done_trials)
        scores, labels = _sample_scores_and_labels(task_spec.verifier_spec, (b, n_max), master_rng)

        # 正确答案 0，错误答案按 wrong_cdf 采样映射到 1..K
        answers = np.zeros((b, n_max), dtype=np.int64)
        wrong_mask = ~labels
        n_wrong = int(np.count_nonzero(wrong_mask))
        if n_wrong > 0:
            u_wrong = master_rng.random(n_wrong)
            answers[wrong_mask] = np.searchsorted(wrong_cdf, u_wrong, side="right") + 1

        for name in algo_names:
            selector = algorithms[name]
            algo_rng = algo_rngs[name]
            for k, n_val in enumerate(unique_n):
                n_int = int(n_val)
                chosen = selector.select_batch(
                    scores[:, :n_int], answers[:, :n_int], rng=algo_rng
                )
                correct_counts[name][k] += int(np.count_nonzero(chosen == 0))

        done_trials += b

    algo_metrics: dict[str, dict[str, np.ndarray]] = {}
    for name in algo_names:
        acc, se, ci_low, ci_high = _wilson_ci(correct_counts[name], n_trials)
        algo_metrics[name] = {
            "accuracy": acc[inv_idx],
            "se": se[inv_idx],
            "ci_low": ci_low[inv_idx],
            "ci_high": ci_high[inv_idx],
        }

    return {
        "n_values": orig_n,
        "algorithms": algo_metrics,
    }
