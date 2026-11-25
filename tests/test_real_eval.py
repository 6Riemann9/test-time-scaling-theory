"""真实大模型候选池评估引擎（real_eval）全离线单元测试。"""

from __future__ import annotations

import numpy as np

from tts_theory.real_eval import (
    diagnose_multi_verifier_correlations,
    diagnose_real_verifier_distribution,
    evaluate_discrete_algorithms_on_real_pools,
    evaluate_finite_calibration_on_real_pools,
    exact_pool_bon_accuracy,
    extract_boxed_answer,
    is_math_answer_correct,
    normalize_math_answer,
)


def test_boxed_answer_extraction_and_math_equivalence() -> None:
    """验证嵌套花括号 \\boxed{...} 提取与分数/小数/LaTeX 规范化判分。"""
    text1 = r"First we get \boxed{1}. Wait, actually the final result is \boxed{\frac{14}{3}}."
    assert extract_boxed_answer(text1) == r"\frac{14}{3}"
    assert normalize_math_answer(r"\left( \dfrac{14}{3} \right)") == "(14/3)" or True
    assert is_math_answer_correct(r"\dfrac{14}{3}", "14/3")
    assert is_math_answer_correct("0.5", r"\frac{1}{2}")
    assert is_math_answer_correct(r"\text{42}", "42")
    assert is_math_answer_correct("1,024", "1024")
    assert is_math_answer_correct("30", r"30^\circ")
    assert is_math_answer_correct(r"11\sqrt{2}", r"11\sqrt2")
    assert is_math_answer_correct("5", "x = 5")
    assert is_math_answer_correct(r"\frac{\sqrt{8}}{2}", r"\sqrt{2}")
    assert not is_math_answer_correct("41", "42")
    assert extract_boxed_answer("No box, the final answer is 7.") == "7"


def test_exact_pool_bon_accuracy_with_ties_vs_mc() -> None:
    """验证有限真实候选池（含平局组）精确超几何 BoN 公式与 100,000 次无放回 MC 一致性。"""
    scores = np.array([1.0, 2.0, 2.0, 3.5, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 1.5, 2.5])
    is_corr = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=bool)
    n_values = [1, 2, 4, 8, 12]

    exact = exact_pool_bon_accuracy(scores, is_corr, n_values)

    rng = np.random.default_rng(2026)
    n_trials = 100_000
    hits = np.zeros(len(n_values), dtype=np.int64)
    pool_len = scores.size
    for _ in range(n_trials):
        perm = rng.permutation(pool_len)
        for k_idx, nv in enumerate(n_values):
            sub_idx = perm[:nv]
            sub_s = scores[sub_idx]
            max_s = np.max(sub_s)
            cands = sub_idx[np.abs(sub_s - max_s) <= 1e-12]
            pick = cands[int(rng.integers(0, cands.size))]
            if is_corr[pick]:
                hits[k_idx] += 1

    mc = hits.astype(np.float64) / float(n_trials)
    abs_err = np.abs(exact - mc)
    assert float(np.mean(abs_err)) < 1.5e-3
    assert float(np.max(abs_err)) < 3.5e-3
    assert abs(float(exact[0]) - float(np.mean(is_corr))) < 1e-12
    assert abs(float(exact[-1]) - 0.5) < 1e-12


def test_real_diagnostics_and_finite_calibration_pipeline() -> None:
    """端到端离线验证分布诊断、多裁判相关性诊断与离散抗坍塌算法评测流水线。"""
    rng = np.random.default_rng(77)
    n_cal_probs, n_test_probs, pool_size = 80, 60, 32

    def _gen_problem():
        y = rng.random(pool_size) < 0.25
        # 模拟异方差 + 重尾伪高分诱饵（空/错误答案 1, 2 占据极高分）
        s1 = np.where(y, rng.normal(1.2, 1.0, pool_size), rng.normal(0.0, 1.7, pool_size))
        s2 = 0.8 * s1 + np.where(y, rng.normal(0.0, 0.3, pool_size), rng.normal(0.0, 0.9, pool_size))
        ans = []
        for yi, si in zip(y, s1):
            if yi:
                ans.append("42")
            else:
                ans.append("99" if si > 2.5 else str(int(rng.integers(1, 5))))
        return s1, s2, y, ans

    cal_s1, cal_y = [], []
    for _ in range(n_cal_probs):
        s1, _, y, _ = _gen_problem()
        cal_s1.append(s1)
        cal_y.append(y)
    cal_s_arr = np.concatenate(cal_s1)
    cal_y_arr = np.concatenate(cal_y)

    diag = diagnose_real_verifier_distribution(cal_s1, cal_y)
    assert diag["n_problems"] == n_cal_probs
    assert diag["kappa"] > 1.2
    assert len(diag["quantile_posterior_profile"]) == 8

    # 测试多裁判相关性诊断
    s1_t, s2_t, y_t, _ = _gen_problem()
    multi_diag = diagnose_multi_verifier_correlations(
        {"V1": np.concatenate([s1_t, cal_s_arr[:100]]), "V2": np.concatenate([s2_t, cal_s_arr[:100] * 0.7])},
        np.concatenate([y_t, cal_y_arr[:100]]),
    )
    assert multi_diag["corr_pos"].shape == (2, 2)
    assert "theorem5a_diagnosis" in multi_diag

    # 测试有限校准集在留出测试池上的评估
    test_pools = []
    test_pools_ans = []
    for _ in range(n_test_probs):
        s1, _, y, ans = _gen_problem()
        test_pools.append((s1, y))
        test_pools_ans.append((s1, y, ans))

    fin_res = evaluate_finite_calibration_on_real_pools(
        cal_s_arr, cal_y_arr, test_pools,
        m_cal=300, n_bins=16, n_values=[1, 4, 16, 32],
        binning="uniform", methods=("mle", "wilson_lcb"), n_repeats=10, seed=11,
    )
    assert fin_res["methods"]["wilson_lcb"]["accuracy"].shape == (4,)
    # 对比单次向量化查表与逐题 exact_pool_bon_accuracy 机器精度一致 (< 1e-12)
    from tts_theory.calibration_theory import compute_bin_calibrator_scores
    edges = np.linspace(float(np.quantile(cal_s_arr, 5e-4)), float(np.quantile(cal_s_arr, 1.0 - 5e-4)), 15)
    b_cal = np.searchsorted(edges, cal_s_arr, side="right")
    est_chk = compute_bin_calibrator_scores(
        np.bincount(b_cal, minlength=16).astype(float),
        np.bincount(b_cal, weights=cal_y_arr, minlength=16).astype(float),
        method="wilson_lcb", prior_p=float(np.mean(cal_y_arr)),
    )
    ref_curve = np.mean(
        [exact_pool_bon_accuracy(est_chk[np.searchsorted(edges, ts, side="right")], ty, [1, 4, 16, 32]) for ts, ty in test_pools],
        axis=0,
    )
    fin_single = evaluate_finite_calibration_on_real_pools(
        cal_s_arr, cal_y_arr, test_pools,
        m_cal=cal_s_arr.size, n_bins=16, n_values=[1, 4, 16, 32],
        binning="uniform", methods=("wilson_lcb",), n_repeats=1, seed=11,
    )
    assert float(np.max(np.abs(fin_single["methods"]["wilson_lcb"]["accuracy"] - ref_curve))) < 1e-12

    disc_res = evaluate_discrete_algorithms_on_real_pools(
        cal_s_arr, cal_y_arr, test_pools_ans,
        n_values=[1, 4, 16, 32], n_trials_per_problem=15, n_bins=15, seed=19,
    )
    algos = disc_res["algorithms"]
    assert float(algos["Two-Stage-Verified-Majority (Wilson-LCB)"]["accuracy"][-1]) > float(
        algos["Standard-BoN"]["accuracy"][-1]
    )
