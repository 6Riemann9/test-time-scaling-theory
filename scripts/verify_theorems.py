"""定理 1-4 自动化端到端交叉验证脚本。

验证内容：
1. 定理 1：连续打分 Verifier 精确顺序统计量积分 vs 100,000 次蒙特卡洛（同方差、异方差、Reward Hacking 混合尾部）及高斯-勒让德/自适应 quad 交叉一致性；
2. 定理 2：极值相变坍塌趋势（N=1..10,000）、相界判据（kappa_c = 1）与闭式最优预算 N* vs 精确数值最优 N*_exact 对比；
3. 定理 3：二元 Verifier 解析公式 vs 100,000 次蒙特卡洛、边际增益指数衰减与闭式预算检验；
4. 定理 4：校准后验 BoN 无坍塌单调收敛，以及离散答案空间下校准加权多数决与两阶段多数决在大 N 下的抗坍塌收敛性。

运行方式：
    python scripts/verify_theorems.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_theory import algorithms as algo_mod
from tts_theory.integrals import (
    exact_bon_accuracy,
    exact_bon_curve,
    exact_calibrated_bon_accuracy,
    numerical_optimal_n,
)
from tts_theory.simulator import (
    DiscreteAnswerTaskSpec,
    simulate_algorithms_comparison,
    simulate_binary_bon,
    simulate_bon,
)
from tts_theory.theorems import (
    BinaryVerifierSpec,
    ContaminatedVerifierSpec,
    GaussianVerifierSpec,
    binary_asymptotic_ceiling,
    binary_bon_accuracy,
    binary_marginal_gain,
    binary_optimal_budget,
    calibrated_bon_asymptotic_limit,
    check_collapse_criterion,
    closed_form_optimal_n,
    weighted_majority_convergence_bound,
)


def to_json_safe(obj: Any) -> Any:
    """将 NumPy 标量/数组递归转换为 JSON 可序列化的原生 Python 类型。"""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _build_majority_voting_baseline() -> Any:
    """获取普通无权 Majority Voting 基线选择器。"""
    return algo_mod.MajorityVoting()


def verify_theorem_1(
    n_trials: int = 100_000,
    n_trials_high_precision: int = 1_500_000,
    seed: int = 42,
) -> dict[str, Any]:
    """验证定理 1：精确顺序统计量积分 vs 100,000 次与 1,500,000 次蒙特卡洛及 quad 交叉校验。"""
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    quad_check_ns = [1, 16, 256, 1000]
    scenarios = {
        "homoscedastic_kappa_1.0": GaussianVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0
        ),
        "heteroscedastic_kappa_1.5": GaussianVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5
        ),
        "reward_hacking_eps_0.08": ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    }

    results: dict[str, Any] = {}
    all_passed_100k_stat = True
    all_passed_1500k_strict = True

    for idx_s, (name, spec) in enumerate(scenarios.items()):
        exact_vals = np.asarray(
            exact_bon_curve(spec, n_values, method="gauss_legendre", n_nodes=256),
            dtype=float,
        )
        mc_res = simulate_bon(spec, n_values, n_trials=n_trials, seed=seed, batch_size=10_000)
        mc_vals = np.asarray(mc_res["accuracy"], dtype=float)
        mc_se = np.asarray(mc_res["se"], dtype=float)
        ci_low = np.asarray(mc_res["ci_low"], dtype=float)
        ci_high = np.asarray(mc_res["ci_high"], dtype=float)

        abs_errors = np.abs(exact_vals - mc_vals)
        max_abs_err = float(np.max(abs_errors))
        mean_abs_err = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(abs_errors**2)))
        ci_coverage = float(np.mean((exact_vals >= ci_low) & (exact_vals <= ci_high)))
        max_z_score = float(np.max(abs_errors / np.maximum(mc_se, 1e-12)))

        # 大样本无偏渐近检验（1.5M 次抽样使二项标准误降至 ~4e-4，从而严格检验 < 1e-3）
        hp_res = simulate_bon(
            spec, n_values, n_trials=n_trials_high_precision, seed=seed + 100 + idx_s, batch_size=10_000
        )
        hp_vals = np.asarray(hp_res["accuracy"], dtype=float)
        hp_abs_errors = np.abs(exact_vals - hp_vals)
        hp_max_err = float(np.max(hp_abs_errors))
        hp_mean_err = float(np.mean(hp_abs_errors))

        quad_diffs = []
        for qn in quad_check_ns:
            idx = n_values.index(qn)
            q_val = float(exact_bon_accuracy(spec, qn, method="quad"))
            quad_diffs.append(abs(q_val - float(exact_vals[idx])))
        max_quad_diff = float(max(quad_diffs))

        stat_100k_ok = (mean_abs_err < 1e-3) and (max_z_score < 3.5) and (max_quad_diff < 1e-8)
        strict_1500k_ok = (hp_max_err < 1e-3) and (max_quad_diff < 1e-8)
        all_passed_100k_stat = all_passed_100k_stat and stat_100k_ok
        all_passed_1500k_strict = all_passed_1500k_strict and strict_1500k_ok

        results[name] = {
            "n_trials": n_trials,
            "n_trials_high_precision": n_trials_high_precision,
            "n_values": n_values,
            "exact_accuracy": exact_vals.tolist(),
            "mc_accuracy": mc_vals.tolist(),
            "mc_se": mc_se.tolist(),
            "mc_ci_low": ci_low.tolist(),
            "mc_ci_high": ci_high.tolist(),
            "abs_errors": abs_errors.tolist(),
            "max_abs_error": max_abs_err,
            "mean_abs_error": mean_abs_err,
            "rmse": rmse,
            "ci_95_coverage": ci_coverage,
            "max_z_score": max_z_score,
            "hp_mc_accuracy": hp_vals.tolist(),
            "hp_max_abs_error": hp_max_err,
            "hp_mean_abs_error": hp_mean_err,
            "max_quad_vs_gl_diff": max_quad_diff,
            "passed_100k_stat": stat_100k_ok,
            "passed_1500k_strict_1e3": strict_1500k_ok,
        }

    return {
        "scenarios": results,
        "all_passed_100k_stat": all_passed_100k_stat,
        "all_passed_1500k_strict": all_passed_1500k_strict,
    }


def verify_theorem_2() -> dict[str, Any]:
    """验证定理 2：极值相变坍塌趋势、相界判据与闭式 N* 精度对比。"""
    n_trend = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    trend_specs = {
        "subcritical_kappa_0.8": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=0.8),
        "critical_kappa_1.0": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0),
        "collapse_kappa_1.2": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.2),
        "collapse_kappa_1.5": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5),
        "collapse_kappa_1.8": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.8),
        "collapse_hack_eps_0.08": ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    }

    trend_results: dict[str, Any] = {}
    for name, spec in trend_specs.items():
        accs = np.asarray(exact_bon_curve(spec, n_trend, method="gauss_legendre", n_nodes=256), dtype=float)
        peak_idx = int(np.argmax(accs))
        criterion = check_collapse_criterion(spec) if isinstance(spec, GaussianVerifierSpec) else {
            "regime": "contaminated_heavy_tail",
            "will_collapse": True,
            "is_monotone_increasing": False,
            "asymptotic_limit": 0.0,
            "x_peak": None,
        }
        collapsed_to_tail = bool(accs[-1] < accs[peak_idx] and accs[-1] < accs[0] * 0.9 + 1e-12) if criterion["will_collapse"] else False
        trend_results[name] = {
            "n_values": n_trend,
            "accuracy": accs.tolist(),
            "peak_n_on_grid": n_trend[peak_idx],
            "peak_accuracy_on_grid": float(accs[peak_idx]),
            "n_1_accuracy": float(accs[0]),
            "n_10000_accuracy": float(accs[-1]),
            "criterion": criterion,
            "collapsed_after_peak": bool(accs[-1] < accs[peak_idx]),
            "dropped_below_baseline_at_10000": bool(accs[-1] < accs[0]),
            "trend_consistent": (
                (accs[-1] < accs[peak_idx]) if criterion["will_collapse"] else (accs[-1] >= accs[0])
            ),
        }

    # 闭式最优 N* vs 精确数值最优 N*_exact 对照（36 组参数）
    opt_comparisons = []
    rel_errors_tail = []
    rel_errors_saddle = []
    acc_gaps_saddle = []

    for p in (0.1, 0.25, 0.4):
        for dmu in (0.8, 1.2, 1.6):
            for kappa in (1.2, 1.4, 1.6, 1.9):
                spec = GaussianVerifierSpec(p=p, mu1=dmu, sigma1=1.0, mu0=0.0, sigma0=kappa)
                num_opt = numerical_optimal_n(spec, n_max=100_000)
                cf_tail = closed_form_optimal_n(spec, method="tail_quantile")
                cf_saddle = closed_form_optimal_n(spec, method="saddle_refined")

                n_exact = int(num_opt["n_star"])
                max_acc = float(num_opt["max_accuracy"])
                n_tail = int(cf_tail["n_star"])
                n_saddle = int(cf_saddle["n_star"])

                acc_at_saddle = float(exact_bon_accuracy(spec, n_saddle, method="gauss_legendre", n_nodes=256))
                acc_gap = max_acc - acc_at_saddle

                # 排除到达搜索上界 100000 的截断情况再统计相对误差
                if n_exact < 100_000:
                    rel_errors_tail.append(abs(n_tail - n_exact) / max(1, n_exact))
                    rel_errors_saddle.append(abs(n_saddle - n_exact) / max(1, n_exact))
                acc_gaps_saddle.append(acc_gap)

                opt_comparisons.append({
                    "p": p,
                    "delta_mu": dmu,
                    "kappa": kappa,
                    "n_exact": n_exact,
                    "max_accuracy": max_acc,
                    "n_tail_quantile": n_tail,
                    "n_tail_float": cf_tail["n_star_float"],
                    "n_saddle_refined": n_saddle,
                    "n_saddle_float": cf_saddle["n_star_float"],
                    "acc_at_saddle": acc_at_saddle,
                    "acc_gap_saddle": acc_gap,
                    "rel_err_tail": abs(n_tail - n_exact) / max(1, n_exact),
                    "rel_err_saddle": abs(n_saddle - n_exact) / max(1, n_exact),
                })

    summary = {
        "n_cases": len(opt_comparisons),
        "n_interior_cases": len(rel_errors_saddle),
        "median_rel_err_tail_quantile": float(np.median(rel_errors_tail)) if rel_errors_tail else None,
        "median_rel_err_saddle_refined": float(np.median(rel_errors_saddle)) if rel_errors_saddle else None,
        "mean_acc_gap_saddle_refined": float(np.mean(acc_gaps_saddle)),
        "max_acc_gap_saddle_refined": float(np.max(acc_gaps_saddle)),
    }

    return {
        "collapse_trends": trend_results,
        "optimal_n_comparison": opt_comparisons,
        "summary": summary,
    }


def verify_theorem_3(
    n_trials: int = 100_000,
    n_trials_high_precision: int = 1_500_000,
    seed: int = 42,
) -> dict[str, Any]:
    """验证定理 3：二元 Verifier 解析公式 vs 100k/1.5M MC、边际增益与最优预算。"""
    configs = [
        {"name": "moderate_verifier", "p": 0.20, "tpr": 0.85, "fpr": 0.10},
        {"name": "strong_verifier", "p": 0.35, "tpr": 0.90, "fpr": 0.05},
        {"name": "weak_verifier", "p": 0.15, "tpr": 0.75, "fpr": 0.15},
    ]
    n_values = [1, 2, 4, 8, 16, 32, 64, 128]
    results = {}
    all_passed = True

    for i, cfg in enumerate(configs):
        spec = BinaryVerifierSpec(p=cfg["p"], tpr=cfg["tpr"], fpr=cfg["fpr"])
        exact_vals = np.asarray(binary_bon_accuracy(spec, np.asarray(n_values)), dtype=float)
        mc_res = simulate_binary_bon(spec, n_values, n_trials=n_trials, seed=seed + i, batch_size=20_000)
        mc_vals = np.asarray(mc_res["accuracy"], dtype=float)
        mc_se = np.asarray(mc_res["se"], dtype=float)
        abs_errors = np.abs(exact_vals - mc_vals)
        max_err = float(np.max(abs_errors))
        mean_err = float(np.mean(abs_errors))
        max_z = float(np.max(abs_errors / np.maximum(mc_se, 1e-12)))

        hp_res = simulate_binary_bon(
            spec, n_values, n_trials=n_trials_high_precision, seed=seed + 500 + i, batch_size=50_000
        )
        hp_vals = np.asarray(hp_res["accuracy"], dtype=float)
        hp_max_err = float(np.max(np.abs(exact_vals - hp_vals)))

        # 边际增益与一阶差分一致性
        n_check = np.arange(1, 33)
        mg_formula = np.asarray(binary_marginal_gain(spec, n_check), dtype=float)
        mg_diff = np.asarray(binary_bon_accuracy(spec, n_check + 1), dtype=float) - np.asarray(
            binary_bon_accuracy(spec, n_check), dtype=float
        )
        max_mg_diff = float(np.max(np.abs(mg_formula - mg_diff)))

        ceiling = float(binary_asymptotic_ceiling(spec))
        n_eps_001 = int(binary_optimal_budget(spec, eps=0.01))
        n_cost_0002 = int(binary_optimal_budget(spec, cost_per_sample=0.002))
        gap_at_n_eps = ceiling - float(binary_bon_accuracy(spec, n_eps_001))

        passed = (
            (max_z < 3.5)
            and (hp_max_err < 1e-3)
            and (max_mg_diff < 1e-12)
            and (gap_at_n_eps <= 0.01 + 1e-12)
        )
        all_passed = all_passed and passed

        results[cfg["name"]] = {
            "params": cfg,
            "n_values": n_values,
            "exact_accuracy": exact_vals.tolist(),
            "mc_accuracy": mc_vals.tolist(),
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "max_z_score": max_z,
            "hp_max_abs_error": hp_max_err,
            "max_marginal_gain_diff": max_mg_diff,
            "asymptotic_ceiling_ppv": ceiling,
            "n_star_eps_0.01": n_eps_001,
            "gap_at_n_eps_0.01": gap_at_n_eps,
            "n_star_cost_0.002": n_cost_0002,
            "passed": passed,
        }

    return {"configs": results, "all_passed": all_passed}


def verify_theorem_4(n_trials: int = 20_000, seed: int = 42) -> dict[str, Any]:
    """验证定理 4：连续校准极限 + 离散答案空间加权多数决大 N 无坍塌收敛。"""
    continuous_specs = {
        "heteroscedastic_kappa_1.5": GaussianVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5
        ),
        "reward_hacking_eps_0.08": ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    }
    n_cont = [1, 10, 100, 1000, 5000]
    cont_results = {}

    for name, spec in continuous_specs.items():
        uncal = [float(exact_bon_accuracy(spec, n, method="gauss_legendre", n_nodes=256)) for n in n_cont]
        cal = [float(exact_calibrated_bon_accuracy(spec, n, n_grid=4096)) for n in n_cont]
        asymp = float(calibrated_bon_asymptotic_limit(spec))
        cont_results[name] = {
            "n_values": n_cont,
            "uncalibrated_bon": uncal,
            "calibrated_bon": cal,
            "asymptotic_limit": asymp,
            "collapse_eliminated": bool(cal[-1] > cal[0] and uncal[-1] < uncal[1]),
        }

    # 离散答案空间对比（包含高诱饵错误答案分布）
    n_disc = [1, 8, 32, 128, 512, 1000]
    hack_spec = ContaminatedVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5,
        epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
    )
    task_specs = {
        "uniform_wrong_answers": DiscreteAnswerTaskSpec(
            verifier_spec=hack_spec,
            num_wrong_answers=5,
            wrong_answer_probs=None,
        ),
        "decoy_heavy_wrong_answers": DiscreteAnswerTaskSpec(
            verifier_spec=hack_spec,
            num_wrong_answers=5,
            wrong_answer_probs=(0.45, 0.20, 0.15, 0.10, 0.10),
        ),
    }

    # 额外验证基于 EmpiricalScoreCalibrator（无解析分布先验）拟合后的抗坍塌表现
    rng = np.random.default_rng(seed)
    n_cal_train = 50_000
    y_cal = rng.random(n_cal_train) < hack_spec.p
    s_cal = np.empty(n_cal_train, dtype=np.float64)
    n_pos_cal = int(np.count_nonzero(y_cal))
    s_cal[y_cal] = _sample_from_spec(hack_spec, y=1, size=n_pos_cal, rng=rng)
    s_cal[~y_cal] = _sample_from_spec(hack_spec, y=0, size=n_cal_train - n_pos_cal, rng=rng)
    emp_cal = algo_mod.EmpiricalScoreCalibrator(n_bins=60, method="quantile_hist").fit(s_cal, y_cal)

    algos = {
        "Standard-BoN": algo_mod.StandardBoN(),
        "Majority-Voting": _build_majority_voting_baseline(),
        "Calibrated-Weighted-BoN": algo_mod.CalibratedWeightedBoN(spec=hack_spec),
        "Two-Stage-Verified-Majority": algo_mod.TwoStageVerifiedMajority(spec=hack_spec),
        "Empirical-TwoStage": algo_mod.TwoStageVerifiedMajority(calibrator=emp_cal),
    }

    disc_results = {}
    for task_name, task_spec in task_specs.items():
        sim_out = simulate_algorithms_comparison(
            task_spec=task_spec,
            n_values=n_disc,
            algorithms=algos,
            n_trials=n_trials,
            seed=seed,
            batch_size=2_000,
        )
        disc_results[task_name] = sim_out

    # 计算理论 Hoeffding/Bernstein 边界（对比截断赔率 w_max=20.0 与紧致后验加权 w_max=1.0）
    n_est = 200_000
    s1 = _sample_from_spec(hack_spec, y=1, size=n_est, rng=rng)
    s0 = _sample_from_spec(hack_spec, y=0, size=n_est, rng=rng)
    w_max = 20.0
    q1 = hack_spec.posterior(s1)
    q0 = hack_spec.posterior(s0)
    w1 = np.minimum(w_max, q1 / np.maximum(1.0 - q1 + 1e-6, 1e-6))
    w0 = np.minimum(w_max, q0 / np.maximum(1.0 - q0 + 1e-6, 1e-6))
    bound_odds = weighted_majority_convergence_bound(
        p=hack_spec.p,
        p_wrong_max=(1.0 - hack_spec.p) * 0.45,
        mu_w_pos=float(np.mean(w1)),
        mu_w_neg=float(np.mean(w0)),
        w_max=w_max,
    )
    bound_posterior = weighted_majority_convergence_bound(
        p=hack_spec.p,
        p_wrong_max=(1.0 - hack_spec.p) * 0.45,
        mu_w_pos=float(np.mean(q1)),
        mu_w_neg=float(np.mean(q0)),
        w_max=1.0,
    )

    return {
        "continuous_calibrated_bon": cont_results,
        "discrete_algorithms": disc_results,
        "theoretical_hoeffding_bound": bound_odds,
        "theoretical_posterior_bound": bound_posterior,
    }


def _sample_from_spec(
    spec: GaussianVerifierSpec | ContaminatedVerifierSpec,
    y: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """从条件打分分布抽取样本，用于估算加权期望与 Hoeffding 收敛边界。"""
    if y == 1:
        return rng.normal(spec.mu1, spec.sigma1, size=size)
    if isinstance(spec, ContaminatedVerifierSpec):
        is_hack = rng.random(size) < spec.epsilon
        samples = rng.normal(spec.mu0, spec.sigma0, size=size)
        n_hack = int(np.sum(is_hack))
        if n_hack > 0:
            samples[is_hack] = rng.normal(spec.mu_hack, spec.sigma_hack, size=n_hack)
        return samples
    return rng.normal(spec.mu0, spec.sigma0, size=size)


def print_summary(report: dict[str, Any]) -> None:
    """在终端清晰打印定理 1-4 自动化交叉验证汇总表。"""
    print("=" * 80)
    print("Test-Time Scaling 理论验证汇总报告 (Theorems 1-4)")
    print("=" * 80)

    print("\n[定理 1] 精确顺序统计量积分 vs 100k/1.5M 蒙特卡洛:")
    print(f"{'场景':<28} {'100k均值误差':>12} {'100k最大误差':>12} {'1.5M最大误差':>12} {'95%CI覆盖':>10} {'quad差':>10} {'状态':>6}")
    for name, item in report["theorem_1"]["scenarios"].items():
        status = "PASS" if item["passed_1500k_strict_1e3"] and item["passed_100k_stat"] else "WARN"
        print(
            f"{name:<28} {item['mean_abs_error']:>12.6f} {item['max_abs_error']:>12.6f} "
            f"{item['hp_max_abs_error']:>12.6f} {item['ci_95_coverage']:>9.1%} "
            f"{item['max_quad_vs_gl_diff']:>10.2e} {status:>6}"
        )

    print("\n[定理 2] 极值坍塌趋势与闭式最优预算 N* 精度:")
    t2_sum = report["theorem_2"]["summary"]
    print(
        f"参数扫描共 {t2_sum['n_cases']} 组 (非截断内点 {t2_sum['n_interior_cases']} 组) | "
        f"Saddle-Refined N* 中位相对误差: {t2_sum['median_rel_err_saddle_refined']:.2%} | "
        f"Tail-Quantile 中位相对误差: {t2_sum['median_rel_err_tail_quantile']:.2%} | "
        f"Saddle-Refined 准确率损失均值: {t2_sum['mean_acc_gap_saddle_refined']:.6f}"
    )

    print("\n[定理 3] 二元 Verifier 解析公式 vs 100k/1.5M 蒙特卡洛:")
    for name, item in report["theorem_3"]["configs"].items():
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{name:<20} | 100kMax={item['max_abs_error']:.6f} | 1.5MMax={item['hp_max_abs_error']:.6f} | "
            f"PPV={item['asymptotic_ceiling_ppv']:.4f} | "
            f"N*(eps=0.01)={item['n_star_eps_0.01']} | {status}"
        )

    print("\n[定理 4] 校准加权与两阶段多数决无坍塌收敛:")
    for name, item in report["theorem_4"]["continuous_calibrated_bon"].items():
        print(
            f"{name:<28} | Uncal(N=5000)={item['uncalibrated_bon'][-1]:.4f} | "
            f"Cal(N=5000)={item['calibrated_bon'][-1]:.4f} | "
            f"Sup q(x)={item['asymptotic_limit']:.4f}"
        )
    print("=" * 80)


def main() -> None:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("正在验证定理 1（精确积分 vs 100,000 次蒙特卡洛）...")
    t1 = verify_theorem_1(n_trials=100_000, seed=42)

    print("正在验证定理 2（极值坍塌趋势与闭式 N* 精度）...")
    t2 = verify_theorem_2()

    print("正在验证定理 3（二元 Verifier 解析式 vs 100,000 次蒙特卡洛）...")
    t3 = verify_theorem_3(n_trials=100_000, seed=42)

    print("正在验证定理 4（校准后验与加权多数决大 N 收敛）...")
    t4 = verify_theorem_4(n_trials=20_000, seed=42)

    elapsed = time.perf_counter() - t0
    report = {
        "metadata": {
            "elapsed_seconds": elapsed,
            "seed": 42,
        },
        "theorem_1": t1,
        "theorem_2": t2,
        "theorem_3": t3,
        "theorem_4": t4,
    }

    out_path = results_dir / "theorem_verification.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(report), f, ensure_ascii=False, indent=2)

    print_summary(report)
    print(f"完整验证结果已保存至: {out_path} (耗时 {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
