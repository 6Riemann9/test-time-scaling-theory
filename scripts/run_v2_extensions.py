"""V2 原创扩展理论自动化实验脚本：定理 5 与定理 6 端到端量化验证。

生成三组 V2 实验数据并输出至 `results/v2_extension_results.json`：
- 实验 E（定理 5A）：对称相关高斯 K-Verifier 的精确相变逆转曲线（精确积分 vs 100k MC）与相界扫描；
- 实验 F（定理 5B）：Student-t 幂律重尾下 Mean vs Min 相变撕裂、同源漏洞击穿与异质漏洞分离；
- 实验 G（定理 6A/6B）：有限校准样本 M 下的 MLE 二次坍塌、Wilson/Bernstein-LCB 根治效果与 Minimax (M/ln M)^{-2/5} 标度律。

运行方式：
    python scripts/run_v2_extensions.py
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

from tts_theory.calibration_theory import (
    calibration_minimax_scaling,
    simulate_finite_calibration_scaling,
)
from tts_theory.ensemble_theory import (
    HeavyTailMultiVerifierSpec,
    HeterogeneousVulnerabilitySpec,
    MultiVerifierGaussianSpec,
    ensemble_phase_reversal_criterion,
    heavy_tail_separation_criterion,
    simulate_multi_verifier_bon,
)
from tts_theory.integrals import exact_bon_curve
from tts_theory.simulator import simulate_bon
from tts_theory.theorems import GaussianVerifierSpec, calibrated_bon_asymptotic_limit


def to_json_safe(obj: Any) -> Any:
    """将 NumPy 标量与数组递归转为原生 JSON 类型。"""
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
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def run_experiment_e(n_trials: int = 100_000, seed: int = 42) -> dict[str, Any]:
    """实验 E：高斯 K-Verifier 相变逆转曲线（精确积分 vs 100,000 次 MC）及相边界扫描。"""
    n_dense = sorted(set(np.unique(np.round(np.geomspace(1, 5000, 55)).astype(int)).tolist()))
    n_mc = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # 基线参数：kappa=1.65, mu1=0.9, rho1=0.85, rho0=0.10 => 理论临界裁判数 K_c = 4
    configs = [
        {"id": "k1_single", "label": "Single Verifier (K=1)", "k": 1, "rho1": 0.85, "rho0": 0.10, "agg": "min"},
        {"id": "k2_min", "label": "Min-Ensemble (K=2 < K_c=4)", "k": 2, "rho1": 0.85, "rho0": 0.10, "agg": "min"},
        {"id": "k4_min", "label": "Min-Ensemble (K=4 = K_c)", "k": 4, "rho1": 0.85, "rho0": 0.10, "agg": "min"},
        {"id": "k6_min", "label": "Min-Ensemble (K=6 > K_c)", "k": 6, "rho1": 0.85, "rho0": 0.10, "agg": "min"},
        {"id": "k6_mean", "label": "Mean-Ensemble (K=6)", "k": 6, "rho1": 0.85, "rho0": 0.10, "agg": "mean"},
        {"id": "k6_irrev", "label": "Irreversible (K=6, rho0=0.60)", "k": 6, "rho1": 0.70, "rho0": 0.60, "agg": "min"},
    ]

    curves: dict[str, Any] = {}
    for idx, cfg in enumerate(configs):
        spec = MultiVerifierGaussianSpec(
            p=0.25,
            mu1=0.9,
            sigma1=1.0,
            mu0=0.0,
            sigma0=1.65,
            num_verifiers=cfg["k"],
            rho1=cfg["rho1"],
            rho0=cfg["rho0"],
        )
        crit = ensemble_phase_reversal_criterion(spec)
        if cfg["agg"] == "mean":
            exact_dense = exact_bon_curve(spec.mean_verifier_spec(), n_dense)
            exact_mc_grid = exact_bon_curve(spec.mean_verifier_spec(), n_mc)
            mc_res = simulate_bon(spec.mean_verifier_spec(), n_mc, n_trials=n_trials, seed=seed + idx)
        else:
            min_spec = spec.min_verifier_spec()
            exact_dense = exact_bon_curve(min_spec, n_dense)
            exact_mc_grid = exact_bon_curve(min_spec, n_mc)
            mc_res = simulate_multi_verifier_bon(
                spec, n_mc, aggregation="min", n_trials=n_trials, seed=seed + idx, batch_size=4_000
            )

        abs_err = np.abs(exact_mc_grid - mc_res["accuracy"])
        curves[cfg["id"]] = {
            "label": cfg["label"],
            "k": cfg["k"],
            "rho1": cfg["rho1"],
            "rho0": cfg["rho0"],
            "aggregation": cfg["agg"],
            "criterion": crit,
            "n_dense": n_dense,
            "exact_dense": exact_dense,
            "n_mc": n_mc,
            "exact_at_mc": exact_mc_grid,
            "mc": mc_res,
            "mae_vs_exact": float(np.mean(abs_err)),
            "max_abs_err_vs_exact": float(np.max(abs_err)),
        }

    # 相界扫描：(rho0, kappa) 二维网格下的最小临界 K_c（rho1=0.80）
    rho0_grid = np.linspace(0.0, 0.75, 16)
    kappa_grid = np.linspace(1.05, 2.0, 20)
    kc_matrix = np.full((len(rho0_grid), len(kappa_grid)), np.nan, dtype=float)
    for i, r0 in enumerate(rho0_grid):
        for j, kap in enumerate(kappa_grid):
            sp = MultiVerifierGaussianSpec(
                p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=float(kap),
                num_verifiers=2, rho1=0.80, rho0=float(r0),
            )
            cr = ensemble_phase_reversal_criterion(sp)
            if cr["critical_k"] is not None:
                kc_matrix[i, j] = float(cr["critical_k"])

    return {
        "curves": curves,
        "phase_scan": {
            "rho1": 0.80,
            "rho0_grid": rho0_grid,
            "kappa_grid": kappa_grid,
            "critical_k_matrix": kc_matrix,
        },
    }


def run_experiment_f(n_trials: int = 100_000, seed: int = 2026) -> dict[str, Any]:
    """实验 F：幂律重尾（Student-t）下 Mean vs Min 分离、同源击穿与异质污染对比。"""
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    spec_indep = HeavyTailMultiVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        num_verifiers=4, alpha1=5.0, alpha0=1.8, eta1=0.15, eta0=0.15, epsilon_shared=0.0,
    )
    spec_single = HeavyTailMultiVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        num_verifiers=1, alpha1=5.0, alpha0=1.8, eta1=0.15, eta0=0.15, epsilon_shared=0.0,
    )
    spec_shared = HeavyTailMultiVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        num_verifiers=4, alpha1=5.0, alpha0=1.8, eta1=0.15, eta0=0.15, epsilon_shared=0.04,
    )
    spec_het_1bad = HeterogeneousVulnerabilitySpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=0.8,
        num_verifiers=4, n_vulnerable=1, alpha1=6.0, alpha0=2.0, sigma_vuln=1.2, eta1=0.15,
    )

    jobs = [
        ("single_k1", "Single Verifier (K=1, heavy-tail)", spec_single, "min", heavy_tail_separation_criterion(spec_single)),
        ("mean_k4_indep", "Mean-Ensemble (K=4, indep hack)", spec_indep, "mean", heavy_tail_separation_criterion(spec_indep)),
        ("min_k4_indep", "Min-Ensemble (K=4, indep hack)", spec_indep, "min", heavy_tail_separation_criterion(spec_indep)),
        ("min_k4_shared_0.04", "Min-Ensemble (K=4, shared eps=0.04)", spec_shared, "min", heavy_tail_separation_criterion(spec_shared)),
        ("het_mean_1vuln", "Heterogeneous Mean (3 Good + 1 Vuln)", spec_het_1bad, "mean", {"regime": "1_bad_kills_mean"}),
        ("het_min_1vuln", "Heterogeneous Min (3 Good + 1 Vuln)", spec_het_1bad, "min", {"regime": "1_good_saves_min"}),
    ]

    out = {}
    for idx, (key, label, spec, agg, crit) in enumerate(jobs):
        res = simulate_multi_verifier_bon(
            spec, n_values, aggregation=agg, n_trials=n_trials, seed=seed + idx, batch_size=4_000
        )
        out[key] = {
            "label": label,
            "aggregation": agg,
            "criterion": crit,
            "n_values": res["n_values"],
            "accuracy": res["accuracy"],
            "se": res["se"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
        }
    return out


def run_experiment_g(seed: int = 777) -> dict[str, Any]:
    """实验 G：有限校准样本 M 二次坍塌对比与 Minimax (M / ln M)^{2/5} 标度律验证。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    q_peak_true = calibrated_bon_asymptotic_limit(spec)

    # G1: 二次坍塌曲线对比（M=400, 等宽 K=20 与 细分等频 K=40）
    n_values_g1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    g1_uniform = simulate_finite_calibration_scaling(
        spec,
        m_cal=400,
        n_bins=20,
        n_values=n_values_g1,
        binning="uniform",
        methods=("mle", "laplace", "wilson_lcb", "empirical_bernstein_lcb"),
        n_repeats=2_000,
        delta=0.05,
        seed=seed,
    )
    g1_quantile = simulate_finite_calibration_scaling(
        spec,
        m_cal=400,
        n_bins=40,
        n_values=n_values_g1,
        binning="quantile",
        methods=("mle", "laplace", "wilson_lcb", "empirical_bernstein_lcb"),
        n_repeats=2_000,
        delta=0.05,
        seed=seed + 1,
    )

    # G2: Minimax 标度律验证（固定大预算 N=5000，扫描 M）
    m_grid = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000]
    scaling_records = []
    for idx, m_val in enumerate(m_grid):
        mm = calibration_minimax_scaling(m_val, curvature_c=0.8, q_peak=q_peak_true, delta=0.05)
        k_opt = max(5, int(round(2.0 * (m_val / max(1.0, np.log(m_val))) ** 0.4)))
        res_opt = simulate_finite_calibration_scaling(
            spec,
            m_cal=m_val,
            n_bins=k_opt,
            n_values=[5000],
            binning="quantile",
            methods=("mle", "wilson_lcb", "empirical_bernstein_lcb"),
            n_repeats=1_500,
            delta=0.05,
            seed=seed + 100 + idx,
        )
        res_fixed = simulate_finite_calibration_scaling(
            spec,
            m_cal=m_val,
            n_bins=80,
            n_values=[5000],
            binning="quantile",
            methods=("mle", "wilson_lcb"),
            n_repeats=1_500,
            delta=0.05,
            seed=seed + 200 + idx,
        )
        res_coarse = simulate_finite_calibration_scaling(
            spec,
            m_cal=m_val,
            n_bins=6,
            n_values=[5000],
            binning="quantile",
            methods=("wilson_lcb",),
            n_repeats=1_500,
            delta=0.05,
            seed=seed + 300 + idx,
        )

        regret_opt_wilson = float(q_peak_true - res_opt["methods"]["wilson_lcb"]["accuracy"][0])
        regret_opt_bern = float(q_peak_true - res_opt["methods"]["empirical_bernstein_lcb"]["accuracy"][0])
        regret_opt_mle = float(q_peak_true - res_opt["methods"]["mle"]["accuracy"][0])
        regret_fixed_wilson = float(q_peak_true - res_fixed["methods"]["wilson_lcb"]["accuracy"][0])
        regret_fixed_mle = float(q_peak_true - res_fixed["methods"]["mle"]["accuracy"][0])
        regret_coarse_wilson = float(q_peak_true - res_coarse["methods"]["wilson_lcb"]["accuracy"][0])

        scaling_records.append({
            "m_cal": m_val,
            "k_opt": k_opt,
            "regret_rate_theory": mm["regret_rate_theory"],
            "regret_opt_wilson": regret_opt_wilson,
            "regret_opt_bernstein": regret_opt_bern,
            "regret_opt_mle": regret_opt_mle,
            "regret_fixed80_wilson": regret_fixed_wilson,
            "regret_fixed80_mle": regret_fixed_mle,
            "regret_coarse6_wilson": regret_coarse_wilson,
            "acc_opt_wilson": float(res_opt["methods"]["wilson_lcb"]["accuracy"][0]),
            "acc_opt_mle": float(res_opt["methods"]["mle"]["accuracy"][0]),
            "acc_fixed80_wilson": float(res_fixed["methods"]["wilson_lcb"]["accuracy"][0]),
            "acc_fixed80_mle": float(res_fixed["methods"]["mle"]["accuracy"][0]),
            "acc_coarse6_wilson": float(res_coarse["methods"]["wilson_lcb"]["accuracy"][0]),
        })

    # 在双对数坐标下拟合实测遗憾收敛斜率 ln(regret) vs ln(M / ln M)
    x_log = np.array([np.log(r["m_cal"] / np.log(r["m_cal"])) for r in scaling_records])
    y_wilson = np.log(np.maximum(1e-6, [r["regret_opt_wilson"] for r in scaling_records]))
    slope_wilson, intercept_wilson = np.polyfit(x_log, y_wilson, 1)
    # Bernstein-LCB 在小 M 因 7/3 常数保守触发全零截断，在充分样本区 M >= 5000 拟合斜率
    large_mask = np.array([r["m_cal"] >= 5000 for r in scaling_records])
    y_bern = np.log(np.maximum(1e-6, [r["regret_opt_bernstein"] for r in scaling_records]))
    slope_bern, intercept_bern = np.polyfit(x_log[large_mask], y_bern[large_mask], 1)

    return {
        "q_peak_true": float(q_peak_true),
        "g1_uniform_m400_k20": g1_uniform,
        "g1_quantile_m400_k40": g1_quantile,
        "g2_minimax_scaling": {
            "records": scaling_records,
            "fitted_slope_wilson": float(slope_wilson),
            "fitted_intercept_wilson": float(intercept_wilson),
            "fitted_slope_bernstein": float(slope_bern),
            "fitted_intercept_bernstein": float(intercept_bern),
            "theoretical_slope": -0.4,
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    print("运行实验 E（定理 5A 高斯多裁判相变逆转，100k MC）...")
    exp_e = run_experiment_e(n_trials=100_000, seed=42)
    print("运行实验 F（定理 5B 重尾 Student-t 下 Mean vs Min 分离与同源击穿，100k MC）...")
    exp_f = run_experiment_f(n_trials=100_000, seed=2026)
    print("运行实验 G（定理 6 有限校准集 M 二次坍塌与 Minimax 标度律验证）...")
    exp_g = run_experiment_g(seed=777)

    elapsed = time.perf_counter() - t0
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "v2_extension_results.json"

    payload = to_json_safe({
        "metadata": {
            "elapsed_seconds": elapsed,
            "n_trials_ensemble": 100_000,
        },
        "experiment_e_gaussian_ensemble": exp_e,
        "experiment_f_heavy_tail_ensemble": exp_f,
        "experiment_g_finite_calibration": exp_g,
    })
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"V2 原创扩展实验已完成并保存至: {out_path} (耗时 {elapsed:.2f}s)")
    for k, d in exp_e["curves"].items():
        print(
            f"[Exp E] {k:14s} | MAE vs Exact = {d['mae_vs_exact']:.6f} | "
            f"N=4096 Acc = {d['mc']['accuracy'][-1]:.4f} | WillCollapse={d['criterion']['will_collapse']}"
        )
    for k, d in exp_f.items():
        print(f"[Exp F] {k:20s} | N=1 -> {d['accuracy'][0]:.4f} | N=4096 -> {d['accuracy'][-1]:.4f}")
    g1u = exp_g["g1_uniform_m400_k20"]["methods"]
    print(
        f"[Exp G1 Uniform M=400, K=20] N=8192: "
        f"MLE={g1u['mle']['accuracy'][-1]:.4f} (Peak={np.max(g1u['mle']['accuracy']):.4f}), "
        f"Wilson-LCB={g1u['wilson_lcb']['accuracy'][-1]:.4f}, "
        f"Bernstein-LCB={g1u['empirical_bernstein_lcb']['accuracy'][-1]:.4f}"
    )
    g2 = exp_g["g2_minimax_scaling"]
    print(
        f"[Exp G2 Minimax Slope] Wilson-LCB={g2['fitted_slope_wilson']:.4f}, "
        f"Bernstein-LCB={g2['fitted_slope_bernstein']:.4f} (Theory=-0.4000)"
    )


if __name__ == "__main__":
    main()
