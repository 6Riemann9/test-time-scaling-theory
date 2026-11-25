"""系统实验与参数扫描脚本。

生成四组核心实验数据并输出到 `results/experiment_results.json` 及 CSV 汇总表：
- 实验 A：同方差/异方差与 Reward Hacking 污染下的 BoN Scaling 曲线（精确积分 vs 100,000 次 MC）；
- 实验 B：全参数网格扫描 (p, delta_mu, kappa) 与 (delta_mu, kappa) 二维相变图数据 + 闭式/数值 N* 一致性数据集；
- 实验 C：强 Reward Hacking 混合尾部 (epsilon in {0.02, 0.08, 0.15}) 下 5 大算法对比；
- 实验 D：二元 Verifier 边际收益衰减与最优采样预算扫描。

运行方式：
    python scripts/run_experiments.py
"""

from __future__ import annotations

import csv
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
from tts_theory.integrals import exact_bon_accuracy, exact_bon_curve, numerical_optimal_n
from tts_theory.simulator import (
    DiscreteAnswerTaskSpec,
    _sample_scores_and_labels,
    simulate_algorithms_comparison,
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
    check_collapse_criterion,
    closed_form_optimal_n,
)


def to_json_safe(obj: Any) -> Any:
    """将 NumPy 标量/数组递归转换为 JSON 可序列化的原生 Python 类型。"""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _build_majority_voting_baseline() -> Any:
    """获取普通无权 Majority Voting 基线选择器。"""
    return algo_mod.MajorityVoting()


def _build_optimal_n_star_selector(spec: GaussianVerifierSpec | ContaminatedVerifierSpec) -> Any:
    """构造 Optimal-N* 算法实例（支持高斯闭式解与混合尾部数值精确峰值）。"""
    return algo_mod.OptimalNStarBoN(spec=spec, max_search_n=10_000)


def run_experiment_a(n_trials_mc: int = 100_000, seed: int = 42) -> dict[str, Any]:
    """实验 A：精确积分曲线 vs 100,000 次蒙特卡洛叠图数据。"""
    n_dense = sorted(set(np.unique(np.round(np.geomspace(1, 10000, 65)).astype(int)).tolist()))
    n_mc = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    configs = [
        {
            "id": "kappa_0.75",
            "label": "Subcritical (kappa=0.75)",
            "group": "heteroscedastic",
            "spec": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=0.75),
        },
        {
            "id": "kappa_1.00",
            "label": "Homoscedastic (kappa=1.00)",
            "group": "both",
            "spec": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.00),
        },
        {
            "id": "kappa_1.35",
            "label": "Heteroscedastic (kappa=1.35)",
            "group": "heteroscedastic",
            "spec": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.35),
        },
        {
            "id": "kappa_1.70",
            "label": "Heteroscedastic (kappa=1.70)",
            "group": "heteroscedastic",
            "spec": GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.70),
        },
        {
            "id": "hack_eps_0.05",
            "label": "Reward Hacking (eps=0.05)",
            "group": "contaminated",
            "spec": ContaminatedVerifierSpec(
                p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
                epsilon=0.05, mu_hack=3.5, sigma_hack=1.5,
            ),
        },
        {
            "id": "hack_eps_0.12",
            "label": "Reward Hacking (eps=0.12)",
            "group": "contaminated",
            "spec": ContaminatedVerifierSpec(
                p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
                epsilon=0.12, mu_hack=3.5, sigma_hack=1.5,
            ),
        },
    ]

    out = {}
    for idx, cfg in enumerate(configs):
        spec = cfg["spec"]
        exact_dense = np.asarray(
            exact_bon_curve(spec, n_dense, method="gauss_legendre", n_nodes=256),
            dtype=float,
        )
        mc_res = simulate_bon(
            spec, n_mc, n_trials=n_trials_mc, seed=seed + idx, batch_size=10_000
        )
        num_opt = numerical_optimal_n(spec, n_max=100_000)

        if isinstance(spec, GaussianVerifierSpec):
            cf_saddle = closed_form_optimal_n(spec, method="saddle_refined")
            cf_tail = closed_form_optimal_n(spec, method="tail_quantile")
            criterion = check_collapse_criterion(spec)
        else:
            cf_saddle = {
                "n_star_float": None,
                "n_star": None,
                "x_peak": None,
                "peak_posterior": None,
                "will_collapse": True,
                "regime": "contaminated_heavy_tail",
            }
            cf_tail = cf_saddle
            criterion = {
                "regime": "contaminated_heavy_tail",
                "will_collapse": True,
                "is_monotone_increasing": False,
                "asymptotic_limit": 0.0,
                "x_peak": None,
            }

        out[cfg["id"]] = {
            "label": cfg["label"],
            "group": cfg["group"],
            "n_dense": n_dense,
            "exact_dense": exact_dense.tolist(),
            "n_mc": n_mc,
            "mc": mc_res,
            "numerical_optimal": num_opt,
            "closed_form_saddle": cf_saddle,
            "closed_form_tail": cf_tail,
            "criterion": criterion,
        }
    return out


def run_experiment_b() -> dict[str, Any]:
    """实验 B：全参数空间扫描与 (delta_mu, kappa) 二维相变相图。"""
    p_list = [0.1, 0.2, 0.35, 0.5]
    dmu_list = [0.5, 1.0, 1.5, 2.0]
    kappa_list = [0.6, 0.8, 0.95, 1.0, 1.05, 1.15, 1.25, 1.4, 1.6, 1.8, 2.0, 2.2]
    probe_ns = [1, 16, 64, 256, 1024, 4096]

    sweep_records = []
    for p in p_list:
        for dmu in dmu_list:
            for kappa in kappa_list:
                spec = GaussianVerifierSpec(p=p, mu1=dmu, sigma1=1.0, mu0=0.0, sigma0=kappa)
                crit = check_collapse_criterion(spec)
                cf_saddle = closed_form_optimal_n(spec, method="saddle_refined")
                cf_tail = closed_form_optimal_n(spec, method="tail_quantile")
                probe_accs = exact_bon_curve(spec, probe_ns, method="gauss_legendre", n_nodes=256)

                if crit["will_collapse"]:
                    num_opt = numerical_optimal_n(spec, n_max=100_000)
                    n_exact = int(num_opt["n_star"])
                    max_acc = float(num_opt["max_accuracy"])
                    acc_saddle = float(
                        exact_bon_accuracy(spec, int(cf_saddle["n_star"]), method="gauss_legendre", n_nodes=256)
                    )
                else:
                    n_exact = None
                    max_acc = float(probe_accs[-1])
                    acc_saddle = None

                sweep_records.append({
                    "p": p,
                    "delta_mu": dmu,
                    "kappa": kappa,
                    "regime": crit["regime"],
                    "will_collapse": crit["will_collapse"],
                    "is_monotone_increasing": crit["is_monotone_increasing"],
                    "asymptotic_limit": crit["asymptotic_limit"],
                    "x_peak": crit["x_peak"],
                    "n_star_exact": n_exact,
                    "max_accuracy_exact": max_acc,
                    "n_star_saddle": cf_saddle["n_star"],
                    "n_star_saddle_float": cf_saddle["n_star_float"],
                    "n_star_tail": cf_tail["n_star"],
                    "n_star_tail_float": cf_tail["n_star_float"],
                    "acc_at_saddle": acc_saddle,
                    "probe_ns": probe_ns,
                    "probe_accuracies": np.asarray(probe_accs, dtype=float).tolist(),
                })

    # 高分辨率二维相图 (delta_mu, kappa)，固定 p = 0.25
    dmu_grid = np.linspace(0.4, 2.2, 29)
    kappa_grid = np.linspace(0.6, 2.2, 33)
    log10_n_saddle = np.full((len(kappa_grid), len(dmu_grid)), np.nan, dtype=float)
    log10_n_tail = np.full((len(kappa_grid), len(dmu_grid)), np.nan, dtype=float)
    collapse_mask = np.zeros((len(kappa_grid), len(dmu_grid)), dtype=bool)

    for i, kappa in enumerate(kappa_grid):
        for j, dmu in enumerate(dmu_grid):
            spec = GaussianVerifierSpec(p=0.25, mu1=float(dmu), sigma1=1.0, mu0=0.0, sigma0=float(kappa))
            cf_s = closed_form_optimal_n(spec, method="saddle_refined")
            cf_t = closed_form_optimal_n(spec, method="tail_quantile")
            collapse_mask[i, j] = bool(cf_s["will_collapse"])
            if cf_s["will_collapse"] and cf_s["n_star_float"] is not None:
                log10_n_saddle[i, j] = float(np.log10(max(1.0, cf_s["n_star_float"])))
                log10_n_tail[i, j] = float(np.log10(max(1.0, cf_t["n_star_float"])))

    return {
        "parameter_grid": {
            "p": p_list,
            "delta_mu": dmu_list,
            "kappa": kappa_list,
        },
        "sweep_records": sweep_records,
        "heatmap_2d": {
            "fixed_p": 0.25,
            "dmu_grid": dmu_grid.tolist(),
            "kappa_grid": kappa_grid.tolist(),
            "log10_n_saddle": log10_n_saddle.tolist(),
            "log10_n_tail": log10_n_tail.tolist(),
            "collapse_mask": collapse_mask.tolist(),
        },
    }


def run_experiment_c(n_trials: int = 30_000, seed: int = 1024) -> dict[str, Any]:
    """实验 C：强 Reward Hacking 混合尾部下 5 种推理扩展算法全面对比。"""
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    epsilons = [0.02, 0.08, 0.15]
    wrong_probs = (0.40, 0.25, 0.15, 0.10, 0.10)

    scenarios = {}
    for idx, eps in enumerate(epsilons):
        spec = ContaminatedVerifierSpec(
            p=0.25,
            mu1=1.2,
            sigma1=1.0,
            mu0=0.0,
            sigma0=1.5,
            epsilon=eps,
            mu_hack=3.5,
            sigma_hack=1.5,
        )
        task_spec = DiscreteAnswerTaskSpec(
            verifier_spec=spec,
            num_wrong_answers=5,
            wrong_answer_probs=wrong_probs,
        )
        cal_rng = np.random.default_rng(seed + idx * 100 + 7)
        s_cal, y_cal = _sample_scores_and_labels(spec, (50_000,), cal_rng)
        emp_cal = algo_mod.EmpiricalScoreCalibrator(n_bins=60, method="quantile_hist").fit(s_cal, y_cal)

        algos = {
            "Standard-BoN": algo_mod.StandardBoN(),
            "Optimal-N*": _build_optimal_n_star_selector(spec),
            "Majority-Voting": _build_majority_voting_baseline(),
            "Calibrated-Weighted-BoN": algo_mod.CalibratedWeightedBoN(spec=spec),
            "Two-Stage-Verified-Majority": algo_mod.TwoStageVerifiedMajority(spec=spec),
            "Empirical-TwoStage": algo_mod.TwoStageVerifiedMajority(calibrator=emp_cal),
        }
        sim_out = simulate_algorithms_comparison(
            task_spec=task_spec,
            n_values=n_values,
            algorithms=algos,
            n_trials=n_trials,
            seed=seed + idx * 100,
            batch_size=2_000,
        )
        num_opt = numerical_optimal_n(spec, n_max=5000)
        scenarios[f"eps_{eps:.2f}"] = {
            "epsilon": eps,
            "kappa": 1.5,
            "p": 0.25,
            "delta_mu": 1.2,
            "mu_hack": 3.5,
            "sigma_hack": 1.5,
            "wrong_answer_probs": list(wrong_probs),
            "n_values": n_values,
            "n_trials": n_trials,
            "numerical_optimal_bon": num_opt,
            "results": sim_out,
        }

    return {"scenarios": scenarios}


def run_experiment_d() -> dict[str, Any]:
    """实验 D：二元 Verifier 边际信息增益与最优计算预算分析。"""
    n_values = list(range(1, 65))
    specs = [
        {"name": "high_precision", "p": 0.25, "tpr": 0.85, "fpr": 0.03},
        {"name": "balanced", "p": 0.25, "tpr": 0.85, "fpr": 0.10},
        {"name": "noisy", "p": 0.25, "tpr": 0.70, "fpr": 0.20},
    ]
    out = {}
    for cfg in specs:
        spec = BinaryVerifierSpec(p=cfg["p"], tpr=cfg["tpr"], fpr=cfg["fpr"])
        n_arr = np.asarray(n_values)
        acc = np.asarray(binary_bon_accuracy(spec, n_arr), dtype=float)
        mg = np.asarray(binary_marginal_gain(spec, n_arr), dtype=float)
        out[cfg["name"]] = {
            "params": cfg,
            "n_values": n_values,
            "accuracy": acc.tolist(),
            "marginal_gain": mg.tolist(),
            "asymptotic_ppv": float(binary_asymptotic_ceiling(spec)),
            "budget_eps_0.02": int(binary_optimal_budget(spec, eps=0.02)),
            "budget_eps_0.005": int(binary_optimal_budget(spec, eps=0.005)),
            "budget_cost_0.005": int(binary_optimal_budget(spec, cost_per_sample=0.005)),
            "budget_cost_0.001": int(binary_optimal_budget(spec, cost_per_sample=0.001)),
        }
    return out


def export_csv_summaries(results_dir: Path, exp_b: dict[str, Any], exp_c: dict[str, Any]) -> None:
    """导出参数扫描汇总 CSV 与算法对比汇总 CSV 便于审阅。"""
    sweep_csv = results_dir / "param_sweep_summary.csv"
    with sweep_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p", "delta_mu", "kappa", "regime", "will_collapse",
            "n_star_exact", "max_accuracy_exact",
            "n_star_saddle", "n_star_tail", "acc_at_saddle",
            "acc_N1", "acc_N64", "acc_N1024", "acc_N4096",
        ])
        for r in exp_b["sweep_records"]:
            probes = dict(zip(r["probe_ns"], r["probe_accuracies"]))
            writer.writerow([
                r["p"], r["delta_mu"], r["kappa"], r["regime"], r["will_collapse"],
                r["n_star_exact"], r["max_accuracy_exact"],
                r["n_star_saddle"], r["n_star_tail"], r["acc_at_saddle"],
                probes.get(1), probes.get(64), probes.get(1024), probes.get(4096),
            ])

    algo_csv = results_dir / "algorithm_comparison_summary.csv"
    with algo_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "algorithm", "N", "accuracy", "se"])
        for _, sc in exp_c["scenarios"].items():
            eps = sc["epsilon"]
            res = sc["results"]
            # 兼容两种常见嵌套结构
            algos_dict = res.get("algorithms", res)
            n_vals = res.get("n_values", sc["n_values"])
            for algo_name, metrics in algos_dict.items():
                if not isinstance(metrics, dict) or "accuracy" not in metrics:
                    continue
                accs = metrics["accuracy"]
                ses = metrics.get("se", [None] * len(accs))
                for n_val, acc, se in zip(n_vals, accs, ses):
                    writer.writerow([eps, algo_name, n_val, acc, se])


def main() -> None:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("运行实验 A：精确顺序统计量积分 vs 100,000 次蒙特卡洛 Scaling 曲线...")
    exp_a = run_experiment_a(n_trials_mc=100_000, seed=42)

    print("运行实验 B：多维参数扫描与 (delta_mu, kappa) 相变图构建...")
    exp_b = run_experiment_b()

    print("运行实验 C：强 Reward Hacking 混合尾部下 5 大算法对比...")
    exp_c = run_experiment_c(n_trials=30_000, seed=1024)

    print("运行实验 D：二元 Verifier 边际增益与预算截断扫描...")
    exp_d = run_experiment_d()

    elapsed = time.perf_counter() - t0
    payload = {
        "metadata": {
            "elapsed_seconds": elapsed,
        },
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "experiment_c": exp_c,
        "experiment_d": exp_d,
    }

    out_path = results_dir / "experiment_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, ensure_ascii=False, indent=2)

    export_csv_summaries(results_dir, exp_b, exp_c)
    print(f"全部实验已完成并保存至: {out_path} (耗时 {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
