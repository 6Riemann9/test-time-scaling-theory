"""V6 核心理论深度优化数值与真实数据交叉验证脚本。

生成 results/v6_theory_verification.json，覆盖：
1. 定理 5A-III（Part 1）：S-Lemma 凸化帕累托路径、Fisher/Safe-Fisher/GenEig 对比与退化边界；
2. 定理 5A-III（Part 2）：不可逆相 Bayes QDA K 维天花板递增性及 N^{-2/K} MC 精确幂率匹配；
3. 定理 6C（V6）：零对数间隙 N-无关 Minimax 上下界与大 M 小 N 下对截断策略的实际胜率增益；
4. 定理 7C：同方差单调 Verifier 含边界原子时 Z-空间自诱导 kappa_Z > 1、尾极限与诱导坍塌；
5. 若本地缓存 data/ 存在，附带 RewardBench-2 (hetero6) Safe-Fisher/QDA 5-Fold 对比与 AceMath RM-B Z-空间坍塌实测。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tts_theory import (
    GaussianVerifierSpec,
    GeneralGaussianMultiVerifierSpec,
    bayes_qda_multi_verifier_analysis,
    calibration_minimax_scaling,
    diagnose_zscore_induced_collapse,
    exact_calibrated_bon_accuracy,
    exact_pool_bon_accuracy,
    optimal_linear_ensemble_criterion,
    pareto_linear_ensemble_frontier,
    simulate_finite_calibration_scaling,
    simulate_multi_verifier_bon,
    simulate_zscore_induced_collapse,
    tight_joint_minimax_scaling,
)


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def verify_thm5a3_pareto_and_qda() -> dict:
    """验证定理 5A-III（Part 1 凸化帕累托 & Part 2 Bayes QDA 维度诅咒）。"""
    rng = np.random.default_rng(101)
    q_rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    cov1 = q_rot @ np.eye(3) @ q_rot.T
    cov0 = q_rot @ np.diag([0.45, 2.2, 3.0]) @ q_rot.T
    mu1 = q_rot @ np.array([0.22, 1.4, 1.1])
    mu0 = np.zeros(3)
    spec = GeneralGaussianMultiVerifierSpec(p=0.25, mu1=mu1, cov1=cov1, mu0=mu0, cov0=cov0)

    pareto = pareto_linear_ensemble_frontier(spec, kappa_max=1.0, n_path_points=30)
    opt_lin = optimal_linear_ensemble_criterion(spec)
    w_geneig = np.asarray(opt_lin["w_star"], dtype=np.float64)
    w_safe = np.asarray(pareto["w_safe"], dtype=np.float64)
    w_fisher = np.asarray(pareto["w_fisher"], dtype=np.float64)

    n_grid = [2, 8, 64, 512, 4096]
    mc_fisher = simulate_multi_verifier_bon(
        spec, n_grid, aggregation="linear", weights=w_fisher, n_trials=40_000, seed=11
    )
    mc_safe = simulate_multi_verifier_bon(
        spec, n_grid, aggregation="linear", weights=w_safe, n_trials=40_000, seed=12
    )
    mc_geneig = simulate_multi_verifier_bon(
        spec, n_grid, aggregation="linear", weights=w_geneig, n_trials=40_000, seed=13
    )

    # 退化正交边界检验
    q2, _ = np.linalg.qr(np.random.default_rng(303).normal(size=(2, 2)))
    spec_deg = GeneralGaussianMultiVerifierSpec(
        p=0.3,
        mu1=q2 @ np.array([0.0, 1.3]),
        cov1=q2 @ np.eye(2) @ q2.T,
        mu0=np.zeros(2),
        cov0=q2 @ np.diag([0.5, 2.4]) @ q2.T,
    )
    pareto_deg = pareto_linear_ensemble_frontier(spec_deg, kappa_max=1.0)

    # Part 2A: K=1, 2, 3 天花板单调递增检验
    qda_monotonicity: dict[str, dict] = {}
    for k_dim in (1, 2, 3):
        mu1_k = np.array([0.8, 0.5, 0.35][:k_dim], dtype=np.float64)
        spec_k = GeneralGaussianMultiVerifierSpec(
            p=0.25,
            mu1=mu1_k,
            cov1=np.eye(k_dim),
            mu0=np.zeros(k_dim),
            cov0=(1.6**2) * np.eye(k_dim),
        )
        qda_info = bayes_qda_multi_verifier_analysis(spec_k)
        qda_monotonicity[f"K={k_dim}"] = {
            "num_verifiers": k_dim,
            "bayes_ceiling_q": qda_info["bayes_ceiling_q"],
            "ell_peak_qda": qda_info["ell_peak_qda"],
            "asymptotic_coef_a_k": qda_info["asymptotic_coef_a_k"],
            "curse_rate_exponent": qda_info["curse_rate_exponent"],
        }

    # Part 2B: 大 A_K 规格下验证 K=1 (精确数值积分 N^{-2}) 与 K=2 (60k MC N^{-1})
    qda_rate_checks: dict[str, dict] = {}
    for k_dim, n_list in [(1, [16, 64, 256, 1024]), (2, [32, 64, 128])]:
        mu1_r = np.array([1.2, 1.0][:k_dim], dtype=np.float64)
        spec_r = GeneralGaussianMultiVerifierSpec(
            p=0.20,
            mu1=mu1_r,
            cov1=np.eye(k_dim),
            mu0=np.zeros(k_dim),
            cov0=(2.2**2) * np.eye(k_dim),
        )
        q_info = bayes_qda_multi_verifier_analysis(spec_r)
        q_star = float(q_info["bayes_ceiling_q"])
        a_k = float(q_info["asymptotic_coef_a_k"])
        exp_k = float(q_info["curse_rate_exponent"])
        reg_th = np.array([a_k * (float(nv) ** (-exp_k)) for nv in n_list])
        if k_dim == 1:
            g1 = GaussianVerifierSpec(p=0.20, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=2.2)
            exact_acc = np.array(
                [exact_calibrated_bon_accuracy(g1, int(nv), n_grid=12_000) for nv in n_list],
                dtype=np.float64,
            )
            reg_obs = q_star - exact_acc
            eval_mode = "exact_numerical_integral"
        else:
            mc_r = simulate_multi_verifier_bon(
                spec_r, n_list, aggregation="qda", n_trials=60_000, seed=50 + k_dim, batch_size=3_000
            )
            exact_acc = mc_r["accuracy"]
            reg_obs = q_star - exact_acc
            eval_mode = "monte_carlo_60k"
        qda_rate_checks[f"K={k_dim}"] = {
            "eval_mode": eval_mode,
            "bayes_ceiling_q": q_star,
            "c_k": q_info["c_k"],
            "asymptotic_coef_a_k": a_k,
            "curse_rate_exponent": exp_k,
            "n_values": n_list,
            "observed_accuracy": exact_acc,
            "observed_regret": reg_obs,
            "theory_regret": reg_th,
            "regret_ratio_obs_over_theory": reg_obs / reg_th,
        }

    return {
        "pareto_interior": {
            "kappa_fisher": pareto["kappa_fisher"],
            "kappa_safe": pareto["kappa_safe"],
            "kappa_geneig": opt_lin["kappa_linear_min"],
            "snr2_fisher": pareto["snr2_fisher"],
            "snr2_safe": pareto["snr2_safe"],
            "tau_star": pareto["tau_star"],
            "snr2_deriv_nonpositive": pareto["snr2_deriv_nonpositive"],
            "ratio_deriv_nonpositive": pareto["ratio_deriv_nonpositive"],
            "n_values": n_grid,
            "fisher_mc_acc": mc_fisher["accuracy"],
            "safe_fisher_mc_acc": mc_safe["accuracy"],
            "geneig_mc_acc": mc_geneig["accuracy"],
        },
        "degenerate_orthogonal_boundary": {
            "regime": pareto_deg["regime"],
            "tau_star": pareto_deg["tau_star"],
            "kappa_safe": pareto_deg["kappa_safe"],
            "snr2_safe": pareto_deg["snr2_safe"],
        },
        "qda_ceiling_monotonicity": qda_monotonicity,
        "qda_curse_rate_verification": qda_rate_checks,
    }


def verify_thm6c_zero_gap_minimax() -> dict:
    """验证定理 6C（V6）：零对数间隙全平面 Minimax 上界与大 M 小 N 下实证对比。"""
    grid_cases = []
    for m_val, n_val in [(1_000, 4), (10_000, 8), (100_000, 8), (100_000, 512)]:
        res = tight_joint_minimax_scaling(
            m_total=m_val, n_budget=n_val, beta=2.0, curvature_c=0.1, f_min=0.25, eta_radius=1.5
        )
        grid_cases.append(res)

    spec = GaussianVerifierSpec(p=0.30, mu1=0.6, sigma1=1.0, mu0=0.0, sigma0=1.8)
    k_opt_50k = int(calibration_minimax_scaling(50_000, beta=2.0)["k_opt_theory"])
    n_grid = [2, 4, 8, 16, 32]
    sim_ag = simulate_finite_calibration_scaling(
        spec, m_cal=50_000, n_bins=k_opt_50k, n_values=n_grid, methods=("kl_lcb",), n_repeats=150, delta=0.05, seed=91
    )
    sim_tr = simulate_finite_calibration_scaling(
        spec, m_cal=50_000, n_bins=2, n_values=n_grid, methods=("kl_lcb",), n_repeats=150, delta=0.05, seed=91
    )
    return {
        "explicit_bounds": grid_cases,
        "simulation_m50k_small_n": {
            "n_values": n_grid,
            "k_n_agnostic": k_opt_50k,
            "k_truncated_baseline": 2,
            "n_agnostic_kl_lcb_acc": sim_ag["methods"]["kl_lcb"]["accuracy"],
            "truncated_kl_lcb_acc": sim_tr["methods"]["kl_lcb"]["accuracy"],
        },
    }


def verify_thm7c_zscore_induced_collapse() -> dict:
    """验证定理 7C：同方差单调 Verifier 含边界原子时 Z-空间自诱导 kappa_Z > 1 与诱导坍塌。"""
    z_g = np.linspace(-3.5, 9.0, 1251)
    diag_no1 = diagnose_zscore_induced_collapse(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.50],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    diag_with1 = diagnose_zscore_induced_collapse(
        prompt_weights=[0.15, 0.05, 0.40, 0.40],
        prompt_p=[0.0, 1.0, 0.35, 0.80],
        mu1_base=2.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    n_grid = [1, 8, 32, 64, 128, 256]
    sim = simulate_zscore_induced_collapse(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.50],
        n_values=n_grid,
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        m_cal_prompts=2_000,
        n_cal_per_prompt=64,
        n_eval_prompts=2_000,
        n_bins=30,
        seed=42,
    )
    return {
        "analytic_no_trivial_pi1_zero": {
            "pi_0_unsolvable": diag_no1["pi_0_unsolvable"],
            "pi_1_trivial": diag_no1["pi_1_trivial"],
            "var_z_pos": diag_no1["var_z_pos"],
            "var_z_neg": diag_no1["var_z_neg"],
            "kappa_z": diag_no1["kappa_z"],
            "z_peak": diag_no1["z_peak"],
            "q_z_peak": diag_no1["q_z_peak"],
            "z_tail_value": diag_no1["z_tail_value"],
            "z_tail_posterior_limit": diag_no1["z_tail_posterior_limit"],
            "raw_bon_asymptotic_acc": diag_no1["raw_bon_asymptotic_acc"],
            "oracle_z_cal_asymptotic_acc": diag_no1["oracle_z_cal_asymptotic_acc"],
            "z_induces_collapse": diag_no1["z_induces_collapse"],
        },
        "analytic_with_trivial_pi1_005": {
            "pi_0_unsolvable": diag_with1["pi_0_unsolvable"],
            "pi_1_trivial": diag_with1["pi_1_trivial"],
            "kappa_z": diag_with1["kappa_z"],
            "z_tail_value": diag_with1["z_tail_value"],
            "z_tail_posterior_limit": diag_with1["z_tail_posterior_limit"],
            "z_induces_collapse": diag_with1["z_induces_collapse"],
        },
        "simulation": sim,
    }


def verify_real_data_checks(root: Path) -> dict:
    """若本地 data/ 缓存存在，附带 RB2 hetero6 5-Fold 与 AceMath RM-B Z-空间实证诊断。"""
    out: dict[str, Any] = {}

    rb2_base = root / "data/stage2/reward_bench_2/eval-set-scores"
    rb2_models = [
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B.json",
        "Qwen/WorldPM-72B.json",
        "internlm/internlm2-20b-reward.json",
        "Nexusflow/Starling-RM-34B.json",
        "Ray2333/GRM-gemma2-2B-rewardmodel-ft.json",
        "hendrydong/Mistral-RM-for-RAFT-GSHF-v0.json",
    ]
    if all((rb2_base / m).exists() for m in rb2_models):
        dfs = {}
        for m in rb2_models:
            obj = json.loads((rb2_base / m).read_text(encoding="utf-8"))
            dfs[m] = {pid: sc for pid, sc in zip(obj["id"], obj["scores"]) if len(sc) == 4}
        common_ids = sorted(set.intersection(*(set(v.keys()) for v in dfs.values())))
        X_list = [np.column_stack([dfs[m][pid] for m in rb2_models]) for pid in common_ids]
        y_list = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in common_ids]
        P = len(X_list)
        idx = np.arange(P)
        np.random.default_rng(42).shuffle(idx)
        folds = np.array_split(idx, 5)
        accs = {k: [] for k in ["Best1(Skywork)", "Fisher(tau=-1)", "Safe-Fisher(kappa<=1)", "QDA", "GenEig"]}
        tau_list = []
        for f in range(5):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(5) if j != f])
            X_tr = np.vstack([X_list[i] for i in tr])
            y_tr = np.concatenate([y_list[i] for i in tr])
            m_tr, s_tr = X_tr.mean(0), X_tr.std(0) + 1e-8
            Xs = (X_tr - m_tr) / s_tr
            X1, X0 = Xs[y_tr == 1], Xs[y_tr == 0]
            spec_tr = GeneralGaussianMultiVerifierSpec(
                p=0.25,
                mu1=X1.mean(0),
                cov1=np.cov(X1, rowvar=False) + 1e-5 * np.eye(6),
                mu0=X0.mean(0),
                cov0=np.cov(X0, rowvar=False) + 1e-5 * np.eye(6),
            )
            par = pareto_linear_ensemble_frontier(spec_tr, kappa_max=1.0)
            qda = bayes_qda_multi_verifier_analysis(spec_tr)
            opt = optimal_linear_ensemble_criterion(spec_tr)
            tau_list.append(par["tau_star"])
            wf = np.asarray(par["w_fisher"])
            ws = np.asarray(par["w_safe"])
            wg = np.asarray(opt["w_star"])
            aq = np.asarray(qda["a_qda"])
            bq = np.asarray(qda["b_qda"])
            for i in te:
                Xi = (X_list[i] - m_tr) / s_tr
                yi = y_list[i]
                accs["Best1(Skywork)"].append(float(exact_pool_bon_accuracy(Xi[:, 0], yi, [4])[0]))
                accs["Fisher(tau=-1)"].append(float(exact_pool_bon_accuracy(Xi @ wf, yi, [4])[0]))
                accs["Safe-Fisher(kappa<=1)"].append(float(exact_pool_bon_accuracy(Xi @ ws, yi, [4])[0]))
                accs["GenEig"].append(float(exact_pool_bon_accuracy(Xi @ wg, yi, [4])[0]))
                qscore = np.einsum("ij,jk,ik->i", Xi, aq, Xi) + Xi @ bq
                accs["QDA"].append(float(exact_pool_bon_accuracy(qscore, yi, [4])[0]))
        out["rb2_hetero6_n4_5fold"] = {
            "n_problems": P,
            "mean_tau_star": float(np.mean(tau_list)),
            "accuracies": {k: float(np.mean(v)) for k, v in accs.items()},
        }

    acemath_pq = root / "data/acemath_rewardbench/default_test_0.parquet"
    if acemath_pq.exists():
        df = pd.read_parquet(acemath_pq)
        # 与 run_real_llm_experiments.py 对齐：后半 offset_b=8987 起完整 8,987 题为 AceMath RM-B
        sub = df.iloc[8987 : 8987 * 2]
        rm_b = np.vstack([np.asarray(x, dtype=float).ravel() for x in sub["rm_score"]])
        labels = np.vstack([np.asarray(x, dtype=float).ravel() for x in sub["score"]])
        p_prompt = labels.mean(axis=1)
        mu = rm_b.mean(axis=1, keepdims=True)
        sd = rm_b.std(axis=1, ddof=1, keepdims=True)
        z = np.where(sd > 1e-12, (rm_b - mu) / sd, 0.0)
        zf, yf, pf = z.ravel(), labels.ravel(), np.repeat(p_prompt, rm_b.shape[1])
        qs = np.quantile(zf, [0.0, 0.4, 0.8, 0.9, 0.95, 0.99, 1.0])
        prof = []
        for i in range(len(qs) - 1):
            m = (zf >= qs[i]) & (zf <= qs[i + 1] if i == len(qs) - 2 else zf < qs[i + 1])
            prof.append(
                {
                    "z_range": [float(qs[i]), float(qs[i + 1])],
                    "q_z": float(yf[m].mean()),
                    "frac_p0": float(np.mean(pf[m] == 0.0)),
                    "mean_p": float(pf[m].mean()),
                }
            )
        out["acemath_rm_b_zscore_tail"] = {
            "n_problems": int(len(p_prompt)),
            "frac_p0": float(np.mean(p_prompt == 0.0)),
            "var_z_pos": float(np.var(zf[yf == 1.0])),
            "var_z_neg": float(np.var(zf[yf == 0.0])),
            "kappa_z": float(np.std(zf[yf == 0.0]) / np.std(zf[yf == 1.0])),
            "quantile_profile": prof,
        }
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = {
        "thm5a3": verify_thm5a3_pareto_and_qda(),
        "thm6c_zero_gap": verify_thm6c_zero_gap_minimax(),
        "thm7c_zscore_collapse": verify_thm7c_zscore_induced_collapse(),
        "real_data": verify_real_data_checks(root),
    }
    out_path = root / "results/v6_theory_verification.json"
    out_path.write_text(json.dumps(_to_json_safe(res), indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)
    p = res["thm5a3"]["pareto_interior"]
    print(
        f"[5A-III Pareto] Fisher={p['fisher_mc_acc'][-1]:.4f} (kappa={p['kappa_fisher']:.3f}) | "
        f"Safe-Fisher={p['safe_fisher_mc_acc'][-1]:.4f} (kappa={p['kappa_safe']:.3f}) | "
        f"GenEig={p['geneig_mc_acc'][-1]:.4f}"
    )
    for k, v in res["thm5a3"]["qda_curse_rate_verification"].items():
        print(
            f"[5A-III QDA {k}] ceiling={v['bayes_ceiling_q']:.4f}, A_K={v['asymptotic_coef_a_k']:.4f}, "
            f"Obs/Theory={[round(float(x), 4) for x in v['regret_ratio_obs_over_theory']]}"
        )
    t6 = res["thm6c_zero_gap"]["simulation_m50k_small_n"]
    print(
        f"[6C Zero-Gap M=50k] N=8: N-Agnostic(K={t6['k_n_agnostic']})={t6['n_agnostic_kl_lcb_acc'][2]:.4f} "
        f"vs Truncated(K=2)={t6['truncated_kl_lcb_acc'][2]:.4f}"
    )
    t7a = res["thm7c_zscore_collapse"]["analytic_no_trivial_pi1_zero"]
    t7s = res["thm7c_zscore_collapse"]["simulation"]
    print(
        f"[7C Z-Collapse] kappa_Z={t7a['kappa_z']:.3f}, q_z_tail(z=9)={t7a['z_tail_value']:.6f}, "
        f"N=256 MC: Raw={t7s['standard_bon_mc'][-1]:.4f}, Pool={t7s['pooled_kl_lcb_mc'][-1]:.4f}, Within-Z={t7s['within_prompt_z_kl_lcb_mc'][-1]:.4f}"
    )
    if res["real_data"]:
        print("[Real Data Checks]", json.dumps(_to_json_safe(res["real_data"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
