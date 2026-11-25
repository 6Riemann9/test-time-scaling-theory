"""V2 原创扩展实验 300 DPI 英文出版级图表绘制脚本。

从 `results/v2_extension_results.json` 读取真实实验数据，生成 2 张学术图表：
1. `results/figures/ensemble_phase_reversal.png`
2. `results/figures/finite_calibration_secondary_collapse.png`

运行方式：
    python scripts/plot_v2_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.2,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45, color="#B0B8C4")
    ax.set_axisbelow(True)


def plot_ensemble_phase_reversal(exp_e: dict[str, Any], exp_f: dict[str, Any], out_path: Path) -> None:
    """绘制图 4：多裁判高斯相变逆转（左）与重尾 Student-t 下 Mean/Min 分离及同源击穿（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharey=True)
    fig.subplots_adjust(wspace=0.12, bottom=0.29, top=0.86, left=0.065, right=0.98)

    # Panel (a): 高斯对称相关 Min-Ensemble 相变逆转 (kappa=1.65, rho1=0.85, rho0=0.10 => K_c=4)
    ax = axes[0]
    _style_axis(ax)
    curves_e = exp_e["curves"]
    styles_e = [
        ("k1_single", r"Single Verifier ($K=1$, $\kappa=1.65$)", "#D55E00", "D", "-"),
        ("k2_min", r"Min-Ensemble ($K=2 < K_c=4$)", "#E69F00", "^", "-"),
        ("k4_min", r"Min-Ensemble ($K=4 = K_c$, Reversed)", "#009E73", "s", "-"),
        ("k6_min", r"Min-Ensemble ($K=6 > K_c$, Reversed)", "#0072B2", "o", "-"),
        ("k6_irrev", r"Irreversible ($\rho_0=0.60 \geq \rho_1/\kappa^2$, $K=6$)", "#7F8C8D", "v", "--"),
    ]
    for key, label, color, marker, ls in styles_e:
        d = curves_e[key]
        n_dense = np.asarray(d["n_dense"], dtype=float)
        ex_dense = np.asarray(d["exact_dense"], dtype=float)
        n_mc = np.asarray(d["n_mc"], dtype=float)
        mc_acc = np.asarray(d["mc"]["accuracy"], dtype=float)
        mc_err = 1.96 * np.asarray(d["mc"]["se"], dtype=float)

        ax.plot(n_dense, ex_dense, color=color, linestyle=ls, linewidth=2.0, label=label, zorder=3)
        ax.errorbar(
            n_mc, mc_acc, yerr=mc_err, fmt=marker, color=color,
            markersize=4.2, elinewidth=0.95, capsize=2.0, alpha=0.88, zorder=4,
            markeredgecolor="white", markeredgewidth=0.45,
        )

    ax.axhline(0.25, color="#555555", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.annotate(
        r"Phase Reversal across $K_c = 4$" + "\n" + r"($\kappa_{\mathrm{eff}}$ crosses $1.0$)",
        xy=(512, 0.98), xytext=(140, 0.77),
        fontsize=8.3, fontweight="bold", color="#00695C",
        arrowprops=dict(arrowstyle="->", color="#00695C", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#009E73", alpha=0.92, lw=0.8),
        zorder=6,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.9, 5200)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Sampling Budget $N$ (log scale)")
    ax.set_ylabel("Best-of-$N$ Pass@1 Accuracy")
    ax.set_title(r"(a) Correlated Gaussian Ensemble Reversal ($\kappa=1.65, \rho_1=0.85, \rho_0=0.10$)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): 重尾 Student-t 下 Mean vs Min 分离与同源漏洞击穿
    ax = axes[1]
    _style_axis(ax)
    styles_f = [
        ("min_k4_indep", r"Min-Ensemble ($K=4$, Indep., $\epsilon_s=0$)", "#0072B2", "o", "-"),
        ("het_min_1vuln", r"Min-Ensemble (3 Healthy + 1 Vuln.)", "#009E73", "s", "-."),
        ("mean_k4_indep", r"Mean-Ensemble ($K=4$, Indep.)", "#E69F00", "^", "-"),
        ("min_k4_shared_0.04", r"Min-Ensemble ($K=4$, Shared $\epsilon_s=0.04$)", "#882255", "D", "--"),
        ("single_k1", r"Single Verifier ($K=1$, $\alpha_0=1.8$)", "#D55E00", "X", ":"),
    ]
    for key, label, color, marker, ls in styles_f:
        d = exp_f[key]
        n_vals = np.asarray(d["n_values"], dtype=float)
        acc = np.asarray(d["accuracy"], dtype=float)
        err = 1.96 * np.asarray(d["se"], dtype=float)
        ax.errorbar(
            n_vals, acc, yerr=err, fmt=marker, linestyle=ls, linewidth=2.0, color=color,
            markersize=4.5, elinewidth=0.95, capsize=2.0, alpha=0.92, label=label, zorder=4,
            markeredgecolor="white", markeredgewidth=0.45,
        )

    ax.axhline(0.25, color="#555555", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.annotate(
        r"Shared Vulnerability ($\epsilon_s=0.04$)" + "\nbreaks Min-Ensemble!",
        xy=(1024, 0.59), xytext=(150, 0.36),
        fontsize=8.3, fontweight="bold", color="#882255",
        arrowprops=dict(arrowstyle="->", color="#882255", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#882255", alpha=0.92, lw=0.8),
        zorder=6,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.9, 4600)
    ax.set_xlabel("Sampling Budget $N$ (log scale)")
    ax.set_title(r"(b) Heavy-Tail Separation & Shared-Vulnerability Breakdown (100k MC)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Theorem 5: Multi-Verifier Phase Reversal, Mean-vs-Min Tail Separation, and Shared-Flaw Collapse",
        fontsize=12.4, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_finite_calibration_collapse(exp_g: dict[str, Any], out_path: Path) -> None:
    """绘制图 5：有限校准集二次坍塌（左）与 Minimax 最优分桶双对数遗憾收敛（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4))
    fig.subplots_adjust(wspace=0.23, bottom=0.29, top=0.86, left=0.07, right=0.97)

    # Panel (a): M=400 等宽 K=20 下二次坍塌与 LCB 消除
    ax = axes[0]
    _style_axis(ax)
    g1 = exp_g["g1_uniform_m400_k20"]
    n_vals = np.asarray(g1["n_values"], dtype=float)
    oracle = np.asarray(g1["oracle_bin_accuracy"], dtype=float)
    q_peak_true = float(exp_g["q_peak_true"])

    ax.axhline(q_peak_true, color="#2C3E50", linestyle=":", linewidth=1.2, alpha=0.8,
               label=rf"Continuous Peak $\sup_x q(x)={q_peak_true:.3f}$")
    ax.plot(n_vals, oracle, color="#1F2937", linestyle="--", linewidth=1.9,
            label="Oracle Bin Ordering", zorder=4)

    m_styles = [
        ("wilson_lcb", "Wilson-LCB (Eliminates Collapse)", "#0072B2", "o"),
        ("empirical_bernstein_lcb", "Empirical-Bernstein LCB", "#009E73", "s"),
        ("laplace", "Laplace Smoothed (s=2.0)", "#E69F00", "^"),
        ("mle", "Naive Empirical MLE (Secondary Collapse)", "#D55E00", "D"),
    ]
    for key, label, color, marker in m_styles:
        md = g1["methods"][key]
        acc = np.asarray(md["accuracy"], dtype=float)
        ci_lo = np.asarray(md["ci_low"], dtype=float)
        ci_hi = np.asarray(md["ci_high"], dtype=float)
        ax.plot(n_vals, acc, marker=marker, markersize=4.3, linewidth=2.0, color=color, label=label, zorder=3)
        ax.fill_between(n_vals, ci_lo, ci_hi, color=color, alpha=0.12, zorder=2)

    mle_acc = np.asarray(g1["methods"]["mle"]["accuracy"], dtype=float)
    peak_i = int(np.argmax(mle_acc))
    ax.axvspan(180, 9000, color="#FADBD8", alpha=0.18, zorder=1)
    ax.annotate(
        f"Secondary Collapse\n($N={int(n_vals[-1])}$ drops to {mle_acc[-1]:.3f})",
        xy=(n_vals[-1], mle_acc[-1]),
        xytext=(380, 0.315),
        fontsize=8.2, fontweight="bold", color="#A93226",
        arrowprops=dict(arrowstyle="->", color="#A93226", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D55E00", alpha=0.92, lw=0.8),
        zorder=6,
    )
    ax.scatter([n_vals[peak_i]], [mle_acc[peak_i]], marker="*", s=145, color="#D55E00", edgecolor="black", zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 9500)
    ax.set_ylim(0.22, 0.49)
    ax.set_xlabel("Inference Sampling Budget $N$ (log scale)")
    ax.set_ylabel("Expected Best-of-$N$ Accuracy")
    ax.set_title(r"(a) Finite Calibration Set ($M=400$, Uniform $K=20$ Bins)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): 双对数遗憾收敛图 (N=5000, M scan)
    ax = axes[1]
    _style_axis(ax)
    g2 = exp_g["g2_minimax_scaling"]
    recs = g2["records"]
    m_arr = np.array([r["m_cal"] for r in recs], dtype=float)
    r_opt_w = np.array([r["regret_opt_wilson"] for r in recs], dtype=float)
    r_coarse_w = np.array([r["regret_coarse6_wilson"] for r in recs], dtype=float)
    r_fix_w = np.array([r["regret_fixed80_wilson"] for r in recs], dtype=float)
    r_fix_m = np.array([r["regret_fixed80_mle"] for r in recs], dtype=float)

    ax.loglog(m_arr, r_fix_m, "D:", color="#D55E00", linewidth=1.8, markersize=4.8,
              label="Fixed Over-Fine $K=80$ + MLE")
    ax.loglog(m_arr, r_coarse_w, "s-.", color="#E69F00", linewidth=1.8, markersize=4.8,
              label="Fixed Coarse $K=6$ + Wilson-LCB")
    ax.loglog(m_arr, r_fix_w, "^--", color="#CC79A7", linewidth=1.8, markersize=4.8,
              label="Fixed $K=80$ + Wilson-LCB")
    ax.loglog(m_arr, r_opt_w, "o-", color="#0072B2", linewidth=2.2, markersize=5.2,
              label=r"Optimal $K^*(M) \propto (M/\ln M)^{2/5}$ + Wilson-LCB")

    # 理论斜率 -2/5 参考线
    ref_x = m_arr
    c_ref = r_opt_w[2] / ((m_arr[2] / np.log(m_arr[2])) ** (-0.4))
    ref_y = c_ref * ((ref_x / np.log(ref_x)) ** (-0.4)) * 0.72
    ax.loglog(ref_x, ref_y, "k--", linewidth=1.4, alpha=0.75,
              label=r"Minimax Theory Slope $(M/\ln M)^{-2/5}$")

    slope_w = float(g2["fitted_slope_wilson"])
    ax.text(
        0.04, 0.06,
        f"Log-Log Fitted Slope vs $(M/\\ln M)$:\n"
        f"Optimal-Bin Wilson-LCB: {slope_w:.3f}\n"
        f"Minimax Theory: -0.400",
        transform=ax.transAxes, fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8F9F9", ec="#BDC3C7", lw=0.85),
    )

    ax.set_xlabel("Offline Calibration Sample Size $M$ (log scale)")
    ax.set_ylabel(r"Asymptotic Regret $q^* - \mathbb{E}[P(N=5000 \mid \mathcal{D}_M)]$ (log scale)")
    ax.set_title(r"(b) Minimax Regret Scaling at Large Inference Budget ($N=5000$)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Theorem 6: Finite-Calibration Secondary Collapse and Minimax LCB Convergence",
        fontsize=12.4, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    v2_path = RESULTS_DIR / "v2_extension_results.json"
    if not v2_path.exists():
        raise FileNotFoundError(f"未找到 {v2_path}，请先运行 python scripts/run_v2_extensions.py")

    data = json.loads(v2_path.read_text(encoding="utf-8"))

    fig4_path = FIGURES_DIR / "ensemble_phase_reversal.png"
    print(f"正在绘制 {fig4_path} ...")
    plot_ensemble_phase_reversal(
        data["experiment_e_gaussian_ensemble"],
        data["experiment_f_heavy_tail_ensemble"],
        fig4_path,
    )

    fig5_path = FIGURES_DIR / "finite_calibration_secondary_collapse.png"
    print(f"正在绘制 {fig5_path} ...")
    plot_finite_calibration_collapse(data["experiment_g_finite_calibration"], fig5_path)

    print("V2 两张 300 DPI 学术图表已生成完毕。")


if __name__ == "__main__":
    main()
