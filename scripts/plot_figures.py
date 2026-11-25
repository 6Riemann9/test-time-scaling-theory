"""出版级 300 DPI 英文学术图表绘制脚本。

从 `results/experiment_results.json` 与 `results/theorem_verification.json` 读取真实实验数据，
生成 3 张高分辨率英文图表：
1. `results/figures/scaling_collapse_curves.png`
2. `results/figures/optimal_n_phase_diagram.png`
3. `results/figures/calibrated_vs_bon.png`

运行方式：
    python scripts/plot_figures.py
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

# 统一出版级排版参数
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _style_axis(ax: plt.Axes) -> None:
    """设置清爽网格线与轴样式。"""
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45, color="#B0B8C4")
    ax.set_axisbelow(True)


def plot_scaling_collapse_curves(exp_a: dict[str, Any], out_path: Path) -> None:
    """绘制图 1：异方差与 Reward Hacking 污染下的 BoN Scaling 曲线与 100k MC 叠图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.14, bottom=0.15, top=0.86, left=0.07, right=0.98)

    # Panel A: 异方差方差比 kappa = sigma0 / sigma1
    ax = axes[0]
    _style_axis(ax)
    het_specs = [
        ("kappa_0.75", r"$\kappa = 0.75$ ($\sigma_0 < \sigma_1$, Safe)", "#0072B2", "o"),
        ("kappa_1.00", r"$\kappa = 1.00$ (Critical $\kappa_c=1$)", "#009E73", "s"),
        ("kappa_1.35", r"$\kappa = 1.35$ (Mild Collapse)", "#E69F00", "^"),
        ("kappa_1.70", r"$\kappa = 1.70$ (Severe Collapse)", "#D55E00", "D"),
    ]
    for key, label, color, marker in het_specs:
        d = exp_a[key]
        n_dense = np.asarray(d["n_dense"], dtype=float)
        ex_dense = np.asarray(d["exact_dense"], dtype=float)
        n_mc = np.asarray(d["n_mc"], dtype=float)
        mc_acc = np.asarray(d["mc"]["accuracy"], dtype=float)
        mc_err = 1.96 * np.asarray(d["mc"]["se"], dtype=float)

        ax.plot(n_dense, ex_dense, color=color, linewidth=2.1, label=label, zorder=3)
        ax.errorbar(
            n_mc, mc_acc, yerr=mc_err, fmt=marker, color=color,
            markersize=4.5, elinewidth=1.0, capsize=2.2, alpha=0.88, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5,
        )

        # 标注闭式最优预算 N*
        cf = d["closed_form_saddle"]
        if cf.get("will_collapse") and cf.get("n_star") is not None:
            n_star = int(cf["n_star"])
            if 1 <= n_star <= 10000:
                y_star = float(np.interp(np.log10(n_star), np.log10(n_dense), ex_dense))
                ax.scatter([n_star], [y_star], marker="*", s=160, color=color,
                           edgecolor="black", linewidth=0.8, zorder=6)
                ax.axvline(n_star, color=color, linestyle=":", linewidth=1.1, alpha=0.65, zorder=2)
                offset_x = 1.25 if key == "kappa_1.70" else 1.15
                offset_y = -0.055 if key == "kappa_1.70" else 0.045
                ax.annotate(
                    f"$N^*={n_star}$",
                    xy=(n_star, y_star),
                    xytext=(n_star * offset_x, np.clip(y_star + offset_y, 0.08, 0.94)),
                    fontsize=8.5, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.88, lw=0.7),
                    zorder=7,
                )

    ax.axhline(0.25, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, label="Baseline $p = 0.25$")
    ax.axvspan(180, 10500, color="#FADBD8", alpha=0.22, zorder=1)
    ax.text(1400, 0.11, "Tail-Dominated\nCollapse Regime", fontsize=8.5, color="#922B21",
            ha="center", style="italic")

    ax.set_xscale("log")
    ax.set_xlim(0.9, 11000)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Sampling Budget $N$ (log scale)")
    ax.set_ylabel("Best-of-$N$ Pass@1 Accuracy")
    ax.set_title("(a) Heteroscedastic Score Variance ($\\kappa = \\sigma_0 / \\sigma_1$)")
    ax.legend(
        loc="center right", bbox_to_anchor=(0.985, 0.72),
        frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D5D8DC",
    )

    # Panel B: Reward Hacking 混合重尾污染 epsilon
    ax = axes[1]
    _style_axis(ax)
    hack_specs = [
        ("kappa_1.00", r"Clean ($\epsilon = 0.00, \kappa = 1.0$)", "#009E73", "s"),
        ("hack_eps_0.05", r"Reward Hacking ($\epsilon = 0.05$)", "#CC79A7", "^"),
        ("hack_eps_0.12", r"Reward Hacking ($\epsilon = 0.12$)", "#882255", "D"),
    ]
    for key, label, color, marker in hack_specs:
        d = exp_a[key]
        n_dense = np.asarray(d["n_dense"], dtype=float)
        ex_dense = np.asarray(d["exact_dense"], dtype=float)
        n_mc = np.asarray(d["n_mc"], dtype=float)
        mc_acc = np.asarray(d["mc"]["accuracy"], dtype=float)
        mc_err = 1.96 * np.asarray(d["mc"]["se"], dtype=float)

        ax.plot(n_dense, ex_dense, color=color, linewidth=2.1, label=label, zorder=3)
        ax.errorbar(
            n_mc, mc_acc, yerr=mc_err, fmt=marker, color=color,
            markersize=4.5, elinewidth=1.0, capsize=2.2, alpha=0.88, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5,
        )

        num_opt = d["numerical_optimal"]
        if key != "kappa_1.00":
            n_star = int(num_opt["n_star"])
            y_star = float(num_opt["max_accuracy"])
            ax.scatter([n_star], [y_star], marker="*", s=160, color=color,
                       edgecolor="black", linewidth=0.8, zorder=6)
            ax.axvline(n_star, color=color, linestyle=":", linewidth=1.1, alpha=0.65, zorder=2)
            y_text = y_star + 0.065 if key == "hack_eps_0.05" else y_star - 0.19
            x_text = n_star * 2.2 if key == "hack_eps_0.05" else n_star * 2.6
            ax.annotate(
                f"$N^*_{{\\mathrm{{exact}}}}={n_star}$",
                xy=(n_star, y_star),
                xytext=(x_text, np.clip(y_text, 0.10, 0.94)),
                fontsize=8.5, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.88, lw=0.7),
                zorder=7,
            )

    ax.axhline(0.25, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, label="Baseline $p = 0.25$")
    ax.text(
        1800, 0.72,
        "Solid lines: Exact Order-Stat Integral\nMarkers: 100k Monte Carlo (95% CI)",
        fontsize=8.2, color="#2C3E50", ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9F9", ec="#BDC3C7", lw=0.8),
    )

    ax.set_xscale("log")
    ax.set_xlim(0.9, 11000)
    ax.set_xlabel("Sampling Budget $N$ (log scale)")
    ax.set_title("(b) Reward Hacking Heavy-Tail Contamination ($\\epsilon$)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC")

    fig.suptitle(
        "Extreme-Value Phase Transition and Scaling Collapse in Best-of-$N$ Inference ($p=0.25, \\Delta\\mu=1.2$)",
        fontsize=12.5, fontweight="bold", y=0.96,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_optimal_n_phase_diagram(exp_b: dict[str, Any], out_path: Path) -> None:
    r"""绘制图 2：$(\Delta\mu, \sigma_0/\sigma_1)$ 相变热图与闭式解 vs 数值解散点一致性检验。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))
    fig.subplots_adjust(wspace=0.26, bottom=0.14, top=0.85, left=0.07, right=0.96)

    # Panel A: 二维相图
    ax = axes[0]
    hm = exp_b["heatmap_2d"]
    dmu_grid = np.asarray(hm["dmu_grid"], dtype=float)
    kappa_grid = np.asarray(hm["kappa_grid"], dtype=float)
    log_n = np.asarray(hm["log10_n_saddle"], dtype=float)
    # 屏蔽 NaN 区域以便用底色渲染无坍塌相
    masked_log_n = np.ma.masked_invalid(log_n)

    ax.set_facecolor("#EBF5FB")
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#EBF5FB")

    mesh = ax.pcolormesh(
        dmu_grid, kappa_grid, masked_log_n,
        cmap=cmap, shading="auto", vmin=0.0, vmax=5.0,
    )
    # 添加等高线（仅在坍塌区 kappa > 1.02），手动错开标签位置防止重叠
    valid_mask = kappa_grid > 1.02
    if np.any(valid_mask):
        cs = ax.contour(
            dmu_grid, kappa_grid[valid_mask], log_n[valid_mask, :],
            levels=[1.0, 2.0, 3.0, 4.0], colors="white", linewidths=0.95, linestyles="--", alpha=0.88,
        )
        ax.clabel(
            cs,
            fmt=lambda v: f"$10^{{{int(v)}}}$",
            fontsize=8.2,
            colors="white",
            manual=[(0.72, 1.33), (1.12, 1.24), (1.58, 1.24), (1.96, 1.23)],
        )

    # 相界 kappa_c = 1.0
    ax.axhline(1.0, color="#C0392B", linestyle="--", linewidth=2.2, zorder=5)
    ax.text(
        1.30, 1.025, r"Phase Boundary $\kappa_c = \sigma_0 / \sigma_1 = 1.0$",
        color="#922B21", fontsize=9.2, fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.9, lw=0.8),
        zorder=6,
    )
    ax.text(
        1.30, 0.78, "Non-Collapsing Phase\n($\\kappa \\leq 1.0,\\ N^* = \\infty$)",
        color="#1B4F72", fontsize=9.5, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2E86C1", alpha=0.85, lw=0.8),
    )
    ax.text(
        0.72, 1.92, "Collapsing Phase\n($\\kappa > 1.0,\\ N^* < \\infty$)",
        color="white", fontsize=9.2, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1C2833", ec="white", alpha=0.55, lw=0.7),
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.025, extend="max")
    cbar.set_label(r"Closed-Form Optimal Budget $\log_{10} N^*$ (clipped at 5)", fontsize=9.2)
    ax.set_xlabel(r"Mean Separation $\Delta\mu = (\mu_1 - \mu_0) / \sigma_1$")
    ax.set_ylabel(r"Noise Std Ratio $\kappa = \sigma_0 / \sigma_1$")
    ax.set_title("(a) Optimal Budget Phase Diagram ($p = 0.25$)")

    # Panel B: 闭式解 vs 精确数值解一致性检验
    ax = axes[1]
    _style_axis(ax)

    recs = [
        r for r in exp_b["sweep_records"]
        if r["will_collapse"] and r["n_star_exact"] is not None and 2 <= r["n_star_exact"] < 100_000
    ]
    n_exact = np.asarray([r["n_star_exact"] for r in recs], dtype=float)
    n_saddle = np.asarray([r["n_star_saddle_float"] for r in recs], dtype=float)
    n_tail = np.asarray([r["n_star_tail_float"] for r in recs], dtype=float)

    lims = [1.0, max(np.max(n_exact), np.max(n_saddle), np.max(n_tail)) * 1.4]
    x_ref = np.geomspace(lims[0], lims[1], 100)
    ax.plot(x_ref, x_ref, color="#2C3E50", linestyle="-", linewidth=1.5, label="Exact Identity ($y = x$)", zorder=2)
    ax.fill_between(x_ref, x_ref * 0.7, x_ref * 1.3, color="#D5D8DC", alpha=0.35, label=r"$\pm 30\%$ Band", zorder=1)

    ax.scatter(
        n_exact, n_tail, s=34, marker="o", color="#E69F00", alpha=0.70,
        edgecolor="white", linewidth=0.5, label="Zero-Order Tail Quantile", zorder=3,
    )
    ax.scatter(
        n_exact, n_saddle, s=42, marker="^", color="#0072B2", alpha=0.88,
        edgecolor="white", linewidth=0.5, label="First-Order Saddle-Refined", zorder=4,
    )

    mre_saddle = float(np.median(np.abs(n_saddle - n_exact) / n_exact)) * 100.0
    mre_tail = float(np.median(np.abs(n_tail - n_exact) / n_exact)) * 100.0
    r2_log = float(np.corrcoef(np.log10(n_exact), np.log10(n_saddle))[0, 1] ** 2)

    stats_text = (
        f"Sweep Points: $n = {len(recs)}$\n"
        f"Saddle-Refined MRE: {mre_saddle:.1f}%\n"
        f"Tail-Quantile MRE: {mre_tail:.1f}%\n"
        f"Log-Scale $R^2$: {r2_log:.4f}"
    )
    ax.text(
        0.96, 0.06, stats_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#AEB6BF", alpha=0.92, lw=0.8),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Exact Numerical Optimal $N^*_{\mathrm{exact}}$")
    ax.set_ylabel(r"Closed-Form Predicted $N^*_{\mathrm{closed}}$")
    ax.set_title("(b) Closed-Form vs. Exact Numerical $N^*$")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC")

    fig.suptitle(
        "Analytical Phase Boundary and Closed-Form Optimal Budget Verification",
        fontsize=12.5, fontweight="bold", y=0.96,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibrated_vs_bon(exp_c: dict[str, Any], out_path: Path) -> None:
    """绘制图 3：强异方差与 Reward Hacking 混合尾部下的五大算法 Scaling 曲线对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=True)
    fig.subplots_adjust(wspace=0.14, bottom=0.13, top=0.76, left=0.07, right=0.98)

    panel_keys = [
        ("eps_0.02", r"(a) Moderate Hacking ($\epsilon = 0.02,\ \kappa = 1.5$)"),
        ("eps_0.15", r"(b) Strong Reward Hacking ($\epsilon = 0.15,\ \kappa = 1.5$)"),
    ]
    algo_styles = [
        ("Standard-BoN", "#D55E00", "-", "D"),
        ("Optimal-N*", "#E69F00", "--", "^"),
        ("Majority-Voting", "#7F8C8D", "-.", "s"),
        ("Calibrated-Weighted-BoN", "#0072B2", "-", "o"),
        ("Two-Stage-Verified-Majority", "#009E73", "-", "*"),
    ]

    for ax, (sc_key, title) in zip(axes, panel_keys):
        _style_axis(ax)
        sc = exp_c["scenarios"][sc_key]
        res = sc["results"]
        n_vals = np.asarray(res.get("n_values", sc["n_values"]), dtype=float)
        algos_dict = res.get("algorithms", res)

        for algo_name, color, ls, marker in algo_styles:
            if algo_name not in algos_dict:
                continue
            metrics = algos_dict[algo_name]
            acc = np.asarray(metrics["accuracy"], dtype=float)
            se = np.asarray(metrics["se"], dtype=float)
            ms = 6.5 if marker == "*" else 4.5

            ax.plot(
                n_vals, acc, color=color, linestyle=ls, marker=marker,
                markersize=ms, linewidth=2.0, label=algo_name, zorder=4 if "Two-Stage" in algo_name else 3,
                markeredgecolor="white", markeredgewidth=0.45,
            )
            ax.fill_between(
                n_vals, np.clip(acc - 1.96 * se, 0.0, 1.0), np.clip(acc + 1.96 * se, 0.0, 1.0),
                color=color, alpha=0.14, zorder=2,
            )

        ax.axhline(0.25, color="#555555", linestyle=":", linewidth=1.1, alpha=0.75, label="Single-Sample $p=0.25$")
        ax.set_xscale("log")
        ax.set_xlim(0.9, 1150)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Sampling Budget $N$ (log scale)")
        ax.set_title(title)

    axes[0].set_ylabel("Selected Answer Accuracy (5 Decoys)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.525, 0.915),
        ncol=3, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
        fontsize=9.2, columnspacing=1.8, handlelength=2.6,
    )

    fig.suptitle(
        "Eliminating Test-Time Scaling Collapse via Posterior Calibration & Two-Stage Majority Voting",
        fontsize=12.4, fontweight="bold", y=0.975,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    exp_path = RESULTS_DIR / "experiment_results.json"
    if not exp_path.exists():
        raise FileNotFoundError(f"未找到实验数据文件: {exp_path}，请先运行 scripts/run_experiments.py")

    with exp_path.open("r", encoding="utf-8") as f:
        exp_data = json.load(f)

    fig1 = FIGURES_DIR / "scaling_collapse_curves.png"
    fig2 = FIGURES_DIR / "optimal_n_phase_diagram.png"
    fig3 = FIGURES_DIR / "calibrated_vs_bon.png"

    print(f"正在绘制 {fig1} ...")
    plot_scaling_collapse_curves(exp_data["experiment_a"], fig1)

    print(f"正在绘制 {fig2} ...")
    plot_optimal_n_phase_diagram(exp_data["experiment_b"], fig2)

    print(f"正在绘制 {fig3} ...")
    plot_calibrated_vs_bon(exp_data["experiment_c"], fig3)

    print("三张 300 DPI 学术图表已生成完毕。")


if __name__ == "__main__":
    main()
