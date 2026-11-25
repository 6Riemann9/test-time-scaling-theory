# Test-Time Scaling 相变理论、多裁判逆转与抗坍塌算法库 (`tts-theory`)

面向大语言模型（LLM）推理阶段的**测试时计算缩放（Test-Time Scaling, TTS）极限行为、极值相变机理、多噪声裁判一般融合逆转、有限校准集与有限算力联合标度律、以及题目异质性辛普森悖论**的严格概率论证明与高精度实证验证库。

> **核心结构与演化（V1 $\to$ V5）**：
> - **第一部分：预备理论与单裁判基础重构（定理 1–4）**：对单裁判 Test-Time Scaling 坍塌与校准理论做系统性统一概率推导（含 $C^0/C^1$ 分离 MLRP 充要判据、固定峰值高度机制下的五步 Gumbel $e^{-\gamma}$ 极限证明与附录 Padé 近似、$\text{PPV}=1$ 边界完备散度界、以及异质单错类条件权重大数定律；明确不与 2022–2026 先发单裁判文献抢占原创），并实现双精度 PIT 数值积分引擎与流式蒙特卡洛仿真；
> - **第二部分：三大核心原创理论支柱（定理 5–7，经 V5 审稿级数学严格化重构）**：
>   - **定理 5（一般异构多裁判集成相变与 Copula 统一重尾理论）**：推导一般非对称协方差矩阵 $(\Sigma_1, \Sigma_0)$ 下的广义特征值最优线性融合（含高错相关下**负权差值共模抑制**）、基于 Hashorva-Hüsler 正交象限 QP 的严格密度级 `Min-Ensemble` 激活集相变（精确常数 $C_c^{\text{pdf}} = \gamma_c C_c^{\text{sf}}$），以及**任意正定相关矩阵 Gaussian-Copula + Student-$t$** 统一剩余尾相依系数（一般正定 $\eta_c \in (0, 1]$ 含负相关超独立加速；非负相关经 Slepian 引理限定于 $[1/K, 1]$）；
>   - **定理 6（有限校准集 $M$ 与有限推理算力 $N$ 联合 Minimax 定理）**：导出含平局组精确条件闭式解，证得 $\beta$-光滑峰最适分桶律 $K^*(M) \asymp (M/\ln M)^{\frac{1}{2\beta+1}}$（二次峰为 **`1/5`**）；严格重述非渐近 **`KL-LCB`** 为**基于三事件并界的安全底线悲观封杀与高概率遗憾界**（取代“全桶完美排序”过强宣称）；建立全局唯一分离峰下的 $\Theta(N^{-\beta})$ 速率、含 $\delta = M^{-1}$ 失败项联合上界，并证明**全策略空间 $(M, N)$ 联合信息论 Minimax 下界** $\Omega\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$；
>   - **定理 7（题目异质性、辛普森伪单调遮蔽与题内标准化交叉权衡）**：分解全局池化后验导数恒等式揭示假单调遮蔽陷阱；严格刻画 $N=2$ 时 Z-Score 退化为三类**纯秩规则天花板** $\max(P_{\text{Raw}}(2), 2\bar{p}-P_{\text{Raw}}(2), \bar{p})$ 以及异质先验 $p(X)$ 诱导的混合矩残差边界；
> - **V3 文献排雷审计与真实大模型/RM 实证验证**：系统对比 12 篇截至 2026-11 的 arXiv / 顶会核心文献（[RELATED_WORK.md](RELATED_WORK.md)），并在 `nvidia/AceMath-RewardBench`（8,987 题 × 64 候选 = 575,168 条双 RM 打分）与 `HuggingFaceH4/MATH-500`（`Llama-3.2-1B` / `3B` × 256 候选 = 256,000 条 `RLHFlow-8B-PRM` 轨迹）上完成端到端真实数据验证。

---

## 1. 核心理论七大定理概览

完整严格证明见学术证明稿 [PROOF.md](PROOF.md)，解析实现分别位于 [theorems.py](src/tts_theory/theorems.py)、[ensemble_theory.py](src/tts_theory/ensemble_theory.py)、[calibration_theory.py](src/tts_theory/calibration_theory.py) 与 [heterogeneity_theory.py](src/tts_theory/heterogeneity_theory.py)：

| 层次 | 定理 | 数学命题与工程意义 | 核心公式 / 判据 | 对应实现 |
| :---: | :---: | :--- | :--- | :--- |
| **预备** | **定理 1** | **连续与含原子 BoN 精确顺序统计量积分** | $P_{\text{BoN}}(N) = \int_0^1 q\big(\bar{F}_{\text{mix}}^{-1}(-\operatorname{expm1}(\tfrac{\ln v}{N}))\big)\,dv$ | `exact_bon_accuracy`, `ContinuousVerifierSpec` |
| **预备** | **定理 2** | **$C^1/C^0$ MLRP 判据、固定峰值高度 $e^{-\gamma}$ 与附录 Padé** | $\kappa_c = 1$；固定峰高 $\lim N^*_{\text{exact}}/N^*_0 = e^{-\gamma}$ | `check_collapse_criterion`, `closed_form_optimal_n` |
| **预备** | **定理 3** | **二元缩放律、$\text{PPV}=1$ 边界分情况散度与闭式预算** | $P_{\text{bin}}(N)=\text{PPV}-C(1-q_1)^N$；$\beta=0$ 给 $\text{TV}$ 与反向 $\text{KL}$ | `BinaryVerifierSpec`, `binary_bon_divergence_bounds` |
| **预备** | **定理 4** | **校准后验 & 异质单错类约束有界加权多数决** | $\Delta_{\min}=p\mu_{w,+}-\sup_j(p_j\mu_{w,j})>0$ 致 $O(e^{-cN})$ | `weighted_majority_convergence_bound`, `TwoStageVerifiedMajority` |
| **原创 1** | **定理 5** | **广义特征值负权逆转、QP `Min` ($\gamma_c$ 因子) 与 Copula** | $\lambda_{\min}\le 1$；$C_c^{\text{pdf}}=\gamma_c C_c^{\text{sf}}$；$\eta_c \in (0, 1]$（非负 $[1/K,1]$） | `general_min_tail_asymptotic_constants`, `gaussian_copula_heavy_tail_criterion` |
| **原创 2** | **定理 6** | **联合 $(M,N)$ Minimax 上下界与 `KL-LCB` 悲观保底** | 下界 $\Omega((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}+N^{-\beta})$；三事件安全底线 | `calibration_minimax_scaling`, `joint_calibration_compute_scaling` |
| **原创 3** | **定理 7** | **辛普森伪单调遮蔽、$N=2$ 纯秩天花板与先验残差** | $q'_{\text{pool}}=\mathbb{E}[\partial_s q_X\mid S]+\operatorname{Cov}(Y,g_X\mid S)$ | `diagnose_simpson_confounding`, `diagnose_within_prompt_residual` |

---

## 2. 仓库结构

```text
test-time-scaling-theory/
├── LICENSE                                    # MIT 开源许可证
├── pyproject.toml                             # 项目构建与 pytest 配置 (v0.5.0)
├── PROOF.md                                   # 预备定理 1–4 与三大原创支柱 5–7 严格证明稿（V5）
├── RELATED_WORK.md                            # 12 篇 2022–2026 顶会/arXiv 文献排雷与原创边界审计
├── README.md                                  # 项目总览与复现指南
├── src/tts_theory/
│   ├── __init__.py                            # 统一公开 API 导出
│   ├── theorems.py                            # V1/V5 定理 1–4 分布规格、散度边界与异质权重接口
│   ├── ensemble_theory.py                     # V2/V4/V5 定理 5 一般 QP (含 C_pdf=gamma*C_sf) / Copula 重尾
│   ├── calibration_theory.py                  # V2/V4/V5 定理 6 条件公式、KL-LCB 悲观保底与联合 Minimax
│   ├── heterogeneity_theory.py                # V4/V5 定理 7 辛普森遮蔽、纯秩天花板与先验残差诊断
│   ├── real_eval.py                           # V3 真实大模型/RM 离线精确池计算与 LaTeX/SymPy 判分引擎
│   ├── integrals.py                           # 双精度 PIT 分段 Gauss-Legendre / quad 引擎
│   ├── simulator.py                           # 低内存流式分块向量化蒙特卡洛仿真器
│   └── algorithms.py                          # StandardBoN, OptimalNStarBoN, EmpiricalScoreCalibrator
│                                              # (支持 wilson/kl/bernstein LCB), CalibratedWeightedBoN, TwoStageVerifiedMajority
├── tests/
│   ├── test_theorems.py                       # 定理 1–4 数值精度与 100k–1.5M MC 交叉验证
│   ├── test_algorithms.py                     # 四大算法与经验校准器单元测试
│   ├── test_extensions.py                     # V2 定理 5 & 定理 6 单元测试
│   ├── test_real_eval.py                      # V3 真实池评估与等价判分单元测试
│   ├── test_v4_theory.py                      # V4 定理 5–7 推广、Fano 1/5、KL-LCB、N^{-2} 与辛普森单测
│   └── test_v5_rigor.py                       # V5 审稿修补全部 9 项数学缺陷专项验证单测
├── scripts/
│   ├── verify_theorems.py                     # 定理 1–4 验证 -> results/theorem_verification.json
│   ├── run_experiments.py                     # V1 参数扫描与五算法对比 -> results/experiment_results.json
│   ├── plot_figures.py                        # V1 图 1–3 绘图 -> results/figures/*.png
│   ├── run_v2_extensions.py                   # V2 实验 E/F/G -> results/v2_extension_results.json
│   ├── plot_v2_figures.py                     # V2 图 4–5 绘图 -> results/figures/*.png
│   ├── run_real_llm_experiments.py            # V3 真实大模型/RM 实验 R1–R4 -> results/real_llm_results.json
│   ├── plot_real_figures.py                   # V3 图 6–7 绘图 -> results/figures/*.png
│   └── verify_v4_theory.py                    # V4 定理 5–7 定量交叉验证 -> results/v4_theory_verification.json
└── results/
    ├── EXPERIMENT_REPORT.md                   # V1–V5 完整实验与审稿修补验证分析报告
    ├── theorem_verification.json / experiment_results.json
    ├── v2_extension_results.json / real_llm_results.json / v4_theory_verification.json
    ├── param_sweep_summary.csv / algorithm_comparison_summary.csv
    └── figures/
        ├── scaling_collapse_curves.png        # 图 1：异方差与污染尾部下的 BoN 坍塌曲线
        ├── optimal_n_phase_diagram.png        # 图 2：相变相图与闭式/数值 N* 一致性检验
        ├── calibrated_vs_bon.png              # 图 3：校准加权与两阶段多数决消除坍塌对比
        ├── ensemble_phase_reversal.png        # 图 4：多裁判相变逆转与重尾 Mean/Min 分离
        ├── finite_calibration_secondary_collapse.png # 图 5：有限校准二次坍塌与 LCB 收敛标度
        ├── real_llm_theorem6_calibration.png  # 图 6：AceMath-RewardBench 真实 RM 校准验证
        └── real_llm_math500_scaling.png       # 图 7：MATH-500 (Llama-3.2-1B/3B + RLHFlow-8B-PRM) 实证
```

---

## 3. 快速安装与一键复现

本核心理论库仅依赖 `numpy`, `scipy`, `matplotlib`, `pytest`；若复现 V3 真实 Parquet 数据集实验，可安装可选依赖组 `[real]`（`pandas`, `pyarrow`, `sympy`，数据自动缓存于 `.gitignore` 忽略的 `data/` 目录）：

```bash
# 1. 克隆并离线可编辑安装
git clone https://github.com/6Riemann9/test-time-scaling-theory.git
cd test-time-scaling-theory
python -m pip install --no-deps --no-build-isolation -e .

# 2. 运行全量单元测试（V1–V5 共 39 项）
python -m pytest -v

# 3. 运行 V1–V4 实验验证与绘图脚本
python scripts/verify_theorems.py
python scripts/run_experiments.py
python scripts/plot_figures.py
python scripts/run_v2_extensions.py
python scripts/plot_v2_figures.py
python scripts/verify_v4_theory.py

# 4. 运行 V3 真实大模型/RM 实证实验（首次运行将自动从 HuggingFace 缓存约 140MB 数据至 data/）
python scripts/run_real_llm_experiments.py
python scripts/plot_real_figures.py
```

---

## 4. 关键实验结论与学术图表展示

详细量化分析请参阅 [results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md)。

### 4.1 核心量化指标摘要
- **V1 定理基准精度**：分段 Gauss-Legendre 与自适应 `quad` 最大偏差 `≤ 3.47e-13`；与 $1.5\times 10^6$ 次 MC 全点误差 `≤ 7.68e-4`；一阶鞍点预算公式 `saddle_refined` 跨 104 组参数 MRE `8.28%`（$R^2 = 0.9989$）；强诱饵 + $\epsilon=0.15$ 污染下 `Two-Stage-Verified-Majority` 将 $N=1000$ 准确率从 `0.00%` 恢复至 `99.983%`。
- **V2 多裁判相变逆转（定理 5 对称基准）**：在 $\kappa=1.65, \rho_1=0.85, \rho_0=0.10$（理论临界 $K_c=4$）下，$K=2$ 在 $N=4096$ 跌至 `51.62%`，跨过 $K=4$ 即刻逆转为 **`99.36%`**；Student-$t$ 重尾下 `Mean (K=4)` 崩溃至 **`0.28%`**，无同源漏洞 `Min (K=4)` 逆转达 **`99.99%`**，注入 4% 同源漏洞后 `Min` 再次跌至 `28.27%`。
- **V4 理论升级全量定量检验（定理 5–7 重构验证）**：
  - **5A 负权差值逆转 & 纯高斯 QP `Min`**：在高错相关（$\rho_1=0, \rho_0=0.85, \kappa=1.2$）下等权 `Mean` 跌至 **`3.16%`** 且全非负组合必坍塌，最优特征向量 $\mathbf{w}^*\propto(1,-1)$ 在 $N=2000$ 逆转为 **`100.00%`**；纯高斯非对称 Active-Set QP 剔除高噪冗余裁判后，将 `Mean`（`27.84%`）逆转为 **`99.91%`**；
  - **5B Copula 统一重尾**：Gaussian-Copula + Student-$t$ 下 $\alpha_0/\eta_0=8.0 > \alpha_1/\eta_1=5.07$ 预测 `Min` 达 **`99.96%`**（`Mean = 4.08%`），注入 6% 同源漏洞后跌回 **`23.02%`**；
  - **6A/6B/6C 修正标度律与非渐近 `KL-LCB`**：修正 $K^*(M)\propto(M/\ln M)^{1/5}$ 在 $M=5\times 10^6$ 将遗憾压低至 `0.00067`（击败错标 `2/5` 的 `0.00088`）；`M=400, K=40` 上 `KL-LCB = 43.44%`、`Wilson-LCB = 43.43%` 全面克服 `Empirical-Bernstein LCB` 过保守崩溃（`25.00%`）；完美校准内点峰值收敛速率在 $N=20{,}000$ 处与闭式 $\frac{C}{2(f^*)^2}N^{-2}$ 之比达 **`0.9997`**；
  - **7A/7B 题目异质性与辛普森遮蔽**：解析导数分解误差仅 `2.26e-10`；强跨题偏移 $N=256$ 下 `Within-Prompt-Z-KL-LCB`（**`57.00%`**）大幅优于坍塌的 `Pooled-KL-LCB`（`34.70%`）与 `Raw-BoN`（`17.80%`），而无偏移非单调 $N=2$ 下 `Pooled-KL-LCB`（**`39.05%`**）反向领先 `Within-Z`（`31.95%`）。
- **V5 审稿级数学严格化重构验证（9 项缺陷全部清零，[test_v5_rigor.py](tests/test_v5_rigor.py) `9/9 PASSED in 0.86s`）**：
  - 逐项验证 $C^0$ 双闭子区间非单调构造、固定峰值高度机制下 $N^*_{\text{exact}}/N^*_0$ 在 $\kappa=1.01$ 处趋近 `0.5682`（目标 $e^{-\gamma}\approx 0.5615$，误差 `<1.5%`）、$\text{PPV}=1$ 边界散度分离、单高权错误诱饵反例检出、$C_c^{\text{pdf}}=\gamma_c C_c^{\text{sf}}$（$t=100$ 密度比落入 `[0.99, 1.01]`）、负相关反例 $\eta=0.25<0.5$ 与 20 组非负相关 Slepian 界、`KL-LCB` 安全底线与非全排序实证、双峰分离假设 $2^{-2}$ 缩放以及 $N=2$ 纯秩天花板与先验诱导残差（详见 [results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md) 第 8 节）。
- **V3 文献排雷与真实大模型/RM 实证（83 万+ 真实候选）**：
  - [RELATED_WORK.md](RELATED_WORK.md) 核验 12 篇 2022–2026 arXiv/顶会文献，精准划定 V1 统一重构与 V2/V4 原创定理边界；
  - `AceMath-RewardBench`（8,987 题 × 64 候选 = 575,168 条双 RM 打分）：`RM-B` 在 `college_math` 上因右尾后验倒挂从 $N=16$ (`46.40%`) 坍塌至 $N=64$ (`45.79%`)；在 4,494 题 holdout 上于 $M=200, N=64$ 用 `Optimal K*(M)=9 + Wilson-LCB`（`69.55%`）挽回过细 `Fixed K=60 + MLE` 退化（`66.01%`, **`+3.53%`**）；
  - `HuggingFaceH4/MATH-500`（`Llama-3.2-1B/3B` × 256 候选 = 256,000 条 `RLHFlow-8B-PRM` 轨迹）：4 类 PRM 视图因高错例相关（$\bar{\rho}_0 \approx 0.617\text{--}0.644$）均触发定理 5A `irreversible_collapse`；5-Fold CV 下 `Calibrated-Weighted-BoN (Wilson-LCB)` 在 $N=256$ 将 Pass@1 从 `Standard-BoN` 的 `42.08% / 53.52%` 大幅提升至 **`52.80% / 65.20%`**（**`+10.72% / +11.68%`**）。

### 4.2 核心图表

#### (1) 异方差与 Reward Hacking 极值坍塌曲线
![Scaling Collapse Curves](results/figures/scaling_collapse_curves.png)

#### (2) 相变相图与闭式最优预算 $N^*$ 一致性检验
![Optimal N Phase Diagram](results/figures/optimal_n_phase_diagram.png)

#### (3) 后验校准与两阶段多数决彻底消除坍塌
![Calibrated vs Standard BoN](results/figures/calibrated_vs_bon.png)

#### (4) V2 定理 5：多裁判高斯相变逆转与重尾 `Mean`/`Min` 分离
![Multi-Verifier Phase Reversal](results/figures/ensemble_phase_reversal.png)

#### (5) V2 定理 6：有限校准样本二次坍塌与 Minimax LCB 收敛标度
![Finite Calibration Secondary Collapse](results/figures/finite_calibration_secondary_collapse.png)

#### (6) V3 真实验证：`AceMath-RewardBench` 有限校准与 $M$ 标度律
![Real LLM Theorem 6 Calibration](results/figures/real_llm_theorem6_calibration.png)

#### (7) V3 真实验证：`MATH-500` (`Llama-3.2-1B / 3B` + `RLHFlow-8B-PRM`) 算法对比与多 PRM 相关性诊断
![Real LLM MATH-500 Scaling](results/figures/real_llm_math500_scaling.png)
