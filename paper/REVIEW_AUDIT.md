# 多工具联合论文质量审计与修正备忘录 (`REVIEW_AUDIT`)

> **审计基准版本**：`v1.0.0`（优化前 `173d565`） $\longrightarrow$ **顶会投稿就绪优化版**  
> **协同工具链**：`citecheck` / `arxiv` / `paper-search` MCP + `ml-paper-writing`、`research-paper-writing`、`sciwrite`、`academic-plotting`、`paper-review-zh-cn`、`academic-writing-skills-zh-cn` Skills。

---

## 1. 关键缺陷定位与修订对照总表

| 严重度 | 发现来源工具 | 原版本 (`v1.0.0`) 问题诊断 | 修正措施与最终验证状态 |
| :---: | :--- | :--- | :--- |
| **CRITICAL** | `sciwrite` Pass 5 + JSON 对账 | `paper/main.tex` Table 2 最后一行 `college_math RM-B` 将 `Within-Z` 错录为 `0.4366`，且省略最后一列 `vs. Raw` 的 $p_t$ 值与完整统计显著性解释 | 核对 `v10_release_verification.json` 后精确更正为 **`0.4591`**，补全 **$p_t = 0.1810$** 并加表注明确区分全量 `P=2,818` 显著回落（$p_t=0.0208$）与 845 题单项子集测试无显著性；明确加粗/下划线标记规则 |
| **CRITICAL** | `ml-paper-writing` + `academic-plotting` | 论文仅嵌入 1 张 PNG 位图（`fig:simpson` 且全文未引用），缺少核心概览图 Figure 1，无矢量 `.pdf` 图表；`scripts/plot_stage2_figures.py` 图表渲染中残留旧符号 `N_match` 与旧名 `Hybrid-Adaptive` | 修复 `scripts/plot_stage2_figures.py` 展示标签并重绘；新建 `scripts/plot_paper_figures.py` 导出 4 组 Serif 字体 + 色盲安全矢量 `.pdf` 与 300 DPI `.png`（Figure 1–4 涵盖模型边界/Minimax 相图/Holdout 控制流、协议/同 $K$ 消融/2D 表面、单次划分净增益/自然坍塌/四相漂移、多 RM 相关性/spectral path/OlympiadBench PRM）并通过逐图视觉检查 |
| **CRITICAL** | `paper-review-zh-cn` + `research-paper-writing` | `paper/main.tex` 存在 **13 个未引用死标签**（含附录 `app:minimax`, `app:ensemble`, `app:index` 与主图 `fig:simpson`、方程 `eq:cond-bin`, `eq:n-balance`, `eq:joint-lower`, `eq:unif-class`） | 重构全部 `\Cref`/`\eqref` 交叉引用网络，实现 **0 个未定义引用、0 个未引用死标签**（经 `audit_paper_artifacts.py` 自动验证） |
| **MAJOR** | `sciwrite` Pass 6 + `ml-paper-writing` | 原 Abstract 高达 **362 词**，含密集复杂符号串，却缺失最关键的实证定量收益与 $p$ 值 | 遵循 5-Sentence 摘要范式压炼至约 **200 词**，保留核心 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}+N^{-\beta}$、$O_p(\sqrt{(\ln N)/N})$ 定理率并前置 `+2.41 pp / +1.83 pp / +42.29 pp` 核心实证结果 |
| **MAJOR** | `citecheck` + `arxiv` + `paper-search` + `search_web` | 全部 27 篇文献 100% 真实存在（0 伪造、0 撤稿），但部分条目的会议发表状态或卷页/URL/arXiv eprint 未及时更新 | 精确升级 `snell2024scaling`（ICLR 2025 Oral）、`dorner2025roc`（ICLR 2026）、`rame2024warm`（补 `pages={42048--42073}`）、`huang2025best`（补 `volume={267}, pages={25075--25126}`）、`garivier2011kl`、`bubeck2011xarmed`、`minsker2013estimation`、`jin2021pessimism`、`muller1985kernel`；同步更新 `RELATED_WORK.md` 并实现 27/27 全文引用 |
| **MAJOR** | `ml-paper-writing` + `research-paper-writing` | 论文缺少独立 Related Work 章节、缺少核心算法伪代码块、缺少 NeurIPS 标准 `.sty` 模板/匿名切换、缺少 Reproducibility/Broader Impact 与 NeurIPS Checklist | 集成标准 `paper/neurips_2025.sty` 与 `paper/Makefile`（支持 `[preprint]` 与双盲提交一键切换）；新增 §2 Related Work、Algorithm 1（`Sample-Split KL-LCB`）、Algorithm 2（`Hybrid-Fixed`）、§7.4 多 RM 实证检验、§8 Limitations/Broader Impact/Reproducibility 与 Appendix F NeurIPS Checklist |
| **MINOR** | `paper-review` `display-notation-provenance` | `README.md` 仍残留旧版本标题 `V9 主次聚焦`、旧计数 `22 篇文献`，且 `q'_pool` 下界与论文略微差 $0.01\times 10^{-3}$ | 更新 `README.md` 至 `v1.0.0` 顶会优化版、`27 篇文献`、完整文件清单，并在两处同步明确精确下界 $+7.044\times 10^{-3} > +7.03\times 10^{-3}$ |

---

## 2. 核心主张—证据—代码与图表映射表（Claim–Evidence Map）

| 论文主命题 / 实证主张 | 数学证明与算法位置 | 实证图表 / 表格证据 | 核心校验 JSON 与源码实现 |
| :--- | :--- | :--- | :--- |
| 联合 $(M,N)$ Minimax 率 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$ 与平衡线 $N_{\mathrm{balance}}(M)$ | `paper/main.tex` §4 (`Theorem 1`), `Appendix B` | Figure 1(b), Figure 2(c) (`N_balance(M)`) | `src/tts_theory/calibration_theory.py`: `tight_joint_minimax_scaling` |
| 全 $\beta>0$ 样本分割 Pyke 乘性间距界 $\hat{\pi}_k \in [\frac{1}{2K}, \frac{143}{72K})$ | `paper/main.tex` §4 (`Proposition 1`, `Algorithm 1`) | Table 1, Figure 2(a)(b) | `calibration_theory.py`: `empirical_quantile_spacings_bounds`; `v9` & `v10` |
| 跨题池化辛普森解析遮蔽 ($q'_{\mathrm{pool}} > +7.044\times 10^{-3}$ on $[3.5,4.2]$) | `paper/main.tex` §5.1 (`Theorem 2`), `Appendix C` | Table 2 (`S3`, `RB2`), Figure 3(a) | `heterogeneity_theory.py`: `diagnose_simpson_confounding` |
| $N=2$ 样本 $Z$ 秩上界与 $\pi_0>0$ 含不可解题在线 $Z_{i,N}$ 极值收敛与上限损失 | `paper/main.tex` §5.2 (`Theorems 3–4`), `Appendix C` | Table 2 (`AceMath`, `OlympiadBench`), Figure 3(a) | `heterogeneity_theory.py`: `verify_sample_zscore_extremal_convergence` |
| 三分支留出门控风险控制 ($\mathrm{Hybrid\text{-}Fixed}$, $\varepsilon_{\mathrm{refit}}\equiv 0$) 与未见漂移对策 | `paper/main.tex` §5.3 (`Proposition 2`, `Algorithm 2`) | Table 2, Table 3, Figure 3(a)(b)(c) | `real_eval.py`: `evaluate_single_split_prompt_heterogeneity`; `v9` & `v10` |
| 异构多裁判 `Safe-Fisher`、Bayes QDA $N^{-2/K}$、零乘子 QP `Min` 与 Copula | `paper/main.tex` §6 (`Theorems 5–6`), `Appendix D` | Figure 4(a)(b)(c) (`RB2`, `OlympiadBench`) | `ensemble_theory.py`: `solve_positive_quadrant_qp`; `v6` & `stage2` JSON |

---

## 3. 自动化核验与投稿前诚实说明

1. **参考文献外部核验结果**：本轮已通过 `citecheck.check_bibliography` + `arxiv.get_abstract` + `paper-search.search_openalex` + 会议官网检索核实 `paper/references.bib` 全部 27 篇条目，并将升级后文本与 `RELATED_WORK.md` 逐字节对齐；更新后已再次调用 `citecheck.check_bibliography` 复核无异常。
2. **代码与脚本回归测试**：全部图表的重绘脚本与仓库 `tests/` 下全部 **76 项**离线单元测试（其中 `tests/test_v10_release.py` 扩展至 9 项）保持 100% 兼容通过。
3. **官方 NeurIPS 2026 本地真实编译与页界核查**：已通过本地 **Tectonic 0.17.0**（基于官方 `paper/neurips_2026.sty`）完成 `[preprint]` 模式与 `\usepackage{neurips_2026}` 双盲投稿模式的完整交叉编译，生成 `paper/main.pdf`（21 页）；末轮 pass 实现 **0 `Overfull \hbox`、0 `Undefined references/citations`、0 `hyperref` Unicode 警告**，正文（含 `Section 1–8`、全部 4 图 3 表及 2 个算法框）严格处于 **第 1–9 页**以内，`Acknowledgments` 与 `References` 自第 10 页起页。

---

## 4. 第五轮学术审稿（v1.0.0）全部 9 项最终整改台账

| 整改编号 | 严重度 | 审稿发现问题 | 实施修正与核验闭环 |
| :--- | :---: | :--- | :--- |
| **`P0-01`** | **P0** | `Algorithm 1` 允许按 raw score 打破平局，破坏条件期望恒等式 $\mathbb{E}[Y_{\hat{i}^*}\mid\mathcal{F}_{\mathrm{bins}}]=q_{\hat{k}^*}$ | 统一修订 `paper/main.tex` §4 与 `Algorithm 1` L205 为严格均匀随机破平局（`strictly uniform random tie-breaking`），并新增 `tests/test_v10_release.py` 单测构造右端错误窄区反例实证校验 |
| **`P0-02`** | **P0** | 英文 `Theorem 5`（`Safe-Fisher`）遗漏退化正交边界分支条件 | 在 `paper/main.tex` §6 与 `Appendix D` 同步恢复退化正交混合分支（$\mathbf{w}^*=\mathbf{w}_{\lambda_{\min}^-}+\alpha\mathbf{u}_{\min}$），并增加反例数值验证测试 |
| **`P0-07`** | **P0** | 需要本地真实编译 `paper/main.pdf` 并控制正文严格不超过 9 页且零溢出 | 使用 Tectonic 0.17.0 在不改动官方 `.sty` 边距字号的前提下重构 `Table 1` 并调整 `fig1`–`fig4` 与浮动体排布，达成 **Pages 1–9** 全正文完结、**0 `Overfull \hbox`** 并完成全 21 页视觉巡检 |
| **`P0-08`** | **P0** | 需迁移至 NeurIPS 2026 官方样式并提供官方完整 16 项 Paper Checklist | 删除旧 `neurips_2025.sty`、部署官方 `paper/neurips_2026.sty` 并更新 `paper/Makefile`；补充完整 16 项 `NeurIPS Paper Checklist`（第 15–21 页）并验证双盲/Preprint 双模式编译 |
| **`P1-01`** | **P1** | 摘要及部分行文使用过宽表述 `whenever unsolvable prompts exist`，遗漏内点可解题条件 | 统一在 Abstract、§1、§5.2 `Theorem 4` 与 §8 限定 $\pi_0>0=\pi_1$ 且存在正概率内点可解题 $\mathbb{P}(p(X)\in(0,1))>0$，并说明双点反例 |
| **`P1-04`** | **P1** | `college_math RM-B` 自然峰选中（$N=16$）存在事后峰选偏差与 14 组多重比较问题 | 重跑 `verify_v10_release.py` 增加 14 组 Bonferroni/Holm 校正（$p_{\mathrm{Bonf},14}=0.2908$）与无偏 50/50 样本分割确认检验，论文重绘 `Figure 3(b)` 并明确降格为探索性候选案例 |
| **`P1-05`** | **P1** | `Table 3` 漂移矩阵原仅跑连续插值基线，需显式对齐 `Algorithm 1` 离散分割实现 | 扩增 `Alg1-SampleSplit`（50% 边+50% `KL-LCB`）同台对照并在 `Table 3` 括号标注（`0.7101`, `0.3500`） |
| **`P1-06`** | **P1** | `Table 3` 仅测超界 OOD 尖峰，未显式暴露分布内高分污染下 `Support-Clipped` 的失效边界 | 增补 `[q_0.90, q_0.985]` 分布内 6% 污染对照、`Table 3` 下半区与 `Figure 3(c)` `R5: In-Sup Unseen` 柱组，诚实报告 `Support-Clipped` 退化为 `0.3629 / 0.3785` |
| **`P1-07`** | **P1** | 独立重跑全量单元测试与静态质量审计脚本并同步仓库文档 | 升级 `scripts/audit_paper_artifacts.py`、同步 `EXPERIMENT_REPORT.md` / `README.md` 并完整复跑全部 **76 项** `pytest` 与审计检查 100% PASS |

---

## 5. TMLR 深度批判与诚实诊断升级台账（`paper/tmlr/`）

在保持 NeurIPS 2026 归档（`paper/main.tex`、`paper/main.pdf` 与 `audit_paper_artifacts.py`）零退化前提下，针对 TMLR“理论诊断与负面/边界实证审计”定位完成以下 4 项全面重构：

1. **理论推广至一般 $(\pi_0, \pi_1)$ 与实证边界核验**：将原要求 $\pi_1 = 0$ 的样本标准化尾反转定理推广至任意 $\pi_1 \ge 0$（给出精确两尾极限 $L_{\mathrm{tail}} = \frac{\pi_1}{\pi_0+\pi_1}$、$\pi_0=0,\pi_1>0$ 零损失相以及 $\pi_0>0$ 一般紧致峰值上限损失证明），并在五大真实基准上实测 $(\hat{\pi}_0, \hat{\pi}_1, \hat{L}_{\mathrm{tail}}, \max_k \hat{q}_Z, \hat{q}_Z(20))$，精准解释从 `RB2 (-0.92 pp)`、`S3 (-1.37 pp)`、`AceMath (-2.89/-2.84 pp)` 到 `OlympiadBench (-8.69 pp)` 的实测跌落梯度。
2. **反事实 $\hat{\pi}_0 = 0$ 因果归因消融**：在 `AceMath RM-B` 相同 $50/20/30$ 划分与完整 $2{,}695$ 题测试集上验证过滤全错题（$\hat{\pi}_0^{\mathrm{tr}}=0$）及全边界原子仅留内点题（$\hat{\pi}_0^{\mathrm{tr}}=\hat{\pi}_1^{\mathrm{tr}}=0$）：剔除全错题将右尾 LCB 抬高 $\approx +12\,\mathrm{pp}$；在修复早期离散 `Algorithm 1` 非随机序列切片 Bug 后（改为真 i.i.d. 置换 50/50 划分与 `-1` 支撑截断），8 个随机划分种子下无论是连续分段线性校准器（`3a OOD`: $-0.33 \pm 0.40\,\mathrm{pp}$）还是真 i.i.d. 离散 `Algorithm 1`（`3a OOD`: $-0.28 \pm 0.46\,\mathrm{pp}$），在移除 $\hat{\pi}_0^{\mathrm{tr}}$ 后均保持稳健防御能力——原因在于极低先验难内点题（$\bar{p}_{16}=0.659 \to \bar{p}_{20}=0.446$）在方差压缩分母作用下继续聚集最右端，使两种校准器仍维持峰后负斜率下压。
3. **诚实叙事重定位与割裂消解**：将 $N=2$ 样本 $Z$ 秩天花板降格为 `Proposition`；将 `Hybrid-Fixed` 诚实定性为“零干净净增益、仅作防退化保护”的最小留出门控基线；在 §3 建立跨题池化错配 + 有限 $(M,N)$ 估计误差分解公式，并以“安全税（safety tax）”统一解释 §7.4 `Safe-Fisher` 与 `Min-Ensemble` 在低 $N$/无坍塌基准上的保守让利。
4. **官方 TMLR 格式双盲编译与零泄密扫描**：部署官方 `tmlr.sty` 与 `tmlr.bst`，生成双盲投稿版 `paper/tmlr/main_tmlr.pdf` 与预印本版 `paper/tmlr/main_tmlr_preprint.pdf`，并通过 `scripts/audit_tmlr_paper.py` 强制验证双盲版 0 身份线索泄漏与 0 死标签/未定义引用。

---

## 6. 多轮 Swarm 对抗审稿与迭代收敛台账（Round 1–3）

为彻底排查理论盲区、代码实现瑕疵与 TMLR 双盲投稿风险，启动由理论数学审稿人（`math-critic`）、实证账实审稿人（`emp-critic`）、TMLR AE 审稿人（`tmlr-ae`）与修复验证工程师（`author-fixer`）组成的 4 角色对抗 Swarm，完成 **3 轮“对抗挑刺 $\to$ 协调者仲裁 $\to$ 实质修补与重验 $\to$ 验尸复审”**闭环并最终收敛签字：

| 轮次 | 暴露并解决的核心 `S0` / `S1` 关键缺陷 | 实施修补与复核结果 |
| :--- | :--- | :--- |
| **Round 1** | 1. **`Theorem 3(1)(a)` 左尾反超陷阱（`[S0]`）**：无约束双尾极大化在多内点高先验层满足 $(1-v_{\max})(\|m_0(p_C)\|-m_1(p_C)) > 2\|m_0(p^*)\|$ 时会将选择推向 $Z^* \to -\infty$（如 `Case 4` 渐近跌至 `0.55`）。<br>2. **`_fit_alg1_discrete` 非 i.i.d. 序列切分 Bug（`[S0]`）**：`train_idx` 按 `AceMath` 7 个子集排序拼接直接取 `[:m_half]` 切分，伪造了 `-7.60 pp` 假暴跌。<br>3. **`Theorem 4(1)` 归一化错误 & `eq:regret-decomp` 基准错配（`[S0]`）**：原 `App. D` 归一化写为 $\sum x_k=1$（应为 $\sum(1+\lambda_k)x_k=1$）；遗憾分解遗漏 $\Theta(N^{-\beta})$ 在线算力项桥接。<br>4. **图内编号错位、双盲工程痕迹与文献缺口（`[S0]`）**：`fig1` 残留 NeurIPS 定理号；双盲稿残留 `Repo Theorems`/`verify_v9`/`v10` 等可追踪文件名；缺 14 篇核心文献及 TMLR 专属诊断图。 | 1. 将 `Theorem 3(1)(a)` 收紧为右半轴 $Z_{i,N} \ge 0$ 约束并新增 `Case 4` 左尾陷阱分析与 `Reverse Fatou` 证明；<br>2. 将 `verify_tmlr_diagnostics.py` 修复为真 i.i.d. 随机置乱 50/50 划分与 `-1` 截断，并补跑 8 随机划分种子稳健性；<br>3. 修正 `App. D` 完整证明与 §6 全符号定义（`\theta_{c,I_0}` 消碰撞）；<br>4. 新建 `paper/tmlr/figures/`（重绘 `fig1..fig4` + 新增 `fig5` 3 联诊断图）与 `paper/tmlr/references_tmlr.bib`（41 篇 100% 闭环），双盲与 Preprint 隔离并通过审计。 |
| **Round 2** | 1. **`P^{-\top} V_{\min}` 线性代数子空间误写（`[S0]`）**：Round 1 修改时误将 $P V_{\min}$ 写成 $P^{-\top} V_{\min}$（非正交 $P$ 反例残差 `1.5051` vs `4.14e-16`）。<br>2. **`Table 5` 行 `2a` `Val \hat{m}` 错位 & `Table 4` 元组陈旧（`[S0]`/`[S1]`）**：`2a` 误印 `Pooled-LCB`（实为 `Within-Z`），且 `1/3a/3b` 干净验证集 `[Alg. 1]` 选中 `Alg1-Pooled` 未标注；`Size-Match` 先验元组陈旧。<br>3. **`Case 2` 单调区间误报 & `heterogeneity_theory.py` 早退（`[S1]`）**：`Case 2` 在 `z \approx -3.68` 存在谷底，仅在 `[-3.6, 18.8]` 单调上升；代码缺 `constrain_nonnegative_z` 开关。 | 1. 修正 `main_tmlr.tex` 与 `PROOF.md` 为 $P V_{\min}$ 与归一化 $\mathbf{u}_{\min}^\top \Sigma_1 \mathbf{u}_{\min}=1$；<br>2. 对齐 `Table 5` 全部 `Val \hat{m}`（`Within-Z` / `Raw-BoN [Pooled]`）及 `Table 4` `(0.1468, 0.0216)`；<br>3. 为 `diagnose_zscore_induced_collapse` 新增 `constrain_nonnegative_z` 与严格差分单调校验，精准区分 `Case 2` 渐近与有限 $N$ 零损失，同步完善图注自包含、`\hat{\kappa}_Z` 定义与全 23 页零警告排版。 |
| **Round 3** | **全面验尸终审**：三方只读复查最新版 `main_tmlr.tex`、双版 23 页 PDF、`heterogeneity_theory.py`、`tmlr_diagnostics.json` 与 `PROOF.md`。 | **全部阻断项清零签字**：`math-critic` (`S0=0, S1=0`)、`emp-critic` (`S0=0, S1=0`)、`tmlr-ae` (`S0=0, S1=0, S2=0`) 一致签发 TMLR `Accept / Accept with Minor Revisions` 收敛放行（Coordinator 随手补齐 `App. B` 构造长度区间 `L_0 \in (2\eta_0, 1/f_{\min})` 与 $\pi_0 \le 10^{-15}$ 边界守卫后完成全量重编与回归）。 |
