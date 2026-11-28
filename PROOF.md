# 测试时计算缩放的联合校准—算力极限：Minimax 理论与裁判选择（概率论证明稿）

**摘要（Abstract）.** 本文围绕大语言模型（LLM）推理阶段的测试时计算缩放（Test-Time Scaling, TTS；*Joint Calibration–Compute Limits of Test-Time Scaling: Minimax Theory and Verifier Selection*）建立两层概率论与统计决策框架。为突出本文的核心理论贡献并保持单裁判预备工具的数学自包含性，全文按以下主次层次组织：

1. **问题设定、适用域界定与单裁判预备引理（第 1 节 & 附录 A）**：在两级层次生成—验证概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$ 中区分**固定题目条件模型**（定理 1–6）与**跨题层次异质模型**（定理 7）；汇总单裁判顺序统计量积分、$C^1/C^0$ 似然比单调判据（MLRP）、固定峰值高度机制下的远尾 Gumbel 极限 $e^{-\gamma}N^*_0$、二元缩放律及异质单错类加权多数决预备命题（完整证明置于**附录 A**，保持定理 1–4 原始编号以便代码与文献交叉检索）。
2. **主定理 I：有限校准集 $M$ 与在线算力 $N$ 的联合 Minimax 理论（第 2 节：定理 6A–6C & 命题 6.1）**：严格分离逐分布渐近类 $\mathcal{F}_\beta^{\text{sep}}$ 与带公共正则常数的**一致正则分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$**；通过三事件离线并界（联合覆盖 + 桶计数 + Hoeffding 下侧偏差）与最近峰在线样本耦合，证明 $N$-无关等频 `KL-LCB` 在一切 $\beta > 0$ 上达到双阶段解耦率 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$，导出**校准—算力预算平衡律** $N_{\text{balance}}(M) \asymp (\frac{M}{\ln M})^{\frac{1}{2\beta+1}}$；进一步利用样本分割 Uniform 顺序统计量 Rényi–Pyke Gamma 间距浓缩界（**命题 6.1**），在未知总体分位数且零额外无标签数据下将等频分桶推广至一切 $\beta > 0$（消解加性 DKW 界在 $0 < \beta \le \frac{1}{2}$ 处的相对误差发散）。
3. **主定理 II：跨题异质性、在线样本标准化尾反转与风险控制门控（第 3 节：定理 7A–7C & 命题 7.1）**：建立池化后验导数协方差分解，并在非退化闭区间 $s \in [3.5, 4.2]$ 上给出**无网格纯解析区间上下界**的二层异方差高斯伪单调遮蔽实例（定理 7A）；给出 $N=2$ 样本 Z-Score 纯秩非严格界及严格优势双向序反转充要条件（定理 7B）；在含不可解题原子 $\pi_0 > 0$ 的同方差单调基准上，严格跟踪原始分数仿射坐标变换 $S_i = b(X) + a(X)\tilde{S}_i$，证明总体标准化 $Z^*$ 诱导出右尾极限 $\frac{\pi_1}{\pi_0+\pi_1}$，且实际在线样本标准化满足一致极值收敛 $\max_{1\le i\le N}|Z_{i,N} - Z_i^*(X)| = O_p(\sqrt{(\ln N)/N}) \xrightarrow{p} 0$ 从而继承渐近准确率上限损失（定理 7C）；最后给出覆盖全体 $N \ge 2$ 的三分支留出门控风险控制不等式及选后重拟合误差分解（**命题 7.1**）。
4. **补充理论：异构多裁判集成相变、帕累托光谱与 Copula 尾相依（第 4 节：定理 5A–5B）**：刻画一般非对称高斯 $(\Sigma_1, \Sigma_0)$ 下的广义特征值负权逆转、带渐近尾约束的 $N=2$ 线性信噪比光谱路径 `Safe-Fisher`、不可逆相 $\Sigma_0 \succ \Sigma_1$ 下 Bayes QDA 天花板与 $N^{-2/K}$ 收敛阶，以及含零 KKT 乘子弱激活退化的多元高斯 `Min-Ensemble` 与 Gaussian-Copula + Student-$t$ 精确对数因子 $(\ln t)^{(\gamma_c - |I_{+,c}|)/2}$。

---

## 1. 层次生成—验证概率模型、适用域界定与预备引理

### 1.1 生成—验证层次概率模型与定理适用域统一界定

真实推理与评测任务服从两级**层次生成—验证概率模型** $(\Omega, \mathcal{F}, \mathbb{P})$：首先从题目总体抽取问题 $X \sim \mathcal{P}_X$；再在给定 $X$ 的条件下从生成器独立同分布（i.i.d.）抽取 $N \ge 1$ 条候选样本 $\{(Y_i, S_i)\}_{i=1}^N \mid X \stackrel{\text{i.i.d.}}{\sim} P_{Y, S \mid X}$，以及（若使用离线校准）基于独立抽取的有标签校准集 $\mathcal{D}_M$。其中：
- $Y_i \in \{0, 1\}$ 为样本 $i$ 的真实正确性示性变量，$p(X) := \mathbb{P}(Y_i = 1 \mid X)$ 为题内先验正确率；
- $S_i \in \mathbb{R}$ 为 Verifier 标量打分（第 4 节定理 5 推广至 $\mathbf{S}_i \in \mathbb{R}^K$）。给定 $(X, Y_i = c)$（$c \in \{0, 1\}$），条件分布函数记为 $F_{c, X}(s) := \mathbb{P}(S_i \le s \mid Y_i = c, X)$。

> **模型适用域统一界定（固定题条件保证 vs. 跨题池化校准）**：
> - 在 **预备定理 1–4、补充定理 5 与主定理 6** 中，我们固定单个题目（或考察条件分布 $P_{Y, S \mid X}$ 不随 $X$ 变化的同质总体），简化记号为 $p = p(x)$、$F_c = F_{c, x}$，对应单题内部条件采样模型 $(Y_i, S_i) \stackrel{\text{i.i.d.}}{\sim} P$。在此条件意义下，单题内各候选样本独立同分布，后验排序最优性、置信覆盖与 Minimax 遗憾界严格成立。
> - 当题目存在跨题异质性（$P_{Y, S \mid X}$ 随 $X$ 变化）时，样本对 $(Y_i, S_i)$ 边际上仅条件独立（可交换）而非 i.i.d.，池化边际后验 $q_{\text{pool}}(s) = \mathbb{P}(Y=1 \mid S=s)$ 通常**不等于**给定全池后的条件后验 $\mathbb{P}(Y_i=1 \mid S_1,\dots,S_N, X)$。因此，单分布校准保证**不能未经检验地直接迁移至跨题共享的池化校准器**。**第 3 节（定理 7 & 命题 7.1）** 专门刻画这种层次错配引发的辛普森遮蔽、在线样本标准化尾反转与门控选择边界。

对固定分布 $P$（给定 $X=x$ 或同质基准），令 $p \in (0, 1)$，对应概率测度 $\mu_c$ 与生存函数 $\bar{F}_c(x) := 1 - F_c(x)$（绝对连续时密度记为 $f_c(x)$）。混合分布函数、生存函数与密度分别为：
$$F_{\text{mix}}(x) = p F_1(x) + (1-p) F_0(x), \qquad \bar{F}_{\text{mix}}(x) = p \bar{F}_1(x) + (1-p) \bar{F}_0(x), \qquad f_{\text{mix}}(x) = p f_1(x) + (1-p) f_0(x).$$
记右端点 $x_R := \sup\{x \in \mathbb{R} : F_{\text{mix}}(x) < 1\} \in (-\infty, +\infty]$，左端点 $x_L := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) > 0\} \in [-\infty, +\infty)$，左连续广义分位数逆函数与上分位数逆函数：
$$F_{\text{mix}}^{-1}(u) := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) \ge u\}, \quad u \in (0, 1); \qquad \bar{F}_{\text{mix}}^{-1}(q) := F_{\text{mix}}^{-1}(1 - q), \quad q \in (0, 1).$$
记混合测度 $\mu_{\text{mix}} := p \mu_1 + (1-p) \mu_0$。由 Radon–Nikodym 定理，给定打分 $S = x$ 时的贝叶斯后验正确率为 $q(x) := \mathbb{P}(Y = 1 \mid S = x) = \frac{d(p \mu_1)}{d\mu_{\text{mix}}}(x) \in [0, 1]$（$\mu_{\text{mix}}$-a.e.）。在正密度支撑集 $\{x : f_{\text{mix}}(x) > 0, f_0(x) > 0\}$ 上定义似然比 $\Lambda(x) := \frac{f_1(x)}{f_0(x)}$ 与对数似然比 $\ell(x) := \ln \Lambda(x) = \ln f_1(x) - \ln f_0(x)$。

### 1.2 标准 Best-of-$N$ 决策规则与基础分析引理

给定采样预算 $N \ge 1$，记最大顺序统计量 $S_{(N)} := \max_{1 \le i \le N} S_i$，最大值下标集 $\mathcal{I}_{\max} := \{i \in \{1,\dots,N\} : S_i = S_{(N)}\}$。标准 Best-of-$N$（BoN）从 $\mathcal{I}_{\max}$ 中均匀随机抽取下标 $i^*$，其精确正确率记为 $P_{\text{BoN}}(N) := \mathbb{P}(Y_{i^*} = 1)$。

**引理 1.1（后验与对数似然比的 Logistic 映射恒等式）.** *设 $f_1, f_0 \in C^1(I)$ 且严格为正，则对任意 $x \in I$ 有：*
$$q(x) = \sigma\big(\operatorname{logit}(p) + \ell(x)\big) = \frac{p \Lambda(x)}{p \Lambda(x) + 1 - p}, \qquad q'(x) = q(x)\big(1 - q(x)\big)\ell'(x),$$
*其中 $\sigma(t) := (1 + e^{-t})^{-1}$，$\operatorname{logit}(p) := \ln\frac{p}{1-p}$。若仅假设 $f_1, f_0 \in C^0(I)$ 且正，则 $q(x_2) > q(x_1) \iff \Lambda(x_2) > \Lambda(x_1)$ 严格保序。*（由 Bayes 公式及 $\sigma' = \sigma(1-\sigma)$ 立得。）

**引理 1.2（离散差分与连续预算导数恒等式）.** *设 $F_{\text{mix}}$ 连续且 $q(x)$ 局部有界变差，$P_{\text{BoN}}(N) = \int_{\mathbb{R}} q(x)\, d([F_{\text{mix}}(x)]^N)$。则：（1）对任意整数 $N \ge 1$，$P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N) = \int_{-\infty}^{\infty} [F_{\text{mix}}(x)]^N (1 - F_{\text{mix}}(x))\, dq(x)$；（2）若 $\int_{\mathbb{R}} |dq| < \infty$，则延拓至 $N \in [1, \infty)$ 上有 $\frac{\partial P_{\text{BoN}}(N)}{\partial N} = \int_{-\infty}^{\infty} (-\ln F_{\text{mix}}(x)) [F_{\text{mix}}(x)]^N\, dq(x)$。*（由 Lebesgue–Stieltjes 分部积分与控制收敛定理立得。）

**引理 2.1（Karlin 单交叉保号引理）.** *设 $F_{\text{mix}}$ 连续严格递增，$q \in C^1$ 且 $q'(x)$ 至多在唯一阈值 $x_c \in \mathbb{R}$ 变号，$u_c := F_{\text{mix}}(x_c) \in (0, 1)$，$I(N) := P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N)$。若 $(x - x_c)q'(x) < 0$（$x \neq x_c$），则对任意 $N > N_0 \ge 1$ 有 $I(N) < u_c^{N-N_0} I(N_0)$，故 $\{P_{\text{BoN}}(N)\}_{N \ge 1}$ 严格单峰；若 $(x - x_c)q'(x) > 0$（$x \neq x_c$），则不等号反向（当 $I(1)\ge 0$ 时全局严格递增，当 $I(1) < 0 < \lim_{N\to\infty}P_{\text{BoN}}(N)-p$ 时呈严格单谷 U 型）。*

### 1.3 单裁判预备定理（定理 1–4）核心结论摘要

> **文献定位说明**：单裁判基础命题整理并细化既有 TTS 极值与统计工具（详细对照见 [RELATED_WORK.md](RELATED_WORK.md)），不主张对既有单裁判核心发现的首创归属，主要作为第 2–4 节主定理的预备工具。定理 1–4 的完整陈述与严格证明见**附录 A**。

| 预备编号 | 核心结论摘要 | 关键精确公式 / 判据 |
| :---: | :--- | :--- |
| **定理 1 & 推论 1.1** | 含平局原子的顺序统计量积分与双样本高斯精确差值 | $P_{\text{BoN}}(N) = \int_0^1 q\big(F_{\text{mix}}^{-1}(v^{1/N})\big)\,dv$；$P_{\text{BoN}}(2)-p = p(1-p)[2\Phi(\tfrac{\Delta\mu}{\sqrt{\sigma_1^2+\sigma_0^2}})-1]$；$P_{\text{Anti-BoN}}(2)=2p-P_{\text{BoN}}(2)$ |
| **定理 2A–2C** | 重错尾坍塌、$C^1/C^0$ MLRP 判据、高斯五相分类及固定峰高 Gumbel 极限 | $\lim_{x\uparrow x_R}\Lambda(x)=0 \implies P_{\text{BoN}}(N)\to 0$；固定峰高 $\Delta\mu(\kappa)=\sqrt{c_1(\kappa^2-1)}\sigma_1$ 下 $\lim_{\kappa\to 1^+} N^*_{\text{exact}}/N^*_0 = e^{-\gamma}$ |
| **定理 3** | 离散二元 Verifier 闭式缩放律与 $\text{PPV}=1$ 边界散度分类 | $P_{\text{BoN,bin}}(N)=\text{PPV}-C(1-q_1)^N$；区分 $\beta>0$ 前向 $\text{KL}/\chi^2$ 与 $\beta=0$（$\text{PPV}=1$）的 $\text{TV}$ 及有限反向 $\text{KL}$ |
| **定理 4A–4B** | 后验校准单调性与可数无穷异质错类有界加权多数决 | $P_{\text{Cal-BoN}}(N)=\mathbb{E}[Q_{(N)}]\uparrow \operatorname{ess\,sup} q$；单错类约束 $p\mu_{w,+} > \sup_j(p_j\mu_{w,j})$ 与超级尾桶给出 $O(e^{-cN})$ |

---

## 2. 主定理 I：有限校准集 $M$ 与在线算力 $N$ 的联合 Minimax 理论（定理 6 & 命题 6.1）

固定单个问题条件分布 $P$，将打分空间 $\mathbb{R}$ 划分为 $K$ 个桶 $\{B_k\}_{k=1}^K$（边际概率 $\pi_k > 0$，桶内真正确率 $q_k := \mathbb{P}(Y=1 \mid S \in B_k)$）。由离线独立同分布校准集 $\mathcal{D}_M \sim P^{\otimes M}$ 统计各桶计数 $m_k$、正确数 $S_k$ 并构造估值 $\hat{\mathbf{s}} \in \mathbb{R}^K$，在线阶段从 $N$ 个无标签候选打分 $S_1, \dots, S_N \stackrel{\text{i.i.d.}}{\sim} F_{\text{mix}}$ 中选择最大化 $\hat{s}_{k(S_i)}$ 的样本（同分均匀随机打破）。

### 2.1 定理 6A：条件闭式公式、稀疏桶高估与 Fano 信息论下界

**定理 6A（精确平局闭式公式、MLE 失配与 $\beta$-光滑峰 Fano 下界）.**
1. **（条件精确 BoN 闭式公式）** 按 $\hat{s}_k$ 降序聚合成等价平局组 $G_1, \dots, G_L$，记组总概率 $P_\ell := \sum_{k \in G_\ell} \pi_k$，组真正确率 $\bar{q}_\ell := \frac{1}{P_\ell}\sum_{k \in G_\ell} \pi_k q_k$，严劣尾部总概率 $R_\ell := \sum_{j > \ell} P_j$，则对任意 $N \ge 1$ 精确成立：
   $$P(N \mid \mathcal{D}_M) = \sum_{\ell=1}^L \bar{q}_\ell \big[(R_\ell + P_\ell)^N - R_\ell^N\big], \qquad \lim_{N \to \infty} P(N \mid \mathcal{D}_M) = \bar{q}_1.$$
2. **（朴素 MLE 稀疏桶高估导致渐近上限降低与倒 U 风险）** 在朴素 MLE（$\hat{s}_k = S_k/m_k$）下，观测较少、真后验欠佳的稀疏尾桶 $k_0$（$q_{k_0} < q^*$）以正概率获满分估值 $\hat{s}_{k_0}=1$ 并进入最优组 $G_1$，从而压低大 $N$ 渐近上限至 $\bar{q}_1 < q^*$；当 $P_1 \ll P_{\text{main}}$ 且次优高质组在中等 $N$ 主导选样时，准确率随 $N \gtrsim 1/P_1$ 可呈倒 U 型回撤。
3. **（$\beta$-光滑极值分桶标度律与 Fano 信息论下界）** 在统一正则分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（定义见 §2.3，固定常数 $C_1, f_{\min}, \eta_0 > 0$ 与 $\beta > 0$）上，桶离散偏差 $O(K^{-\beta})$ 与极值统计偏差 $O\big(\sqrt{\frac{K\ln M}{M}}\big)$ 平衡解得最优分桶阶 $K^*(M) \asymp (\frac{M}{\ln M})^{\frac{1}{2\beta + 1}}$（$\beta=2$ 时为 $(\frac{M}{\ln M})^{1/5}$）；且对任何基于 $\mathcal{D}_M$ 的点估计器 $\hat{x}(\mathcal{D}_M)$，必存在类内成员满足：
   $$\inf_{\hat{x}} \sup_{P \in \mathcal{F}_\beta^{\text{unif}}} \Big(q^*_P - \mathbb{E}_{\mathcal{D}_M}\big[q_P(\hat{x}(\mathcal{D}_M))\big]\Big) = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right).$$

*证明.* (1)–(2) 由水平组排列公式直接计算即证。(3) 设 $2\eta_0 f_{\min} < 1$（保证支持长度容许滑动鼓包），对任意 $\beta > 0$，取支持于 $[-1, 1]$ 的对称核 $\psi(u) := \max(0, 1 - |u|^\beta)$，在 $[0, 1/f_{\min}]$ 的子区间 $[0.2/f_{\min},\, 0.8/f_{\min}]$ 上划分 $J = \Theta((h f_{\min})^{-1})$ 个互不相交中心 $t_1, \dots, t_J$（半径 $h < \eta_0$），令 $f_{\text{mix}} \equiv f_{\min}$ 并把剩余概率放在 $[0.8/f_{\min}, 1/f_{\min}]$ 上（或直接以 $f_{\text{mix}} \equiv f_{\min}$ 支承于 $[0, 1/f_{\min}]$），构造假设族 $q_j(x) := \frac{1}{2} + a_\beta h^\beta \psi\big(\frac{x-t_j}{h}\big)$，其中 $a_\beta \in (0, C_1]$ 使 $q_j \in [\frac{1}{4}, \frac{3}{4}]$（不与凸性常数 $c_\beta := \max(1, 2^{\beta-1})$ 冲突）。对每个 $j$ 及一切 $x$，精确成立 $q_j^* - q_j(x) = a_\beta \min(|x - t_j|^\beta,\, h^\beta) \le C_1 |x - t_j|^\beta$（$h^\beta$ 精确对消，不因 $h\to 0$ 破坏曲率上界），故全体假设严格落入 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$。单样本散度满足 $D_{\text{KL}}(P_j \| P_0) \le C_2 f_{\min} a_\beta^2 h^{2\beta+1}$，取 $h = c_h (\frac{\ln M}{M})^{\frac{1}{2\beta+1}}$ 使 $M D_{\text{KL}}(P_j\|P_0) \le \frac{1}{8}\ln J$，由广义 Fano 不等式即证。$\blacksquare$

---

### 2.2 定理 6B：非渐近 `KL-LCB` 悲观保底安全界与三事件遗憾定理

针对各非空分桶（$m_k \ge 1, \hat{q}_k = S_k/m_k$）定义联合置信水平 $1 - \delta_{\text{cov}}$ 的二项 **`KL-LCB`**（Garivier & Cappé 2011）：
$$\hat{s}_k^{\text{KL-LCB}}(\delta_{\text{cov}}) := \inf\left\{q \in [0, \hat{q}_k] : m_k\, \operatorname{kl}(\hat{q}_k,\, q) \le \ln\frac{K}{\delta_{\text{cov}}}\right\} \qquad (\text{空桶 }m_k=0\text{ 时置 }0).$$
定义该桶置信半宽为 $r_k(\delta_{\text{cov}}) := \max\big(\hat{q}_k - \hat{s}_k^{\text{KL-LCB}}(\delta_{\text{cov}}),\, \hat{s}_k^{\text{KL-UCB}}(\delta_{\text{cov}}) - \hat{q}_k\big)$。

> **覆盖保证与排序范围**：非渐近置信下界仅保证 $\hat{s}_k^{\text{KL-LCB}} \le q_k$ 同步成立，并不保证恢复全部桶间的真后验全序（如极稀疏高真均值桶 $m_A=1, q_A=0.90$ 可能排在中质多样本桶 $m_B=100, q_B=0.60$ 之后）。其数学作用是以高概率阻止真实质量劣于最高经验下界的低质稀疏桶进入 $G_1$，从而锁定悲观安全底线并控制简单遗憾。

**定理 6B（非渐近 `KL-LCB` 悲观保底安全界、三事件高概率遗憾与 Bernstein 阈值）.**
1. **（非渐近联合悲观安全底线）** 对任意有限分桶 $K \ge 1$、任意有限样本与任意 $\delta_1 \in (0, 1)$：在单侧安全事件 $\mathcal{E}_1 := \{\forall\, k \le K,\ \hat{s}_k^{\text{KL-LCB}}(\delta_1) \le q_k\}$ 上（$\mathbb{P}(\mathcal{E}_1) \ge 1 - \delta_1$），$\bar{q}_1 := \lim_{N\to\infty} P_{\text{LCB}}(N \mid \mathcal{D}_M) \ge \max_{1 \le k \le K} \hat{s}_k^{\text{KL-LCB}}(\delta_1)$；若以 $\delta_1/2$ 同步覆盖上下置信带（概率 $\ge 1 - \delta_1$），则 $\bar{q}_1 \ge \max_{1 \le k \le K}\big(q_k - 2 r_k(\delta_1/2)\big)$。
2. **（等频分桶三事件高概率遗憾界）** 若 $P \in \mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$，采用 $K > \frac{2}{\eta_0 f_{\min}}$ 阶等频分桶（$\pi_k = 1/K$），且给定 $\delta \in (0, 1)$ 与 $M \ge 8 K \ln(3/\delta)$。令单侧 `KL-LCB` 采用 $\delta_{\text{cov}} = \delta/3$（惩罚阈值 $\ln(3K/\delta)$），则以至少 $1 - \delta$ 的概率有：
   $$q^* - \bar{q}_1 \le C_1 (K f_{\min})^{-\beta} + 2\sqrt{\frac{K \ln(3K/\delta)}{M}}.$$
3. **（`Wilson-LCB` 近似与 `Empirical-Bernstein LCB` 归零阈值）** 将 $\operatorname{kl}(\hat{q}, q)$ 二阶展开代以 $\frac{z^2}{2}$（$z=\Phi^{-1}(1-\frac{\delta}{2K})$）得 `Wilson-LCB`；而经验 Bernstein LCB 对给定经验桶均值 $\hat{q}_k > 0$ 的**确定性归零条件**为 $m_k < 1 + \frac{7}{3\,\hat{q}_k} \ln(\frac{2K}{\delta}) \implies \hat{s}_k^{\text{Bern-LCB}} = 0$。

*证明.* (1) 由 Cramér–Chernoff 上界与 $K$ 桶并界立证 $\mathbb{P}(\mathcal{E}_1) \ge 1 - \delta_1$，组内加权平均得下界。(2) 结合最优桶 $k^*$ 样本量乘法 Chernoff 界 $\mathbb{P}(m_{k^*} < \frac{M}{2K}) \le e^{-M/(8K)} \le \delta/3$、Hoeffding 下侧偏差 $q_{k^*} - \hat{q}_{k^*} \le \sqrt{\frac{\ln(3K/\delta)}{2 m_{k^*}}}$（概率 $\le \delta/3$）及 Pinsker 不等式 $\hat{q}_{k^*} - \hat{s}_{k^*}^{\text{KL-LCB}} \le \sqrt{\frac{\ln(3K/\delta)}{2 m_{k^*}}}$，由三事件并界完证。(3) 直接比对线性惩罚项 $\frac{7\ln(2K/\delta)}{3(m_k-1)} > \hat{q}_k$ 即得。$\blacksquare$

---

### 2.3 定理 6C：一致正则类 $\mathcal{F}_\beta^{\text{unif}}$、联合 $(M, N)$ Minimax 定理与预算平衡律

为区分逐分布渐近收敛与一致 Minimax 收敛，定义两类分布空间：

- **（逐点渐近类 $\mathcal{F}_\beta^{\text{sep}}$，固定单个分布 $P$）**：具有连续正混合密度（$f_{\text{mix}}(x^*) = f^* > 0$），在唯一全局峰值 $x^*$ 邻域满足局部 $\beta$-阶渐近 $q^* - q(x) = C |x - x^*|^\beta (1 + o(1))$（$\beta > 0, C > 0$），且对任意 $\eta > 0$ 存在单分布分离常数 $\varepsilon_\eta > 0$ 使 $\sup_{|x-x^*|\ge\eta} q(x) \le q^* - \varepsilon_\eta$。
- **（一致正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$，固定正类常数 $C_1, f_{\min}, \eta_0 > 0$ 与阶数 $\beta > 0$）**：分布 $P$ 具有正混合密度 $f_{\text{mix}}$（$q^* := \sup_x q(x)$），并存在全局最大值点 $x^*$（$q(x^*)=q^*$）使得在公共邻域 $U_{\eta_0} := [x^* - \eta_0, x^* + \eta_0]$ 上一致成立：
  $$f_{\text{mix}}(x) \ge f_{\min} > 0, \qquad q^* - q(x) \le C_1 |x - x^*|^\beta \qquad (\forall\, x \in U_{\eta_0}).$$

> **类定义必要性与上下界相容性说明**：
> 1. **为何必须对 $(C_1, f_{\min}, \eta_0)$ 施加公共约束**：若仅逐分布要求局部曲率存在（如 $\mathcal{F}_\beta^{\text{sep}}$），窄峰族 $S\sim\mathcal{N}(0,1),\ q_h(s)=\frac{1}{4}+\frac{1}{2}e^{-s^2/h^2}$ 均属于 $\mathcal{F}_2^{\text{sep}}$，但固定 $N$ 下 $\mathcal{R}_h(N) \ge \frac{1}{2} - \frac{Nh}{2\sqrt{h^2+2}} \xrightarrow{h\to 0^+} \frac{1}{2}$，无一致收敛率；固定上界常数 $C_1$ 排除了峰宽缩至零的退化序列。
> 2. **为何一致类不能施加全局分离间隙 $\varepsilon_0 > 0$**：若强加公共全局高差 $\varepsilon_0 > 0$，则 Fano 下界中高度 $\Delta = c_\beta h^\beta \to 0$ 的鼓包假设族将被排除。在 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 中，上界仅凭 $U_{\eta_0}$ 内密度下界 $f_{\min}$ 与 $C_1|x-x^*|^\beta$ 锁定最近峰样本 $D_N \le \eta_0/2$，同时一切 $c_\beta \le C_1$ 的 Fano 鼓包假设均严格落入同一类内。

**定理 6C（逐点 $N^{-\beta}$ 极限、一致正则类联合 $(M, N)$ Minimax 最优性与校准—算力平衡律）.**
1. **（逐点精确渐近收敛主项 $\Theta(N^{-\beta})$）** *对任意固定分布 $P \in \mathcal{F}_\beta^{\text{sep}}$（$\beta > 0$），完美后验排序（$M=\infty$）精确满足逐点渐近恒等式：*
   $$q^*_P - P_{\text{Cal-BoN}}(N) = \big(1 + o(1)\big)\, \frac{\Gamma(\beta + 1)\, C}{(2 f^*)^\beta}\, N^{-\beta} \qquad \Big(\beta = 2\text{ 时为 }\frac{C}{2(f^*)^2}\, N^{-2}\Big).$$
2. **（一致正则类 $\mathcal{F}_\beta^{\text{unif}}$ 上静态等频 `KL-LCB` 的非渐近显式上界——覆盖所有 $\beta > 0$）** *设分布属于 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（$\beta > 0$），记凸性常数 $c_\beta := \max(1,\, 2^{\beta - 1})$。在总体分位数等频分桶 $K^*(M) = \lfloor c_0 (M/\ln M)^{\frac{1}{2\beta+1}}\rfloor$（满足 $M \ge 8 K \ln(3KM)$ 且 $K > \frac{2}{\eta_0 f_{\min}}$）及置信水平 $\delta_{\text{cov}} = \frac{1}{3M}$ 参数化的 `KL-LCB` 下，对一切合法 $(M, N)$ 一致成立非渐近期望遗憾上界：*
   $$\sup_{P \in \mathcal{F}_\beta^{\text{unif}}}\Big(q^*_P - \mathbb{E}_{\mathcal{D}_M}\big[P_{\text{KL-LCB}}(N \mid \mathcal{D}_M)\big]\Big) \le \frac{1}{M} + e^{-\eta_0 f_{\min} N} + \frac{C_1 c_\beta}{(K f_{\min})^\beta} + 2\sqrt{\frac{K \ln(3KM)}{M}} + \frac{C_1 c_\beta\, \Gamma(\beta + 1)}{(2 f_{\min})^\beta}\, N^{-\beta} = O\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}} + N^{-\beta}\right).$$
3. **（全策略空间联合 $(M, N)$ 信息论 Minimax 下界与预算平衡律 $N_{\text{balance}}(M)$）** *对同一类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（固定支持使对照分布 $P^* \in \mathcal{F}_\beta^{\text{unif}} \cap \mathcal{F}_\beta^{\text{sep}}$ 与 Fano 鼓包族同时落入类内），考虑**任意**可联合使用离线有标签样本 $\mathcal{D}_M$ 与在线无标签打分 $(S_1, \dots, S_N)$ 选出下标 $i^* \in \{1,\dots,N\}$ 的策略 $\pi(\mathcal{D}_M, S_{1:N})$：*
   $$\inf_{\pi}\ \sup_{P \in \mathcal{F}_\beta^{\text{unif}}} \Big(q^*_P - \mathbb{E}_{P^{\otimes(M+N)}}\big[q_P(S_{i^*})\big]\Big) = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}} + N^{-\beta}\right).$$
   *令两项主导阶相等，即导出测试时计算缩放的**校准—算力预算平衡律（Calibration–Compute Balance Law）**：*
   $$N_{\text{balance}}(M) \asymp \left(\frac{M}{\ln M}\right)^{\frac{1}{2\beta + 1}} \asymp K^*(M) \qquad \Big(\beta = 2\text{ 光滑内点峰时为 }\left(\frac{M}{\ln M}\right)^{1/5}\Big).$$
   *当 $N \ll N_{\text{balance}}(M)$ 时系统受限于在线候选覆盖密度（算力瓶颈相，$\mathcal{R} \asymp N^{-\beta}$）；当 $N \gg N_{\text{balance}}(M)$ 时继续盲目扩大在线采样预算 $N$ 的边际收益饱和，系统转为受限于离线校准样本量 $M$（校准瓶颈相，$\mathcal{R} \asymp (\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$）。*

*定理 6C 的完整证明.*
- **(1) 逐点渐近常数**：在 $P \in \mathcal{F}_\beta^{\text{sep}}$ 上，对充分小 $\eta > 0$ 与任意 $\varepsilon \in (0, \varepsilon_\eta)$，全局分离保证 $\{x : q(x) > q^* - \varepsilon\} \subset U_\eta$，故 $\bar{F}_Q(q^*-\varepsilon) = (1+o(1))\frac{2f^*}{C^{1/\beta}}\varepsilon^{1/\beta}$。将积分拆为 $[0, N^\beta\varepsilon_\eta]$ 与 $[N^\beta\varepsilon_\eta, N^\beta q^*]$，后段指数衰减为 $o(1)$，前段依控制收敛趋于 $\int_0^\infty e^{-c_0 u^{1/\beta}} du = \frac{\Gamma(\beta+1)C}{(2f^*)^\beta}$。
- **(2) 一致正则类上三事件并界与最近峰耦合**：对 $K > \frac{2}{\eta_0 f_{\min}}$，每个完全落于 $U_{\eta_0} = [x^*-\eta_0, x^*+\eta_0]$ 内的等频桶长度至多为 $\frac{1}{K f_{\min}} < \frac{\eta_0}{2}$。定义最近峰在线样本下标与桶指标：
  $$i^{\text{near}} := \arg\min_{1 \le i \le N} |S_i - x^*|, \qquad D_N := |S_{i^{\text{near}}} - x^*|, \qquad k^{\text{near}} := k(S_{i^{\text{near}}}), \qquad \hat{k}^* := k(S_{\hat{i}^*}),$$
  及过滤代数 $\mathcal{F}_{\text{bins}} := \sigma(\mathcal{D}_M,\, k(S_1),\dots,k(S_N))$。因给定桶序列后在线同桶样本彼此独立且均服从该桶条件分布，塔性质给出 $\mathbb{E}[Y_{\hat{i}^*} \mid \mathcal{F}_{\text{bins}}] = q_{\hat{k}^*}$。划分**三个离线失效事件**（每个概率 $\le \frac{1}{3M}$）：
  1. $\mathcal{B}_1 := \{\exists k \le K:\ \hat{s}_k^{\text{KL-LCB}}(\frac{1}{3M}) > q_k\}$：由 Cramér–Chernoff 界与并界，$\mathbb{P}(\mathcal{B}_1) \le K \cdot \frac{1}{3KM} = \frac{1}{3M}$；
  2. $\mathcal{B}_2 := \{\exists k \le K:\ m_k < \frac{M}{2K}\}$：$m_k \sim \operatorname{Binomial}(M, 1/K)$，在 $M \ge 8 K \ln(3KM)$ 下乘法 Chernoff 界给出 $\mathbb{P}(\mathcal{B}_2) \le K e^{-M/(8K)} \le \frac{1}{3M}$；
  3. $\mathcal{B}_3 := \{\exists k \le K:\ m_k \ge 1\ \text{且}\ q_k - \hat{q}_k > \sqrt{\frac{\ln(3KM)}{2 m_k}}\}$：条件于 $m_k \ge 1$ 由 Hoeffding 不等式得条件概率 $\le \frac{1}{3KM}$，取期望再并界得 $\mathbb{P}(\mathcal{B}_3) \le \frac{1}{3M}$。
  合记 $\mathcal{B}_{\text{off}} := \mathcal{B}_1 \cup \mathcal{B}_2 \cup \mathcal{B}_3$（$\mathbb{P}(\mathcal{B}_{\text{off}}) \le \frac{1}{M}$），在线越界事件 $\mathcal{B}_{\text{miss}} := \{D_N > \frac{\eta_0}{2}\}$（$\mathbb{P}(\mathcal{B}_{\text{miss}}) \le (1-\eta_0 f_{\min})^N \le e^{-\eta_0 f_{\min} N}$），好事件 $\mathcal{G} := \mathcal{B}_{\text{off}}^c \cap \mathcal{B}_{\text{miss}}^c$。在 $\mathcal{G}$ 上，桶 $k^{\text{near}} \subset U_{\eta_0}$ 且桶内点距 $x^*$ 至多为 $D_N + \frac{1}{Kf_{\min}}$，由 $(a+b)^\beta \le c_\beta(a^\beta + b^\beta)$（$\forall \beta > 0$）得 $q^* - q_{k^{\text{near}}} \le C_1 c_\beta D_N^\beta + \frac{C_1 c_\beta}{(K f_{\min})^\beta}$。同时由 $\mathcal{B}_1^c, \mathcal{B}_2^c, \mathcal{B}_3^c$ 及 Pinsker 不等式得：
  $$q_k - \hat{s}_k^{\text{KL-LCB}} = (q_k - \hat{q}_k) + (\hat{q}_k - \hat{s}_k^{\text{KL-LCB}}) \le 2\sqrt{\frac{\ln(3KM)}{2 m_k}} = \sqrt{\frac{2\ln(3KM)}{m_k}} \le 2\sqrt{\frac{K\ln(3KM)}{M}},$$
  故 $q_{\hat{k}^*} \ge \hat{s}_{\hat{k}^*}^{\text{KL-LCB}} \ge \hat{s}_{k^{\text{near}}}^{\text{KL-LCB}} \ge q_{k^{\text{near}}} - 2\sqrt{\frac{K\ln(3KM)}{M}}$。最后利用 $\mathbb{E}[\mathbf{1}_{\mathcal{G}} D_N^\beta] \le \int_0^{\eta_0/2} \beta r^{\beta-1} e^{-2f_{\min} N r} dr \le \frac{\Gamma(\beta+1)}{(2f_{\min})^\beta} N^{-\beta}$ 汇总五项即得一致上界。
- **(3) 全策略 $(M, N)$ Minimax 下界与平衡律**：在定理 6A 均匀边际鼓包族 $\{P_1, \dots, P_J\} \subset \mathcal{F}_\beta^{\text{unif}}$ 上，因无标签在线分数 $S_{1:N} \sim \operatorname{Uniform}(0,1)^N$ 的边际分布与假设编号 $j$ 统计独立，联合积测度散度精确满足 $D_{\text{KL}}(P_j^{\otimes M}\otimes \operatorname{Unif}^N \| P_0^{\otimes M}\otimes \operatorname{Unif}^N) = M D_{\text{KL}}(P_j\|P_0)$（完全不随 $N$ 增加信息量），广义 Fano 不等式给出 $\Omega((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}})$；再在单分布 $P^* \in \mathcal{F}_\beta^{\text{unif}} \cap \mathcal{F}_\beta^{\text{sep}}$ 上由第 (1) 步 Oracle 极值间距给出 $\Omega(N^{-\beta})$。合并两项并令两主导速率相等即得 $N_{\text{balance}}(M) \asymp (\frac{M}{\ln M})^{\frac{1}{2\beta+1}}$。$\blacksquare$

---

### 2.4 命题 6.1：未知总体分位数下样本分割经验等频分桶的全 $\beta > 0$ 一致浓缩定理

**命题 6.1（样本分割顺序统计量分桶的紧 Cramér 间距界与精确惩罚放大常数）.** *在实际未知连续总体分布函数 $F_{\text{mix}}$ 且不引入额外无标签样本时，若朴素调用加性 DKW 不等式 $|\hat{\pi}_k - \frac{1}{K}| = O(\sqrt{\frac{\ln M}{M}})$，则相对误差阶 $K^*(M)\sqrt{\frac{\ln M}{M}} \asymp (\frac{M}{\ln M})^{\frac{1-2\beta}{2(2\beta+1)}}$ 在粗糙峰区域 $0 < \beta \le \frac{1}{2}$ 不趋于零（当 $0<\beta<\frac{1}{2}$ 时发散）。改用如下**纯 $M$-预算样本分割顺序统计量估计器**：将 $\mathcal{D}_M$ 对半划分为独立边界子集 $\mathcal{D}_{M_1}^{\text{edge}}$（$M_1 = \lfloor M/2\rfloor$）与估计子集 $\mathcal{D}_{M_2}^{\text{est}}$（$M_2 = M - M_1 \ge M/2$）；对 $\mathcal{D}_{M_1}^{\text{edge}}$ 升序统计量 $S_{(1)}^{\text{edge}} < \dots < S_{(M_1)}^{\text{edge}}$ 取内部节点下标 $r_k := \lfloor \frac{k(M_1 + 1)}{K}\rfloor$（$k=1,\dots,K-1$），定义经验桶 $\hat{B}_k := (\hat{e}_{k-1}, \hat{e}_k]$（$\hat{e}_0=-\infty,\ \hat{e}_k = S_{(r_k)}^{\text{edge}},\ \hat{e}_K=+\infty$），并在独立的 $\mathcal{D}_{M_2}^{\text{est}}$ 上以置信水平 $\delta_{\text{cov}} = \frac{1}{4M}$ 计算 `KL-LCB`。则对**一切 $\beta > 0$**：*
1. **（紧 Cramér 乘性质量界）** *只要 $M_1 + 1 \ge \max\big(12 K,\, 214\ln(16 M)\big)$ 且最小阶差 $d_{\min} := \min_{1\le k\le K}(r_k - r_{k-1}) \ge 9.1 \ln(16 K M)$（对 $K^*(M) \asymp (\frac{M}{\ln M})^{\frac{1}{2\beta+1}}$，因 $\frac{1}{2\beta+1} < 1 \iff \beta > 0$ 在一切 $\beta > 0$ 下随 $M$ 增大自动满足，例如 $\beta=0.25, M=2.5\times 10^6, K=3{,}065$ 时 $d_{\min}=407 \ge 232.34$ 已直接满足完整 $\delta = \frac{1}{4M}$ 条件），以至少 $1 - \frac{1}{4M}$ 的概率对全体 $k \in \{1,\dots,K\}$ 一致成立：*
   $$\frac{1}{2K} \le \hat{\pi}_k := F_{\text{mix}}(\hat{e}_k) - F_{\text{mix}}(\hat{e}_{k-1}) \le \frac{143}{72 K} < \frac{2}{K};$$
2. **（四事件并界与精确常数放大因子）** *当同时满足 $M_2 \ge 16 K \ln(4 K M)$（即 $M \ge 32 K \ln(4 K M)$）且 $K > \frac{4}{\eta_0 f_{\min}}$ 时，对一切 $P \in \mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 成立非渐近上界：*
   $$q^*_P - \mathbb{E}_{\mathcal{D}_M}\big[P_{\text{Split-KL-LCB}}(N \mid \mathcal{D}_M)\big] \le \frac{1}{M} + e^{-\eta_0 f_{\min} N} + \frac{2^\beta C_1 c_\beta}{(K f_{\min})^\beta} + 4\sqrt{\frac{K \ln(4KM)}{M}} + \frac{C_1 c_\beta\, \Gamma(\beta + 1)}{(2 f_{\min})^\beta}\, N^{-\beta}.$$
   *（注：相较于定理 6C(2) 已知总体分位数情形，样本分割使落入 $U_{\eta_0}$ 的经验桶宽度上界由 $\frac{1}{K f_{\min}}$ 放宽至 $\frac{2}{K f_{\min}}$ 带来 **$2^\beta$** 倍离散偏差常数，而桶样本数下界由 $m_k \ge \frac{M}{2K}$ 缩减为 $m_k \ge \frac{M_2}{4K} \ge \frac{M}{8K}$ 带来 $\sqrt{\frac{M/(2K)}{M/(8K)}} = \mathbf{2}$ 倍统计惩罚常数——由 $2\sqrt{\frac{K\ln(\cdot)}{M}}$ 变为 $4\sqrt{\frac{K\ln(\cdot)}{M}}$——从而在一切 $\beta > 0$ 上严格保持同一 Minimax 阶 $O\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$。）*

*证明.* (1) 因 $F_{\text{mix}}$ 连续，$U_i := F_{\text{mix}}(S_i^{\text{edge}}) \stackrel{\text{i.i.d.}}{\sim} \operatorname{Uniform}(0, 1)$，$\hat{\pi}_k = U_{(r_k)} - U_{(r_{k-1})}$（$U_{(0)}=0, U_{(M_1+1)}=1$）。由 Rényi–Pyke 指数间距表示，存在 $E_1, \dots, E_{M_1+1} \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(1)$ 使 $(\hat{\pi}_1, \dots, \hat{\pi}_K) \stackrel{d}{=} (G_1/G_{\text{tot}}, \dots, G_K/G_{\text{tot}})$，其中 $G_k := \sum_{j=r_{k-1}+1}^{r_k} E_j \sim \operatorname{Gamma}(d_k, 1)$，$G_{\text{tot}} := \sum_{j=1}^{M_1+1} E_j \sim \operatorname{Gamma}(M_1 + 1, 1)$。当 $M_1 + 1 \ge 12 K$ 时，$\frac{11}{12}\frac{M_1+1}{K} \le \lfloor\frac{M_1+1}{K}\rfloor \le d_k \le \lceil\frac{M_1+1}{K}\rceil \le \frac{13}{12}\frac{M_1+1}{K}$。对 $G \sim \operatorname{Gamma}(d, 1)$ 应用精确 Cramér–Chernoff 大偏差界 $\mathbb{P}(G < \alpha d) \le e^{-\psi_-(\alpha)d}$ 与 $\mathbb{P}(G > \beta d) \le e^{-\psi_+(\beta)d}$（$\psi(c) := c - 1 - \ln c$）：
- 对各 $G_k$ 取 $[\alpha, \beta] = [0.60, 1.65]$：$\psi_-(0.60) = -0.40 - \ln 0.60 \approx 0.11082 > \frac{1}{9.1}$，$\psi_+(1.65) = 0.65 - \ln 1.65 \approx 0.14922 > \frac{1}{9.1}$；
- 对 $G_{\text{tot}}$ 取 $[\alpha_{\text{tot}}, \beta_{\text{tot}}] = [0.90, 1.10]$：$\psi_-(0.90) = -0.10 - \ln 0.90 \approx 0.005360 > \frac{1}{214}$，$\psi_+(1.10) = 0.10 - \ln 1.10 \approx 0.0046898 > \frac{1}{214}$。
因此在 $d_{\min} \ge 9.1\ln(16KM)$ 与 $M_1 + 1 \ge 214\ln(16M)$ 下，事件 $\mathcal{E}_{\text{sp}} := \{\forall k,\ 0.60 d_k \le G_k \le 1.65 d_k\} \cap \{0.90(M_1+1) \le G_{\text{tot}} \le 1.10(M_1+1)\}$ 的余集概率至多为 $K(\frac{1}{16KM} + \frac{1}{16KM}) + (\frac{1}{16M} + \frac{1}{16M}) = \frac{1}{4M}$。在 $\mathcal{E}_{\text{sp}}$ 上对一切 $k$ 同步成立：
$$\hat{\pi}_k = \frac{G_k}{G_{\text{tot}}} \ge \frac{0.60 \times \frac{11}{12}\frac{M_1+1}{K}}{1.10(M_1 + 1)} = \frac{1}{2K}, \qquad \hat{\pi}_k \le \frac{1.65 \times \frac{13}{12}\frac{M_1+1}{K}}{0.90(M_1 + 1)} = \frac{143}{72 K} < \frac{2}{K}.$$
(2) 条件于满足 $\mathcal{E}_{\text{sp}}$ 的独立第一半样本 $\mathcal{D}_{M_1}^{\text{edge}}$，每个落入 $U_{\eta_0}$ 的经验桶长度至多为 $\frac{\hat{\pi}_k}{f_{\min}} < \frac{2}{K f_{\min}} < \frac{\eta_0}{2}$，且第二半样本中 $m_k \sim \operatorname{Binomial}(M_2, \hat{\pi}_k)$ 的条件均值满足 $\mathbb{E}[m_k \mid \mathcal{D}_{M_1}^{\text{edge}}] = M_2 \hat{\pi}_k \ge \frac{M_2}{2K}$。当 $M_2 \ge 16 K \ln(4KM)$ 时，乘法 Chernoff 界给出 $\mathbb{P}(\exists k:\ m_k < \frac{M_2}{4K} \mid \mathcal{D}_{M_1}^{\text{edge}}) \le K e^{-M_2/(16K)} \le \frac{1}{4M}$。将间距失效事件 $\mathcal{E}_{\text{sp}}^c$（$\le \frac{1}{4M}$）与以 $\frac{1}{4M}$ 参数化的三个估计失效事件（LCB 上越界、桶计数 $< \frac{M_2}{4K}$、Hoeffding 下侧偏差 $> \sqrt{\frac{\ln(4KM)}{2m_k}}$，各 $\le \frac{1}{4M}$）取四事件并界得总离线失效概率 $\le \frac{1}{M}$；在好事件上由 $m_k \ge \frac{M_2}{4K} \ge \frac{M}{8K}$ 代入统计项 $\sqrt{\frac{2\ln(4KM)}{m_k}} \le \sqrt{\frac{16 K \ln(4KM)}{M}} = 4\sqrt{\frac{K\ln(4KM)}{M}}$ 即证。$\blacksquare$

---

## 3. 主定理 II：跨题异质性、在线样本标准化尾反转与风险控制门控（定理 7 & 命题 7.1）

回到 §1.1 的层次生成模型：$X \sim \mathcal{P}_X$，给定 $X$ 抽取 $(Y_i, S_i) \stackrel{\text{i.i.d.}}{\sim} P_{Y, S \mid X}$，题内先验 $p(X) := \mathbb{P}(Y=1 \mid X)$，题内混合密度 $f_{\text{mix}, X}(s)$，题内真实后验 $q_X(s) := \mathbb{P}(Y=1 \mid S=s, X)$。跨题池化密度与池化后验定义为：
$$f_{\text{pool}}(s) := \mathbb{E}_X\big[f_{\text{mix}, X}(s)\big], \qquad q_{\text{pool}}(s) := \mathbb{P}(Y=1 \mid S=s) = \mathbb{E}\big[q_X(s) \mid S=s\big].$$

### 3.1 定理 7A：辛普森混杂导数分解与无网格解析伪单调遮蔽

**定理 7A（池化后验导数的辛普森精确分解与闭区间纯解析伪单调遮蔽构造）.** *假设对 $\mathcal{P}_X$-a.e. $X$，密度 $f_{c, X}(s)$ 在 $s$ 上连续可微正定且满足积分号下求导正则性，记题内对数混合密度得分函数 $g_X(s) := \frac{\partial}{\partial s}\ln f_{\text{mix}, X}(s)$。*
1. **（精确分解恒等式）** *全局池化后验导数精确等于题内斜率条件期望与跨题难度协方差之和：*
   $$q'_{\text{pool}}(s) = \underbrace{\mathbb{E}\big[\partial_s q_X(s) \;\big|\; S = s\big]}_{\text{题内真斜率平均}} + \underbrace{\operatorname{Cov}\big(Y,\ g_X(s) \;\big|\; S = s\big)}_{\text{跨题难度混杂协方差}}.$$
2. **（二层有限参数闭区间纯解析遮蔽构造）** *取共享异方差高斯基底 $\mu_{1,\text{base}} = 0.3,\ \mu_{0,\text{base}} = 0.0,\ \sigma_1 = 1.0,\ \sigma_0 = 2.0$（$\kappa = 2 > 1$，基底峰值 $x_{\text{peak,base}} = 0.4$），层权重 $(w_1, w_2) = (0.85, 0.15)$，先验 $(p_1, p_2) = (0.05, 0.95)$，位置平移 $(b_1, b_2) = (0.0, 3.0)$（两层题内峰值分别为 $x_{\text{peak}, 1} = 0.4$ 与 $x_{\text{peak}, 2} = 3.4$）。则在连续闭区间 $s \in [3.5, 4.2]$ 上**无需任何网格离散检验**即可纯解析证明处处成立：*
   $$\partial_s q_1(s) < -1.10 \times 10^{-3} < 0, \qquad \partial_s q_2(s) < -1.85 \times 10^{-3} < 0, \qquad q'_{\text{pool}}(s) > +7.03 \times 10^{-3} > 0.$$

*完全解析证明（零网格扫描）.*
- **(1) 协方差分解**：记后验层密度权重 $w(x\mid s) := \frac{f_{\text{mix},x}(s)}{f_{\text{pool}}(s)}$，求导得 $\partial_s w(x\mid s) = w(x\mid s)\big(g_x(s) - \mathbb{E}[g_X(s)\mid S=s]\big)$。对 $q_{\text{pool}}(s)=\int w(x\mid s)q_x(s)d\mathcal{P}_X(x)$ 求导并应用塔性质 $\operatorname{Cov}(q_X(s), g_X(s)\mid S=s)=\operatorname{Cov}(Y, g_X(s)\mid S=s)$ 立证。
- **(2) 闭区间 $[3.5, 4.2]$ 上的纯解析端点单调上下界**：在二层情形下，两点分布协方差恒等于 $\operatorname{Cov}(Y, g_X(s)\mid S=s) = w_1(s)w_2(s)\big(q_2(s)-q_1(s)\big)\big(g_2(s)-g_1(s)\big)$。对任意 $s \in [3.5, 4.2]$ 逐项锁定解析界：
  1. **题内后验单调区间界与斜率上界**：直接计算两层对数似然比导数：
     $$\ell'_1(s) = -(s - 0.3) + \frac{s}{4} = -0.75(s - 0.4) \in [-2.85,\, -2.325] < 0, \qquad \ell'_2(s) = -(s - 3.3) + \frac{s - 3.0}{4} = -0.75(s - 3.4) \in [-0.60,\, -0.075] < 0.$$
     因 $\ell'_1, \ell'_2$ 在 $[3.5, 4.2]$ 上处处严格为负，$q_1(s)$ 与 $q_2(s)$ 在整个连续区间 $[3.5, 4.2]$ 上均**严格单调递减**，故区间取值范围由闭区间两端点精确闭式锁定：
     $$q_1(s) \in \big[q_1(4.2),\, q_1(3.5)\big] = \big[4.750\times 10^{-4},\; 2.900\times 10^{-3}\big], \qquad q_2(s) \in \big[q_2(4.2),\, q_2(3.5)\big] = \big[0.96810,\; 0.97464\big],$$
     由此立即给出差值精确下界 $q_2(s) - q_1(s) \ge q_2(4.2) - q_1(3.5) > 0.96519$。同时注意在 $(0.5, 1)$ 上函数 $u \mapsto u(1-u)$ **严格单调递减**，故 $q_2(s)\big(1-q_2(s)\big)$ 在 $[3.5, 4.2]$ 上的最小值于左端点 $s=3.5$ 取得、最大值于右端点 $s=4.2$ 取得；而 $q_1(s) < 0.5$ 时 $u(1-u)$ 严格递增。代入 $\partial_s q_g(s) = q_g(s)(1-q_g(s))\ell'_g(s)$ 得：
     $$\partial_s q_1(s) \le -q_1(4.2)\big(1-q_1(4.2)\big)\times 2.325 < -1.10\times 10^{-3} < 0, \qquad |\partial_s q_1(s)| \le q_1(3.5)\times 2.85 < 0.00827,$$
     $$\partial_s q_2(s) \le -0.075\, q_2(3.5)\big(1-q_2(3.5)\big) \approx -1.8539\times 10^{-3} < -1.85\times 10^{-3} < 0, \qquad |\partial_s q_2(s)| \le 0.60\, q_2(4.2)\big(1-q_2(4.2)\big) < 0.60\big(1 - q_2(4.2)\big) < 0.01915.$$
     因凸组合满足 $|\mathbb{E}[\partial_s q_X(s)\mid S=s]| \le \max\big(|\partial_s q_1(s)|, |\partial_s q_2(s)|\big)$，故题内平均斜率绝对值处处严格小于 $0.01915$。
  2. **后验层权重乘积 $w_1(s)w_2(s)$ 解析下界**：因四个高斯分量均值 $\{0.3, 0.0, 3.3, 3.0\}$ 全部严格小于 $3.5$，两层混合密度 $f_{\text{mix}, 1}(s)$ 与 $f_{\text{mix}, 2}(s)$ 在 $[3.5, 4.2]$ 上均**严格单调递减**。从而密度比 $r(s) := \frac{w_2 f_{\text{mix}, 2}(s)}{w_1 f_{\text{mix}, 1}(s)}$ 处处受控于端点交叉比：
     $$r(s) \in \left[\frac{0.15\, f_{\text{mix}, 2}(4.2)}{0.85\, f_{\text{mix}, 1}(3.5)},\; \frac{0.15\, f_{\text{mix}, 2}(3.5)}{0.85\, f_{\text{mix}, 1}(4.2)}\right] \subset [1.1211,\; 3.2180].$$
     因函数 $r \mapsto \frac{r}{(1+r)^2}$ 在 $[1, +\infty)$ 上严格单调递减，必有 $w_1(s)w_2(s) = \frac{r(s)}{(1+r(s))^2} \ge \frac{3.2180}{(4.2180)^2} > 0.18087$。
  3. **对数密度得分差 $g_2(s) - g_1(s)$ 解析下界**：对每一层 $g \in \{1, 2\}$，得分函数 $g_g(s) = \frac{f'_{\text{mix}, g}(s)}{f_{\text{mix}, g}(s)}$ 是正类高斯得分 $-(s - b_g - 0.3)$ 与错类高斯得分 $-\frac{s - b_g}{4}$ 的凸组合。在 $s \in [3.5, 4.2]$ 上：
     - 对第 1 层（$b_1 = 0$）：因 $s - 0.3 > \frac{s}{4} \iff 0.75 s > 0.3$ 恒成立，两分量中较大者为 $-\frac{s}{4}$，故 $g_1(s) \le -0.25 s$；
     - 对第 2 层（$b_2 = 3.0$）：因 $(s - 3.3) - \frac{s - 3.0}{4} = 0.75(s - 3.4) > 0$ 在 $s \ge 3.5$ 上恒成立，两分量中较小者为 $-(s - 3.3)$，故 $g_2(s) \ge -(s - 3.3)$。
     两式相减即得线性解析下界：
     $$g_2(s) - g_1(s) \ge -(s - 3.3) - (-0.25 s) = 3.3 - 0.75 s \ge 3.3 - 0.75 \times 4.2 = +0.15 > 0.$$
  4. **汇总符号判定**：将三项解析正下界相乘得 $\operatorname{Cov}(Y, g_X(s)\mid S=s) > 0.18087 \times 0.96519 \times 0.15 > 0.02618$，从而对**一切**连续点 $s \in [3.5, 4.2]$ 成立：
     $$q'_{\text{pool}}(s) \ge \operatorname{Cov}(Y, g_X(s)\mid S=s) - \big|\mathbb{E}[\partial_s q_X(s)\mid S=s]\big| > 0.02618 - 0.01915 = +7.03 \times 10^{-3} > 0,$$
     证毕。$\blacksquare$

---

### 3.2 定理 7B：题内 Z-Score 标准化的非退化条件、纯秩 $N=2$ 天花板与异质先验残差

考虑仿射题异质生成模型：$S_i = b(X) + a(X)\tilde{S}_i$（$a(X) > 0$，共享基底分布 $\tilde{S}_i \mid (Y_i=c, X) \sim \tilde{F}_c$ 满足**有限四阶矩 $\mathbb{E}[\tilde{S}^4 \mid X] < \infty$** 及**严格正条件方差非退化条件 $\tilde{\sigma}^2(X) := \operatorname{Var}(\tilde{S} \mid X) > 0$（$\mathcal{P}_X$-a.s.）**；基底混合均值 $\tilde{\mu}(X) := \tilde{\mu}_0 + p(X)(\tilde{\mu}_1 - \tilde{\mu}_0)$，基底混合方差 $\tilde{\sigma}^2(X) = p(X)\tilde{\sigma}_1^2 + (1-p(X))\tilde{\sigma}_0^2 + p(X)(1-p(X))(\tilde{\mu}_1-\tilde{\mu}_0)^2 > 0$，基底条件后验记为 $\tilde{q}_X(\tilde{s}) := \mathbb{P}(Y_i = 1 \mid \tilde{S}_i = \tilde{s}, X) \equiv q_X(b(X) + a(X)\tilde{s})$）。对单题内抽取的 $N \ge 2$ 个分数定义样本矩 $\bar{S}_N := \frac{1}{N}\sum_{i=1}^N S_i$、$\hat{s}_N := \sqrt{\frac{1}{N-1}\sum_{i=1}^N (S_i - \bar{S}_N)^2}$，并在有限样本下显式约定零标准差平局规则（同分最大值下标一律均匀随机打破）：
$$Z_{i, N} := \begin{cases} \dfrac{S_i - \bar{S}_N}{\hat{s}_N}, & \hat{s}_N > 0, \\[4pt] 0, & \hat{s}_N = 0\ (\text{即 }S_1 = \dots = S_N). \end{cases}$$

**定理 7B（题内 Z-Score 的先验残差边界与 $N=2$ 纯秩规则非严格界及充要判据）.**
1. **（固定 $X$ 逐下标收敛与异质先验 $p(X)$ 残差）** *在 $\mathbb{E}[\tilde{S}^4 \mid X] < \infty$ 与 $\tilde{\sigma}^2(X) > 0$（$\mathcal{P}_X$-a.s.）下，对 $\mathcal{P}_X$-a.e. 固定题目 $X$ 与固定下标 $i$，当 $N \to \infty$ 时 $\mathbb{P}(\hat{s}_N = 0 \mid X) \to 0$，$Z_{i, N} \xrightarrow{\text{a.s.}} Z_i^*(X) := \frac{\tilde{S}_i - \tilde{\mu}(X)}{\tilde{\sigma}(X)}$ 且 $Z_{i, N} = Z_i^*(X) + O_p(N^{-1/2})$。若先验恒定 $p(X) \equiv p_0$，标准化后验跨题一致对齐；若 $p(X)$ 随题波动，即便消去了 $(b(X), a(X))$，Z-Score 仍保留由 $p(X)$ 诱导的均值漂移 $\tilde{\mu}(X)$ 与尺度变化 $\tilde{\sigma}(X)$。*
2. **（$N=2$ 纯秩规则非严格界与严格优势充要条件）** *当 $N=2$ 时，对 $S_1 \neq S_2$ 恒有 $(Z_{1,2}, Z_{2,2}) \in \{(\pm \frac{1}{\sqrt{2}}, \mp \frac{1}{\sqrt{2}})\}$，对平局事件 $S_1 = S_2$ 有 $Z_{1,2}=Z_{2,2}=0$（此时 `Raw-BoN(2)`、`Anti-BoN(2)`、`Within-Z(2)` 与 Oracle 在平局处均退化为均匀随机择一，条件正确率同为 $\frac{Y_1+Y_2}{2}$）。因此无论分布连续或含原子，任何基于 $Z_{i,2}$（平局均匀打破）的确定性或随机化校准器均退化为 `Raw-BoN(2)`、`Anti-BoN(2)` 与随机择一的凸组合，满足非严格不等式：*
   $$P_{\text{Within-Z}}(2) \le \max\!\Big(P_{\text{Raw}}(2),\ P_{\text{Anti-BoN}}(2),\ \bar{p}\Big) \le P_{\text{Cal-BoN}}^{\text{oracle}}(2),$$
   *其中连续无平局分布下 $P_{\text{Anti-BoN}}(2) = 2\bar{p} - P_{\text{Raw}}(2)$。*
   - **（等号可达性）** 若 $\mathcal{P}_X$-a.e. 题内后验 $q_X(s)$ 关于 $s$ 单调非降（如 $\mathcal{N}(1,1)/\mathcal{N}(0,1),\ p=0.5$），则三者精确相等（$\approx 0.630125$）；若处处单调非增则与 $2\bar{p}-P_{\text{Raw}}(2)$ 取等。
   - **（严格不等号 `<` 成立的充要条件）** 对连续条件分布 $F_{\text{mix}, X}$，右侧取严格不等号 $\max(P_{\text{Raw}}(2), 2\bar{p}-P_{\text{Raw}}(2), \bar{p}) < P_{\text{Cal-BoN}}^{\text{oracle}}(2)$ **当且仅当**在联合抽样 $(X, S_1, S_2)$ 下正向与反向序反转事件 $\{S_1 < S_2,\ q_X(S_1) > q_X(S_2)\}$ 与 $\{S_1 < S_2,\ q_X(S_1) < q_X(S_2)\}$ 的概率均严格大于零。

*证明.* (1) 由强大数定律、四阶矩有界下的样本方差中心极限定理与 Delta 方法约去 $(b(X), a(X))$ 立得。(2) 由交换对称性直接相减得精确恒等式 $P_{\text{Cal-BoN}}^{\text{oracle}}(2) - P_{\text{Raw}}(2) = 2\mathbb{E}[(q_X(S_1) - q_X(S_2))\mathbf{1}\{S_1 < S_2, q_X(S_1) > q_X(S_2)\}] \ge 0$ 与 $P_{\text{Cal-BoN}}^{\text{oracle}}(2) - (2\bar{p} - P_{\text{Raw}}(2)) = 2\mathbb{E}[(q_X(S_2) - q_X(S_1))\mathbf{1}\{S_1 < S_2, q_X(S_1) < q_X(S_2)\}] \ge 0$，且 $P_{\text{Cal-BoN}}^{\text{oracle}}(2) - \bar{p} = \frac{1}{2}\mathbb{E}[|q_X(S_1) - q_X(S_2)|]$，两非负期望同时严格为正当且仅当双向序反转事件均具正概率。$\blacksquare$

---

### 3.3 定理 7C：一般边界 $(\pi_0, \pi_1)$ 下的样本标准化极值收敛、仿射坐标变换与上限损失

将题库先验分布 $p(X) \in [0, 1]$ 扩展至一般边界原子：记**不可解题原子** $\pi_0 := \mathbb{P}(p(X)=0) \ge 0$ 与**全对题原子** $\pi_1 := \mathbb{P}(p(X)=1) \ge 0$（$\pi_0 + \pi_1 > 0$），内点可解题落于 $[p_{\min}, p_{\max}] \subset (0, 1)$ 且概率 $1 - \pi_0 - \pi_1 > 0$。考察原始打分 $S_i = b(X) + a(X)\tilde{S}_i$（允许任意未知题间偏移 $b(X) \in \mathbb{R}$ 与尺度 $a(X) > 0$），其中基底 Verifier 为标准**同方差单调 MLRP 高斯**：
$$\tilde{S}_i \mid (Y_i = 1, X) \sim \mathcal{N}(\mu_1, \sigma^2), \qquad \tilde{S}_i \mid (Y_i = 0, X) \sim \mathcal{N}(\mu_0, \sigma^2), \qquad \kappa_{\text{base}} = 1,\ \Delta := \mu_1 - \mu_0 > 0.$$
此时 `Standard-BoN` 在每一道 $p(X)>0$ 可解题上均渐近趋近 $1$，总体渐近上限为 $P_{\text{Raw}}(\infty) = 1 - \pi_0$。注意由 $S_i = b(X) + a(X)\tilde{S}_i = b(X) + a(X)\big[\tilde{\mu}(X) + \tilde{\sigma}(X) Z_i^*(X)\big]$，原始分数后验 $q_X(s) = \mathbb{P}(Y=1 \mid S=s, X)$ 与基底分数后验 $\tilde{q}_X(\tilde{s}) = \mathbb{P}(Y=1 \mid \tilde{S}=\tilde{s}, X)$ 满足精确**仿射坐标变换恒等式**：
$$q_X(S_i) = q_X\!\Big(b(X) + a(X)\big[\tilde{\mu}(X) + \tilde{\sigma}(X) Z_i^*(X)\big]\Big) \equiv \tilde{q}_X\!\big(\tilde{\mu}(X) + \tilde{\sigma}(X) Z_i^*(X)\big).$$

**定理 7C（一般 $(\pi_0, \pi_1)$ 标准化尾竞争极限 $L_{\text{tail}} = \frac{\pi_1}{\pi_0+\pi_1}$、在线样本 $Z_{i,N}$ 一致极值收敛与相变界）.** *考察总体标准化变量 $Z^*_i(X) = \frac{\tilde{S}_i - \tilde{\mu}(X)}{\tilde{\sigma}(X)}$ 及其池化边际后验 $q_{\text{pool}, Z^*}(z) := \mathbb{P}(Y = 1 \mid Z^* = z)$，并定义边界尾比 $L_{\text{tail}} := \frac{\pi_1}{\pi_0 + \pi_1}$：*
1. **（无条件双尾竞争极限与两相界）** *记 $v(p) := \frac{\sigma^2}{\sigma^2 + p(1-p)\Delta^2}$。对边界原子 $p \in \{0, 1\}$ 有 $v(0) = v(1) = 1$，而对一切内点可解题 $p \in [p_{\min}, p_{\max}]$ 严格有 $v(p) \le v_{\max} < 1$。因此无需额外方差假设，标准化池化后验的左右双尾极限精确等于边界比：*
   $$\lim_{z \to \pm\infty} q_{\text{pool}, Z^*}(z) = L_{\text{tail}} = \frac{\pi_1}{\pi_0 + \pi_1}.$$
   - **（a）无不可解题界（$\pi_0 = 0, \pi_1 > 0$）**：*此时 $L_{\text{tail}} = 1 > q_{\text{pool}, Z^*}(z)$（对一切有限 $z \in \mathbb{R}$）。若仅在右半轴/非负域约束最大化（仅从满足 $Z_{i,N} \ge 0$ 的候选中最大化 $q_{\text{pool}, Z^*}(Z_{i,N})$，或补充排斥左尾反超的充分条件：单内点层或 $p_{\max} \le \frac{1}{2}$），则全局最大化导向 $Z^* \to +\infty$，在线样本 $Z_{i,N}$ 校准器渐近收敛至 $1 = P_{\text{Raw}}(\infty)$，**天花板损失严格等于 $0$**。反之，若允许无约束双尾极大化且存在多个内点层，当高先验层 $p_C$ 满足 $(1 - v_{\max})(|m_0(p_C)| - m_1(p_C)) > 2|m_0(p^*)|$（$p^*$ 达到 $v_{\max}=\max_p v(p)$）时，极值左尾 $\ln(1 - q_{\text{pool}, Z^*}(Z_{\min}))$ 比右尾衰减更快，可触发“左尾反超陷阱”将选择拉向 $Z^* \to -\infty$ 并造成严格正的渐近损失（如 $w=(0.10, 0.45, 0.45), p=(1.0, 0.01, 0.95), \Delta=15$ 总体极限跌至 $0.55 < 1$）。*
   - **（b）一般不可解题共存相（$\pi_0 > 0, \pi_1 \ge 0$ 与 $1 - \pi_0 - \pi_1 > 0$）**：*因为每道内点题的条件后验 $\tilde{q}_X(\tilde{\mu}(X)+\tilde{\sigma}(X)z) \to 1 > L_{\text{tail}}$，数学上必存在有限点 $z_0 \in \mathbb{R}$ 使 $q_{\text{pool}, Z^*}(z_0) > L_{\text{tail}}$，从而全局最大值集 $K^* := \arg\max_{z \in \mathbb{R}} q_{\text{pool}, Z^*}(z) \subset [-M_Z, M_Z]$ 对一切 $\pi_1 \ge 0$ 均为非空紧集（即使净总体条件方差比反例给出 $\kappa_Z < 1$，如 $\pi_0=0.01, \pi_1=0, p_1=0.1, p_2=0.2, \sigma=1, \Delta=100$ 得 $\kappa_Z \approx 0.3103$；单内点层 $\{0, p_1\}$ 下 $\operatorname{Var}(m_1\mid Y=1)=0$ 保证 $\kappa_Z > 1$）。*
2. **（一般 $\pi_1 \ge 0$ 下的在线样本 $Z_{i,N}$ 一致极值收敛与渐近上限损失）** *在 $\pi_0 > 0, \pi_1 \ge 0$ 且 $1-\pi_0-\pi_1 > 0$（或对一般非高斯混合，存在有限 $z_0$ 使 $q_{\text{pool}, Z^*}(z_0) > L_{\text{tail}}$）下，在线选择 $\hat{i}_N \in \arg\max_{1 \le i \le N} q_{\text{pool}, Z^*}(Z_{i,N})$（$Z_{i,N} = \frac{S_i - \bar{S}_N}{\hat{s}_N}$ 严格消去 $(a(X), b(X))$）对任意固定题目 $X = x$ 满足：*
   $$\Delta_N(x) := \max_{1 \le i \le N} \big|Z_{i, N} - Z_i^*(x)\big| = O_p\!\left(\sqrt{\frac{\ln N}{N}}\right) \xrightarrow{p} 0 \qquad (N \to \infty),$$
   *且其渐近准确率上界严格低于 `Standard-BoN`（对任意题间仿射变换 $b(X), a(X) > 0$ 完全不变）：*
   $$\limsup_{N \to \infty} P_{\text{Sample-Z-Cal}}^{\text{oracle}}(N) \le \pi_1 + \mathbb{E}_X\!\left[\max_{z \in K^*} q_X\!\Big(b(X) + a(X)\big[\tilde{\mu}(X) + \tilde{\sigma}(X)z\big]\Big)\, \mathbf{1}\{p(X)\in(0,1)\}\right] \equiv \pi_1 + \mathbb{E}_X\!\left[\max_{z \in K^*} \tilde{q}_X\!\big(\tilde{\mu}(X) + \tilde{\sigma}(X)z\big)\, \mathbf{1}\{p(X)\in(0,1)\}\right] < 1 - \pi_0 = P_{\text{Raw}}(\infty).$$
   *（有限窗口缓解机制：随着 $\pi_1/\pi_0$ 增大，$K^*$ 向遥远右尾外推，上限损失随 $\pi_1$ 增大而收窄——真实基准实测见 `results/tmlr_diagnostics.json`。）*

> **适用域与曲线形态区分注记**：定理 7C 评估的是将理想极限池化映射 $q_{\text{pool}, Z^*}$ 作用于在线有限样本 $Z_{i,N}$ 的内在结构上限；在真实连续难度的基准中（如 `AceMath RM-B` 即使移除严格 $\hat{\pi}_0 = 0$ 后），极低先验内点难题 $p(X) \to 0^+$ 同样具有近 $1$ 的相对方差压缩比 $v(p) \approx 1$，从而在实用分数分位区间持续维持非单调峰值倒挂。

*定理 7C 的完整证明.*
- **(1)**：各层内边界原子（$p_g \in \{0, 1\}$）精确服从 $\mathcal{N}(0, 1)$ 密度 $\phi(z)$，而所有内点层 $v_g \le v_{\max} < 1$ 的高斯密度随 $|z| \to +\infty$ 以 $o(\phi(z))$ 衰减，代入 Bayes 公式立得 $\lim_{z\to\pm\infty} q_{\text{pool}, Z^*}(z) = \frac{\pi_1}{\pi_0+\pi_1} = L_{\text{tail}}$。当 $\pi_0 = 0, \pi_1 > 0$ 时，$L_{\text{tail}} = 1$ 而内点层存在负类正密度使 $q_{\text{pool}, Z^*}(z) < 1$ 处处成立。若限制于右半轴 $Z_{i,N} \ge 0$（注意由 $\sum_i Z_{i,N} = 0$，样本内必存在至少一个候选满足 $Z_{i,N} \ge 0$）或单内点层/$p_{\max} \le \frac{1}{2}$ 使 $|m_0(p)| \le m_1(p)$ 处处阻止左尾超车，则最大化必然导向 $Z^*_{\hat{i}_N} \to +\infty$，全对题与内点题均渐近命中 `1`；相反，当高先验内点层 $p_C$ 满足 $(1 - v_{\max})(|m_0(p_C)| - m_1(p_C)) > 2|m_0(p^*)|$ 时，经由与 $|z|$ 成比例的一阶指数项比对可见左尾在极值分位处压倒右尾，导致无约束双尾极大化落入 $Z^* \to -\infty$ 选中错样本。当 $\pi_0 > 0, \pi_1 \ge 0$ 时，$L_{\text{tail}} < 1$；由于内点先验紧支撑于 $[p_{\min}, p_{\max}] \subset (0, 1)$，对数优势下界 $\operatorname{logit}(\tilde{q}_X(\tilde{\mu}(X) + \tilde{\sigma}(X) z)) \ge \operatorname{logit}(p_{\min}) - \frac{\Delta^2}{2\sigma^2} + \frac{\Delta}{\sigma} z \to +\infty$ 一致成立，故存在足够大有限点 $z_0$ 使全体内点一致满足 $\tilde{q}_X > L_{\text{tail}}$，将内点整体正高斯密度平均与边界原子作凸组合即证 $q_{\text{pool}, Z^*}(z_0) > L_{\text{tail}}$，连续性与左右收敛于 $L_{\text{tail}}$ 保证全局最大值集 $K^*$ 为非空紧集。
- **(2) 在线样本 $Z_{i,N}$ 一致极值收敛与仿射坐标还原**：由位置—尺度不变性 $Z_{i,N} = \frac{S_i - \bar{S}_N}{\hat{s}_N} = \frac{\tilde{S}_i - \bar{\tilde{S}}_N}{\hat{\tilde{s}}_N}$。固定任意题目 $X = x$：
  1. 因 $\tilde{\sigma}(x) = \sqrt{\sigma^2 + p(x)(1-p(x))\Delta^2} \ge \sigma > 0$，条件于 $X=x$ 时 $\tilde{S}_i - \tilde{\mu}(x)$ 为次高斯变量，有限最大值次高斯并界得 $\max_{1\le i\le N}|Z_i^*(x)| = O_p(\sqrt{\ln N})$；同时 $|\bar{\tilde{S}}_N - \tilde{\mu}(x)| = O_p(N^{-1/2})$、$|\hat{\tilde{s}}_N - \tilde{\sigma}(x)| = O_p(N^{-1/2})$ 且 $\mathbb{P}(\hat{\tilde{s}}_N < \sigma/2 \mid X=x) \to 0$，故代数恒等式导出 $\Delta_N(x) = O_p(\sqrt{\frac{\ln N}{N}}) \xrightarrow{p} 0$。
  2. 因 $q_{\text{pool}, Z^*}$ 在 $\mathbb{R}$ 上连续且在 $\pm\infty$ 处趋于 $L_{\text{tail}} < q_Z^*$，故在 $\mathbb{R}$ 上一致连续，结合 $\Delta_N(x) \xrightarrow{p} 0$ 得 $\operatorname{dist}(Z_{\hat{i}_N}^*(x), K^*) \xrightarrow{p} 0$。
  3. **仿射坐标精确还原**：由条件期望塔性质与仿射恒等式 $\mathbb{P}(Y_{\hat{i}_N} = 1 \mid X = x) = \mathbb{E}\big[\tilde{q}_x\big(\tilde{\mu}(x) + \tilde{\sigma}(x) Z_{\hat{i}_N}^*(x)\big) \mid X = x\big]$。对 $p(x) = 0$ 恒为 $0$，对 $p(x) = 1$ 恒为 $1$（总贡献 $\pi_1$）；对任意内点题 $p(x) \in (0, 1)$，连续函数 $z \mapsto \tilde{q}_x(\tilde{\mu}(x) + \tilde{\sigma}(x) z) \in (0, 1)$ 在紧集 $K^*_\varepsilon$ 上严格小于 $1$。取上极限 $\limsup_{N\to\infty}$ 并应用**反向 Fatou 引理（Reverse Fatou's Lemma）**对 $\mathcal{P}_X$ 积分（即使 $|K^*|\ge 2$ 导致逐点极限未必存在亦严格成立），即证得一般 $\pi_1 \ge 0$ 下与 $(b(X), a(X))$ 无关的严格上界 $\pi_1 + \mathbb{E}_X[\max_{z\in K^*} \tilde{q}_X(\tilde{\mu}(X)+\tilde{\sigma}(X)z)\,\mathbf{1}\{p(X)\in(0,1)\}] < \pi_1 + (1 - \pi_0 - \pi_1) = 1 - \pi_0 = P_{\text{Raw}}(\infty)$。$\blacksquare$

---

### 3.4 命题 7.1：三分支留出门控的风险控制不等式与重拟合误差分解

**命题 7.1（覆盖全体 $N \ge 2$ 的三分支留出门控选择遗憾界：固定预测器 vs 选后重拟合）.** *由于定理 7A–7C 证明对任意 $N \ge 2$，`Standard-BoN`（无分桶平局损失且免疫边界原子反转）、`Pooled-KL-LCB`（抑制题内重错尾假高分且保留绝对分信息）与 `Within-Prompt-Z-KL-LCB`（消去跨题仿射偏移，且在 $N=2$ 处涵盖 `Anti-BoN(2)` 反向打分恢复能力）在不同异质性相区互无一致占优关系，对全体 $N \ge 2$ 统一定义三元候选策略族 $\mathcal{M} := \{\text{Standard-BoN},\ \text{Pooled-KL-LCB},\ \text{Within-Prompt-Z-KL-LCB}\}$（$|\mathcal{M}| = 3$）。设数据按独立题目划分为互不相交的训练集 $\mathcal{D}_{\text{train}}$ 与留出验证集 $\mathcal{D}_{\text{val}}$（含 $M_{\text{val}}$ 道独立题目）：*
1. **（固定内部预测器协议 `Hybrid-Fixed`，零重拟合误差）** *各候选规则 $m \in \mathcal{M}$ 仅在 $\mathcal{D}_{\text{train}}$ 上拟合完毕并固定，在 $\mathcal{D}_{\text{val}}$ 上评估经验均值 $\hat{A}_{\text{val}}(m) \in [0, 1]$ 选出 $\hat{m} \in \arg\max_{m \in \mathcal{M}} \hat{A}_{\text{val}}(m)$ 并直接用于独立测试题目。记总体测试准确率为 $A_{\text{fixed}}(m)$，则条件于 $\mathcal{D}_{\text{train}}$，对任意 $\delta \in (0, 1)$ 以至少 $1 - \delta$ 概率成立：*
   $$\max_{m \in \mathcal{M}} A_{\text{fixed}}(m) - A_{\text{fixed}}(\hat{m}) \le 2\sqrt{\frac{\ln(2|\mathcal{M}|/\delta)}{2\, M_{\text{val}}}}, \qquad \max_{m \in \mathcal{M}} A_{\text{fixed}}(m) - \mathbb{E}_{\text{val}}\big[A_{\text{fixed}}(\hat{m})\big] \le \sqrt{\frac{\ln|\mathcal{M}|}{2\, M_{\text{val}}}}.$$
2. **（选后合并重拟合协议 `Hybrid-Refit` 的附加稳定性项）** *若在选出分支标签 $\hat{m}$ 后，在合并集 $\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}}$ 上重新拟合校准器得到预测器 $m_{\text{refit}}$（总体测试准确率记为 $A_{\text{refit}}(m)$），定义重拟合漂移模量 $\varepsilon_{\text{refit}} := \max_{m \in \mathcal{M}} \big|A_{\text{refit}}(m) - A_{\text{fixed}}(m)\big|$（对无参分支 `Standard-BoN` 恒为 $0$），则以至少 $1 - \delta$ 概率成立：*
   $$\max_{m \in \mathcal{M}} A_{\text{refit}}(m) - A_{\text{refit}}(\hat{m}) \le 2\sqrt{\frac{\ln(2|\mathcal{M}|/\delta)}{2\, M_{\text{val}}}} + 2\,\varepsilon_{\text{refit}}.$$

*证明.* (1) 条件于 $\mathcal{D}_{\text{train}}$，每道验证题在固定规则 $m \in \mathcal{M}$ 下的得分属于 $[0, 1]$ 且跨题独立同分布，期望为 $A_{\text{fixed}}(m)$；由 Hoeffding 双侧 $2|\mathcal{M}|$ 并界及有限族次高斯最大值不等式直接得出。(2) 由三角不等式 $A_{\text{refit}}(m^*) - A_{\text{refit}}(\hat{m}) \le \big(A_{\text{fixed}}(m^*) - A_{\text{fixed}}(\hat{m})\big) + |A_{\text{refit}}(m^*) - A_{\text{fixed}}(m^*)| + |A_{\text{fixed}}(\hat{m}) - A_{\text{refit}}(\hat{m})|$ 代入第 (1) 步上界即证。$\blacksquare$

---

## 4. 补充理论：异构多裁判集成相变、帕累托光谱与 Copula 尾相依（定理 5）

本节在固定问题条件模型下，分析一般非对称正定协方差矩阵 $(\Sigma_1, \Sigma_0)$ 与一般正定相关 Gaussian-Copula + Student-$t$ 多裁判集成的相变边界。

### 4.1 定理 5A：一般非对称高斯 $K$-Verifier 线性融合、含退化 QP `Min` 与 QDA 维度律

考虑固定问题条件下 $K \ge 1$ 个裁判的联合高斯分布 $\mathbf{S} \mid (Y = c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c)$（$\Sigma_c \succ 0,\ c \in \{0, 1\}$）。

#### 定理 5A-I（最优线性集成广义特征值相变与负权共模抑制）
考虑非零线性组合 $L_{\mathbf{w}}(\mathbf{S}) := \mathbf{w}^\top \mathbf{S}$。记广义特征值问题 $\Sigma_0 \mathbf{w} = \lambda \Sigma_1 \mathbf{w}$ 之最小特征值与对应（经定向使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \ge 0$ 的）广义特征向量为 $\lambda_{\min} := \lambda_{\min}(\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2})$ 与 $\mathbf{w}^*$。则：
1. **（一般线性集成渐近一致充要判据）** 存在线性权重 $\mathbf{w}$ 使 $\lim_{N\to\infty} P_{\text{BoN}}(N; L_{\mathbf{w}}) = 1$ 的充分必要条件为：$\lambda_{\min} < 1$（或 $\lambda_{\min} = 1$ 且存在对应广义特征向量使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) > 0$）。
2. **（负权差值共模抑制推论）** 当两裁判受同源错例信号影响（如 $\rho_1 = 0, \rho_0 = 0.85, \kappa = 1.2 > 1$），任意非负权重均满足 $\kappa_{\text{nonneg}} > 1$ 而坍塌；若正类存在均值差，差值融合 $\mathbf{w}^* = \frac{1}{\sqrt{2}}(1, -1)^\top$ 实现 $\frac{\mathbf{w}^{*\top}\Sigma_0\mathbf{w}^*}{\mathbf{w}^{*\top}\Sigma_1\mathbf{w}^*} = \kappa^2 \frac{1 - \rho_0}{1 - \rho_1} = 0.216 < 1$，对消错类共模方差实现渐近一致。

*证明.* 由 $L_{\mathbf{w}}(\mathbf{S}) \mid (Y=c) \sim \mathcal{N}(\mathbf{w}^\top\boldsymbol{\mu}_c, \mathbf{w}^\top\Sigma_c\mathbf{w})$ 结合定理 2B(4) 及对称矩阵 Rayleigh 商最值定理即证。$\blacksquare$

#### 定理 5A-II（一般异构 Min-Ensemble 的 Hashorva–Hüsler QP 密度级定理——含零乘子退化情形）
考察 $M_{\min}(\mathbf{S}) := \min_{1 \le k \le K} S^{(k)}$。对 $c \in \{0, 1\}$，求解严格凸正交象限二次规划（Savage 1962; Hashorva & Hüsler 2003）$\gamma_c := \min_{\mathbf{u} \ge \mathbf{1}_K} \mathbf{u}^\top \Sigma_c^{-1} \mathbf{u}$，记唯一最优解 $\mathbf{u}_c^* \ge \mathbf{1}_K$ 与唯一 KKT 乘子 $\mathbf{w}_c := \Sigma_c^{-1}\mathbf{u}_c^* \ge \mathbf{0}_K$。将指标集 $\{1, \dots, K\}$ 严格三分划为：
- **本质正乘子支撑集** $I_{+,c} := \{k : w_{c, k} > 0\} \neq \emptyset$；**零乘子弱激活集** $I_{0,c} := \{k : u_{c,k}^* = 1,\ w_{c,k} = 0\}$；**严格非激活集** $J_c := \{k : u_{c,k}^* > 1\}$。

记 $\gamma_c = \sum_{k \in I_{+,c}} w_{c,k}$、$\beta_c := \mathbf{w}_{c, I_{+,c}}^\top \boldsymbol{\mu}_{c, I_{+,c}}$，条件 Schur 补高斯向量 $\mathbf{W}_{I_{0,c}} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}_{0|+},\, \tilde{\Sigma}_{0|+})$（其中 $\tilde{\boldsymbol{\mu}}_{0|+} := \boldsymbol{\mu}_{c, I_{0,c}} - \Sigma_{c, I_{0,c} I_{+,c}}\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\boldsymbol{\mu}_{c, I_{+,c}}$，$\tilde{\Sigma}_{0|+} := \Sigma_{c, I_{0,c} I_{0,c}} - \Sigma_{c, I_{0,c} I_{+,c}}\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\Sigma_{c, I_{+,c} I_{0,c}}$）及弱激活条件正象限常数 $\theta_{c, I_0} := \mathbb{P}(\mathbf{W}_{I_{0,c}} > \mathbf{0}) \in (0, 1]$（$I_{0,c}=\emptyset$ 时为 $1$）。则对**任意** $\Sigma_c \succ 0$，当 $t \to +\infty$ 时：
1. **（含弱激活分支的精确密度级右尾渐近式）** 多项式阶由本质基数 $|I_{+,c}|$ 确定：
   $$\bar{F}_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{sf}}_c\, t^{-|I_{+,c}|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right), \qquad f_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{pdf}}_c\, t^{1 - |I_{+,c}|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right),$$
   其中 $C^{\text{sf}}_c := \theta_{c, I_0}\, (2\pi)^{-|I_{+,c}|/2}(\det \Sigma_{c, I_{+,c} I_{+,c}})^{-1/2} (\prod_{k \in I_{+,c}} w_{c, k}^{-1}) \exp(-\frac{1}{2}\boldsymbol{\mu}_{c, I_{+,c}}^\top\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\boldsymbol{\mu}_{c, I_{+,c}}) > 0$，$C^{\text{pdf}}_c := \gamma_c C^{\text{sf}}_c > 0$。
2. **（渐近极限三层级判据）** $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 1 \iff (\gamma_0 > \gamma_1)$ 或 $(\gamma_0 = \gamma_1,\ \beta_1 > \beta_0)$ 或 $(\gamma_0 = \gamma_1,\ \beta_1 = \beta_0,\ |I_{+,0}| > |I_{+,1}|)$；首位不等号反向则趋于 $0$；三阶全相等则趋于 $\frac{p\Lambda_R}{p\Lambda_R+1-p}$（$\Lambda_R = C^{\text{pdf}}_1 / C^{\text{pdf}}_0$）。
3. **（Cauchy–Schwarz 下界）** $\gamma_0 \ge \frac{K^2}{\mathbf{1}_K^\top \Sigma_0 \mathbf{1}_K}$；对称相关下临界人数 $K_c = \lceil 1 + \frac{\kappa^2-1}{\rho_1 - \kappa^2\rho_0} \rceil$ 有限当且仅当 $\rho_0 < \rho_1/\kappa^2$（推论 5.1）。

*证明.* 将 $f_{\min, c}(t) = -\frac{d}{dt}\mathbb{P}(M_{\min} > t \mid Y=c)$ 展开为 $K$ 个角锥侧面通量之和：$k \in J_c$ 侧面贡献指数级可忽略 $O(e^{-\Delta\gamma t^2/2})$；对 $k \in I_{+,c}$ 侧面作尺度替换 $y_j = \frac{\xi_j}{t w_{c,j}}$，利用 KKT 恒等式 $\Sigma_{c, I_{0,c} I_{+,c}}\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\mathbf{1}_{I_{+,c}} = \mathbf{1}_{I_{0,c}}$ 使弱激活块线性阈值精确消去并依分布收敛至 $\mathbf{W}_{I_{0,c}}$（贡献常数因子 $\theta_{c, I_0}$ 而非额外 $t^{-|I_{0,c}|}$），对 $k \in I_{+,c}$ 求和得 $C^{\text{pdf}}_c = \gamma_c C^{\text{sf}}_c$；而 $k \in I_{0,c}$ 侧面因剩余积分仍含全部 $|I_{+,c}|$ 维压缩，体积阶为 $O(t^{-|I_{+,c}|}e^{-\frac{\gamma_c}{2}t^2+\beta_c t}) = o(f_{\min,c}^{\text{main}}(t))$。结合定理 2A/2B 与 $(\mathbf{u}^\top\Sigma_0^{-1}\mathbf{u})(\mathbf{1}^\top\Sigma_0\mathbf{1}) \ge K^2$ 即证。$\blacksquare$

#### 定理 5A-III（带尾部安全约束的 $N=2$ 线性光谱路径 `Safe-Fisher` 与 Bayes QDA 维度律）
做同时合同对角化 $P^\top \Sigma_1 P = I_K$、$P^\top \Sigma_0 P = \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_K)$（$0 < \lambda_1 \le \dots \le \lambda_K$），记 $\mathbf{d} := P^\top \Delta\boldsymbol{\mu}$ 与 $\mathbf{v} := P^{-1}\mathbf{w}$。
1. **（带渐近尾方差比约束的 $N=2$ SNR 光谱路径 `Safe-Fisher`）** 考察最大化 $N=2$ 二样本信噪比 $\operatorname{SNR}_2^2(\mathbf{w}) := \frac{(\mathbf{w}^\top \Delta\boldsymbol{\mu})^2}{\mathbf{w}^\top(\Sigma_1 + \Sigma_0)\mathbf{w}}$ 且满足渐近尾方差比约束 $R(\mathbf{w}) := \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}} \le \kappa_{\max}^2$ 的优化问题（注：仅针对 $N=2$ 目标与渐近约束权衡，不主张对一般 $N > 2$ 最优）。设 Slater 条件 $\lambda_1 < \kappa_{\max}^2 < R(\mathbf{w}_{\text{Fisher}})$ 成立：在非退化主分支下，全局最优解必落于一维光谱路径 $\mathbf{w}_\tau := (\Sigma_0 - \tau \Sigma_1)^{-1}\Delta\boldsymbol{\mu}$（$\tau \in [-1, \lambda_1)$），满足 $R'(\tau) < 0$ 与 $\frac{d}{d\tau}\operatorname{SNR}_2^2(\mathbf{w}_\tau) < 0$（$\tau \in (-1, \lambda_1)$），由单调二分方程 $R(\tau^*) = \kappa_{\max}^2$ 唯一确定；若 $P_{V_{\min}}\mathbf{d} = \mathbf{0}$ 且 $\lim_{\tau\uparrow\lambda_1^-}R(\tau) \ge \kappa_{\max}^2$，则由正交边界混合解 $\mathbf{w}^* = \mathbf{w}_{\lambda_1^-} + \alpha\mathbf{u}_{\min}$（其中 $\mathbf{u}_{\min} \in P V_{\min}\setminus\{\mathbf{0}\}$ 归一化满足 $\mathbf{u}_{\min}^\top \Sigma_1 \mathbf{u}_{\min} = 1$）饱和约束 $R(\mathbf{w}^*) = \kappa_{\max}^2$ 给出。
2. **（不可逆相 $\Sigma_0 \succ \Sigma_1$ 下 Bayes QDA 天花板与 $N^{-2/K}$ 维度律）** 当 $\lambda_1 > 1$（即 $\Sigma_0 \succ \Sigma_1$）时，二次判别后验 $q_{\text{QDA}}(\mathbf{s})$ 在唯一内点峰值 $\mathbf{s}^* = (\Sigma_1^{-1} - \Sigma_0^{-1})^{-1}(\Sigma_1^{-1}\boldsymbol{\mu}_1 - \Sigma_0^{-1}\boldsymbol{\mu}_0)$ 处取得小于 $1$ 的天花板 $q^*_{\text{Bayes}} = \sigma\big(\operatorname{logit}(p) + \frac{1}{2}\ln\frac{\det\Sigma_0}{\det\Sigma_1} + \frac{1}{2}\Delta\boldsymbol{\mu}^\top(\Sigma_0 - \Sigma_1)^{-1}\Delta\boldsymbol{\mu}\big) < 1$（随 $K$ 单调非降），且极值遗憾满足 $q^*_{\text{Bayes}} - P_{\text{QDA}}(N) = (1 + o(1))\frac{\Gamma(1 + 2/K)}{c_K^{2/K}} N^{-2/K}$。

*证明.* Part 1 作平方坐标替换 $x_k := v_k^2 \ge 0$ 并将分母归一化为 $\sum_{k=1}^K (1 + \lambda_k) x_k = 1$（从而 $\operatorname{SNR}_2^2$ 等价于最大化凹目标 $\sum_{k=1}^K |d_k|\sqrt{x_k}$），尾部约束化为线性约束 $\sum_{k=1}^K (\lambda_k - \kappa_{\max}^2) x_k \le 0$；由 Slater 强对偶、KKT 平稳条件 $\frac{|d_k|}{2\sqrt{x_k}} = \alpha(1+\lambda_k) + \mu(\lambda_k - \kappa_{\max}^2)$（对应 $\tau = -\frac{\alpha - \mu\kappa_{\max}^2}{\alpha + \mu} \in [-1, \lambda_1)$）与 Cauchy–Schwarz 不等式 $C(\tau)^2 < A(\tau)D(\tau)$ 得证；退化正交分支下，对偶乘子取边界值 $\tau^* = \lambda_1$，在子空间 $P V_{\min}$ 上以 $\mathbf{u}_{\min}^\top \Sigma_1 \mathbf{u}_{\min} = 1$（即 $\|P^{-1}\mathbf{u}_{\min}\|_2 = 1$）叠加零目标分量 $\alpha \mathbf{u}_{\min}$ 并解 $\sum_{k \notin I_{\min}} (\lambda_k - \kappa_{\max}^2)(v_k^\perp)^2 = \alpha^2(\kappa_{\max}^2 - \lambda_1)$（其中 $v_k^\perp := (P^{-1}\mathbf{w}_{\lambda_1^-})_k$）即饱和 $R(\mathbf{w}^*)=\kappa_{\max}^2$。Part 2 对局部二次型水平椭球求 $K$ 维体积 $\bar{F}_{Q,\text{QDA}}(q^*_{\text{Bayes}}-u) = (1+o(1))c_K u^{K/2}$（局部椭球体积前缀常数 $c_K > 0$）并积分即证。$\blacksquare$

---

### 4.2 定理 5B：正则变化重尾卷积与 Gaussian-Copula 统一剩余尾相依

#### 定理 5B-I（正则变化卷积与共享加性高斯微扰的分离判据）
设正类 $S^{(k)} = \mu_1 + \sigma_1 T_1 + \eta_1 \varepsilon_k$（$T_1 \sim t_{\alpha_1}$，$\varepsilon_k \sim \mathcal{N}(0,1)$），错类以概率 $1-\epsilon_s$ 为独立 $t_{\alpha_0}$（$1 < \alpha_0 < \alpha_1$）、以概率 $\epsilon_s$ 为同源共享分支 $\mu_0 + \sigma_0 T_{\text{shared}} + \eta_0 \varepsilon_k$。则 `Mean` 对任意 $K \ge 1$ 渐近坍塌至 $0$；对 `Min`，当 $\epsilon_s = 0$ 时若 $K\alpha_0 > \alpha_1$ 趋于 $1$、$K\alpha_0 < \alpha_1$ 趋于 $0$、$K\alpha_0 = \alpha_1$ 趋于 $\frac{p\Lambda_R}{p\Lambda_R+1-p}$，而一旦 $\epsilon_s > 0$ 则同源分支产生 $\epsilon_s\Theta(t^{-\alpha_0-1})$ 密度下界从而在 $N\to\infty$ 坍塌至 $0$。（由正则变化卷积一致收敛定理证得。）

#### 定理 5B-II（一般正定 Gaussian-Copula + Student-$t$ 统一剩余尾相依系数与对数修正因子）
设任意正定相关矩阵 $R_1, R_0 \succ 0$（对角元为 $1$），潜向量 $\mathbf{Z} \mid (Y=c) \sim \mathcal{N}(\mathbf{0}, R_c)$，$S^{(k)} = \mu_c + \sigma_c (F_{t_{\alpha_c}}^{-1} \circ \Phi)(Z_k)$（错类含同源率 $\epsilon_s \in [0, 1]$）。设 QP 解为 $(\gamma_c(R_c), I_{+,c})$，定义剩余尾相依系数 $\eta_c := \frac{1}{\gamma_c(R_c)}$（Tankov 2016; Das & Fasen-Hartmann 2024）与 $\eta_{0,\text{eff}} := 1$（当 $\epsilon_s>0$）或 $\frac{1}{\gamma_0(R_0)}$（当 $\epsilon_s=0$）：
1. **（精确四类区间划分与非负相关充分性）**（i）$K=1$ 时 $\eta_c = 1$；（ii）$K \ge 2,\ R_c \succ 0$ 时由二维主子块 Schur 补正定性得 $\gamma_c > 1$，故 $\eta_c \in (0, 1)$；（iii）逐元非负 $(R_c)_{ij} \ge 0$ 作为**充分非必要条件**（反例：块对角 $\operatorname{diag}(R_{-0.2}, R_{+0.5})$ 给 $\eta = \frac{6}{23} > \frac{1}{4}$），由 Slepian 引理得 $\eta_c \in [\frac{1}{K}, 1)$；（iv）$\epsilon_s > 0$ 或奇异共单调极限得 $\eta_{0,\text{eff}} = 1$。
2. **（本质基数决定的对数修正阶与相界）** 对一切 $R_c \succ 0$（含零乘子弱激活退化矩阵），精确尾阶为 $\mathbb{P}(M_{\min} > t \mid Y=c) = \Theta\big(t^{-\alpha_c / \eta_c} (\ln t)^{\frac{\gamma_c(R_c) - |I_{+,c}|}{2}}\big)$ 与 $f_{\min, c}(t) = \Theta\big(t^{-\alpha_c / \eta_c - 1} (\ln t)^{\frac{\gamma_c(R_c) - |I_{+,c}|}{2}}\big)$，主阶收敛判据为 $\frac{\alpha_0}{\eta_{0,\text{eff}}} > \frac{\alpha_1}{\eta_1}$（主阶相等且 $\epsilon_s=0$ 时由 $\gamma_0 - |I_{+,0}| < \gamma_1 - |I_{+,1}|$ 判别）。

*证明.* 由二维主子块 Schur 补 $\mathbf{u}^\top R_c^{-1}\mathbf{u} \ge \frac{2}{1+\rho_{12}} > 1$ 及 Slepian 引理证得第 (1) 步；将边际高斯阈值 $h_c(t) = \sqrt{2\alpha_c\ln t}(1+o(1))$、$e^{-h_c(t)^2/2} = \Theta(t^{-\alpha_c}(\ln t)^{1/2})$ 与 $h_c'(t) = \Theta(t^{-1}(\ln t)^{-1/2})$ 代入定理 5A-II(1) 的 $h_c(t)^{-|I_{+,c}|}$ 主项即得精确对数因子 $(\ln t)^{(\gamma_c - |I_{+,c}|)/2}$。$\blacksquare$

---

## 5. 核心定理与软件实现符号对照索引

| 论文结构层次 | 理论模块 | 核心命题与适用域界定 | 对应 `tts_theory` 实现接口 |
| :---: | :--- | :--- | :--- |
| **主定理 I（§2）** | **定理 6A–6C & 命题 6.1** | 一致正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 联合 $(M,N)$ Minimax 率、预算平衡点 $N_{\text{balance}}(M)$、紧 Cramér 样本分割间距界（常数放大 $2$ 倍）及三类分桶协议 | `calibration_theory.py`: `tight_joint_minimax_scaling`, `empirical_quantile_spacings_bounds`, `calibration_minimax_scaling`; `real_eval.py`: `evaluate_finite_calibration_on_real_pools` |
| **主定理 II（§3）** | **定理 7A–7C & 命题 7.1** | `[3.5, 4.2]` 无网格纯解析辛普森遮蔽；$N=2$ 纯秩界；原始打分仿射变换恒等式与在线样本 $Z_{i,N}$ 一致极值收敛 $O_p(\sqrt{(\ln N)/N})$；全体 $N\ge 2$ 三分支门控与 $\varepsilon_{\text{refit}}$ 分解 | `heterogeneity_theory.py`: `diagnose_simpson_confounding`, `verify_sample_zscore_extremal_convergence`; `real_eval.py`: `evaluate_single_split_prompt_heterogeneity`, `evaluate_real_prompt_heterogeneity` |
| **补充理论（§4）** | **定理 5A–5B** | 广义特征值负权逆转；$N=2$ `Safe-Fisher` 光谱与 QDA $N^{-2/K}$；含零乘子弱激活 QP `Min`（$I_{+,c}, \theta_{c,I_0}$）与 Copula $(\gamma_c-|I_{+,c}|)/2$ | `ensemble_theory.py`: `solve_positive_quadrant_qp`, `pareto_linear_ensemble_frontier`, `bayes_qda_multi_verifier_analysis`, `gaussian_copula_heavy_tail_criterion` |
| **预备基础（§1 & 附录 A）** | **定理 1–4** | 含原子顺序统计量积分；$C^1/C^0$ MLRP、高斯五相与固定峰高 $e^{-\gamma}$；$\text{PPV}=1$ 散度分离；单错类条件权重与超级尾桶多数决 | `theorems.py`, `integrals.py`, `algorithms.py` |

---

## 附录 A：单裁判基础预备定理（定理 1–4）完整陈述与严格证明

### A.1 定理 1：精确顺序统计量积分公式与含原子统一推广

**定理 1（BoN 精确顺序统计量公式、PIT 等价形式与含原子推广）.**
1. **（连续无原子情形）** *若 $F_1, F_0$ 均绝对连续（密度 $f_1, f_0$），则对任意 $N \ge 1$：$P_{\text{BoN}}(N) = \int_{-\infty}^{\infty} N p f_1(x) [p F_1(x) + (1-p) F_0(x)]^{N-1}\, dx = \mathbb{E}[q(S_{(N)})]$。*
2. **（概率积分变换形式）** *令 $v = [F_{\text{mix}}(x)]^N \in (0, 1)$，上式严格等价于 $P_{\text{BoN}}(N) = \int_0^1 q\big(F_{\text{mix}}^{-1}(v^{1/N})\big)\, dv = \int_0^1 q\Big(\bar{F}_{\text{mix}}^{-1}\big(-\operatorname{expm1}(\tfrac{\ln v}{N})\big)\Big)\, dv$。*
3. **（含平局原子的一般分布推广）** *设 $F_1, F_0$ 为任意分布函数（允许可数离散原子集 $\mathcal{A}$），最大平局采用均匀随机打破，则对任意 $N \ge 1$ 统一精确成立 $P_{\text{BoN}}(N) = \int_{\mathbb{R}} q(x)\, d([F_{\text{mix}}(x)]^N) = \int_0^1 q(F_{\text{mix}}^{-1}(v^{1/N}))\, dv$，其中原子点 $a \in \mathcal{A}$ 处 $q(a) = \frac{p \Delta F_1(a)}{\Delta F_{\text{mix}}(a)}$。*

*证明.* (1) 无原子时由下标置换对称性与 Fubini 定理即得。(2) 由 $V := [F_{\text{mix}}(S_{(N)})]^N \sim \operatorname{Uniform}(0,1)$ 及无相消恒等式 $1 - v^{1/N} = -\operatorname{expm1}(\frac{\ln v}{N})$ 即证。(3) 在原子 $a\in\mathcal{A}$ 处平局选出的每条样本条件正确率均为 $q(a)$，且 $\mathbb{P}(S_{(N)}=a) = [F_{\text{mix}}(a)]^N - [F_{\text{mix}}(a-)]^N = \Delta([F_{\text{mix}}]^N)(a)$；由 Galois 联络 $F_{\text{mix}}^{-1}(u)\le x \iff u\le F_{\text{mix}}(x)$ 验证推前测度一致即可。$\blacksquare$

**推论 1.1（高斯双样本精确差值与互补恒等式）.** *设 $S|Y=c \sim \mathcal{N}(\mu_c, \sigma_c^2)$，$\Delta\mu = \mu_1 - \mu_0$，则 $P_{\text{BoN}}(2) - P_{\text{BoN}}(1) = p(1-p)\big[2\,\Phi(\frac{\Delta\mu}{\sqrt{\sigma_1^2 + \sigma_0^2}}) - 1\big]$，且连续无平局双样本恒满足 $P_{\text{Anti-BoN}}(2) = 2p - P_{\text{BoN}}(2)$。*（由 $\int f_1 F_0 dx = \mathbb{P}(S^{(1)} \ge S^{(0)})$ 及 $\mathbf{1}\{Y_{\arg\max}=1\} + \mathbf{1}\{Y_{\arg\min}=1\} = Y_1 + Y_2$ 立得。）

### A.2 定理 2：重错尾坍塌、MLRP 判据与固定峰值高度 Gumbel 定理

**定理 2A（重错尾导致的 BoN 无穷采样坍塌）.** *设 $F_{\text{mix}}$ 连续具正密度 $f_1, f_0$。若 $\lim_{x \uparrow x_R} \frac{f_1(x)}{f_0(x)} = 0$，则 $\lim_{N\to\infty} P_{\text{BoN}}(N) = 0$ 且存在有限最优预算 $N^* \in [1, \infty)$；该条件覆盖高斯异方差 $\sigma_0 > \sigma_1$ 与 Reward Hacking 污染混合尾部 $f_0(x) = (1-\epsilon)f_{0,\text{clean}}(x) + \epsilon g_{\text{hack}}(x)$（$\epsilon > 0, \lim_{x\uparrow x_R} f_1/g_{\text{hack}} = 0$）。*

*证明.* 对任意 $\varepsilon > 0$ 取 $x_\varepsilon < x_R$ 使 $x > x_\varepsilon$ 时 $q(x) < \varepsilon/2$，则 $P_{\text{BoN}}(N) \le [F_{\text{mix}}(x_\varepsilon)]^N + \varepsilon/2 \to \varepsilon/2$；由 $P_{\text{BoN}}(1) = p > 0$ 知最优预算必在有限前缀取得。$\blacksquare$

**定理 2B（MLRP 结构单调充要判据与高斯五相相图）.** *设 $F_1, F_0$ 在公共开区间 $I \subseteq \mathbb{R}$ 上具正密度 $f_1, f_0$。*
1. **（$C^1$ 连通闭子区间判据）** *若 $f_1, f_0 \in C^1(I)$ 且正，则对任意 $p \in (0, 1)$、$N \ge 1$ 及非退化闭区间 $[a, b] \subset I$ 均有 $P_{\text{BoN}}(N+1; [a, b]) \ge P_{\text{BoN}}(N; [a, b])$ 的充要条件是 $\Lambda(x) = f_1(x)/f_0(x)$ 在 $I$ 上单调非降。*
2. **（$C^0$ 双闭子区间判据）** *若仅假设 $f_1, f_0 \in C^0(I)$ 且正，则对任意两不相交闭子区间之并 $A = I_1 \cup I_2 \subset I$ 均有 $P_{\text{BoN}}(N+1; A) \ge P_{\text{BoN}}(N; A)$ 的充要条件亦是 $\Lambda(x)$ 在 $I$ 上单调非降。*
3. **（渐近极限）** *若 $\Lambda_R := \lim_{x \uparrow x_R} \Lambda(x) \in [0, +\infty]$ 存在，则 $\lim_{N\to\infty} P_{\text{BoN}}(N) = \frac{p \Lambda_R}{p \Lambda_R + 1 - p}$。*
4. **（高斯五相分类）** *依 $\kappa = \sigma_0/\sigma_1$ 与 $\Delta\mu = \mu_1 - \mu_0$ 划分为：（i）$\kappa > 1$ 坍塌相；（ii）$\kappa=1, \Delta\mu>0$ 临界单调一致；（iii）$\kappa=1, \Delta\mu=0$ 恒常相；（iv）$\kappa=1, \Delta\mu<0$ 单调反转坍塌；（v）$\kappa < 1$ 轻错尾一致相：$\Delta\mu > 0$ 时对一切 $N \ge 1$ 严格单调增至 $1$；$\Delta\mu = 0$ 时初值平局 $P_{\text{BoN}}(1) = P_{\text{BoN}}(2) = p$ 后对一切 $N \ge 2$ 严格单调递增至 $1$；$\Delta\mu < 0$ 时呈先降后升 U 型。*

*证明.* 充分性由 $S_{(N+1)}$ 对 $S_{(N)}$ 的一阶随机占优及非降函数保序立得。$C^1$ 必要性由中值定理取 $q'(x)<0$ 的闭区间 $[a,b]$ 代入引理 1.2(1) 导出矛盾；$C^0$ 必要性对 $q(x_1) - q(x_2) = \delta > 0$（$x_1 < x_2$）取 $A = [x_1-\epsilon, x_1+\epsilon] \cup [x_2-\epsilon, x_2+\epsilon]$，利用精确恒等式 $P_{\text{BoN}}(2; A) - P_{\text{BoN}}(1; A) = w_1^2(P_{\text{BoN}}(2; I_1) - \bar{q}_1) + w_2^2(P_{\text{BoN}}(2; I_2) - \bar{q}_2) - w_1 w_2(\bar{q}_1 - \bar{q}_2) \xrightarrow{\epsilon \to 0^+} -w_1^* w_2^* \delta < 0$ 导出矛盾。(3)–(4) 由引理 1.1、推论 1.1、引理 2.1 及定理 2A 合推。$\blacksquare$

**定理 2C（高斯异方差闭式最优预算与固定峰值高度渐近极限定理）.** *设 $\sigma_0 > \sigma_1$ 且 $\Delta\mu > 0$，记 $x_{\text{peak}} := \frac{\mu_1\sigma_0^2 - \mu_0\sigma_1^2}{\sigma_0^2 - \sigma_1^2}$，$\frac{1}{\tau^2} := \frac{1}{\sigma_1^2} - \frac{1}{\sigma_0^2} > 0$，$N^*_0 := 1/\bar{F}_{\text{mix}}(x_{\text{peak}})$。固定 $p \in (0, 1), \sigma_1 > 0, c_1 > 0$，在 $\kappa := \sigma_0/\sigma_1 \to 1^+$ 的**固定峰值高度渐近机制** $\Delta\mu(\kappa) := \sqrt{c_1(\kappa^2 - 1)}\,\sigma_1$ 下，$q_{\text{peak}}(\kappa) \to \sigma(\operatorname{logit}(p) + c_1/2) \in (0, 1)$，且由 $h \sim \frac{\sqrt{c_1}}{\sigma_1\sqrt{\kappa^2-1}}$ 与 $\tau = \frac{\kappa\sigma_1}{\sqrt{\kappa^2-1}}$ 得无量纲尺度比 $h\tau \sim \frac{\sqrt{c_1}}{\kappa^2-1} \to +\infty$。令 $N(\delta) := e^\delta N^*_0$，规范化遗憾泛函在任意紧集 $[-D, D]$ 上一致收敛：*
$$\mathcal{J}_\kappa(\delta) := \frac{2(h\tau)^2}{q_{\text{peak}}(1 - q_{\text{peak}})}\Big(q_{\text{peak}} - \mathbb{E}\big[q(S_{(N(\delta))})\big]\Big) \xrightarrow{\kappa \to 1^+} (\delta + \gamma)^2 + \frac{\pi^2}{6}, \qquad \lim_{\kappa \to 1^+} \frac{N^*_{\text{exact}}}{N^*_0} = e^{-\gamma} \approx 0.561459.$$

*证明.* 平移令 $\mu_0 = 0$，则 $x_{\text{peak}} = \frac{\kappa^2\sqrt{c_1}}{\sqrt{\kappa^2-1}}\sigma_1 \to +\infty$，$\ell(x_{\text{peak}}) = \ln\kappa + \frac{c_1}{2} \to \frac{c_1}{2}$，$h \sim \frac{\sqrt{c_1}}{\sigma_1\sqrt{\kappa^2-1}}$，$\tau = \frac{\kappa\sigma_1}{\sqrt{\kappa^2-1}}$，故 $h\tau \sim \frac{\sqrt{c_1}}{\kappa^2-1} \to +\infty$。记 $g(u) := \sigma(L_0 - u)$（$u = \frac{y^2}{2\tau^2}$），由 $|g''(s)| \le g(0)(1-g(0))e^s$ 得近邻带 $u \le 1$ 上四次余项界 $\frac{e}{8\tau^4}y^4$ 与远尾截断界 $\frac{1}{q_{\text{peak}}(1-q_{\text{peak}})} + \frac{y^2}{2\tau^2}$。对 $G_N := h(S_{(N_\delta)} - x_{\text{peak}}) - \delta$，由高斯危险率局部渐近一致趋 $1$ 得 $G_N \xrightarrow{d} \operatorname{Gumbel}$；右尾 $t > 0$ 由危险率递增得 $\mathbb{P}(G_N > t) \le e^{D+1-\frac{1}{2}\max(0, t-D)}$；左尾分中段 $x \in [\frac{1}{2}x_{\text{peak}}, x_{\text{peak}}]$（超指数衰减 $\exp(-c' e^{-D} e^{c_0|t+D|})$）与远左段 $x \le a := \frac{1}{2}x_{\text{peak}}$（利用端点精确相容界 $\mathbb{P}(S_{(N_\delta)} \le x) = [F_{\text{mix}}(a)]^{N_\delta - 1}(\frac{F_{\text{mix}}(x)}{F_{\text{mix}}(a)})^{N_\delta - 1}F_{\text{mix}}(x) \le \exp(-e^{\Omega(h x_{\text{peak}})})F_{\text{mix}}(x)$ 压制四阶矩）。由此一致可积给出 $\sup_{\delta \in [-D, D]}|\mathcal{J}_\kappa(\delta) - [(\delta+\gamma)^2 + \frac{\pi^2}{6}]| \to 0$，最后由引理 2.1（Karlin 严格单峰性）将离散整数全局最优点锁定于 $[e^{-\gamma-\varepsilon}, e^{-\gamma+\varepsilon}]$ 内。$\blacksquare$

### A.3 定理 3：离散二元 Verifier 缩放律与 $\text{PPV}=1$ 边界散度分类

**定理 3（二元 Verifier 精确缩放律、分情况散度界与闭式预算）.** *设 $S_i \in \{0,1\}$，$\alpha = \text{TPR} > \beta = \text{FPR}$，$q_1 = p\alpha + (1-p)\beta$，$\text{PPV} = p\alpha/q_1$，$C = \frac{p(1-p)(\alpha-\beta)}{q_1(1-q_1)} > 0$：*
1. $P_{\text{BoN,bin}}(N) = \text{PPV} - C(1-q_1)^N$，单步增益 $\Delta P_{\text{bin}}(N) = p(1-p)(\alpha-\beta)(1-q_1)^{N-1} > 0$；
2. 记 $P_N = \operatorname{Bernoulli}(P_{\text{BoN,bin}}(N)), P_\infty = \operatorname{Bernoulli}(\text{PPV})$：当 $\beta > 0$ 时 $D_{\text{KL}}(P_N \| P_\infty) \le \frac{C^2}{\text{PPV}(1-\text{PPV})}(1-q_1)^{2N}$；当 $\beta = 0$（$\text{PPV}=1, P_\infty=\delta_1$）时 $D_{\text{KL}}(P_N\|\delta_1)=+\infty$，而 $\text{TV}(P_N, P_\infty) = \frac{1-p}{1-p\alpha}(1-p\alpha)^N$ 且反向相对熵有限 $D_{\text{KL}}(\delta_1\|P_N) = -\ln P_{\text{BoN,bin}}(N) = \Theta((1-p\alpha)^N)$；对任意 $\beta \ge 0$，边际条件互信息满足 $I(\hat{Y}_{N+1}; S_{N+1}\mid E_N) \le H_2(q_1)(1-q_1)^N$；
3. 容差 $\varepsilon$ 与线性成本 $\lambda$ 闭式预算分别为 $N^*(\varepsilon) = \max(1, \lceil\frac{\ln(C/\varepsilon)}{-\ln(1-q_1)}\rceil)$ 与 $N^*(\lambda) = \max(1, 1 + \lceil\frac{\ln(p(1-p)(\alpha-\beta)/\lambda)}{-\ln(1-q_1)}\rceil)$。

*证明.* 按全负事件 $\{E_N=0\}$（概率 $(1-q_1)^N$）全期望分解得 (1)，由 $\operatorname{kl}(u,v)\le\chi^2(u,v)$、Dirac 绝对连续性及马尔可夫链数据处理不等式得 (2)–(3)。$\blacksquare$

### A.4 定理 4：校准后验单调性与异质单错类加权多数决

**定理 4A（Calibrated-BoN 消除右尾坍塌）.** *设 $Q_i := q(S_i)$，选择 $\arg\max_{i\le N} Q_i$。则 $P_{\text{Cal-BoN}}(N) = \mathbb{E}[Q_{(N)}]$，当 $\mathbb{P}(Q_1=p)<1$ 时严格单调递增至 $\operatorname{ess\,sup}_{x \in \operatorname{supp}(F_{\text{mix}})} q(x) > p$。*（由塔性质 $\mathbb{P}(Y=1\mid Q)=Q$ a.s. 及定理 1(3) 立证。）

**定理 4B（可数无穷异质错类下的有界加权多数决与超级尾桶指数一致定理）.** *设正确答案 $a^*$ 生成概率为 $p$，错误答案集 $\{a_j\}_{j\in\mathcal{J}}$（允许可数无穷）生成概率 $p_j$（$\sum_j p_j = 1-p$），有界权重 $W_i = w(S_i) \in [0, w_{\max}]$，正确类条件期望权重 $\mu_{w,+} := \mathbb{E}[W \mid A=a^*]$，每个错类各自条件期望权重 $\mu_{w,j} := \mathbb{E}[W \mid A=a_j]$。若严格间隙条件 $\Delta_{\min} := p\mu_{w,+} - \sup_{j\in\mathcal{J}}(p_j\mu_{w,j}) > 0$ 成立，则对普通加权多数决，存在 $M, c > 0$ 使 $\mathbb{P}(\hat{A}_N \neq a^*) \le M e^{-cN}$ 且 $\hat{A}_N \to a^*$ a.s.。*

*证明.* 对前 $K_0$ 个高频错类逐一应用 Hoeffding/Bernstein 不等式于 $Z_{i,j} = W_i(\mathbf{1}\{A_i=a^*\} - \mathbf{1}\{A_i=a_j\})$（均值 $\Delta_j \ge \Delta_{\min} > 0$）；对无穷尾部 $B_{\text{tail}} = \{j > K_0\}$，取 $K_0$ 使 $p_{\text{tail}}(K_0) = \sum_{j>K_0}p_j < \frac{p\mu_{w,+}}{2 w_{\max}}$，则超级尾桶期望得分 $\sum_{j>K_0}p_j\mu_{w,j} \le w_{\max}p_{\text{tail}}(K_0) < \frac{1}{2}p\mu_{w,+}$，以 $K_0+1$ 项有限指数并界结合 Borel–Cantelli 引理完证。$\blacksquare$

---

## 附录 B：有限 $r$ 区域 `[1/1]` Padé 工程插值公式说明

在非远尾渐近（即无量纲危险率曲率 $r := h'(x_{\text{peak}})/h^2$ 显著正值）的实用工程参数域内，直接截断一阶线性偏移项 $-\gamma + \frac{\pi^2}{6}r$ 会在 $r$ 较大时产生过冲。`theorems.py` 中的 `closed_form_optimal_n(..., method="saddle_refined")` 采用与远尾定理 2C 相匹配的 `[1/1]` Padé 平滑插值：
$$h\,(u^*_{\text{Padé}} - x_{\text{peak}}) := -\gamma + \frac{\frac{\pi^2}{6} r}{1 + \frac{\pi^2}{6} r}, \qquad N^*_{\text{saddle}} := \operatorname{round}\!\left(\frac{-1}{\ln\big(1 - \bar{F}_{\text{mix}}(u^*_{\text{Padé}})\big)}\right).$$
该式属于在 $r \to 0^+$ 时保留一阶远尾极限斜率、在有限 $r$ 上有界化平滑的工程插值公式，而非独立渐近极限定理（在 104 组内点参数扫描中将零阶解 $N^*_0$ 的中位相对误差由 `26.04%` 降低至 `8.28%`，$R^2 = 0.9989$）。
