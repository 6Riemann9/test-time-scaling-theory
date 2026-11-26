# 测试时计算缩放的极限行为与相变理论：概率论证明稿

**摘要（Abstract）.** 本文围绕大语言模型（LLM）推理阶段的测试时计算缩放（Test-Time Scaling, TTS）建立概率论与统计决策分析框架。全文分为两部分：

- **第一部分：预备理论与单裁判基础（定理 1–4）**：在统一生成—打分测度空间中重述并细化单裁判顺序统计量积分（含离散平局原子）、$C^0/C^1$ 分离的似然比单调（MLRP）判据、固定峰值高度机制下的远尾 Gumbel 极限 $e^{-\gamma}N^*_0$、二元 Verifier 缩放律与 $\text{PPV}=1$ 边界散度分类，以及单错类条件期望权重约束下的有界加权多数决指数一致界；
- **第二部分：多裁判融合、有限样本联合标度与题目异质性理论（定理 5–7）**：
  1. **多噪声 Verifier 集成相变、帕累托光谱路径与 QDA 维度律（定理 5）**：建立一般非对称高斯 $(\Sigma_1, \Sigma_0)$ 下的广义特征值负权逆转判据；求解以渐近尾方差比 $\kappa(\mathbf{w})\le\kappa_{\max}$ 为硬约束的 **$N=2$ 线性信噪比最大化**一维光谱路径 $\mathbf{w}_\tau=(\Sigma_0-\tau\Sigma_1)^{-1}\Delta\boldsymbol{\mu}$（`Safe-Fisher`），并刻画不可逆相 $\Sigma_0\succ\Sigma_1$ 下 Bayes QDA 天花板与 $N^{-2/K}$ 收敛阶；利用正乘子本质集 $I_{+,c}$ 与弱激活正象限常数 $\alpha_{c, I_0}$ 将多元高斯最小值 `Min-Ensemble` 与 Gaussian-Copula + Student-$t$ 渐近判据推广至含零乘子退化的任意正定相关矩阵（精确区分 $K=1$ 时 $\eta_c=1$、$K\ge 2$ 且 $R_c\succ 0$ 时 $\eta_c\in(0,1)$，以及非负相关 $(R_c)_{ij}\ge 0$ 作为 $\eta_c\in[1/K,1)$ 的充分非必要条件）；
  2. **有限校准集 $M$ 与推理算力 $N$ 联合 Minimax 标度律（定理 6）**：区分固定分布逐点渐近定理与**一致正则分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$** 上的非渐近 Minimax 理论；利用三事件并界（联合 LCB 上越界 + 桶计数 + Hoeffding 下侧偏差）与最近峰在线样本耦合，证得 $N$-无关等频 `KL-LCB` 在所有 $\beta>0$ 与 $(M,N)$ 区域上达到 $O\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$ 上界并与 Fano / Oracle 联合下界相匹配；同时借助样本分割 Uniform 顺序统计量 Gamma 间距表示（命题 6.1）将未知总体分位数下的经验等频分桶推广覆盖至一切 $\beta > 0$（消解加性 DKW 界在 $0<\beta\le 0.5$ 处的相对误差发散）；
  3. **题目异质性、辛普森遮蔽与样本标准化诱导尾反转（定理 7）**：通过两层层次概率模型明确界定单分布条件结论与跨题共享校准器的适用边界；推导池化后验导数分解并给出显式有限参数二层高斯遮蔽实例；给出 $N=2$ 样本 Z-Score 三类纯秩规则非严格上界及严格不等号成立的双向序反转充要条件；证明在含不可解题原子 $\pi_0>0$ 的同方差单调基准上，总体标准化 $Z^*$ 无条件诱导出右尾极限 $\frac{\pi_1}{\pi_0+\pi_1}$（区分多内点反例 $\kappa_Z<1$ 与单内点充分条件 $\kappa_Z>1$），并进一步通过在线样本标准化极值一致收敛 $\max_{1\le i\le N}|Z_{i,N}-Z_i^*(X)| = O_p(\sqrt{(\ln N)/N})$ 严格证得实际在线样本 $Z_{i,N}$ Oracle 选择器的渐近上限损失（定理 7C(4)）及三分支留出门控选择遗憾界（命题 7.1）；
- **附录 A**：有限 $r$ 区域 `[1/1]` Padé 工程插值公式说明。

---

## 第一部分：预备理论与单裁判基础（定理 1–4）

> **文献定位说明**：本部分定理 1–4 整理并细化单裁判 Test-Time Scaling 的基础数学工具（详细文献对照见 [RELATED_WORK.md](RELATED_WORK.md)），不主张对既有单裁判核心发现的首创归属，主要作为第二部分定理 5–7 的分析基础。

### 1. 概率空间、层次生成结构与预备引理

#### 1.1 生成—验证层次概率模型与定理适用域界定

真实推理与评测任务自然服从两级**层次生成—验证概率模型** $(\Omega, \mathcal{F}, \mathbb{P})$：首先从题目总体抽取问题 $X \sim \mathcal{P}_X$；再在给定 $X$ 的条件下从生成器独立同分布（i.i.d.）抽取 $N \ge 1$ 条候选样本 $\{(Y_i, S_i)\}_{i=1}^N \mid X \stackrel{\text{i.i.d.}}{\sim} P_{Y, S \mid X}$，以及（若使用离线校准）基于独立抽取的有标签校准集 $\mathcal{D}_M$。其中：
- $Y_i \in \{0, 1\}$ 为样本 $i$ 的真实正确性示性变量，$p(X) := \mathbb{P}(Y_i = 1 \mid X)$ 为题内先验正确率；
- $S_i \in \mathbb{R}$ 为 Verifier 标量打分（定理 5 推广至 $\mathbf{S}_i \in \mathbb{R}^K$）。给定 $(X, Y_i = c)$（$c \in \{0, 1\}$），条件分布函数记为 $F_{c, X}(s) := \mathbb{P}(S_i \le s \mid Y_i = c, X)$。

> **模型适用域统一界定（固定题条件保证 vs. 跨题池化校准）**：
> - 在 **定理 1–6** 中，我们固定单个题目（或考察条件分布 $P_{Y, S \mid X}$ 不随 $X$ 变化的同质总体），此时可简化记号为 $p = p(x)$、$F_c = F_{c, x}$，对应单题内部条件采样模型 $(Y_i, S_i) \stackrel{\text{i.i.d.}}{\sim} P$。在此条件意义下，单题内各候选样本严格独立同分布，从而后验排序最优性、置信覆盖与 Minimax 遗憾界成立。
> - 当题目存在跨题异质性（$P_{Y, S \mid X}$ 随 $X$ 变化）时，样本对 $(Y_i, S_i)$ 边际上仅条件独立（可交换）而非 i.i.d.，池化边际后验 $q_{\text{pool}}(s) = \mathbb{P}(Y=1 \mid S=s)$ 通常**不等于**给定全池后的条件后验 $\mathbb{P}(Y_i=1 \mid S_1,\dots,S_N, X)$。因此，定理 4A 与定理 6A–6C 的分布无关或单峰最优保证**不能未经修正地直接迁移至跨题共享的池化校准器**。**定理 7** 专门研究这种跨题层次错配引发的辛普森遮蔽与标准化权衡问题。

对固定分布 $P$（即给定 $X=x$ 或同质基准），令 $p \in (0, 1)$，对应概率测度 $\mu_c$ 与生存函数 $\bar{F}_c(x) := 1 - F_c(x)$（绝对连续时密度记为 $f_c(x)$）。混合分布函数、生存函数与密度分别为：
$$F_{\text{mix}}(x) = p F_1(x) + (1-p) F_0(x), \qquad \bar{F}_{\text{mix}}(x) = p \bar{F}_1(x) + (1-p) \bar{F}_0(x), \qquad f_{\text{mix}}(x) = p f_1(x) + (1-p) f_0(x).$$
记右端点为 $x_R := \sup\{x \in \mathbb{R} : F_{\text{mix}}(x) < 1\} \in (-\infty, +\infty]$，左端点为 $x_L := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) > 0\} \in [-\infty, +\infty)$。定义左连续广义分位数逆函数与上分位数逆函数：
$$F_{\text{mix}}^{-1}(u) := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) \ge u\}, \quad u \in (0, 1); \qquad \bar{F}_{\text{mix}}^{-1}(q) := F_{\text{mix}}^{-1}(1 - q), \quad q \in (0, 1).$$
记混合测度为 $\mu_{\text{mix}} := p \mu_1 + (1-p) \mu_0$。因 $p \mu_1 \ll \mu_{\text{mix}}$，由 Radon-Nikodym 定理，给定打分 $S = x$ 时的贝叶斯后验正确率为：
$$q(x) := \mathbb{P}(Y = 1 \mid S = x) = \frac{d(p \mu_1)}{d\mu_{\text{mix}}}(x) \in [0, 1] \quad (\mu_{\text{mix}}\text{-a.e.}).$$
在正密度支撑集 $\{x : f_{\text{mix}}(x) > 0, f_0(x) > 0\}$ 上定义似然比 $\Lambda(x) := \frac{f_1(x)}{f_0(x)} \in [0, +\infty)$ 与对数似然比 $\ell(x) := \ln \Lambda(x) = \ln f_1(x) - \ln f_0(x)$。

#### 1.2 标准 Best-of-$N$ 决策规则与预备引理

给定采样预算 $N \ge 1$，记最大顺序统计量 $S_{(N)} := \max_{1 \le i \le N} S_i$，最大值下标集 $\mathcal{I}_{\max} := \{i \in \{1,\dots,N\} : S_i = S_{(N)}\}$。标准 Best-of-$N$（BoN）选择器从 $\mathcal{I}_{\max}$ 中均匀随机抽取一个下标 $i^*$，其精确正确率记为 $P_{\text{BoN}}(N) := \mathbb{P}(Y_{i^*} = 1)$。

**引理 1.1（后验与对数似然比的 Logistic 映射恒等式）.** *设 $f_1(x), f_0(x)$ 在区间 $I$ 上正且连续可微（$f_1, f_0 \in C^1(I)$），则对任意 $x \in I$ 有：*
$$q(x) = \sigma\big(\operatorname{logit}(p) + \ell(x)\big) = \frac{p \Lambda(x)}{p \Lambda(x) + 1 - p}, \qquad q'(x) = q(x)\big(1 - q(x)\big)\ell'(x),$$
*其中 $\sigma(t) := (1 + e^{-t})^{-1}$，$\operatorname{logit}(p) := \ln\frac{p}{1-p}$。特别地，$q'(x)$ 与 $\ell'(x)$ 严格同号；若仅假设 $f_1, f_0 \in C^0(I)$ 且正，则对任意 $x_1, x_2 \in I$ 仍有 $q(x_2) > q(x_1) \iff \Lambda(x_2) > \Lambda(x_1)$ 严格保序。*

*证明.* 由 Bayes 公式及 $\sigma$ 的严格递增性与导数 $\sigma'(t) = \sigma(t)(1-\sigma(t))$ 立得。$\blacksquare$

**引理 1.2（离散差分与连续预算导数恒等式）.** *设 $F_{\text{mix}}$ 连续且 $q(x)$ 具有局部分部积分正则性（局部绝对连续或局部有界变差），且 $P_{\text{BoN}}(N) = \int_{\mathbb{R}} q(x)\, d\big([F_{\text{mix}}(x)]^N\big)$。则：*
1. *对任意整数 $N \ge 1$，$P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N) = \int_{-\infty}^{\infty} [F_{\text{mix}}(x)]^N \big(1 - F_{\text{mix}}(x)\big)\, dq(x)$；特别若 $q \in C^1$ 且有界，则 $\int_{-\infty}^\infty [F_{\text{mix}}(x)]^N(1-F_{\text{mix}}(x))q'(x)\,dx$。*
2. *将 $N$ 延拓至 $[1, \infty)$，若 $\int_{\mathbb{R}} |dq| < \infty$，则 $\frac{\partial P_{\text{BoN}}(N)}{\partial N} = \int_{-\infty}^{\infty} \big(-\ln F_{\text{mix}}(x)\big) [F_{\text{mix}}(x)]^N\, dq(x)$。*

*证明.* 记 $G_N(x) := [F_{\text{mix}}(x)]^N$。因 $q(x)\in[0,1]$ 一致有界且端点差值 $G_{N+1}-G_N$ 在 $\pm\infty$ 均为 $0$，由 Lebesgue-Stieltjes 分部积分公式立得 (1)。对 (2)，在任意闭区间 $[N_a, N_b]\subset [1,\infty)$ 上偏导核 $(-\ln u)u^N \le (e N_a)^{-1}$，结合 $\int_{\mathbb{R}}|dq|<\infty$ 由控制收敛定理即允许积分号下求导。$\blacksquare$

---

### 2. 定理 1：精确顺序统计量积分公式与含原子统一推广

**定理 1（BoN 精确顺序统计量公式、PIT 等价形式与含原子推广）.**
1. **（连续无原子情形）** *若 $F_1, F_0$ 均绝对连续（密度 $f_1, f_0$），则对任意 $N \ge 1$：*
   $$P_{\text{BoN}}(N) = \int_{-\infty}^{\infty} N p f_1(x) \big[p F_1(x) + (1-p) F_0(x)\big]^{N-1}\, dx = \mathbb{E}\big[q(S_{(N)})\big].$$
2. **（概率积分变换形式）** *令 $v = [F_{\text{mix}}(x)]^N \in (0, 1)$，上式严格等价于：*
   $$P_{\text{BoN}}(N) = \int_0^1 q\big(F_{\text{mix}}^{-1}(v^{1/N})\big)\, dv = \int_0^1 q\Big(\bar{F}_{\text{mix}}^{-1}\big(-\operatorname{expm1}(\tfrac{\ln v}{N})\big)\Big)\, dv.$$
3. **（含平局原子的一般分布推广）** *设 $F_1, F_0$ 为任意分布函数（允许可数离散原子集 $\mathcal{A}$），最大平局采用均匀随机打破，则对任意 $N \ge 1$ 统一精确成立：*
   $$P_{\text{BoN}}(N) = \int_{\mathbb{R}} q(x)\, d\big([F_{\text{mix}}(x)]^N\big) = \int_0^1 q\big(F_{\text{mix}}^{-1}(v^{1/N})\big)\, dv,$$
   *其中在原子点 $a \in \mathcal{A}$ 处 $q(a) = \frac{p \Delta F_1(a)}{\Delta F_{\text{mix}}(a)}$。*

*证明.* (1) 无原子时几乎必然无平局，由下标置换对称性与 Fubini 定理即得，结合 $pf_1(x)=q(x)f_{\text{mix}}(x)$ 即为 $\mathbb{E}[q(S_{(N)})]$。(2) 由 $U := F_{\text{mix}}(S_{(N)})$ 满足 $\mathbb{P}(U\le u)=u^N$、$V:=U^N\sim\text{Uniform}(0,1)$ 及上尾无相消恒等式 $1 - v^{1/N} = -\operatorname{expm1}(\frac{\ln v}{N})$ 即证。(3) 在每个原子 $a\in\mathcal{A}$ 处平局选出的每条样本条件正确概率均等于 $q(a)$，且 $\mathbb{P}(S_{(N)}=a) = \sum_{k=1}^N \binom{N}{k}[\Delta F_{\text{mix}}(a)]^k [F_{\text{mix}}(a-)]^{N-k} = [F_{\text{mix}}(a)]^N - [F_{\text{mix}}(a-)]^N = \Delta([F_{\text{mix}}]^N)(a)$；再由 Galois 联络 $F_{\text{mix}}^{-1}(u)\le x \iff u\le F_{\text{mix}}(x)$ 验证推前测度一致性即可。$\blacksquare$

**推论 1.1（高斯 Verifier 两样本精确差值恒等式）.** *设 $S|Y=1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$，$S|Y=0 \sim \mathcal{N}(\mu_0, \sigma_0^2)$，$\Delta\mu = \mu_1 - \mu_0$，则：*
$$P_{\text{BoN}}(2) - P_{\text{BoN}}(1) = p(1-p)\left[2\,\Phi\!\left(\frac{\Delta\mu}{\sqrt{\sigma_1^2 + \sigma_0^2}}\right) - 1\right],$$
*且同时选择低分者（$\text{Anti-BoN}(2)$）满足精确互补恒等式 $P_{\text{Anti-BoN}}(2) = 2p - P_{\text{BoN}}(2)$。*

*证明.* 由 $\int f_1 F_1\, dx = 1/2$、$\int f_1 F_0\, dx = \mathbb{P}(S^{(1)} \ge S^{(0)})$ 减去 $p = p^2 + p(1-p)$ 得前式；由对任意连续无平局双样本 $\mathbf{1}\{Y_{\arg\max}=1\} + \mathbf{1}\{Y_{\arg\min}=1\} = Y_1 + Y_2$ 取期望即得后式。$\blacksquare$

---

### 3. 定理 2：极值相变坍塌、MLRP 结构判据与固定峰值高度 Gumbel 定理

#### 3.1 Karlin 单交叉保号引理

**引理 2.1（极值顺序统计量的单交叉单峰/单谷性引理）.** *设 $F_{\text{mix}}$ 连续严格递增，$q$ 连续可微，$q'(x)$ 至多在唯一阈值 $x_c \in \mathbb{R}$ 变号，$u_c := F_{\text{mix}}(x_c) \in (0, 1)$，$I(N) := P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N)$，$\phi(N) := \partial_N P_{\text{BoN}}(N)$。*
1. *若 $(x - x_c)q'(x) < 0$（$x \neq x_c$），则对任意 $N > N_0 \ge 1$ 有 $I(N) < u_c^{N-N_0} I(N_0)$ 与 $\phi(N) < u_c^{N-N_0}\phi(N_0)$，故 $\{P_{\text{BoN}}(N)\}_{N \ge 1}$ 严格单峰（存在唯一或至多相邻两个整数全局最大点）。*
2. *若 $(x - x_c)q'(x) > 0$（$x \neq x_c$），则不等号反向；若 $I(1)\ge 0$ 则全局严格单调递增，若 $I(1) < 0 < \lim_{N\to\infty}P_{\text{BoN}}(N)-p$ 则呈严格单谷 U 型。*

*证明.* 核函数 $\big([F_{\text{mix}}(x)]^{N-N_0} - u_c^{N-N_0}\big)q'(x)$ 在 $x \neq x_c$ 处处严格同号，乘以正权后由引理 1.2 积分立得。$\blacksquare$

#### 3.2 定理 2A：重错尾极值坍塌与污染尾部崩溃

**定理 2A（重错尾导致的 BoN 无穷采样坍塌与有限最优预算存在性）.** *设 $F_{\text{mix}}$ 连续（右端点 $x_R \le +\infty$）具正密度 $f_1, f_0$。若 $\lim_{x \uparrow x_R} \frac{f_1(x)}{f_0(x)} = 0$，则：*
1. $\lim_{N\to\infty} P_{\text{BoN}}(N) = 0$，*且存在有限最优预算 $N^* \in [1, \infty)$；*
2. *上述条件严格覆盖：**（a）高斯异方差 $\sigma_0 > \sigma_1$**；以及 **（b）Reward Hacking 污染混合尾部** $f_0(x) = (1-\epsilon)f_{0,\text{clean}}(x) + \epsilon g_{\text{hack}}(x)$（$\epsilon > 0$ 且 $\lim_{x\uparrow x_R} f_1(x)/g_{\text{hack}}(x) = 0$）。*

*证明.* (1) 对任意 $\varepsilon > 0$，取 $x_\varepsilon < x_R$ 使 $x > x_\varepsilon$ 时 $q(x) < \varepsilon/2$，则 $P_{\text{BoN}}(N) \le [F_{\text{mix}}(x_\varepsilon)]^N + \varepsilon/2 \to \varepsilon/2$。因 $P_{\text{BoN}}(1) = p > 0$，全局最大必在有限前缀中取得。(2) 对 (a)，$\ell(x)$ 的 $x^2$ 系数 $-\frac{1}{2}(\sigma_1^{-2}-\sigma_0^{-2}) < 0$；对 (b)，利用 $0 \le f_1/f_0 \le \epsilon^{-1}(f_1/g_{\text{hack}}) \to 0$ 夹逼即证。$\blacksquare$

#### 3.3 定理 2B：MLRP 结构单调充要判据（$C^1$ 与 $C^0$ 严格分离版）与高斯五相相图

**定理 2B（MLRP 结构单调充要判据与高斯完整相图）.** *设 $F_1, F_0$ 在公共开区间 $I \subseteq \mathbb{R}$ 上具正密度 $f_1, f_0$。*
1. **（$C^1$ 密度下的连通闭子区间判据）** *若 $f_1, f_0 \in C^1(I)$ 且处处严格为正，则“对任意先验 $p \in (0, 1)$、任意 $N \ge 1$ 及任意非退化连通闭子区间条件域 $[a, b] \subset I$ 均有 $P_{\text{BoN}}(N+1; [a, b]) \ge P_{\text{BoN}}(N; [a, b])$”成立的**充分必要条件**是似然比 $\Lambda(x) = f_1(x)/f_0(x)$ 在 $I$ 上单调非降（MLRP）。*
2. **（仅需 $C^0$ 连续密度的正测度子集判据）** *若仅假设 $f_1, f_0 \in C^0(I)$ 且处处严格为正（无需可微性！），则“对任意先验 $p \in (0, 1)$、任意 $N \ge 1$ 及任意正测度 Borel 条件子集 $A \subset I$（等价地，即使只对两不相交闭子区间之并 $A = I_1 \cup I_2 \subset I$）均有 $P_{\text{BoN}}(N+1; A) \ge P_{\text{BoN}}(N; A)$”成立的**充分必要条件**亦是 $\Lambda(x)$ 在 $I$ 上单调非降。*
3. **（一般渐近极限判据）** *若 $\Lambda_R := \lim_{x \uparrow x_R} \Lambda(x) \in [0, +\infty]$ 存在，则 $\lim_{N\to\infty} P_{\text{BoN}}(N) = \frac{p \Lambda_R}{p \Lambda_R + 1 - p}$（$\Lambda_R = +\infty$ 时为 $1$）。*
4. **（高斯族五相分类及 $\kappa < 1, \Delta\mu = 0$ 边界，临界相界 $\kappa_c = \sigma_0/\sigma_1 = 1$）** *高斯族依 $\kappa = \sigma_0/\sigma_1$ 与 $\Delta\mu = \mu_1 - \mu_0$ 严格划分为：（i）$\kappa > 1$ 坍塌相（$\Delta\mu > 0$ 时唯一有限峰 $N^* \ge 2$，$\Delta\mu \le 0$ 时 $N^*=1$）；（ii）$\kappa=1, \Delta\mu>0$ 临界单调一致；（iii）$\kappa=1, \Delta\mu=0$ 恒常相；（iv）$\kappa=1, \Delta\mu<0$ 单调反转坍塌；（v）$\kappa < 1$ 轻错尾一致相：当 $\Delta\mu > 0$ 时对一切 $N \ge 1$ 严格单调增至 $1$；当 $\Delta\mu = 0$ 时初值平局 $P_{\text{BoN}}(1) = P_{\text{BoN}}(2) = p$（由推论 1.1），随后对一切 $N \ge 2$ 严格单调递增至 $1$；当 $\Delta\mu < 0$ 时呈严格先降后升 U 型（$P_{\text{BoN}}(2) < P_{\text{BoN}}(1) = p$，经唯一有限谷底后升向 $1$）。*

*证明.*
- **充分性（对 (1) 与 (2) 同成立）**：限制到任意正测度 Borel 子集 $A \subset I$（含 $[a,b]$）时条件后验 $q|_{A}$ 保持 $q$ 自身在 $A$ 上不变；若 $\Lambda$ 非降，则由引理 1.1 知 $q$ 在 $A$ 上单调非降。注意 $S_{(N+1)}$ 一阶随机占优于 $S_{(N)}$（对所有 $x$，$\mathbb{P}(S_{(N+1)} > x \mid S \in A) = 1 - [F_A(x)]^{N+1} \ge 1 - [F_A(x)]^N = \mathbb{P}(S_{(N)} > x \mid S \in A)$），结合非降有界函数对一阶随机占优保序即得 $P_{\text{BoN}}(N+1; A) \ge P_{\text{BoN}}(N; A)$（无需 $q$ 可微）。
- **必要性（1：$f_1, f_0 \in C^1(I)$ 连通区间论证）**：由引理 1.1 知 $q \in C^1(I)$。反设 $\Lambda$ 非单调非降，则存在 $x_1 < x_2$ 使 $q(x_2) < q(x_1)$。由 Lagrange 中值定理，存在 $x_0 \in (x_1, x_2)$ 使 $q'(x_0) = \frac{q(x_2)-q(x_1)}{x_2-x_1} < 0$；再由 $q'$ 连续性，存在闭子区间 $[a, b] \subset I$（$a < b$）使得 $q'(x) < 0$ 在 $[a, b]$ 上**处处严格成立**。在该连通区间上对 $N=1$ 应用引理 1.2(1)：
  $$P_{\text{BoN}}(2; [a, b]) - P_{\text{BoN}}(1; [a, b]) = \int_a^b F_{[a, b]}(x)\big(1 - F_{[a, b]}(x)\big)\, q'(x)\, dx < 0,$$
  矛盾！
- **必要性（2：仅需 $f_1, f_0 \in C^0(I)$ 的双子区间论证，无需可微性）**：反设 $\Lambda$ 非单调非降，存在 $x_1 < x_2 \in I$ 使 $\delta := q(x_1) - q(x_2) > 0$。取充分小 $\epsilon > 0$ 令两个闭子区间 $I_1 := [x_1 - \epsilon, x_1 + \epsilon]$ 与 $I_2 := [x_2 - \epsilon, x_2 + \epsilon]$ 完全落入 $I$ 内且互不相交（$x_1 + \epsilon < x_2 - \epsilon$）。令条件子集 $A := I_1 \cup I_2$。记 $w_k := \frac{\mu_{\text{mix}}(I_k)}{\mu_{\text{mix}}(A)} \in (0, 1)$（$w_1 + w_2 = 1$）与区间条件均值 $\bar{q}_k := \mathbb{E}[q(S) \mid S \in I_k]$。给定两独立条件样本 $S_1, S_2 \in A$，按各样本落入 $I_1$ 或 $I_2$ 分解全期望（当恰有一条落入 $I_2$ 而另一条落入 $I_1$ 时，因 $I_2$ 整体位于 $I_1$ 右侧，最大值必为来自 $I_2$ 的样本，贡献 $2 w_1 w_2 \bar{q}_2$）：
  $$P_{\text{BoN}}(2; A) = w_1^2\, P_{\text{BoN}}(2; I_1) + w_2^2\, P_{\text{BoN}}(2; I_2) + 2 w_1 w_2\, \bar{q}_2,$$
  而 $P_{\text{BoN}}(1; A) = w_1 \bar{q}_1 + w_2 \bar{q}_2 = w_1^2 \bar{q}_1 + w_2^2 \bar{q}_2 + w_1 w_2 (\bar{q}_1 + \bar{q}_2)$，相减得精确代数恒等式：
  $$P_{\text{BoN}}(2; A) - P_{\text{BoN}}(1; A) = w_1^2\big(P_{\text{BoN}}(2; I_1) - \bar{q}_1\big) + w_2^2\big(P_{\text{BoN}}(2; I_2) - \bar{q}_2\big) - w_1 w_2\big(\bar{q}_1 - \bar{q}_2\big).$$
  当 $\epsilon \to 0^+$ 时，因 $f_{\text{mix}}$ 在 $I$ 正且连续，有 $w_k \to w_k^* := \frac{f_{\text{mix}}(x_k)}{f_{\text{mix}}(x_1)+f_{\text{mix}}(x_2)} \in (0, 1)$；由 $q \in C^0(I)$ 知 $|P_{\text{BoN}}(2; I_k) - \bar{q}_k| \le \sup_{u, v \in I_k}|q(u)-q(v)| \to 0$ 且 $\bar{q}_1 - \bar{q}_2 \to q(x_1) - q(x_2) = \delta > 0$。从而当 $\epsilon$ 足够小时上式严格趋近 $-w_1^* w_2^* \delta < 0$，导出 $P_{\text{BoN}}(2; A) < P_{\text{BoN}}(1; A)$ 的严格矛盾！
- (3)–(4) 由引理 1.1、推论 1.1、引理 2.1 及定理 2A 直接合推。$\blacksquare$

#### 3.4 定理 2C：闭式最优预算 $N^*$ 与固定峰值高度下的远尾 Gumbel 渐近定理

**定理 2C（高斯异方差闭式最优预算与固定峰值高度渐近极限定理）.** *设 $\sigma_0 > \sigma_1$ 且 $\Delta\mu > 0$，记 $x_{\text{peak}} := \frac{\mu_1\sigma_0^2 - \mu_0\sigma_1^2}{\sigma_0^2 - \sigma_1^2}$，$\frac{1}{\tau^2} := \frac{1}{\sigma_1^2} - \frac{1}{\sigma_0^2} > 0$，$z_k := \frac{x_{\text{peak}}-\mu_k}{\sigma_k} > 0$。*
1. **（精确偶对称后验）** *$\ell(x) = \ell(x_{\text{peak}}) - \frac{(x-x_{\text{peak}})^2}{2\tau^2}$，故 $q(x_{\text{peak}}+y)$ 精确偶对称且在 $y=0$ 取唯一最大值 $q_{\text{peak}} := q(x_{\text{peak}})$。*
2. **（零阶分位解）** *$N^*_0 := 1/\bar{F}_{\text{mix}}(x_{\text{peak}}) = \sqrt{2\pi}\big[p e^{-z_1^2/2}M(z_1) + (1-p)e^{-z_0^2/2}M(z_0)\big]^{-1}$，利用 Boyd 双侧 Mills 不等式即得显式可计算上下界。*
3. **（固定峰值高度机制下的严格 Gumbel 渐近极限定理）** *固定任意 $p \in (0, 1)$、$\sigma_1 > 0$ 与形状参数 $c_1 > 0$，考虑令异方差比 $\kappa := \sigma_0/\sigma_1 \to 1^+$ 的**固定峰值高度渐近机制**：*
   $$\Delta\mu(\kappa) := \sqrt{c_1(\kappa^2 - 1)}\,\sigma_1, \qquad x_{\text{peak}}(\kappa) - \mu_0 = \frac{\kappa^2 \sqrt{c_1}}{\sqrt{\kappa^2 - 1}}\,\sigma_1 \to +\infty.$$
   *在此机制下，$q_{\text{peak}}(\kappa) \to \sigma(\operatorname{logit}(p) + c_1/2) \in (0, 1)$ 严格远离 $\{0, 1\}$，$x_{\text{peak}}$ 处混合危险率 $h := f_{\text{mix}}(x_{\text{peak}})/\bar{F}_{\text{mix}}(x_{\text{peak}})$ 满足无量纲尺度比 $h\tau \to +\infty$。令 $N(\delta) := e^\delta N^*_0$（$\delta \in \mathbb{R}$），规范化遗憾泛函在任意紧集 $[-D, D]$ 上一致收敛至纯二次 Gumbel 泛函：*
   $$\mathcal{J}_\kappa(\delta) := \frac{2(h\tau)^2}{q_{\text{peak}}(1 - q_{\text{peak}})}\Big(q_{\text{peak}} - \mathbb{E}\big[q(S_{(N(\delta))})\big]\Big) \xrightarrow{\kappa \to 1^+} (\delta + \gamma)^2 + \frac{\pi^2}{6},$$
   *结合引理 2.1（Karlin 严格单峰性），全局唯一最优离散预算 $N^*_{\text{exact}} := \arg\max_{N \ge 1} P_{\text{BoN}}(N)$ 严格满足：*
   $$\lim_{\kappa \to 1^+} \frac{N^*_{\text{exact}}}{N^*_0} = e^{-\gamma} \approx 0.561459.$$

*五步证明（控制四阶泰勒余项与极值尾矩）.*

- **第 1 步（参数渐近尺度控制）.** 不失一般性平移行轴令 $\mu_0 = 0$。在 $\Delta\mu(\kappa) = \sqrt{c_1(\kappa^2-1)}\,\sigma_1$ 下直接代入解析式：
  $$x_{\text{peak}} = \frac{\kappa^2\sqrt{c_1}}{\sqrt{\kappa^2-1}}\sigma_1 \to +\infty, \qquad z_0 = \frac{x_{\text{peak}}}{\kappa\sigma_1} = \frac{\kappa\sqrt{c_1}}{\sqrt{\kappa^2-1}} \to +\infty, \qquad z_1 = \frac{\sqrt{c_1}}{\sqrt{\kappa^2-1}} \to +\infty.$$
  而对数似然比峰值 $\ell(x_{\text{peak}}) = \ln\kappa + \frac{1}{2}(z_0^2 - z_1^2) = \ln\kappa + \frac{c_1}{2} \to \frac{c_1}{2}$，故 $q_{\text{peak}} = \sigma(\operatorname{logit}(p) + \ell(x_{\text{peak}})) \to q_\infty := \sigma(\operatorname{logit}(p) + c_1/2) \in [\epsilon_0, 1-\epsilon_0]$（对某个固定 $\epsilon_0 \in (0, 1/2)$ 一致有界）。混合危险率 $h = w_1 h_1 + w_0 h_0$，其中 $h_k := \frac{\phi(z_k)}{\sigma_k \bar{\Phi}(z_k)} = \frac{z_k}{\sigma_k}(1 + O(z_k^{-2})) = \frac{\sqrt{c_1}}{\sigma_1\sqrt{\kappa^2-1}}(1 + o(1))$（注意因 $\kappa \to 1^+$，$h_1 \sim h_0 \sim h$ 同阶！），且 $\tau = \frac{\kappa\sigma_1}{\sqrt{\kappa^2-1}}$，故核心大参数满足：
  $$h\tau \sim \frac{c_1}{\kappa^2 - 1} \to +\infty, \qquad \frac{\bar{F}_{\text{mix}}(x_{\text{peak}})}{(2\pi)^{-1/2} z_1^{-1} e^{-z_1^2/2}} = \Theta(1) \implies N^*_0 = \exp\!\left(\frac{c_1}{2(\kappa^2-1)}(1+o(1))\right) \to +\infty.$$

- **第 2 步（偶对称后验带四阶 Lagrange / 积分余项的非渐近分段界）.** 记 $L_0 := \operatorname{logit}(p) + \ell(x_{\text{peak}})$，$g(u) := \sigma(L_0 - u)$（对 $u = \frac{y^2}{2\tau^2} \ge 0$，$q(x_{\text{peak}} + y) = g(u)$）。由 $g'(u) = -g(u)(1-g(u))$ 与 $g''(u) = g(u)(1-g(u))\big(1 - 2g(u)\big)$。注意到对一切 $s \ge 0$：因 $L_0 - s \le L_0$ 得 $g(s) \le g(0)$；且 $1 - g(s) = \frac{1}{1 + e^{L_0 - s}} = \frac{e^s}{e^s + e^{L_0}} \le e^s \frac{1}{1 + e^{L_0}} = e^s(1 - g(0))$，因此处处成立非渐近估计：
  $$\big|g''(s)\big| \le g(0)\big(1 - g(0)\big) e^s \qquad (\forall\, s \ge 0).$$
  由积分余项 Taylor 公式 $g(u) = g(0) + g'(0)u + \int_0^u (u-s)g''(s)\,ds$：
  - **在近邻带 $0 \le u \le 1$（对应 $|y| \le \sqrt{2}\tau$）上**：$e^s \le e$，故直接有精确四次方控制：
    $$\left|\frac{q_{\text{peak}} - q(x_{\text{peak}} + y)}{q_{\text{peak}}(1 - q_{\text{peak}})} - \frac{y^2}{2\tau^2}\right| \le \frac{e}{2}\, u^2 = \frac{e}{8\,\tau^4}\, y^4.$$
  - **在远尾截断区 $u > 1$（对应 $|y| > \sqrt{2}\tau$）上**：仅利用后验全局有界性 $0 \le q_{\text{peak}} - q \le 1$，有：
    $$\left|\frac{q_{\text{peak}} - q(x_{\text{peak}} + y)}{q_{\text{peak}}(1 - q_{\text{peak}})} - \frac{y^2}{2\tau^2}\right| \le \frac{1}{q_{\text{peak}}(1-q_{\text{peak}})} + \frac{y^2}{2\tau^2}.$$

- **第 3 步（混合高斯极值规范化变量 $G_N$ 的分布与四阶矩一致收敛）.** 对任意紧区间 $\delta \in [-D, D]$，记整数预算 $N_\delta := \lfloor e^\delta N^*_0 \rceil$（因 $N^*_0 \to \infty$，离散舍入带来 $\ln N_\delta = \ln N^*_0 + \delta + O(1/N^*_0)$，相对误差可一致忽略），定义规范化极值随机变量：
  $$G_N := h\big(S_{(N_\delta)} - x_{\text{peak}}\big) - \delta.$$
  对任意 $x = x_{\text{peak}} + \frac{\delta + t}{h}$，分析 $N_\delta \bar{F}_{\text{mix}}(x)$：注意到混合危险率 $h_{\text{mix}}(z) = -\frac{d}{dz}\ln\bar{F}_{\text{mix}}(z) = w_1(z)h_1(z) + w_0(z)h_0(z)$（凸组合权 $w_1(z) + w_0(z) = 1, w_k \ge 0$）。对于单分量危险率 $h_k(z) = \frac{1}{\sigma_k}\tilde{h}(z_k(z))$（其中标准正态危险率 $\tilde{h}(u) = \phi(u)/\bar{\Phi}(u) = u + u^{-1} + O(u^{-3})$ 满足 $\tilde{h}'(u) = 1 + O(u^{-2})$），在 $|z - x_{\text{peak}}| \le M / h$ 上有 $|z_k(z) - z_k(x_{\text{peak}})| \le \frac{M}{h\sigma_k}$。由第 1 步中 $z_k = \Theta((\kappa^2-1)^{-1/2})$、$h\sigma_k = \Theta((\kappa^2-1)^{-1/2})$ 得：
  $$\frac{h_k(z)}{h_k(x_{\text{peak}})} = 1 + O\!\left(\frac{M}{z_k h \sigma_k}\right) = 1 + O\big(M(\kappa^2 - 1)\big) \qquad (k \in \{0, 1\}).$$
  同时在 $x_{\text{peak}}$ 处有 $\frac{h_1(x_{\text{peak}})}{h_0(x_{\text{peak}})} = \kappa^2 \big(1 + O(z_1^{-2})\big) = 1 + O(\kappa^2 - 1)$，因此对凸组合 $h_{\text{mix}}(z)$ 挟逼成立 $\min(h_1(z), h_0(z)) \le h_{\text{mix}}(z) \le \max(h_1(z), h_0(z))$，即 $h_{\text{mix}}(z)/h = 1 + O(M(\kappa^2-1)) \to 1$ 在 $|z - x_{\text{peak}}| \le M/h$ 上一致成立，进而 $\mathbb{P}(G_N \le t) = [1 - \bar{F}_{\text{mix}}(x_{\text{peak}} + \frac{\delta+t}{h})]^{N_\delta} \to \exp(-e^{-t})$ 逐点收敛于标准 Gumbel 分布 $G$。

  **一致可积控制包络**（对所有充分接近 $1^+$ 的 $\kappa$ 与 $\delta \in [-D, D]$ 一致有效）：
  1. **右尾（$t > 0$，$x > x_{\text{peak}}$）**：因单正态危险率 $\tilde{h}(z) = \phi(z)/\bar{\Phi}(z)$ 全局严格递增，$x \ge x_{\text{peak}}$ 时 $h_k(x) \ge h_k(x_{\text{peak}})$，从而混合危险率 $h_{\text{mix}}(x) = w_1(x)h_1(x) + w_0(x)h_0(x) \ge \min(h_1(x_{\text{peak}}), h_0(x_{\text{peak}})) \ge \frac{1}{2} h$（对充分小 $\kappa-1$，因 $h_1 \sim h_0 \sim h$）。由此 $\bar{F}_{\text{mix}}(x_{\text{peak}} + \frac{\delta+t}{h}) \le \bar{F}_{\text{mix}}(x_{\text{peak}}) e^{-\frac{1}{2}\max(0, \delta+t)}$，得指数尾控制（结合 $N_\delta \bar{F}_{\text{mix}}(x_{\text{peak}}) \le e^{D+1}$）：
     $$\mathbb{P}(G_N > t) \le N_\delta \bar{F}_{\text{mix}}\!\left(x_{\text{peak}} + \frac{\delta + t}{h}\right) \le e^{D + 1 - \frac{1}{2}\max(0, t - D)} \qquad (\forall\, t > 0).$$
  2. **左尾（$t < 0$，三段划分）**：
     - 当 $x \in [\frac{1}{2}x_{\text{peak}}, x_{\text{peak}}]$ 时（即 $- \frac{1}{2}h x_{\text{peak}} - \delta \le t \le -\delta$）：$h_k(x) \ge \frac{1}{3} h$，于是 $\bar{F}_{\text{mix}}(x) \ge \bar{F}_{\text{mix}}(x_{\text{peak}}) e^{\frac{1}{3} h(x_{\text{peak}} - x)}$（注意对 $x \le x_{\text{peak}}$，$h_{\text{mix}}(z) \le \max(h_1(x_{\text{peak}}), h_0(x_{\text{peak}})) \le 2h$，而在下积分界可用局部单调与线性高斯下界：精确地，对任意高斯生存函数 $\ln\bar{\Phi}(z)$ 为严格**凹**函数，故 $\ln\bar{F}_{\text{mix}}(x) \ge \ln(p\bar{\Phi}(z_1)) \lor \ln((1-p)\bar{\Phi}(z_0))$，利用下界端点线性增长率至少为 $h_{\min}(x_{\text{peak}}/2) \ge \frac{1}{3}h$ 得 $\bar{F}_{\text{mix}}(x_{\text{peak}} - u) \ge \bar{F}_{\text{mix}}(x_{\text{peak}}) \cdot \frac{1}{2} e^{c_0 h u}$ 对某固定 $c_0 \in (0, 1/2)$ 成立，或更直接：利用 $\bar{\Phi}(z_k - s) / \bar{\Phi}(z_k) = \exp(\int_{z_k-s}^{z_k} \tilde{h}(v)dv) \ge \exp(s\,\tilde{h}(z_k - s)) \ge \exp(\frac{1}{3} z_k s)$ 当 $z_k - s \ge z_k/2$！）。因而：
       $$\mathbb{P}(G_N \le t) = \big[1 - \bar{F}_{\text{mix}}(x)\big]^{N_\delta} \le \exp\!\left(-c'\, e^{-D}\, e^{c_0 |t + D|}\right) \qquad (-h x_{\text{peak}}/2 \le t + \delta \le 0).$$
     - 当 $x \le a := \frac{1}{2} x_{\text{peak}}$ 时：对充分接近 $1^+$ 的 $\kappa$ 有 $N_\delta \ge 2$，且 $\bar{F}_{\text{mix}}(a) = (N^*_0)^{-1} e^{\Omega(h x_{\text{peak}})}$（其中 $h x_{\text{peak}} = \Theta(\frac{c_1}{\kappa^2-1}) \to \infty$）。因对一切 $x \le a$ 有 $0 \le \frac{F_{\text{mix}}(x)}{F_{\text{mix}}(a)} \le 1$ 且 $N_\delta - 1 \ge 1$，成立在端点 $x = a$ 处亦精确相容的上界：
       $$\mathbb{P}(S_{(N_\delta)} \le x) = [F_{\text{mix}}(x)]^{N_\delta} = [F_{\text{mix}}(a)]^{N_\delta - 1}\!\left(\frac{F_{\text{mix}}(x)}{F_{\text{mix}}(a)}\right)^{N_\delta - 1} F_{\text{mix}}(x) \le [F_{\text{mix}}(a)]^{N_\delta - 1} F_{\text{mix}}(x) \le \exp\!\left(-e^{\Omega(h x_{\text{peak}})}\right) F_{\text{mix}}(x).$$
       因 $F_{\text{mix}}(x)$ 拥有高斯指数左尾与有限四阶矩，且前沿双指数衰减因子 $\exp(-e^{\Omega(h x_{\text{peak}})})$ 在 $h x_{\text{peak}} \to \infty$ 时压制任何 $(h x_{\text{peak}})^4$ 多项式项，故远左尾积分 $\int_{-\infty}^{-\frac{1}{2}h x_{\text{peak}}-\delta} |t|^4\, d\mathbb{P}(G_N \le t) \le e^{-e^{\Omega(1/(\kappa^2-1))}} \cdot O\big((h x_{\text{peak}})^4 + h^4\big) \to 0$。

  上述三段一致可积包络立刻给出（对所有 $\delta \in [-D, D]$ 一致成立）：
  $$\lim_{\kappa \to 1^+} \sup_{\delta \in [-D, D]} \Big|\mathbb{E}\big[G_N^k\big] - \mathbb{E}[G^k]\Big| = 0 \quad (k = 1, 2, 3, 4), \qquad \sup_{\delta \in [-D, D]} \mathbb{P}\big(|\delta + G_N| > \sqrt{2} h \tau\big) \le C\, e^{-c\, h\tau}.$$

- **第 4 步（紧集一致收敛至 $(\delta + \gamma)^2 + \frac{\pi^2}{6}$）.** 将 $y = S_{(N_\delta)} - x_{\text{peak}} = \frac{\delta + G_N}{h}$ 代入第 2 步分段余项界并乘以 $2(h\tau)^2$，取期望并用第 3 步一致矩收敛与尾界：
  $$\sup_{\delta \in [-D, D]} \left|\mathcal{J}_\kappa(\delta) - \mathbb{E}\big[(\delta + G_N)^2\big]\right| \le \frac{e}{4 (h\tau)^2} \sup_{\delta \in [-D, D]} \mathbb{E}\big[(\delta + G_N)^4\big] + O\!\left((h\tau)^2 e^{-c\, h\tau}\right) \xrightarrow{\kappa \to 1^+} 0.$$
  又因 Gumbel 分布满足 $\mathbb{E}[G] = \gamma$ 与 $\operatorname{Var}(G) = \frac{\pi^2}{6}$，故 $\mathbb{E}[(\delta + G)^2] = (\delta + \gamma)^2 + \frac{\pi^2}{6}$，从而在任意 $[-D, D]$ 上严格成立一致收敛 $\sup_{\delta \in [-D, D]} |\mathcal{J}_\kappa(\delta) - [(\delta+\gamma)^2 + \frac{\pi^2}{6}]| \to 0$。

- **第 5 步（Karlin 单峰性锁定全局最优点收敛）.** 取定 $D > 1$ 及任意 $\varepsilon \in (0, 1/2)$，记 $\delta_0 := -\gamma$。因为极限二次凸函数在 $\delta_0$ 处的导数间隙至少为 $\varepsilon^2 > 0$，由一致收敛，对充分接近 $1^+$ 的 $\kappa$，必同时成立：
  $$\mathcal{J}_\kappa(-\gamma) < \mathcal{J}_\kappa(-\gamma - \varepsilon), \qquad \mathcal{J}_\kappa(-\gamma) < \mathcal{J}_\kappa(-\gamma + \varepsilon),$$
  等价于在离散整数点处 $P_{\text{BoN}}(N(-\gamma)) > \max\big(P_{\text{BoN}}(N(-\gamma-\varepsilon)),\, P_{\text{BoN}}(N(-\gamma+\varepsilon))\big)$。此时调用**引理 2.1（Karlin 严格单峰性，在高斯 $\sigma_0 > \sigma_1$ 处精确适用）**：任何关于离散整数变量 $N$ 的严格单峰序列，其全局最大值点 $N^*_{\text{exact}}$ 不可能落到任何由局部下凹三元组所夹区间之外（否则单侧递增/递减被破坏）！从而全局最优点必然满足 $N^*_{\text{exact}} / N^*_0 \in [e^{-\gamma - \varepsilon}, e^{-\gamma + \varepsilon}]$。因 $\varepsilon > 0$ 任意小，立证：
  $$\lim_{\kappa \to 1^+} \frac{N^*_{\text{exact}}}{N^*_0} = e^{-\gamma} \approx 0.561459.$$
  *（注记：若采用固定 $\Delta\mu > 0$ 令 $\kappa \to 1^+$，数值结果虽同样趋近 $e^{-\gamma}$，但该机制下 $q_{\text{peak}}(\kappa) \to 1$ 形成平顶平台，导致 $q_{\text{peak}}(1-q_{\text{peak}})\to 0$ 使一阶常数缩放退化；上述**固定峰值高度机制**是唯一保持后验曲率尺度非退化的严格解析通道。工程有限 $r$ 区域 `[1/1]` Padé 修正见附录 A。）* $\blacksquare$

---

### 4. 定理 3：离散二元 Verifier 缩放律与 $\text{PPV}=1$ 边界完备散度界

设 $S_i \in \{0,1\}$，$\text{TPR} = \alpha \in [0, 1]$，$\text{FPR} = \beta \in [0, 1]$，$\alpha > \beta$，$q_1 := p\alpha + (1-p)\beta \in (0, 1)$，$\text{PPV} := p\alpha / q_1$。

**定理 3（二元 Verifier 精确缩放律、分情况 KL / $\chi^2$ / TV 散度界与闭式预算）.** *采用平局随机打破且全负判时均匀随机择一：*
1. **（精确闭式缩放律）** *记 $C := \frac{p(1-p)(\alpha-\beta)}{q_1(1-q_1)} > 0$，则对一切 $N \ge 1$ 精确成立：*
   $$P_{\text{BoN,bin}}(N) = \text{PPV} - C(1-q_1)^N, \qquad \Delta P_{\text{bin}}(N) := P_{\text{BoN,bin}}(N+1) - P_{\text{BoN,bin}}(N) = p(1-p)(\alpha-\beta)(1-q_1)^{N-1} > 0,$$
   *序列 $\{P_{\text{BoN,bin}}(N)\}_{N \ge 1}$ 全局严格递增且严格凹收敛至 $\text{PPV}$。*
2. **（条件互信息及按 $\beta=0$ 严格分类的信息论散度收敛界）** *记样本判决分布 $P_N := \text{Bernoulli}(P_{\text{BoN,bin}}(N))$ 与渐近极限分布 $P_\infty := \text{Bernoulli}(\text{PPV})$：*
   - **情形 A（存在假阳性，$\beta > 0$，故 $0 < \text{PPV} < 1$）**：$P_N \ll P_\infty$ 严格成立，且满足：
     $$D_{\text{KL}}(P_N \,\|\, P_\infty) \le \chi^2(P_N \,\|\, P_\infty) = \frac{C^2}{\text{PPV}(1 - \text{PPV})}\,(1 - q_1)^{2N},$$
     $$D_{\text{KL}}(P_\infty \,\|\, P_N) \le \frac{C^2}{P_{\text{BoN,bin}}(N)\big(1 - P_{\text{BoN,bin}}(N)\big)}\,(1 - q_1)^{2N}.$$
   - **情形 B（无假阳性边界，$\beta = 0$ 且 $\alpha > 0$，从而 $\text{PPV} = 1$，$q_1 = p\alpha$）**：$P_\infty = \delta_1$ 为单点 Dirac 测度，$P_N$ 不绝对连续于 $P_\infty$，故前向散度 $D_{\text{KL}}(P_N \,\|\, P_\infty) = \chi^2(P_N \,\|\, P_\infty) = +\infty$；精确缩放公式依然完全成立并化简为：
     $$P_{\text{BoN,bin}}(N) = 1 - \frac{1-p}{1-p\alpha}(1 - p\alpha)^N, \qquad \text{TV}(P_N, P_\infty) = 1 - P_{\text{BoN,bin}}(N) = \frac{1-p}{1-p\alpha}(1 - p\alpha)^N,$$
     *且反向相对熵有限，精确等于指数衰减对数形式：$D_{\text{KL}}(P_\infty \,\|\, P_N) = -\ln P_{\text{BoN,bin}}(N) = \Theta\big((1-p\alpha)^N\big)$。*
   - **（边际条件互信息上界，对任意 $\beta \ge 0$ 统一成立）** *令全负/检出指示器 $E_N := \mathbf{1}\{\max_{1 \le i \le N} S_i = 1\}$，则边际条件互信息精确受控于：*
     $$I(\hat{Y}_{N+1}; S_{N+1} \mid E_N) \le I(E_{N+1}; S_{N+1} \mid E_N) = H_2(q_1)(1 - q_1)^N.$$
3. **（容差与成本闭式最优预算）** *给定渐近容差 $\varepsilon > 0$ 与线性样本成本 $\lambda > 0$，最小整数最优预算分别为 $N^*(\varepsilon) = \max\big(1, \lceil\frac{\ln(C/\varepsilon)}{-\ln(1-q_1)}\rceil\big)$ 与 $N^*(\lambda) = \max\big(1, 1 + \lceil\frac{\ln(p(1-p)(\alpha-\beta)/\lambda)}{-\ln(1-q_1)}\rceil\big)$。*

*证明.* (1) 记全负事件 $A_N := \{\max_{i\le N} S_i = 0\}$，概率 $(1-q_1)^N$；全负下各样本标签按 $p_0^{\text{neg}} := \frac{p(1-\alpha)}{1-q_1}$ 独立同分布，否则必从阳性桶均匀抽取获正确率 $\text{PPV}$。直接整理 $P_{\text{BoN,bin}}(N) = \text{PPV}(1-(1-q_1)^N) + p_0^{\text{neg}}(1-q_1)^N$ 并注意到 $\text{PPV} - p_0^{\text{neg}} = \frac{p(1-p)(\alpha-\beta)}{q_1(1-q_1)} = C$ 即证。(2) 情形 A 利用二元 KL-卡方基本不等式 $\operatorname{kl}(u, v) \le \chi^2(u, v) = \frac{(u-v)^2}{v(1-v)}$（需 $v\in(0,1)$）；情形 B 当 $\beta=0$ 时 $P_{\text{BoN,bin}}(N) < 1$ 对一切有限 $N$ 严格成立，故 $P_N(\{0\}) > 0$ 而 $\delta_1(\{0\}) = 0 \implies D_{\text{KL}}(P_N\|\delta_1)=+\infty$，同时 $D_{\text{KL}}(\delta_1 \| P_N) = 1\cdot\ln\frac{1}{P_{\text{BoN,bin}}(N)} = -\ln P_{\text{BoN,bin}}(N)$；条件互信息仅在 $E_N=0$ 支路非零，由马尔可夫链 $S_{N+1}\to E_{N+1}\to \hat{Y}_{N+1}\mid E_N=0$ 数据处理不等式立证。(3) 直接解离散指数容差与离散一阶差分 $\Delta P_{\text{bin}}(N) \le \lambda$。$\blacksquare$

---

### 5. 定理 4：校准后验消除坍塌与异质单错类约束下的加权多数决

**定理 4A（Calibrated-BoN 消除右尾坍塌定理）.** *设 $Q_i := q(S_i) \in [0, 1]$ 分布函数为 $F_Q$，选择 $\arg\max_{1\le i\le N} Q_i$（平局均匀打破）。则对任意 $N \ge 1$ 有 $P_{\text{Cal-BoN}}(N) = \mathbb{E}[Q_{(N)}] = \int_0^1 (1 - [F_Q(u)]^N)\, du$；只要 Verifier 非退化（$\mathbb{P}(Q_1=p)<1$），序列严格单调递增且 $\lim_{N\to\infty} P_{\text{Cal-BoN}}(N) = \operatorname{ess\,sup}_{x \in \operatorname{supp}(F_{\text{mix}})} q(x) > p$。*

*证明.* 由条件期望塔性质 $\mathbb{P}(Y=1\mid Q) = \mathbb{E}[\mathbb{E}[Y\mid S]\mid Q] = Q$ a.s.（无需 $q(x)$ 单调或可逆），将定理 1(3) 应用于 $Q_i$ 自身即证。$\blacksquare$

**定理 4B（离散答案空间异质单错类加权与超级尾桶指数一致定理）.** *设离散答案空间中正确答案 $a^*$ 生成概率 $p\in(0,1)$，错误答案集合 $\{a_j\}_{j\in\mathcal{J}}$（允许无穷多）各成员生成概率 $p_j := \mathbb{P}(A=a_j)$（满足 $\sum_{j\in\mathcal{J}} p_j = 1 - p$）。给定有界非负校准权重 $W_i = w(S_i) \in [0, w_{\max}]$（或门控权 $w_\tau(s) = \mathbf{1}\{q(s)\ge\tau\}\phi(q(s))\in[0, w_{\max}]$），记正确答案条件期望权重 $\mu_{w,+} := \mathbb{E}[W \mid A = a^*]$ 及**每个错误答案 $a_j$ 各自的独立条件期望权重** $\mu_{w,j} := \mathbb{E}[W \mid A = a_j] \in [0, w_{\max}]$。假设严格正极值间隙条件成立：*
$$\Delta_{\min} := p\,\mu_{w,+} - \sup_{j \in \mathcal{J}}\big(p_j\, \mu_{w,j}\big) > 0$$
*（注：上式同时约束具体错类出现概率与单错类欺骗高权；即使高权诱饵频率远高于总体均值，只要所有单个错误答案总期待得分均严劣于 $p\mu_{w,+}$ 即允许全局突破少数群体 $p < \sup_j p_j$）。则在有界加权与两阶段门控加权下（门控下设正确答案幸存率 $\rho_\tau := \mathbb{P}(q(S)\ge\tau, A=a^*) > 0$ 且以门控后期望权重满足 $\Delta_{\min,\tau}>0$），均存在常数 $M, c > 0$ 使：*
$$\mathbb{P}(\hat{A}_N \neq a^*) \le M e^{-c N} + \mathbf{1}_{\text{two-stage}}(1 - \rho_\tau)^N, \qquad \lim_{N\to\infty}\hat{A}_N = a^*\quad (\text{a.s.}).$$

*严格一致性证明（覆盖可数无穷异质错误类）.*
1. **单个错误答案对比较**：对每个固定 $j \in \mathcal{J}$，定义独立同分布增量 $Z_{i,j} := W_i(\mathbf{1}\{A_i = a^*\} - \mathbf{1}\{A_i = a_j\}) \in [-w_{\max}, w_{\max}]$。其期望精确等于 $\Delta_j := p\mu_{w,+} - p_j\mu_{w,j} \ge \Delta_{\min} > 0$（注意此处严格使用了**第 $j$ 类错误答案自身**的条件权重 $\mu_{w,j}$，而非全错类均值 $\mu_{w,-}$！），方差上界 $\operatorname{Var}(Z_{i,j}) \le (p + p_j)w_{\max}^2 - \Delta_j^2$，中心化上界 $|Z_{i,j} - \Delta_j| \le w_{\max} + \Delta_j$。由 Hoeffding 与单侧 Bernstein 不等式直接得：
   $$\mathbb{P}\!\left(\sum_{i=1}^N Z_{i,j} \le 0\right) \le \exp\!\left(-c_j N\right), \qquad c_j := \max\!\left(\frac{\Delta_j^2}{2 w_{\max}^2},\, \frac{\Delta_j^2 / 2}{(p+p_j)w_{\max}^2 - \Delta_j^2 + \frac{1}{3}(w_{\max}+\Delta_j)\Delta_j}\right) > 0.$$
2. **可数无穷集一致性“超级尾桶”界（无需错类打分同分布）**：不妨将 $\mathcal{J} = \{1, 2, \dots\}$ 按 $p_1 \ge p_2 \ge \dots$ 排列。因 $\sum_{j=1}^\infty p_j = 1 - p < \infty$，尾部总概率 $p_{\text{tail}}(K_0) := \sum_{j > K_0} p_j \to 0$（当 $K_0 \to \infty$）。无论剩余无穷多个错答案的条件打分分布各自多么异质或偏态高权，因权重一致有界 $W_i \le w_{\max}$，其尾部总体期望得分被**严格一致封顶**：
   $$\sum_{j > K_0} p_j \mu_{w,j} = \mathbb{E}\big[W_i \mathbf{1}\{A_i \in B_{\text{tail}}\}\big] \le w_{\max}\, p_{\text{tail}}(K_0).$$
   因此只要选取足够大的有限整数 $K_0$ 满足**超级尾桶安全阈值**：
   $$p_{\text{tail}}(K_0) < \frac{p\,\mu_{w,+}}{2\, w_{\max}},$$
   即保证合并超级尾桶变量 $\tilde{Z}_{i,\text{tail}} := W_i(\mathbf{1}\{A_i=a^*\} - \mathbf{1}\{A_i \in B_{\text{tail}}\}) \in [-w_{\max}, w_{\max}]$ 拥有严格正期望间隙 $\Delta_{\text{tail}} \ge p\mu_{w,+} - w_{\max}p_{\text{tail}}(K_0) > \frac{1}{2} p\mu_{w,+} > 0$！进而任意单个尾部错答案 $j > K_0$ 的总分绝不超过整个超级尾桶总分，故失败事件被包含在 $K_0 + 1$ 个有限事件并集中：
   $$\mathbb{P}(\hat{A}_N \neq a^*) \le \sum_{j=1}^{K_0}\mathbb{P}\!\left(\sum_{i=1}^N Z_{i,j} \le 0\right) + \mathbb{P}\!\left(\sum_{i=1}^N \tilde{Z}_{i,\text{tail}} \le 0\right) + \mathbf{1}_{\text{two-stage}}(1 - \rho_\tau)^N \le (K_0 + 1) e^{-c_{\min} N} + \mathbf{1}_{\text{two-stage}}(1 - \rho_\tau)^N.$$
   由 $\sum_{N=1}^\infty \mathbb{P}(\hat{A}_N \neq a^*) < \infty$ 及 Borel-Cantelli 引理立证 $\hat{A}_N \to a^*$ almost surely。$\blacksquare$

---

## 第二部分：多噪声集成、联合标度律与题目异质性理论（定理 5–7）

### 6. 定理 5：一般多噪声 Verifier 集成相变、帕累托光谱与 Copula 尾相依

本节将多裁判理论建立在一般非对称正定协方差矩阵 $(\Sigma_1, \Sigma_0)$ 与一般正定相关 Gaussian-Copula + Student-$t$ 模型上，并分别讨论线性融合、最小值融合与二次 Bayes 判别。

#### 6.1 定理 5A：一般非对称高斯 $K$-Verifier 最优线性融合、QP Min 与光谱路径

考虑固定问题条件下 $K \ge 1$ 个裁判的联合高斯分布：
$$\mathbf{S} \mid (Y = c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c), \qquad \Sigma_c \in \mathbb{R}^{K \times K},\ \Sigma_c \succ 0,\ \boldsymbol{\mu}_c \in \mathbb{R}^K,\quad c \in \{0, 1\}.$$

##### 定理 5A-I（最优线性集成广义特征值相变与负权共模抑制）
考虑非零线性组合 $L_{\mathbf{w}}(\mathbf{S}) := \mathbf{w}^\top \mathbf{S}$。记广义特征值问题 $\Sigma_0 \mathbf{w} = \lambda \Sigma_1 \mathbf{w}$ 之最小特征值与对应（经定向使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \ge 0$ 的）广义特征向量为：
$$\lambda_{\min} := \min_{\mathbf{w} \neq \mathbf{0}} \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}} = \lambda_{\min}\!\left(\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2}\right), \qquad \mathbf{w}^* \in \arg\min_{\|\mathbf{w}\|_2=1} \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}}.$$
则：
1. **（一般线性集成渐近一致充要判据）** 存在线性融合权重 $\mathbf{w}$ 使 $\lim_{N\to\infty} P_{\text{BoN}}(N; L_{\mathbf{w}}) = 1$ 的充分必要条件为：$\lambda_{\min} < 1$（或 $\lambda_{\min} = 1$ 且存在对应广义特征向量使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) > 0$）。此时选 $\mathbf{w}^*$ 使等效方差比 $\kappa_{\text{lin}} = \sqrt{\lambda_{\min}} \le 1$。（注：依定理 2B(4)，当 $\lambda_{\min} < 1$ 时，若 $\mathbf{w}^{*\top}\Delta\boldsymbol{\mu} > 0$ 则对一切 $N \ge 1$ 严格递增至 $1$；若 $\mathbf{w}^{*\top}\Delta\boldsymbol{\mu} = 0$ 则 $P_{\text{BoN}}(1)=P_{\text{BoN}}(2)=p$ 后对一切 $N \ge 2$ 严格递增至 $1$；若 $\mathbf{w}^{*\top}\Delta\boldsymbol{\mu} < 0$ 则呈先降后升的 U 型。）
2. **（负权差值共模抑制推论：$\rho_0 > \rho_1$ 逆转）** 当两个裁判受同源错误信号影响显著（如 $\rho_1 = 0, \rho_0 = 0.85, \kappa = 1.2 > 1$），任意非负单纯形凸组合权重均满足 $\kappa_{\text{nonneg}} > 1$ 而渐近坍塌；若两裁判在正样本上存在均值差异 $\mu_{1,1} - \mu_{1,2} > \mu_{0,1} - \mu_{0,2}$，差值融合 $\mathbf{w}^* = \frac{1}{\sqrt{2}}(1, -1)^\top$ 实现：
   $$\frac{\mathbf{w}^{*\top}\Sigma_0\mathbf{w}^*}{\mathbf{w}^{*\top}\Sigma_1\mathbf{w}^*} = \kappa^2 \frac{1 - \rho_0}{1 - \rho_1} = 1.44 \times \frac{0.15}{1.0} = 0.216 < 1,$$
   通过对消错类共模相关性实现渐近一致。

*证明.* 由 $L_{\mathbf{w}}(\mathbf{S}) \mid (Y=c) \sim \mathcal{N}(\mathbf{w}^\top\boldsymbol{\mu}_c, \mathbf{w}^\top\Sigma_c\mathbf{w})$ 结合定理 2B(4) 一维高斯五相分类及对称实矩阵 Rayleigh 商最值定理即证。$\blacksquare$

##### 定理 5A-II（一般异构 Min-Ensemble 的 Hashorva-Hüsler QP 密度级定理——含零乘子退化情形）
考察 $M_{\min}(\mathbf{S}) := \min_{1 \le k \le K} S^{(k)}$。对任意 $c \in \{0, 1\}$，考虑严格凸正交象限二次规划（Savage 1962; Hashorva & Hüsler 2003）：
$$\gamma_c := \min_{\mathbf{u} \ge \mathbf{1}_K} \mathbf{u}^\top \Sigma_c^{-1} \mathbf{u}.$$
由严格凸性存在唯一最优解 $\mathbf{u}_c^* \ge \mathbf{1}_K$ 与唯一 KKT 乘子 $\mathbf{w}_c := \Sigma_c^{-1}\mathbf{u}_c^* \ge \mathbf{0}_K$。为完整覆盖零 KKT 乘子弱激活退化情形，将指标集 $\{1, \dots, K\}$ 严格三分划为：
- **本质正乘子支撑集**：$I_{+,c} := \{k : w_{c, k} > 0\} \neq \emptyset$（互补松弛保证 $\mathbf{u}_{c, I_{+,c}}^* = \mathbf{1}_{I_{+,c}}$）；
- **零乘子弱激活集**：$I_{0,c} := \{k : u_{c,k}^* = 1,\ w_{c,k} = 0\}$；
- **严格非激活集**：$J_c := \{k : u_{c,k}^* > 1\}$（必然有 $w_{c, J_c} = \mathbf{0}$）。

记本质子块权重 $\mathbf{w}_{c, I_{+,c}} = \Sigma_{c, I_{+,c} I_{+,c}}^{-1}\mathbf{1}_{I_{+,c}} > \mathbf{0}$、$\gamma_c = \sum_{k \in I_{+,c}} w_{c,k}$ 及线性漂移 $\beta_c := \mathbf{w}_{c, I_{+,c}}^\top \boldsymbol{\mu}_{c, I_{+,c}}$。定义由本质集对弱激活集的条件 Schur 补高斯向量 $\mathbf{W}_{I_{0,c}} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}_{0|+},\, \tilde{\Sigma}_{0|+})$，其中：
$$\tilde{\boldsymbol{\mu}}_{0|+} := \boldsymbol{\mu}_{c, I_{0,c}} - \Sigma_{c, I_{0,c} I_{+,c}}\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\boldsymbol{\mu}_{c, I_{+,c}}, \qquad \tilde{\Sigma}_{0|+} := \Sigma_{c, I_{0,c} I_{0,c}} - \Sigma_{c, I_{0,c} I_{+,c}}\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\Sigma_{c, I_{+,c} I_{0,c}},$$
并定义弱激活条件正象限概率常数 $\alpha_{c, I_0} := \mathbb{P}(\mathbf{W}_{I_{0,c}} > \mathbf{0}) \in (0, 1]$（当 $I_{0,c} = \emptyset$ 严格互补成立时，$\alpha_{c, I_0} = 1$）。则对**任意**正定协方差矩阵 $\Sigma_c \succ 0$（无论严格互补是否成立），当 $t \to +\infty$ 时：

1. **（含弱激活分支的精确密度级右尾渐近恒等式）** $M_{\min}$ 条件于 $Y=c$ 的精确生存函数与 Lebesgue 密度多项式阶由本质基数 $|I_{+,c}|$（而非边界等式基数 $|I_{+,c} \cup I_{0,c}|$）确定：
   $$\bar{F}_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{sf}}_c\, t^{-|I_{+,c}|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right),$$
   $$f_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{pdf}}_c\, t^{1 - |I_{+,c}|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right),$$
   其中精确生存常数与精确密度常数分别为：
   $$C^{\text{sf}}_c := \alpha_{c, I_0}\, (2\pi)^{-|I_{+,c}|/2}(\det \Sigma_{c, I_{+,c} I_{+,c}})^{-1/2} \Big(\prod_{k \in I_{+,c}} w_{c, k}^{-1}\Big)\, \exp\!\left(-\frac{1}{2}\boldsymbol{\mu}_{c, I_{+,c}}^\top\Sigma_{c, I_{+,c} I_{+,c}}^{-1}\boldsymbol{\mu}_{c, I_{+,c}}\right) > 0, \qquad C^{\text{pdf}}_c := \gamma_c\, C^{\text{sf}}_c > 0.$$
   *（退化反例检验：对 $R = \begin{pmatrix}1 & 0 & 1/2 \\ 0 & 1 & 1/2 \\ 1/2 & 1/2 & 1\end{pmatrix} \succ 0$ 与 $\boldsymbol{\mu}=\mathbf{0}$，$\mathbf{u}^*=(1,1,1)^\top$、$\mathbf{w}=R^{-1}\mathbf{u}^*=(1,1,0)^\top$，故边界集为 $\{1,2,3\}$，但本质集仅 $|I_+|=2$ 且 $I_0=\{3\}$。条件方差 $\tilde{\Sigma}_{0|+}=\frac{1}{2}$，$\alpha_{I_0}=\Phi(0)=\frac{1}{2}$，正确尾率为 $\frac{1}{4\pi}t^{-2}e^{-t^2}$，区别于误用 $|I|=3$ 所得的 $\Theta(t^{-3}e^{-t^2})$。）*
2. **（一般异构 Min-Ensemble 渐近极限完全判据）**
   - $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 1 \iff \big(\gamma_0 > \gamma_1\big)\ \text{或}\ \big(\gamma_0 = \gamma_1,\ \beta_1 > \beta_0\big)\ \text{或}\ \big(\gamma_0 = \gamma_1,\ \beta_1 = \beta_0,\ |I_{+,0}| > |I_{+,1}|\big)$；
   - 若首位严格不等号反向，则 $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 0$；
   - 若三阶全相等（$\gamma_0=\gamma_1,\ \beta_1=\beta_0,\ |I_{+,0}|=|I_{+,1}|$），则 $\Lambda_R = C^{\text{pdf}}_1 / C^{\text{pdf}}_0 \in (0, \infty)$，$\lim_{N\to\infty} P_{\text{BoN,min}}(N) = \frac{p\Lambda_R}{p\Lambda_R + 1 - p} \in (0, 1)$。
3. **（纯高斯下的 Cauchy-Schwarz 下界）** 对任意协方差矩阵恒有 $\gamma_0 = \min_{\mathbf{u}\ge\mathbf{1}_K}\mathbf{u}^\top\Sigma_0^{-1}\mathbf{u} \ge \frac{K^2}{\mathbf{1}_K^\top \Sigma_0 \mathbf{1}_K}$；因而即使在纯高斯族中，当高噪错类裁判被排除出 $I_{+,0}$ 时，`Min-Ensemble` 亦可获得严格优于等权 `Mean-Ensemble` 的尾部衰减指数。

*证明（完整处理严格互补与零乘子退化情形）.*
**第一步（边界通量分解）：** 对绝对连续高斯向量 $(\mathbf{S}\mid Y=c)$ 记密度 $\phi_{\boldsymbol{\mu}_c, \Sigma_c}(\mathbf{s})$，事件 $\{M_{\min}>t\}$ 对 $t$ 的负导数精确等于 $K$ 个角锥侧面边界通量之和：
$$f_{\min, c}(t) = -\frac{d}{dt}\mathbb{P}(M_{\min} > t \mid Y=c) = \sum_{k=1}^K \int_{(t,\infty)^{K-1}} \phi_{\boldsymbol{\mu}_c, \Sigma_c}(\mathbf{s}^{(k,t)})\, d\mathbf{s}_{-k}.$$
按三分划 $I_{+,c}, I_{0,c}, J_c$ 逐类评估贡献：
1. **非激活侧面（$k \in J_c$）**：因 $u_{c,k}^* > 1$，可行域限制到 $\{u_k = 1, \mathbf{u}\ge \mathbf{1}_K\}$ 后的最小二次型价值严格大于 $\gamma_c$，故该类侧面对密度与生存函数的贡献均为 $O(e^{-\Delta\gamma\,t^2/2})$ 倍指数级可忽略。
2. **本质激活侧面（$k \in I_{+,c}$）与弱激活坐标条件律**：固定某 $k \in I_{+,c}$，写 $\mathbf{s}_{I_{+,c}} = t\mathbf{1}_{I_{+,c}} + \mathbf{y}$，其中 $y_k = 0$ 而其余 $|I_{+,c}|-1$ 个坐标作置换 $y_j = \frac{\xi_j}{t\,w_{c,j}}$（$\xi_j \ge 0$，$w_{c,j} > 0$）。关键利用 KKT 线性恒等式 $\Sigma_c \mathbf{w}_c = \mathbf{u}_c^*$（其中 $\mathbf{w}_{c, I_{0,c} \cup J_c} = \mathbf{0}$）：
   $$\Sigma_{c, I_{0,c} I_{+,c}} \Sigma_{c, I_{+,c} I_{+,c}}^{-1} \mathbf{1}_{I_{+,c}} = \Sigma_{c, I_{0,c} I_{+,c}} \mathbf{w}_{c, I_{+,c}} = \mathbf{u}_{c, I_{0,c}}^* = \mathbf{1}_{I_{0,c}}, \qquad \Sigma_{c, J_c I_{+,c}} \Sigma_{c, I_{+,c} I_{+,c}}^{-1} \mathbf{1}_{I_{+,c}} = \mathbf{u}_{c, J_c}^* > \mathbf{1}_{J_c}.$$
   由此给定 $\mathbf{S}_{I_{+,c}} = t\mathbf{1}_{I_{+,c}} + \mathbf{y}$ 时：
   - 对非激活块 $J_c$：条件均值 $= t\mathbf{u}_{c,J_c}^* + O(1)$，因 $u_{c,m}^*>1$ 且条件协方差有界，故 $\mathbb{P}(\mathbf{S}_{J_c} > t\mathbf{1}_{J_c} \mid \mathbf{S}_{I_{+,c}}, \mathbf{S}_{I_{0,c}}) \to 1$；
   - 对弱激活块 $I_{0,c}$：因系数恰等于 $\mathbf{1}_{I_{0,c}}$，线性 $t\mathbf{1}_{I_{0,c}}$ 项与阈值 $t\mathbf{1}_{I_{0,c}}$ 精确消去，使得：
     $$\mathbf{S}_{I_{0,c}} - t\mathbf{1}_{I_{0,c}} \;\Big|\; \big(\mathbf{S}_{I_{+,c}} = t\mathbf{1}_{I_{+,c}} + \mathbf{y}\big) \xrightarrow{\;d\;} \mathcal{N}(\tilde{\boldsymbol{\mu}}_{0|+},\, \tilde{\Sigma}_{0|+}) = \mathbf{W}_{I_{0,c}},$$
     从而条件生存概率 $\mathbb{P}(\mathbf{S}_{I_{0,c}} > t\mathbf{1}_{I_{0,c}} \mid \mathbf{S}_{I_{+,c}} = t\mathbf{1}_{I_{+,c}} + \mathbf{y}) \to \mathbb{P}(\mathbf{W}_{I_{0,c}} > \mathbf{0}) = \alpha_{c, I_0}$（常数，不产生额外的 $t^{-|I_{0,c}|}$ 衰减！）。
   对本质子块二次型展开并沿 $\boldsymbol{\xi} \in [0,\infty)^{|I_{+,c}|-1}$ 应用控制收敛定理积分得第 $k$ 个侧面贡献：
   $$\alpha_{c, I_0}\,(2\pi)^{-|I_{+,c}|/2}(\det\Sigma_{c, I_{+,c} I_{+,c}})^{-1/2} e^{-\frac{1}{2}\boldsymbol{\mu}_{c,I_{+,c}}^\top\Sigma_{c,I_{+,c}I_{+,c}}^{-1}\boldsymbol{\mu}_{c,I_{+,c}}}\Big(\prod_{j \in I_{+,c}\setminus\{k\}} w_{c,j}^{-1}\Big)\, t^{1 - |I_{+,c}|} e^{-\frac{\gamma_c}{2}t^2 + \beta_c t}.$$
   对 $k \in I_{+,c}$ 求和并用恒等式 $\sum_{k \in I_{+,c}} \prod_{j\in I_{+,c}\setminus\{k\}} w_{c,j}^{-1} = \gamma_c \prod_{j\in I_{+,c}} w_{c,j}^{-1}$ 即得主项系数 $C^{\text{pdf}}_c = \gamma_c C^{\text{sf}}_c$。
3. **弱激活侧面（$k \in I_{0,c}$）**：当将某 $k \in I_{0,c}$ 固定在 $s_k = t$ 时，其余待积分变量仍包含全体 $|I_{+,c}|$ 维（需作 $|I_{+,c}|$ 次 $y_j = \frac{\xi_j}{tw_{c,j}}$ 尺度压缩！），体积微元产生 $t^{-|I_{+,c}|}$ 而非 $t^{1-|I_{+,c}|}$。结合在 $s_k=t$ 处的有界条件密度值 $\phi_{\tilde\mu_{k|I_+},\tilde\sigma^2_{k|I_+}}(0)$，所有 $k \in I_{0,c}$ 侧面之和仅为 $O(t^{-|I_{+,c}|}e^{-\frac{\gamma_c}{2}t^2 + \beta_c t}) = O(t^{-1}\cdot f_{\min, c}^{\text{main}}(t)) = o(f_{\min, c}^{\text{main}}(t))$。
对生存函数 $\bar{F}_{\min, c}(t)$ 在全部 $|I_{+,c}|$ 维上积分同理得 $C^{\text{sf}}_c t^{-|I_{+,c}|}e^{-\frac{\gamma_c}{2}t^2 + \beta_c t}$。
**第二步（比值判据与 Cauchy-Schwarz 界）：** 将精确密度相除并结合定理 2A/2B，再由 $\big(\mathbf{u}^\top\Sigma_0^{-1}\mathbf{u}\big)\big(\mathbf{1}_K^\top \Sigma_0 \mathbf{1}_K\big) \ge (\mathbf{u}^\top\mathbf{1}_K)^2 \ge K^2$ 完成全部证明。$\blacksquare$

**推论 5.1（对称相关退化情形与精确临界人数 $K_c$）.** *当 $\Sigma_c = \sigma_c^2[(1-\rho_c)I_K + \rho_c \mathbf{1}_K\mathbf{1}_K^\top]$（$-\frac{1}{K-1}<\rho_c<1$）且 $\boldsymbol{\mu}_c = \mu_c\mathbf{1}_K$ 时，全本质激活且 $\gamma_c = \frac{K}{\sigma_c^2[1+(K-1)\rho_c]}$。当 $\kappa > 1$ 且 $\Delta\mu > 0$ 时，`Mean` 与 `Min` 共用相界 $\kappa \le \sqrt{\frac{1+(K-1)\rho_1}{1+(K-1)\rho_0}}$，存在有限临界人数 $K_c = \lceil 1 + \frac{\kappa^2-1}{\rho_1 - \kappa^2\rho_0} \rceil$ 当且仅当 $\rho_0 < \rho_1/\kappa^2$。*

##### 定理 5A-III（带尾部安全约束的 $N=2$ 线性光谱路径与 Bayes QDA 维度律）
设一般异构高斯族 $\mathbf{S} \mid (Y = c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c)$（$\Sigma_c \succ 0,\ c \in \{0, 1\}$），均值差 $\Delta\boldsymbol{\mu} := \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$。做同时合同对角化 $P^\top \Sigma_1 P = I_K$、$P^\top \Sigma_0 P = \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_K)$（$0 < \lambda_1 \le \dots \le \lambda_K$），记 $\mathbf{d} := P^\top \Delta\boldsymbol{\mu}$ 与坐标 $\mathbf{v} := P^{-1}\mathbf{w}$。

1. **（带渐近尾方差比约束的 $N=2$ SNR 优化光谱路径 `Safe-Fisher`）** 由推论 1.1，$N=2$ 处线性组合 $L_{\mathbf{w}}(\mathbf{S})=\mathbf{w}^\top\mathbf{S}$ 的单步准确率严单调取决于二样本信噪比 $\operatorname{SNR}_2^2(\mathbf{w}) := \frac{(\mathbf{w}^\top \Delta\boldsymbol{\mu})^2}{\mathbf{w}^\top(\Sigma_1 + \Sigma_0)\mathbf{w}}$，无约束极大值为 Fisher / Logit 方向 $\mathbf{w}_{\text{Fisher}} := (\Sigma_0 + \Sigma_1)^{-1}\Delta\boldsymbol{\mu}$（对应 $\tau = -1$）。现考察带渐近防坍塌方差比硬约束的 $N=2$ 优化问题（$\kappa_{\max} > 0$，通常取 $\kappa_{\max} = 1$）：
   $$\max_{\mathbf{w} \neq \mathbf{0}}\ \operatorname{SNR}_2^2(\mathbf{w}) \qquad \text{s.t.}\qquad R(\mathbf{w}) := \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}} \le \kappa_{\max}^2.$$
   *（注：本命题针对 $N=2$ 二样本目标与渐近约束之间的凸对偶权衡；对于一般有限预算 $N > 2$，精确 BoN 准确率需对高阶顺序统计量积分，故 `Safe-Fisher` 并非一般 $N > 2$ 上的精确最优化方向。）* 设 Slater 条件 $\lambda_1 < \kappa_{\max}^2$ 成立且 Fisher 自身不满足约束（$R(\mathbf{w}_{\text{Fisher}}) > \kappa_{\max}^2$）：
   - **非退化主分支（或正交极限 $R_{\lim} < \kappa_{\max}^2$）**：全部全局最优解必落于参数化一维光谱路径
     $$\mathbf{w}_\tau := (\Sigma_0 - \tau \Sigma_1)^{-1}\Delta\boldsymbol{\mu}, \qquad \tau \in [-1,\ \lambda_1).$$
     记 $A(\tau) := \sum_{k=1}^K \frac{d_k^2}{\lambda_k - \tau},\ C(\tau) := \sum_{k=1}^K \frac{d_k^2}{(\lambda_k-\tau)^2},\ D(\tau) := \sum_{k=1}^K \frac{d_k^2}{(\lambda_k-\tau)^3},\ B(\tau) := A(\tau) + (1+\tau)C(\tau)$ 与权重分布 $p_k(\tau) := \frac{v_k(\tau)^2}{\sum_j v_j(\tau)^2}$。假设 $\lambda_k$ 在 $\operatorname{supp}(\mathbf{d})$ 上不全相等，则：
     $$R'(\tau) = 2\,\operatorname{Cov}_{p(\tau)}\!\left(\lambda_k,\, \frac{1}{\lambda_k - \tau}\right) = \frac{2\big(C(\tau)^2 - A(\tau)D(\tau)\big)}{C(\tau)^2} < 0 \qquad (\forall\, \tau \in [-1, \lambda_1)),$$
     $$\left.\frac{d}{d\tau}\operatorname{SNR}_2^2(\mathbf{w}_\tau)\right|_{\tau=-1} = 0, \qquad \frac{d}{d\tau}\operatorname{SNR}_2^2(\mathbf{w}_\tau) = -\frac{2(1+\tau)A(\tau)}{B(\tau)^2}\big(A(\tau)D(\tau) - C(\tau)^2\big) < 0 \qquad (\forall\, \tau \in (-1, \lambda_1)).$$
     唯一安全边界最优解 **`Safe-Fisher`** $\mathbf{w}_{\tau^*}$ 由单调二分方程 $R(\tau^*) = \kappa_{\max}^2$ 确定。
   - **退化正交边界分支**：若在最小特征子空间 $V_{\min} := \{k : \lambda_k = \lambda_1\}$ 上恰有 $P_{V_{\min}}\mathbf{d} = \mathbf{0}$ 且极限比 $R_{\lim} := \lim_{\tau\uparrow \lambda_1^-} R(\tau) \ge \kappa_{\max}^2$，则边界乘子取 $\tau^* = \lambda_1$，全局最优方向为 $\mathbf{v}^* = \mathbf{v}_\perp + c\,\mathbf{e}_{i_0}$，其中 $v_{\perp,k} = \frac{d_k}{\lambda_k - \lambda_1}$（$k \notin V_{\min}$）、$v_{\perp, V_{\min}} = \mathbf{0}$、$\mathbf{e}_{i_0} \in V_{\min}$ 为单位矢量，混合系数 $c^2 := \frac{\sum_{k \notin V_{\min}}(\lambda_k - \kappa_{\max}^2)v_{\perp,k}^2}{\kappa_{\max}^2 - \lambda_1} \ge 0$。

2. **（不可逆相 $\Sigma_0 \succ \Sigma_1$ 下全局 Bayes 最优 QDA 天花板单调性与 $N^{-2/K}$ 维度律）** 联合 Gaussian Bayes 最优对数似然比为二次判别式：
   $$\ell_{\text{QDA}}(\mathbf{s}) := \ln\frac{f_1(\mathbf{s})}{f_0(\mathbf{s})} = \frac{1}{2}\mathbf{s}^\top(\Sigma_0^{-1} - \Sigma_1^{-1})\mathbf{s} + (\Sigma_1^{-1}\boldsymbol{\mu}_1 - \Sigma_0^{-1}\boldsymbol{\mu}_0)^\top \mathbf{s} + \frac{1}{2}\ln\frac{\det\Sigma_0}{\det\Sigma_1} + \frac{1}{2}\big(\boldsymbol{\mu}_0^\top\Sigma_0^{-1}\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1^\top\Sigma_1^{-1}\boldsymbol{\mu}_1\big).$$
   - 当 $\lambda_1 < 1$（或 $\lambda_1 = 1$ 且 $P_{V_{\min}}\mathbf{d} \neq \mathbf{0}$）时，$\sup_{\mathbf{s}} \ell_{\text{QDA}}(\mathbf{s}) = +\infty \implies \lim_{N\to\infty} P_{\text{QDA}}(N) = 1$；
   - 当进入线性不可逆相 $\Sigma_0 \succ \Sigma_1$（等价于 $\lambda_1 > 1$）时，$H_{\text{neg}} := \Sigma_1^{-1} - \Sigma_0^{-1} \succ 0$，后验 $q_{\text{QDA}}(\mathbf{s}) = \sigma(\operatorname{logit}(p) + \ell_{\text{QDA}}(\mathbf{s}))$ 在唯一全局最大值点 $\mathbf{s}^* = (\Sigma_1^{-1} - \Sigma_0^{-1})^{-1}(\Sigma_1^{-1}\boldsymbol{\mu}_1 - \Sigma_0^{-1}\boldsymbol{\mu}_0)$ 处取得小于 $1$ 的有限天花板：
     $$\ell^*_{\text{QDA}} = \frac{1}{2}\ln\frac{\det\Sigma_0}{\det\Sigma_1} + \frac{1}{2}\Delta\boldsymbol{\mu}^\top(\Sigma_0 - \Sigma_1)^{-1}\Delta\boldsymbol{\mu}, \qquad q^*_{\text{Bayes}} := \sigma\big(\operatorname{logit}(p) + \ell^*_{\text{QDA}}\big) < 1,$$
     且随新增裁判维数 $K$ 单调非降（$q^*_{\text{Bayes}}(K+1) \ge q^*_{\text{Bayes}}(K)$）；以全部 $K$ 维极值逼近该光滑内点峰值的渐近收敛速率为 $N^{-2/K}$：
     $$q^*_{\text{Bayes}} - P_{\text{QDA}}(N) = \big(1 + o(1)\big)\, \frac{\Gamma(1 + 2/K)}{c_K^{2/K}}\, N^{-2/K}, \qquad c_K := f_{\text{mix}}(\mathbf{s}^*)\,\frac{(2\pi)^{K/2}}{\Gamma(K/2 + 1)\,\big[q^*_{\text{Bayes}}(1 - q^*_{\text{Bayes}})\big]^{K/2}\sqrt{\det(\Sigma_1^{-1} - \Sigma_0^{-1})}}.$$

*证明.*
- **Part 1（凸化强对偶与微分恒等式）**：在同时对角化坐标下 $\mathbf{w}^\top\Sigma_1\mathbf{w} = \sum_k v_k^2$、$\mathbf{w}^\top\Sigma_0\mathbf{w} = \sum_k \lambda_k v_k^2$、$\mathbf{w}^\top\Delta\boldsymbol{\mu} = \sum_k d_k v_k$。令符号 $s_k = \operatorname{sgn}(d_k)$ 并作变量替换 $x_k := v_k^2 \ge 0$，归一化分母为 $\sum_k(1+\lambda_k)x_k \le 1$，等价于凹最大化—线性约束凸规划：
  $$\max_{\mathbf{x} \ge \mathbf{0}}\ \sum_{k=1}^K |d_k|\sqrt{x_k} \qquad \text{s.t.}\qquad \sum_{k=1}^K (1 + \lambda_k)x_k \le 1,\quad \sum_{k=1}^K (\lambda_k - \kappa_{\max}^2)x_k \le 0.$$
  在 Slater 条件 $\lambda_1 < \kappa_{\max}^2$ 下强对偶及 KKT 充要条件成立。引入乘子 $\nu_1, \nu_2 > 0$ 得平稳方程 $\frac{|d_k|}{2\sqrt{x_k}} = (\nu_1+\nu_2)(\lambda_k - \tau)$，其中 $\tau := \frac{\nu_2\kappa_{\max}^2-\nu_1}{\nu_1+\nu_2}\in (-1, \lambda_1]$。非退化时必有 $\tau < \lambda_1$；沿路径求导利用 $A'=C, C'=2D$ 得 $R'(\tau)=\frac{2(C^2-AD)}{C^2} < 0$，及 $\frac{d}{d\tau}\operatorname{SNR}_2^2(\tau) = -\frac{2(1+\tau)A}{B^2}(AD-C^2)$，该式于 $\tau=-1$ 处精确取 $0$，而在开区间 $\tau\in(-1,\lambda_1)$ 上严格为负（由 Cauchy-Schwarz $C^2 < AD$）。正交边界分支同理验证互补松弛条件饱和。
- **Part 2（QDA 天花板与 $N^{-2/K}$ 速率）**：配方法简化 $\mathbf{s}^*-\boldsymbol{\mu}_0 = \Sigma_0(\Sigma_0-\Sigma_1)^{-1}\Delta\boldsymbol{\mu}$ 即得 $\ell^*_{\text{QDA}}$ 闭式公式；边际似然比为条件期望的加权平均，立即保证 $q^*_{\text{Bayes}}(K+1) \ge q^*_{\text{Bayes}}(K)$。水平集 $\{q_{\text{QDA}}(\mathbf{s}^*+\boldsymbol{\delta}) > q^*_{\text{Bayes}} - u\}$ 渐近等于半轴椭球 $\{\boldsymbol{\delta}^\top H_{\text{neg}}\boldsymbol{\delta} \le \frac{2u}{q^*(1-q^*)}\}$，$K$ 维球体积公式得 $\bar{F}_{Q,\text{QDA}}(q^*_{\text{Bayes}}-u) = (1+o(1))c_K u^{K/2}$，代回期望积分令 $u=wN^{-2/K}$ 积分 $\int_0^\infty e^{-c_K w^{K/2}}dw = \frac{\Gamma(1+2/K)}{c_K^{2/K}}$ 即证。$\blacksquare$

---

#### 6.2 定理 5B：正则变化重尾卷积与 Gaussian-Copula 统一剩余尾相依

##### 定理 5B-I（正则变化卷积与共享加性高斯微扰的分离判据）
设 $T \sim t_\alpha$ 满足 $\bar{F}_{t_\alpha}(t) = c_\alpha t^{-\alpha}(1+o(1))$（密度 $f_{t_\alpha}(t) = \alpha c_\alpha t^{-\alpha-1}(1+o(1))$），正类 $S^{(k)} = \mu_1 + \sigma_1 T_1 + \eta_1 \varepsilon_k$（$\varepsilon_k \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$），错类以概率 $1-\epsilon_s$ 为独立 $t_{\alpha_0}$（$1 < \alpha_0 < \alpha_1$）、以概率 $\epsilon_s$ 为同源共享分支 $\mu_0 + \sigma_0 T_{\text{shared}} + \eta_0 \varepsilon_k$。则：
1. `Mean` 密度满足 $f_{\text{mean}, 0}(t) = \Theta(t^{-\alpha_0 - 1})$ 严于 $f_{\text{mean}, 1}(t) = \Theta(t^{-\alpha_1 - 1})$，故对任意 $K \ge 1$ 渐近坍塌至 $0$；
2. 对 `Min`，共享加性高斯极小值 $E_{\min} = \min_k \eta_1 \varepsilon_k$ 的卷积密度满足 $f_{\min, 1}(t) = (1+o(1))\alpha_1 c_{\alpha_1}\sigma_1^{\alpha_1} t^{-\alpha_1-1}$。当 $\epsilon_s = 0$ 时，错类 `Min` 密度为 $(1+o(1)) K\alpha_0 (c_{\alpha_0}\sigma_0^{\alpha_0})^K t^{-K\alpha_0 - 1}$：
   - 若 $K\alpha_0 > \alpha_1$，则 $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 1$；
   - 若 $K\alpha_0 < \alpha_1$，则 $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 0$；
   - 若 $K\alpha_0 = \alpha_1$（临界边界相），则渐近极限取决于尺度常数比 $\Lambda_R = \frac{c_{\alpha_1}\sigma_1^{\alpha_1}}{(c_{\alpha_0}\sigma_0^{\alpha_0})^K} \in (0, \infty)$，满足 $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = \frac{p\Lambda_R}{p\Lambda_R + 1 - p} \in (0, 1)$。
   而一旦 $\epsilon_s > 0$，同源分支贡献 $\epsilon_s \Theta(t^{-\alpha_0 - 1})$ 密度下界从而在 $N\to\infty$ 时跌向 $0$。

*证明.* 将卷积积分按 $|u| \le t/2$ 与 $|u| > t/2$ 分割，前者由正则变化一致收敛定理匹配 $f_{t_{\alpha_1}}(t/\sigma_1)/\sigma_1$，后者由高斯极小值超幂律尾控制；当 $\epsilon_s = 0$ 时 $\mathbb{P}(M_{\min}>t\mid Y=0) = [\mathbb{P}(\mu_0+\sigma_0 T_1+\eta_0\varepsilon_1 > t)]^K$ 求导立得密度常数 $(c_{\alpha_0}\sigma_0^{\alpha_0})^K K\alpha_0$ 与临界相 $\Lambda_R$。$\blacksquare$

##### 定理 5B-II（一般正定 Gaussian-Copula + Student-$t$ 统一剩余尾相依系数与对数修正因子）
考虑**任意正定相关矩阵** $R_1, R_0 \in \mathbb{R}^{K \times K}$（对角元全等于 $1$，允许负相关性与零 KKT 乘子退化情形），令潜向量 $\mathbf{Z} \mid (Y=c) \sim \mathcal{N}(\mathbf{0}, R_c)$，经边际分位映射生成重尾打分：$S^{(k)} = \mu_c + \sigma_c (F_{t_{\alpha_c}}^{-1} \circ \Phi)(Z_k)$（若错类含混合同源漏洞，则以概率 $\epsilon_s \in [0, 1]$ 切换至共因子 $Z_1 = \dots = Z_K$）。设对应 QP 解 $(R_c)$ 所得二次指数与**本质正乘子支撑集**为 $(\gamma_c(R_c), I_{+,c})$（依定理 5A-II 定义 $I_{+,c}=\{k : w_{c,k} > 0\}$），定义剩余尾相依系数 $\eta_c := \frac{1}{\gamma_c(R_c)}$（Tankov 2016; Das & Fasen-Hartmann 2024）与有效错类系数 $\eta_{0,\text{eff}}$：

1. **（精确边界区间分类与非负相关的充分条件性质）**
   $$\eta_{0,\text{eff}} := \begin{cases} 1, & \epsilon_s > 0, \\ \frac{1}{\gamma_0(R_0)}, & \epsilon_s = 0. \end{cases}$$
   按维度 $K$ 与正定相关矩阵 $R_c \succ 0$ 的结构，$\eta_c = 1/\gamma_c(R_c)$ 严格划分为四类边界情形：
   - **单裁判（$K = 1$）**：$R_c = (1)$，$\gamma_c = 1$，故 $\eta_c = 1$；
   - **多裁判一般正定（$K \ge 2,\ R_c \succ 0$）**：任意两维主子块 $R_{\{1,2\}} = \begin{pmatrix}1 & \rho_{12} \\ \rho_{12} & 1\end{pmatrix}$ 必满足 $|\rho_{12}| < 1$，由分块矩阵 Schur 补正定性知对一切 $\mathbf{u} \ge \mathbf{1}_K$ 有 $\mathbf{u}^\top R_c^{-1}\mathbf{u} \ge \mathbf{u}_{\{1,2\}}^\top R_{\{1,2\}}^{-1}\mathbf{u}_{\{1,2\}} \ge \frac{2}{1+\rho_{12}} > 1$，从而 $\gamma_c(R_c) \in (1, +\infty)$，$\eta_c$ 严格落于**开区间 $(0, 1)$** 内（含负相关时可取到 $\eta_c < \frac{1}{K}$）；
   - **多裁判逐元非负正定（$K \ge 2,\ R_c \succ 0,\ (R_c)_{ij} \ge 0\ \forall i,j$）**：作为**充分非必要条件**，由 Slepian 引理得 $\gamma_c(R_c) \le K$，从而 $\eta_c$ 严格落于**半开区间 $[\frac{1}{K}, 1)$** 内；
   - **同源上尾漏洞或奇异共单调极限**：当 $\epsilon_s > 0$（或 $R_0$ 退化至奇异全 $1$ 矩阵 $\rho_{ij} \to 1^-$）时，有效系数达到上端点 $\eta_{0,\text{eff}} = 1$。
   *（充分不必要反例：$R = \operatorname{diag}\big(\begin{pmatrix}1 & -0.2 \\ -0.2 & 1\end{pmatrix},\, \begin{pmatrix}1 & 0.5 \\ 0.5 & 1\end{pmatrix}\big) \succ 0$ 含负元 $-0.2$，然由块独立性得 $\gamma(R) = \frac{2}{1-0.2} + \frac{2}{1+0.5} = \frac{23}{6}$，即 $\eta = \frac{6}{23} \approx 0.26087 \in (\frac{1}{4}, 1)$。）*
2. **（修正后精确对数因子耦合阶与统一相界）** 对一切正定相关矩阵 $R_c \succ 0$（含零乘子弱激活退化矩阵），精确幂律—对数渐近阶由**本质基数 $|I_{+,c}|$** 决定：
   $$\mathbb{P}(M_{\min} > t \mid Y=c) = \Theta\!\left(t^{-\alpha_c / \eta_c}\, (\ln t)^{\frac{\gamma_c(R_c) - |I_{+,c}|}{2}}\right), \qquad f_{\min, c}(t) = \Theta\!\left(t^{-\alpha_c / \eta_c - 1}\, (\ln t)^{\frac{\gamma_c(R_c) - |I_{+,c}|}{2}}\right),$$
   从而 `Min-Ensemble` 渐近收敛至 $1$ 的主阶条件为 $\frac{\alpha_0}{\eta_{0,\text{eff}}} > \frac{\alpha_1}{\eta_1}$（主阶相等 $\frac{\alpha_0}{\eta_{0,\text{eff}}} = \frac{\alpha_1}{\eta_1}$ 时，由对数指数 $\gamma_0 - |I_{+,0}| < \gamma_1 - |I_{+,1}|$ 决定趋 $1$，反向则趋 $0$；仅当 $\epsilon_s > 0$ 且 $\alpha_0 = \alpha_1/\eta_1$ 等退化边界按对应同源支路无对数因子 $(\ln t)^0$ 比较）。

*证明.*
- **范围边界与 Slepian 充分性**：$K=1$ 显然；对 $K \ge 2$，将 $R_c = \begin{pmatrix}A & B \\ B^\top & D\end{pmatrix}$（其中 $A = R_{\{1,2\}}$，$S := D - B^\top A^{-1}B \succ 0$）作块分解得 $\mathbf{u}^\top R_c^{-1}\mathbf{u} = \mathbf{u}_1^\top A^{-1}\mathbf{u}_1 + (\mathbf{u}_2 - B^\top A^{-1}\mathbf{u}_1)^\top S^{-1}(\mathbf{u}_2 - B^\top A^{-1}\mathbf{u}_1) \ge \mathbf{u}_1^\top A^{-1}\mathbf{u}_1$。在 $\mathbf{u}_1 \ge \mathbf{1}_2$ 上最小化右侧得 $\frac{2}{1+\rho_{12}} > 1$（因 $R_c \succ 0 \implies \rho_{12} < 1$），立证 $\eta_c < 1$；同时令 $K=2,\ \rho_{12} \to -1^+$ 有 $\eta_c = \frac{1+\rho_{12}}{2} \to 0^+$，令 $\rho_{ij} \to 1^-$ 有 $\eta_c \to 1^-$，故开区间 $(0, 1)$ 紧致可达。若所有 $(R_c)_{ij} \ge 0$，由 Slepian 正态比较引理比较独立标准正态 $\tilde{\mathbf{Z}} \sim \mathcal{N}(\mathbf{0}, I_K)$ 得 $\mathbb{P}_{R_c}(\min Z_k > t) \ge [\bar{\Phi}(t)]^K$，两边取 $-\frac{2}{t^2}\ln(\cdot)$ 令 $t\to\infty$ 即得 $\gamma_c(R_c) \le K \iff \eta_c \ge \frac{1}{K}$。
- **含弱激活情形的 Copula 对数因子推导**：令 $h_c(t) := \bar{\Phi}^{-1}(\bar{F}_{t_{\alpha_c}}(\frac{t-\mu_c}{\sigma_c}))$，由标准高斯 Mills 比渐近式知 $h_c(t) = \sqrt{2\alpha_c \ln t}\,(1 + o(1))$、$e^{-h_c(t)^2 / 2} = \Theta\big(t^{-\alpha_c} (\ln t)^{1/2}\big)$ 且 Jacobian 导数 $h_c'(t) = \Theta\big(t^{-1}(\ln t)^{-1/2}\big)$。将这些关系直接代入**适用一切正定相关矩阵 $R_c$ 的定理 5A-II(1)**（其中高斯最小值为 $t^{-|I_{+,c}|}$ 与密度 $t^{1 - |I_{+,c}|}$ 而非 $|I_c|$！）：
  $$\bar{F}_{\min, c}(t) = \Theta\Big(h_c(t)^{-|I_{+,c}|} \big(e^{-h_c(t)^2/2}\big)^{\gamma_c}\Big) = \Theta\!\left((\ln t)^{-|I_{+,c}|/2}\cdot t^{-\alpha_c \gamma_c} (\ln t)^{\gamma_c/2}\right) = \Theta\!\left(t^{-\alpha_c / \eta_c}(\ln t)^{\frac{\gamma_c - |I_{+,c}|}{2}}\right),$$
  再乘以 $h_c'(t)$ 即得 $f_{\min, c}(t)$ 的精确对数修正指数 $\frac{\gamma_c - |I_{+,c}|}{2}$。当 $\epsilon_s > 0$ 时同源分枝给出 $\epsilon_s \Theta(t^{-\alpha_0-1})$ 对应 $\eta_{0,\text{eff}}=1$ 与对数指数 $0$。$\blacksquare$

---

### 7. 定理 6：有限校准集 $M$ 与推理预算 $N$ 的严格 Minimax 标度律及非渐近 `KL-LCB`

将打分空间 $\mathbb{R}$ 划分为 $K$ 个桶 $\{B_k\}_{k=1}^K$（边际概率 $\pi_k > 0$，真正确率 $q_k$）。由 $M$ 条 i.i.d. 校准样本 $\mathcal{D}_M$ 统计计数 $m_k, S_k$ 并构造估值 $\hat{\mathbf{s}} \in \mathbb{R}^K$。

#### 7.1 定理 6A：条件闭式公式、有限校准失误与逐点 Fano 标度律

本节定理 6A–6C 固定单个分布 $P$（即第 1.1 节中固定问题 $X=x$ 条件模型），基于离线独立同分布样本 $\mathcal{D}_M \sim P^{\otimes M}$ 与在线样本 $S_{1:N}$ 给出分桶校准 BoN 的统计界与 Minimax 速率。

**定理 6A（精确平局闭式公式、MLE 失配与 $\beta$-光滑峰 Minimax 标度率）.**
1. **（条件精确 BoN 闭式公式）** 按 $\hat{s}_k$ 降序聚合成等价平局组 $G_1, \dots, G_L$，记组总概率 $P_\ell := \sum_{k \in G_\ell} \pi_k$，组真正确率 $\bar{q}_\ell := \frac{1}{P_\ell}\sum_{k \in G_\ell} \pi_k q_k$，严劣尾部总概率 $R_\ell := \sum_{j > \ell} P_j$，则对任意 $N \ge 1$ 精确成立：
   $$P(N \mid \mathcal{D}_M) = \sum_{\ell=1}^L \bar{q}_\ell \big[(R_\ell + P_\ell)^N - R_\ell^N\big], \qquad \lim_{N \to \infty} P(N \mid \mathcal{D}_M) = \bar{q}_1.$$
2. **（MLE 稀疏桶高估导致渐近上限降低与倒 U 风险）** 在朴素 MLE（$\hat{s}_k = S_k/m_k$）下，观测较少、真后验欠佳的稀疏尾桶 $k_0$（$q_{k_0} < q^*$）以正概率获满分估值 $\hat{s}_{k_0}=1$ 并进入最优组 $G_1$，从而必然压低大 $N$ 渐近上限至 $\bar{q}_1 < q^*$。进一步地，当该高估组总概率较小（$P_1 \ll P_{\text{main}}$）且次优高质组在中等 $N$ 主导选样时，准确率随 $N$ 增至 $N \gtrsim 1/P_1$ 可呈倒 U 型回撤。
3. **（$\beta$-光滑极值分桶标度律与 Fano 信息论下界）**
   在统一正则分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（定义见定理 6C，要求固定常数 $C_1, f_{\min}, \eta_0 > 0$ 与 $\beta > 0$）上，采用等频分桶 $\pi_k = 1/K$（未知总体分位数时的样本分割顺序统计量推广见命题 6.1）：
   - 桶离散偏差 $O(K^{-\beta})$ 与极值估计偏差 $O\big(\sqrt{\frac{K\ln M}{M}}\big)$ 平衡解得最优分桶指数：
     $$K^*(M) \asymp \left(\frac{M}{\ln M}\right)^{\frac{1}{2\beta + 1}} \quad\Big(\beta=2\text{ 时为 }\left(\frac{M}{\ln M}\right)^{1/5}\Big), \qquad \mathcal{R}(M) = O\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right);$$
   - 对任何基于 $M$ 条样本的估计器 $\hat{x}(\mathcal{D}_M)$，必存在属于统一分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 的成员使得：
     $$\inf_{\hat{x}} \sup_{P \in \mathcal{F}_\beta^{\text{unif}}} \Big(q^*_P - \mathbb{E}_{\mathcal{D}_M}\big[q_P(\hat{x}(\mathcal{D}_M))\big]\Big) = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right).$$

*证明.* (1)–(2) 由水平组排列公式直接计算即证。(3) 对任意 $\beta > 0$，取支持于 $[-1, 1]$ 的对称峰函数 $\psi(u) := \max(0, 1 - |u|^\beta)$（若需 $C^1$ 可取平滑紧支集核）。在 $[0.2, 0.8]$ 上划分 $J = \Theta(h^{-1})$ 个互不相交中心 $t_1, \dots, t_J$，令 $f_{\text{mix}} \equiv 1$（$1 \ge f_{\min}$ 取满足 $[0.2-\eta_0, 0.8+\eta_0] \subset [0, 1]$ 或缩放区间常数），构造假设族 $q_j(x) := \frac{1}{2} + c_\beta h^\beta \psi\big(\frac{x-t_j}{h}\big)$，其中 $c_\beta \in (0, C_1]$ 足够小以使 $q_j \in [\frac{1}{4}, \frac{3}{4}]$。对每个 $j$，在峰值 $t_j$ 的一切点 $x$ 上均精确成立 $q_j^* - q_j(x) = c_\beta \min(|x - t_j|^\beta,\, h^\beta) \le c_\beta |x - t_j|^\beta \le C_1 |x - t_j|^\beta$（$h^\beta$ 对消，不因 $h\to 0$ 破坏曲率常数上界！），故一切假设严格落入 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 内。单样本散度满足 $D_{\text{KL}}(P_j \| P_0) \le C_2 h \Delta^2 = C_2 c_\beta^2 h^{2\beta+1}$，取 $h = c_h (\frac{\ln M}{M})^{\frac{1}{2\beta+1}}$ 使 $M D_{\text{KL}}(P_j\|P_0) \le \frac{1}{8}\ln J$，由广义 Fano 不等式即证 $\Omega\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}\big)$。$\blacksquare$

---

#### 7.2 定理 6B：非渐近 `KL-LCB` 悲观保底安全界与三事件遗憾定理

针对各非空分桶（$m_k \ge 1, \hat{q}_k = S_k/m_k$）定义联合置信水平 $1 - \delta_{\text{cov}}$ 的二项 **`KL-LCB`**（Garivier & Cappé 2011）：
$$\hat{s}_k^{\text{KL-LCB}}(\delta_{\text{cov}}) := \inf\left\{q \in [0, \hat{q}_k] : m_k\, \operatorname{kl}(\hat{q}_k,\, q) \le \ln\frac{K}{\delta_{\text{cov}}}\right\} \qquad (\text{空桶 }m_k=0\text{ 时置 }0).$$
定义该桶置信半宽为 $r_k(\delta_{\text{cov}}) := \max\big(\hat{q}_k - \hat{s}_k^{\text{KL-LCB}}(\delta_{\text{cov}}),\, \hat{s}_k^{\text{KL-UCB}}(\delta_{\text{cov}}) - \hat{q}_k\big)$。

> **覆盖保证与排序范围**：非渐近置信下界仅保证 $\hat{s}_k^{\text{KL-LCB}} \le q_k$ 同步成立，并不保证完美恢复所有桶之间的真后验序（例如极稀疏的高真均值桶 $m_A=1, q_A=0.90$ 可能排在中质多样本桶 $m_B=100, q_B=0.60$ 之后）。其数学保证在于以高概率阻止真实质量劣于最高经验下界的低质罕见桶进入 $G_1$，并控制悲观保底下界与期望遗憾。

**定理 6B（非渐近 `KL-LCB` 悲观保底安全界、三事件高概率遗憾与 Bernstein 阈值）.**
1. **（非渐近联合悲观安全底线）** 对任意有限分桶 $K \ge 1$、任意有限样本与任意置信水平 $\delta_1 \in (0, 1)$：
   - 在单侧安全事件 $\mathcal{E}_1 := \big\{\forall\, k \in \{1,\dots,K\},\ \hat{s}_k^{\text{KL-LCB}}(\delta_1) \le q_k\big\}$ 上（$\mathbb{P}(\mathcal{E}_1) \ge 1 - \delta_1$），$\bar{q}_1 := \lim_{N\to\infty} P_{\text{LCB}}(N \mid \mathcal{D}_M)$ 满足：
     $$\bar{q}_1 \ge \max_{1 \le k \le K} \hat{s}_k^{\text{KL-LCB}}(\delta_1).$$
   - 若同时以双侧联合界参数 $\delta_1/2$ 覆盖全部桶上、下置信带（概率 $\ge 1 - \delta_1$），则 $\bar{q}_1 \ge \max_{1 \le k \le K}\big(q_k - 2 r_k(\delta_1/2)\big)$。
2. **（等频分桶三事件高概率遗憾界）** 若分布落于 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 上，采用 $K > \frac{2}{\eta_0 f_{\min}}$ 阶等频分桶（$\pi_k = 1/K$），且给定 $\delta \in (0, 1)$ 与 $M \ge 8 K \ln(3/\delta)$。令单侧 `KL-LCB` 采用 $\delta_{\text{cov}} = \delta/3$（惩罚阈值 $\ln(3K/\delta)$），则以至少 $1 - \delta$ 的概率有：
   $$q^* - \bar{q}_1 \le C_1 (K f_{\min})^{-\beta} + 2\sqrt{\frac{K \ln(3K/\delta)}{M}}.$$
3. **（`Wilson-LCB` 近似与 `Empirical-Bernstein LCB` 归零阈值）** 将 $\operatorname{kl}(\hat{q}, q)$ 作二阶展开代以 $\frac{z^2}{2}$（$z=\Phi^{-1}(1-\frac{\delta}{2K})$）得 `Wilson-LCB`；而经验 Bernstein LCB 对给定经验桶均值 $\hat{q}_k > 0$ 的**确定性归零崩溃条件**为：
   $$m_k < 1 + \frac{7}{3\,\hat{q}_k} \ln\!\left(\frac{2K}{\delta}\right) \implies \hat{s}_k^{\text{Bern-LCB}} = 0.$$
   在单侧上侧集中事件 $\{\hat{q}_k \le \min(1,\, q_k + \sqrt{\frac{\ln(2K/\delta)}{2 m_k}})\}$（对全部 $k$ 联合成立概率 $\ge 1 - \delta/2$）上，对应的真实均值参考截断阈值即代换 $\hat{q}_k$ 为 $q_{\text{peak}} + \sqrt{\frac{\ln(2K/\delta)}{2 m_k}}$。

*证明.* (1) 由 Cramér-Chernoff 上界与 $K$ 桶并界立证 $\mathbb{P}(\mathcal{E}_1) \ge 1 - \delta_1$，组内加权平均即得下界。(2) 结合最优桶 $k^*$ 样本量乘法 Chernoff 界 $\mathbb{P}(m_{k^*} < \frac{M}{2K}) \le e^{-M/(8K)} \le \delta/3$、Hoeffding 下侧偏差 $q_{k^*} - \hat{q}_{k^*} \le \sqrt{\frac{\ln(3K/\delta)}{2 m_{k^*}}}$（概率 $\le \delta/3$）及 Pinsker 不等式 $\hat{q}_{k^*} - \hat{s}_{k^*}^{\text{KL-LCB}} \le \sqrt{\frac{\ln(3K/\delta)}{2 m_{k^*}}}$，由三事件并界完证。(3) 直接比对 Bernstein 线性惩罚项 $\frac{7\ln(2K/\delta)}{3(m_k-1)} > \hat{q}_k$ 即证确定性阈值，结合 Hoeffding 上偏即得真实均值参考界。$\blacksquare$

---

#### 7.3 定理 6C：逐点渐近常数、一致正则类 $\mathcal{F}_\beta^{\text{unif}}$ 与全策略联合 $(M, N)$ Minimax 定理

为消除逐分布收敛与一致 Minimax 收敛之间的混用风险，本文明确分离**逐点渐近分布类 $\mathcal{F}_\beta^{\text{sep}}$** 与带有全局类常数的**一致正则分布类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$**：

- **（逐点渐近类 $\mathcal{F}_\beta^{\text{sep}}$，固定单个分布 $P$）**：具有连续正混合密度（$f_{\text{mix}}(x^*) = f^* > 0$），在唯一全局峰值 $x^*$ 邻域满足局部 $\beta$-阶渐近 $q^* - q(x) = C |x - x^*|^\beta (1 + o(1))$（$\beta > 0, C > 0$），且对任意 $\eta > 0$ 存在单分布分离常数 $\varepsilon_\eta > 0$ 使 $\sup_{|x-x^*|\ge\eta} q(x) \le q^* - \varepsilon_\eta$。
- **（一致正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$，固定正类常数 $C_1, f_{\min}, \eta_0 > 0$ 与阶数 $\beta > 0$）**：分布 $P$ 具有正混合密度 $f_{\text{mix}}$（$q^* := \sup_x q(x)$），并存在全局最大值点 $x^*$（$q(x^*)=q^*$）使得在公共邻域 $U_{\eta_0} := [x^* - \eta_0, x^* + \eta_0]$ 上一致成立：
  $$f_{\text{mix}}(x) \ge f_{\min} > 0, \qquad q^* - q(x) \le C_1 |x - x^*|^\beta \qquad (\forall\, x \in U_{\eta_0}).$$

> **类定义必要性与一致性说明**：
> 1. **为何必须对 $(C_1, f_{\min}, \eta_0)$ 施加公共约束**：若仅逐分布要求局部曲率存在（如 $\mathcal{F}_\beta^{\text{sep}}$），构造窄峰族 $S\sim\mathcal{N}(0,1),\ q_h(s)=\frac{1}{4}+\frac{1}{2}e^{-s^2/h^2}$ 均满足 $\mathcal{F}_2^{\text{sep}}$，然固定 $N$ 下 $\mathcal{R}_h(N) \ge \frac{1}{2} - \frac{Nh}{2\sqrt{h^2+2}} \xrightarrow{h\to 0^+} \frac{1}{2}$，导致甚至无一致收敛率！引入固定上界常数 $C_1$ 正是合理排除峰宽缩至零病态族的经典正则条件；
> 2. **为何一致 Minimax 上界中不能施加公共全局分离间隙 $\varepsilon_0 > 0$**：若强加全局最小高差 $\varepsilon_0 > 0$，则 Fano 下界中高度 $\Delta = c_\beta h^\beta \to 0$ 的鼓包假设族将被排除。而在上述 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 中，上界仅依靠 $U_{\eta_0}$ 内密度下界 $f_{\min}$ 与 $C_1|x-x^*|^\beta$ 即能锁定近峰耦合样本 $D_N \le \eta_0/2$；同时如定理 6A 证明所核验，只要 $c_\beta \le C_1$，一切 Fano 鼓包假设对任意 $h \to 0$ 均满足 $q_j^* - q_j(x) \le C_1 |x-x^*|^\beta$，从而严格落在同一个类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 中！

**定理 6C（逐点 $N^{-\beta}$ 极限、一致正则类非渐近上界与全策略 $(M, N)$ Minimax 最优性）.**
1. **（逐点精确渐近收敛主项 $\Theta(N^{-\beta})$）** *对任意固定分布 $P \in \mathcal{F}_\beta^{\text{sep}}$（$\beta > 0$），完美后验排序（$M=\infty$）精确满足逐点渐近恒等式：*
   $$q^*_P - P_{\text{Cal-BoN}}(N) = \big(1 + o(1)\big)\, \frac{\Gamma(\beta + 1)\, C}{(2 f^*)^\beta}\, N^{-\beta} \qquad \Big(\beta = 2\text{ 时为 }\frac{C}{2(f^*)^2}\, N^{-2}\Big).$$
2. **（一致正则类 $\mathcal{F}_\beta^{\text{unif}}$ 上静态等频 `KL-LCB` 的非渐近显式上界——覆盖所有 $\beta > 0$）** *设分布属于 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（$\beta > 0$），记统一凸性常数 $c_\beta := \max(1,\, 2^{\beta - 1})$。在总分位数等频分桶 $K^*(M) = \lfloor c_0 (M/\ln M)^{\frac{1}{2\beta+1}}\rfloor$（满足 $M \ge 8 K \ln(3KM)$ 且 $K > \frac{2}{\eta_0 f_{\min}}$）及置信水平 $\delta_{\text{cov}} = \frac{1}{3M}$ 参数化的 `KL-LCB` 下，对一切合法 $(M, N)$ 一致成立显式非渐近期望遗憾上界：*
   $$\sup_{P \in \mathcal{F}_\beta^{\text{unif}}}\Big(q^*_P - \mathbb{E}_{\mathcal{D}_M}\big[P_{\text{KL-LCB}}(N \mid \mathcal{D}_M)\big]\Big) \le \frac{1}{M} + e^{-\eta_0 f_{\min} N} + \frac{C_1 c_\beta}{(K f_{\min})^\beta} + 2\sqrt{\frac{K \ln(3KM)}{M}} + \frac{C_1 c_\beta\, \Gamma(\beta + 1)}{(2 f_{\min})^\beta}\, N^{-\beta} = O\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}} + N^{-\beta}\right).$$
3. **（全策略空间联合 $(M, N)$ 信息论 Minimax 下界与速率匹配）** *对同一类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$（适当固定区间支持使得单点对照分布 $P^* \in \mathcal{F}_\beta^{\text{unif}} \cap \mathcal{F}_\beta^{\text{sep}}$ 与 Fano 鼓包族均合法落入类内，如 $f_{\min} \le 1$ 配合 $\eta_0 \le 0.2$），考虑**任意**可基于离线有标签样本 $\mathcal{D}_M$ 与在线无标签打分 $(S_1, \dots, S_N)$ 选出下标 $i^* \in \{1,\dots,N\}$ 的策略 $\pi(\mathcal{D}_M, S_{1:N})$：*
   $$\inf_{\pi}\ \sup_{P \in \mathcal{F}_\beta^{\text{unif}}} \Big(q^*_P - \mathbb{E}_{P^{\otimes(M+N)}}\big[q_P(S_{i^*})\big]\Big) = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}} + N^{-\beta}\right).$$
   *从而在 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 上，静态等频 `KL-LCB` 搭配 $N$-无关分桶 $K^*(M)$ 在 $M$ 维与 $N$ 维均达到 Minimax 最优速率（无额外 $(\ln N)^\beta$ 因子）。*

**命题 6.1（未知总体分位数下样本分割经验等频分桶的全 $\beta > 0$ 一致浓缩定理）.** *在实际未知总体分布函数 $F_{\text{mix}}$（连续）时，若朴素调用加性 DKW 不等式 $|\hat{\pi}_k - \frac{1}{K}| = O(\sqrt{\frac{\ln M}{M}})$，则相对误差阶 $K^*(M)\sqrt{\frac{\ln M}{M}} \asymp (\frac{M}{\ln M})^{\frac{1-2\beta}{2(2\beta+1)}}$ 在 $0 < \beta \le \frac{1}{2}$ 时不趋于零（当 $0<\beta<\frac{1}{2}$ 时发散）。改用如下**样本分割顺序统计量分桶估计器**：将 $\mathcal{D}_M$ 对半划分为独立边界子集 $\mathcal{D}_{M_1}^{\text{edge}}$（$M_1 = \lfloor M/2\rfloor$）与估计子集 $\mathcal{D}_{M_2}^{\text{est}}$（$M_2 = M - M_1$）；对 $\mathcal{D}_{M_1}^{\text{edge}}$ 升序统计量 $S_{(1)}^{\text{edge}} < \dots < S_{(M_1)}^{\text{edge}}$ 取内部节点下标 $r_k := \lfloor \frac{k(M_1 + 1)}{K}\rfloor$（$k=1,\dots,K-1$），定义经验桶 $\hat{B}_k := (\hat{e}_{k-1}, \hat{e}_k]$（$\hat{e}_0=-\infty,\ \hat{e}_k = S_{(r_k)}^{\text{edge}},\ \hat{e}_K=+\infty$），并在 $\mathcal{D}_{M_2}^{\text{est}}$ 上计算 `KL-LCB`。则对**一切 $\beta > 0$**：*
1. *只要 $M_1 + 1 \ge 6K$ 且最小阶差 $d_{\min} := \min_{1\le k\le K}(r_k - r_{k-1}) \ge 27 \ln(16 K M)$（等价于 $M \ge 66 K \ln(16KM)$，对 $K^*(M) \asymp (\frac{M}{\ln M})^{\frac{1}{2\beta+1}}$ 因 $\frac{1}{2\beta+1} < 1 \iff \beta > 0$ 在一切 $\beta > 0$ 下随 $M$ 增大均自动满足），以至少 $1 - \frac{1}{4M}$ 的概率对全体 $k \in \{1,\dots,K\}$ 一致成立乘性质量界：*
   $$\frac{1}{2K} \le \hat{\pi}_k := F_{\text{mix}}(\hat{e}_k) - F_{\text{mix}}(\hat{e}_{k-1}) \le \frac{35}{18 K} < \frac{2}{K};$$
2. *在该事件上条件于 $\mathcal{D}_{M_1}^{\text{edge}}$，所有落入 $U_{\eta_0}$ 的经验桶宽度至多为 $\frac{2}{K f_{\min}}$（需 $K > \frac{4}{\eta_0 f_{\min}}$），且 $\mathcal{D}_{M_2}^{\text{est}}$ 中各桶样本数 $m_k \sim \operatorname{Binomial}(M_2, \hat{\pi}_k)$ 以至少 $1 - \frac{1}{4M}$ 概率一致满足 $m_k \ge \frac{M_2}{4K} \ge \frac{M}{8K}$，从而经验等频 `KL-LCB` 在**一切 $\beta > 0$** 上保持同一联合速率 $O\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$。*

*命题 6.1 的证明.* 因 $F_{\text{mix}}$ 连续，$U_i := F_{\text{mix}}(S_i^{\text{edge}}) \stackrel{\text{i.i.d.}}{\sim} \operatorname{Uniform}(0, 1)$，故 $\hat{\pi}_k = U_{(r_k)} - U_{(r_{k-1})}$（令 $U_{(0)}=0, U_{(M_1+1)}=1$）。由 Rényi–Pyke 顺序统计量指数间距表示，存在独立同分布 $E_1, \dots, E_{M_1+1} \stackrel{\text{i.i.d.}}{\sim} \operatorname{Exp}(1)$ 使得 $(\hat{\pi}_1, \dots, \hat{\pi}_K) \stackrel{d}{=} (G_1/G_{\text{tot}}, \dots, G_K/G_{\text{tot}})$，其中 $G_k := \sum_{j=r_{k-1}+1}^{r_k} E_j \sim \operatorname{Gamma}(d_k, 1)$（$d_k = r_k - r_{k-1}$），$G_{\text{tot}} := \sum_{j=1}^{M_1+1} E_j \sim \operatorname{Gamma}(M_1 + 1, 1)$。当 $M_1 + 1 \ge 6K$ 时，$\frac{5}{6}\frac{M_1+1}{K} \le \lfloor\frac{M_1+1}{K}\rfloor \le d_k \le \lceil\frac{M_1+1}{K}\rceil \le \frac{7}{6}\frac{M_1+1}{K}$。对 $G \sim \operatorname{Gamma}(d, 1)$ 应用精确 Cramér–Chernoff 大偏差界：
$$\mathbb{P}\!\left(G < \frac{3}{4}d\right) \le e^{-\psi_-(3/4)\, d}, \quad \mathbb{P}\!\left(G > \frac{4}{3}d\right) \le e^{-\psi_+(4/3)\, d},$$
其中 $\psi_-(3/4) = -\frac{1}{4} - \ln\frac{3}{4} \approx 0.03768 > \frac{1}{27}$，$\psi_+(4/3) = \frac{1}{3} - \ln\frac{4}{3} \approx 0.04565 > \frac{1}{27}$；同理对 $G_{\text{tot}}$ 有 $\psi_-(4/5) = -0.2 - \ln 0.8 \approx 0.02314 > \frac{1}{45}$ 与 $\psi_+(5/4) = 0.25 - \ln 1.25 \approx 0.02686 > \frac{1}{45}$。当 $d_{\min} \ge 27 \ln(16 K M)$（此时 $M_1 + 1 \ge K d_{\min} \ge 54 \ln(32 M) > 45 \ln(16 M)$）时，由并界知事件 $\{\forall k,\ \frac{3}{4} d_k \le G_k \le \frac{4}{3} d_k\} \cap \{\frac{4}{5}(M_1+1) \le G_{\text{tot}} \le \frac{5}{4}(M_1+1)\}$ 失效概率不超过 $K(\frac{1}{16KM} + \frac{1}{16KM}) + \frac{2}{16M} = \frac{1}{4M}$。在该事件上，对一切 $k$ 同步成立：
$$\hat{\pi}_k = \frac{G_k}{G_{\text{tot}}} \ge \frac{\frac{3}{4}\cdot \frac{5}{6}\frac{M_1+1}{K}}{\frac{5}{4}(M_1 + 1)} = \frac{1}{2K}, \qquad \hat{\pi}_k \le \frac{\frac{4}{3}\cdot \frac{7}{6}\frac{M_1+1}{K}}{\frac{4}{5}(M_1 + 1)} = \frac{35}{18 K} < \frac{2}{K}.$$
再条件于 $\mathcal{D}_{M_1}^{\text{edge}}$ 将定理 6C(2) 中的 $\pi_k = \frac{1}{K}$ 替换为已知满足 $\hat{\pi}_k \in [\frac{1}{2K}, \frac{2}{K}]$ 的固定桶质量，重复三事件并界即得第 (2) 结论（常数仅相差至多 $2^\beta$ 与 $\sqrt{2}$ 倍）。$\blacksquare$

*完整证明（补齐三事件下侧浓缩与全 $\beta > 0$ 不等式）.*
- **(1) 逐点渐近常数**：在 $P \in \mathcal{F}_\beta^{\text{sep}}$ 上，对于充分小 $\eta > 0$ 与任意 $\varepsilon \in (0, \varepsilon_\eta)$，全局分离保证水平集 $\{x : q(x) > q^* - \varepsilon\} \subset U_\eta$ 仅由单峰邻域构成，从而 $\bar{F}_Q(q^*-\varepsilon) = (1+o(1))\frac{2f^*}{C^{1/\beta}}\varepsilon^{1/\beta}$。将积分拆为 $[0, N^\beta\varepsilon_\eta]$ 与 $[N^\beta\varepsilon_\eta, N^\beta q^*]$，后段指数衰减为 $o(1)$，前段依控制收敛趋于 $\int_0^\infty e^{-c_0 u^{1/\beta}} du = \frac{\Gamma(\beta+1)C}{(2f^*)^\beta}$。
- **(2) 一致正则类上三事件并界与最近峰耦合**：对 $K > \frac{2}{\eta_0 f_{\min}}$，每个完全落于 $U_{\eta_0} = [x^*-\eta_0, x^*+\eta_0]$ 内的等频桶因质量等于 $\frac{1}{K} < \frac{\eta_0 f_{\min}}{2}$ 而长度至多为 $\frac{1}{K f_{\min}} < \frac{\eta_0}{2}$。定义：
  $$i^{\text{near}} := \arg\min_{1 \le i \le N} |S_i - x^*|, \qquad D_N := |S_{i^{\text{near}}} - x^*|, \qquad k^{\text{near}} := k(S_{i^{\text{near}}}), \qquad \hat{k}^* := k(S_{\hat{i}^*}),$$
  及过滤代数 $\mathcal{F}_{\text{bins}} := \sigma(\mathcal{D}_M,\, k(S_1),\dots,k(S_N))$。因给定桶序列后在线同桶样本彼此独立且均服从该桶条件分布，塔性质精确给出：
  $$\mathbb{E}\big[Y_{\hat{i}^*} \mid \mathcal{F}_{\text{bins}}\big] = q_{\hat{k}^*}.$$
  划分**三个离线失效事件**（要求每个概率 $\le \frac{1}{3M}$）：
  1. $\mathcal{B}_1 := \{\exists k \in \{1,\dots,K\}:\ \hat{s}_k^{\text{KL-LCB}}(\frac{1}{3M}) > q_k\}$：由 Cramér-Chernoff 不等式与 $K$ 桶并界，$\mathbb{P}(\mathcal{B}_1) \le K \cdot \frac{1}{3KM} = \frac{1}{3M}$；
  2. $\mathcal{B}_2 := \{\exists k \in \{1,\dots,K\}:\ m_k < \frac{M}{2K}\}$：每个 $m_k \sim \text{Binomial}(M, 1/K)$ 均值 $\frac{M}{K}$，在 $M \ge 8 K \ln(3KM)$ 下乘法 Chernoff 界得 $\mathbb{P}(\mathcal{B}_2) \le K e^{-M/(8K)} \le K e^{-\ln(3KM)} = \frac{1}{3M}$；
  3. $\mathcal{B}_3 := \big\{\exists k \in \{1,\dots,K\}:\ m_k \ge 1\ \text{且}\ q_k - \hat{q}_k > \sqrt{\frac{\ln(3KM)}{2 m_k}}\big\}$：**关键下侧偏差控制**，在给定 $m_k \ge 1$ 条件下由 Hoeffding 不等式得条件概率 $\le \exp\big(-2 m_k \frac{\ln(3KM)}{2m_k}\big) = \frac{1}{3KM}$，对 $m_k$ 取期望再并界得 $\mathbb{P}(\mathcal{B}_3) \le \frac{1}{3M}$！
  合记离线失效事件 $\mathcal{B}_{\text{off}} := \mathcal{B}_1 \cup \mathcal{B}_2 \cup \mathcal{B}_3$（$\mathbb{P}(\mathcal{B}_{\text{off}}) \le \frac{1}{M}$），及在线越界事件 $\mathcal{B}_{\text{miss}} := \{D_N > \frac{\eta_0}{2}\}$（因 $\mathbb{P}(|S_i - x^*|\le\frac{\eta_0}{2})\ge \eta_0 f_{\min}$，故 $\mathbb{P}(\mathcal{B}_{\text{miss}}) \le (1-\eta_0 f_{\min})^N \le e^{-\eta_0 f_{\min} N}$）。令好事件 $\mathcal{G} := \mathcal{B}_{\text{off}}^c \cap \mathcal{B}_{\text{miss}}^c$，故 $\mathbb{P}(\mathcal{G}^c) \le \frac{1}{M} + e^{-\eta_0 f_{\min} N}$。
  在 $\mathcal{G}$ 上：首先因 $D_N \le \frac{\eta_0}{2}$ 且桶长 $\le \frac{1}{Kf_{\min}} < \frac{\eta_0}{2}$，桶 $k^{\text{near}}$ 整体包含于 $U_{\eta_0}$ 中且内部点距 $x^*$ 至多为 $D_N + \frac{1}{Kf_{\min}}$；利用对**所有** $\beta > 0$ 一致成立的初等不等式 $(a+b)^\beta \le c_\beta(a^\beta + b^\beta)$（$c_\beta := \max(1, 2^{\beta-1})$）：
  $$q^* - q_{k^{\text{near}}} \le C_1 c_\beta D_N^\beta + \frac{C_1 c_\beta}{(K f_{\min})^\beta}.$$
  其次由 $\mathcal{B}_1^c, \mathcal{B}_2^c, \mathcal{B}_3^c$ 同时成立及 Pinsker 不等式 $m_k \cdot 2(\hat{q}_k - \hat{s}_k^{\text{KL-LCB}})^2 \le m_k \operatorname{kl}(\hat{q}_k, \hat{s}_k^{\text{KL-LCB}}) \le \ln(3KM)$，对任意 $k$ 完成**两项完整误差分解**：
  $$q_k - \hat{s}_k^{\text{KL-LCB}} = \underbrace{(q_k - \hat{q}_k)}_{\le \sqrt{\frac{\ln(3KM)}{2m_k}}\ (\mathcal{B}_3^c)} + \underbrace{(\hat{q}_k - \hat{s}_k^{\text{KL-LCB}})}_{\le \sqrt{\frac{\ln(3KM)}{2m_k}}\ (\text{Pinsker})} \le 2\sqrt{\frac{\ln(3KM)}{2 m_k}} = \sqrt{\frac{2\ln(3KM)}{m_k}} \le 2\sqrt{\frac{K\ln(3KM)}{M}},$$
  （最后一步用到 $m_k \ge \frac{M}{2K}$）。因 $\hat{k}^*$ 最大化全部在线可见桶（含 $k^{\text{near}}$）之 KL-LCB 且 $\mathcal{B}_1^c$ 保证 $q_{\hat{k}^*} \ge \hat{s}_{\hat{k}^*}^{\text{KL-LCB}}$，即得：
  $$q_{\hat{k}^*} \ge \hat{s}_{\hat{k}^*}^{\text{KL-LCB}} \ge \hat{s}_{k^{\text{near}}}^{\text{KL-LCB}} \ge q_{k^{\text{near}}} - 2\sqrt{\frac{K\ln(3KM)}{M}}.$$
  最后由对一切 $r \in [0, \eta_0/2]$ 成立 $\mathbb{P}(|S_i - x^*| > r) \le 1 - 2f_{\min}r$，计算 $\mathbb{E}[\mathbf{1}_{\mathcal{G}} D_N^\beta] \le \int_0^{\eta_0/2} \beta r^{\beta-1} e^{-2f_{\min} N r} dr \le \frac{\Gamma(\beta+1)}{(2f_{\min})^\beta} N^{-\beta}$。分解总期望遗憾 $q^* - \mathbb{E}[Y_{\hat{i}^*}] = \mathbb{E}[q^* - q_{\hat{k}^*}] \le \mathbb{P}(\mathcal{G}^c) + \mathbb{E}[(q^* - q_{\hat{k}^*})\mathbf{1}_{\mathcal{G}}]$ 汇总五项，即得对所有 $P \in \mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 成立的一致非渐近上界。
- **(3) 全策略 $(M, N)$ Minimax 下界**：沿用定理 6A 证明中构造的属于同一 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 类的 $J = \Theta(h^{-1})$ 个均匀边际鼓包族 $\{P_1, \dots, P_J\}$：因为边际分打分 $S_{1:N} \sim \text{Uniform}(0,1)^N$ 与假设编号 $j$ 统计独立，积测度散度满足 $D_{\text{KL}}(P_j^{\otimes M}\otimes \text{Unif}^N \| P_0^{\otimes M}\otimes \text{Unif}^N) = M D_{\text{KL}}(P_j\|P_0)$，与 $N$ 无关，由广义 Fano 不等式得出第一项 $\Omega((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}})$；再固定一个同时属于 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0) \cap \mathcal{F}_\beta^{\text{sep}}$ 且满足局部下界 $q^* - q(x) \ge c_1|x-x^*|^\beta$ 的分布 $P^*$（例如双侧二次衰减高斯后验适当限定参数），即使在已知完美 $q_{P^*}$ 的情况下，逐样本选择准确率上限由第 (1) 步精确锁定为 $q^*_{P^*} - P_{\text{Cal-BoN}}(N) = \Theta(N^{-\beta})$。合并两项立证 $\Omega\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$。$\blacksquare$

---

### 8. 定理 7：题目异质性、辛普森伪单调遮蔽与标准化尾反转相图

回到第 1.1 节的层次生成模型：$X \sim \mathcal{P}_X$，给定 $X$ 抽取 $(Y_i, S_i) \stackrel{\text{i.i.d.}}{\sim} P_{Y, S \mid X}$，题内先验 $p(X) := \mathbb{P}(Y=1 \mid X)$，题内混合密度 $f_{\text{mix}, X}(s)$，题内真实后验 $q_X(s) := \mathbb{P}(Y=1 \mid S=s, X)$。跨题池化密度与池化后验定义为：
$$f_{\text{pool}}(s) := \mathbb{E}_X\big[f_{\text{mix}, X}(s)\big], \qquad q_{\text{pool}}(s) := \mathbb{P}(Y=1 \mid S=s) = \mathbb{E}\big[q_X(s) \mid S=s\big].$$

#### 8.1 定理 7A：辛普森混杂导数分解与构造性假单调遮蔽

**定理 7A（池化后验导数的辛普森精确分解与假单调遮蔽充分构造）.** *假设对 $\mathcal{P}_X$-a.e. $X$，密度 $f_{c, X}(s)$ 在 $s$ 上连续可微正定且满足积分号下求导正则性，记题内对数密度得分函数 $g_X(s) := \frac{\partial}{\partial s}\ln f_{\text{mix}, X}(s)$。*
1. **（精确分解恒等式）** *全局池化后验导数等于题内斜率条件期望与跨题难度协方差之和：*
   $$q'_{\text{pool}}(s) = \underbrace{\mathbb{E}\big[\partial_s q_X(s) \;\big|\; S = s\big]}_{\text{题内真斜率平均}} + \underbrace{\operatorname{Cov}\big(Y,\ g_X(s) \;\big|\; S = s\big)}_{\text{跨题难度混杂协方差}}.$$
2. **（二层显式有限参数遮蔽构造）** *存在显式有限参数的二层异方差高斯模型（无需令偏置差趋于无穷），在同一非退化闭区间 $[s_a, s_b]$ 上同时满足两层各自题内斜率处处严格为负（$\partial_s q_1(s) < 0,\ \partial_s q_2(s) < 0$），而后验混合权重严格非退化且全局池化斜率处处严格为正（$q'_{\text{pool}}(s) > 0$）。具体取共享异方差基底 $\mu_{1,\text{base}} = 0.3,\ \mu_{0,\text{base}} = 0.0,\ \sigma_1 = 1.0,\ \sigma_0 = 2.0$（$\kappa = 2 > 1$，基底峰值 $x_{\text{peak,base}} = \frac{0.3 \times 4}{4 - 1} = 0.4$），层权重 $(w_1, w_2) = (0.85, 0.15)$，先验 $(p_1, p_2) = (0.05, 0.95)$，位置平移 $(b_1, b_2) = (0.0, 3.0)$（故两层题内峰值分别为 $x_{\text{peak}, 1} = 0.4$ 与 $x_{\text{peak}, 2} = 3.4$）。则在闭区间 $s \in [3.5, 4.2]$（整体严格位于两层峰值右侧）上处处成立：*
   $$\max_{s \in [3.5, 4.2]} \partial_s q_1(s) \approx -1.35 \times 10^{-3} < 0, \qquad \max_{s \in [3.5, 4.2]} \partial_s q_2(s) \approx -1.85 \times 10^{-3} < 0, \qquad \min_{s \in [3.5, 4.2]} q'_{\text{pool}}(s) \approx +2.22 \times 10^{-2} > 0.$$

*证明.* (1) 记 $w(x\mid s) := \frac{f_{\text{mix},x}(s)}{f_{\text{pool}}(s)}$ 得 $\partial_s w(x\mid s) = w(x\mid s)\big(g_x(s) - \mathbb{E}[g_X(s)\mid S=s]\big)$，对 $q_{\text{pool}}(s)=\int w(x\mid s)q_x(s)d\mathcal{P}_X(x)$ 求导并应用 $\operatorname{Cov}(q_X, g_X\mid S=s)=\operatorname{Cov}(Y, g_X\mid S=s)$ 即证。(2) 在二层情形两点协方差恒等于 $\operatorname{Cov}(Y, g_X(s)\mid S=s) = w_1(s)w_2(s)\big(q_2(s)-q_1(s)\big)\big(g_2(s)-g_1(s)\big)$。上述有限参数实例得以成立的解析机制在于：对 $s \in [3.5, 4.2] > 3.4 = x_{\text{peak},2} > x_{\text{peak},1}$，两层对数似然比导数 $\ell'_g(s) < 0$ 虽均为负，但题内后验导数 $\partial_s q_g(s) = q_g(s)(1-q_g(s))\ell'_g(s)$ 被极端先验下的 Bernoulli 方差因子 $q_g(1-q_g)$ 显著压缩（第 1 层 $q_1 \approx 0.02$，第 2 层 $q_2 \approx 0.96$，使 $\mathbb{E}[\partial_s q_X(s)\mid S=s] \in [-3.9\times 10^{-3}, -3.7\times 10^{-3}]$）；与此同时，第 1 层密度由宽方差错类主导使 $g_1(s) \approx -s/\sigma_0^2 = -s/4 \in [-1.05, -0.875]$，第 2 层密度由窄方差正类主导但距均值 $3.3$ 更近使 $g_2(s) \approx -(s-3.3)/\sigma_1^2 \in [-0.90, -0.20] > g_1(s)$，且后验层权重保持均衡 $w_1(s) \in [0.312, 0.379]$、$w_1(s)w_2(s) > 0.21$，使正协方差项 $\ge +3.54 \times 10^{-2}$ 严格主导负的题内期望斜率，令 $q_{\text{pool}}(s)$ 在 $[3.5, 4.2]$ 上从 $0.6061$ 严格单调上升至 $0.6661$。$\blacksquare$

#### 8.2 定理 7B：题内 Z-Score 标准化的纯秩 $N=2$ 天花板与异质先验残差

考虑仿射题异质生成模型：$S_i \stackrel{d}{=} b(X) + a(X)\tilde{S}_i$（$a(X) > 0$，共享基底分布 $\tilde{S}_i \mid (Y_i=c, X) \sim \tilde{F}_c$ 具有**有限四阶矩 $\mathbb{E}[\tilde{S}^4 \mid X] < \infty$**，基混合均值 $\tilde{\mu}(X) := \tilde{\mu}_0 + p(X)(\tilde{\mu}_1 - \tilde{\mu}_0)$，基混合方差 $\tilde{\sigma}^2(X) := p(X)\tilde{\sigma}_1^2 + (1-p(X))\tilde{\sigma}_0^2 + p(X)(1-p(X))(\tilde{\mu}_1-\tilde{\mu}_0)^2$）。对单题内抽取的 $N \ge 2$ 个分数定义样本矩 $\bar{S}_N := \frac{1}{N}\sum_{i=1}^N S_i$、$\hat{s}_N := \sqrt{\frac{1}{N-1}\sum_{i=1}^N (S_i - \bar{S}_N)^2}$ 与题内 Z-Score：
$$Z_{i, N} := \frac{S_i - \bar{S}_N}{\hat{s}_N}.$$

**定理 7B（题内 Z-Score 的先验残差边界与 $N=2$ 纯秩规则天花板）.**
1. **（固定 $X$ 大 $N$ 仿射消去与异质先验 $p(X)$ 残差边界）** *在有限四阶矩 $\mathbb{E}[\tilde{S}^4 \mid X] < \infty$ 条件下，对固定题目 $X$ 与固定下标 $i$，当 $N \to \infty$ 时有：*
   $$Z_{i, N} \xrightarrow{\text{a.s.}} Z_i^*(X) := \frac{\tilde{S}_i - \tilde{\mu}(X)}{\tilde{\sigma}(X)}, \qquad Z_{i, N} = Z_i^*(X) + O_p(N^{-1/2}).$$
   - *若先验恒定 $p(X) \equiv p_0$，则 $(\tilde{\mu}(X), \tilde{\sigma}(X))$ 与 $X$ 无关，标准化后验 $\tilde{q}_X(z) \equiv \tilde{q}(z)$ 跨题一致对齐；*
   - *若 $p(X)$ 随题波动，即便消除了未知 $(b(X), a(X))$，Z-Score 仍保留由 $p(X)$ 诱导的均值漂移 $\tilde{\mu}(X)$、方差变化 $\tilde{\sigma}(X)$ 与峰值位置漂移。*
2. **（$N=2$ 退化为三类纯秩规则及严格优势充要条件）** *当 $N=2$ 时对 $S_1 \neq S_2$ 恒有 $(Z_{1,2}, Z_{2,2}) \in \{(\pm \frac{1}{\sqrt{2}}, \mp \frac{1}{\sqrt{2}})\}$，因此任何基于 $Z_{i,2}$ 的校准器 $\hat{g}(z)$ 必然退化为三类纯秩规则之一：`Raw-BoN(2)`、`Anti-BoN(2)` 或随机选择，准确率受限于非严格纯秩上界：*
   $$P_{\text{Within-Z}}(2) \le \max\!\Big(P_{\text{Raw}}(2),\ 2\bar{p} - P_{\text{Raw}}(2),\ \bar{p}\Big) \le P_{\text{Cal-BoN}}^{\text{oracle}}(2).$$
   - **（等号可达情形）** 若对 $\mathcal{P}_X$-a.e. $X$，题内后验 $q_X(s)$ 关于 $s$ 单调非降（如无异方差同质高斯 $S\mid Y=1 \sim \mathcal{N}(1,1),\ S\mid Y=0 \sim \mathcal{N}(0,1),\ p=0.5$，此时 $q(s)=\sigma(s-0.5)$ 严格递增），则 Oracle 排序等价于原始分数排序，三者精确相等（上例中 $P_{\text{Within-Z}}(2) = P_{\text{Raw}}(2) = P_{\text{Cal-BoN}}^{\text{oracle}}(2) = \frac{1}{2} + \frac{1}{4}[2\Phi(1/\sqrt{2})-1] \approx 0.630125$）；若处处单调非增则与 $2\bar{p}-P_{\text{Raw}}(2)$ 取等。
   - **（严格不等号 `<` 成立的充要条件）** 对连续条件分布 $F_{\text{mix}, X}$，右侧取严格不等号 $\max(P_{\text{Raw}}(2), 2\bar{p}-P_{\text{Raw}}(2), \bar{p}) < P_{\text{Cal-BoN}}^{\text{oracle}}(2)$ **当且仅当**在联合抽样 $(X, S_1, S_2)$ 下双向序反转事件 $\{S_1 < S_2,\ q_X(S_1) > q_X(S_2)\}$ 与 $\{S_1 < S_2,\ q_X(S_1) < q_X(S_2)\}$ 的概率均严格大于零（特别地，对任意单峰非单调异方差高斯 $\sigma_0 \neq \sigma_1$ 恒成立）。

*证明.* (1) 由强极限定理得 $\bar{S}_N \xrightarrow{\text{a.s.}} b(X)+a(X)\tilde{\mu}(X)$；在有限四阶矩 $\mathbb{E}[\tilde{S}^4\mid X] < \infty$ 下样本方差 $\hat{s}_N^2$ 具有有限二阶矩，由中心极限定理与 Delta 方法得 $\hat{s}_N = a(X)\tilde{\sigma}(X) + O_p(N^{-1/2})$，约去 $(b(X), a(X))$ 即证。(2) 对无平局双样本 $(S_1, S_2)$，Oracle 选择 $\max(q_X(S_1), q_X(S_2))$，而 `Raw-BoN(2)` 选择 $q_X(\max(S_1, S_2))$，由交换对称性直接相减得精确恒等式：
$$P_{\text{Cal-BoN}}^{\text{oracle}}(2) - P_{\text{Raw}}(2) = 2\,\mathbb{E}\Big[\big(q_X(S_1) - q_X(S_2)\big)\,\mathbf{1}\{S_1 < S_2,\ q_X(S_1) > q_X(S_2)\}\Big] \ge 0,$$
$$P_{\text{Cal-BoN}}^{\text{oracle}}(2) - \big(2\bar{p} - P_{\text{Raw}}(2)\big) = 2\,\mathbb{E}\Big[\big(q_X(S_2) - q_X(S_1)\big)\,\mathbf{1}\{S_1 < S_2,\ q_X(S_1) < q_X(S_2)\}\Big] \ge 0,$$
且 $P_{\text{Cal-BoN}}^{\text{oracle}}(2) - \bar{p} = \frac{1}{2}\mathbb{E}[|q_X(S_1) - q_X(S_2)|] \ge 0$。由此可见纯秩上界严格小于 $P_{\text{Cal-BoN}}^{\text{oracle}}(2)$ 当且仅当上述两项非负期望同时严格为正，即正向与反向序反转事件均具有正概率。$\blacksquare$

#### 8.3 定理 7C：总体标准化 $Z^*$ 诱导的尾部竞争反转与渐近上限损失

将题库分布 $p(X) \in [0, 1]$ 扩展至允许边界原子：**不可解题原子** $\pi_0 := \mathbb{P}(p(X)=0) > 0$ 与**全对题原子** $\pi_1 := \mathbb{P}(p(X)=1) \ge 0$，内点可解题落于区间 $[p_{\min}, p_{\max}] \subset (0, 1)$ 且概率 $1 - \pi_0 - \pi_1 > 0$。考察基底 Verifier 为标准**同方差单调 MLRP 高斯**（无论存在何种未知题间仿射变换 $a(X) > 0, b(X) \in \mathbb{R}$）：
$$\tilde{S}_i \mid (Y_i = 1, X) \sim \mathcal{N}(\mu_1, \sigma^2), \qquad \tilde{S}_i \mid (Y_i = 0, X) \sim \mathcal{N}(\mu_0, \sigma^2), \qquad \kappa_{\text{base}} = 1,\ \Delta := \mu_1 - \mu_0 > 0.$$
此处在真实池上 `Standard-BoN` 在每一道 $p(X)>0$ 可解题上均渐近趋近 $1$，总渐近上限 $P_{\text{Raw}}(\infty) = 1 - \pi_0$。

**定理 7C（总体标准化 $Z^*$ 的右尾竞争反转 $\frac{\pi_1}{\pi_0+\pi_1}$、渐近上限损失及 $\kappa_Z$ 充分条件）.** *考察大 $N$ 总体标准化变量 $Z^*_i(X) = \frac{\tilde{S}_i - \tilde{\mu}(X)}{\tilde{\sigma}(X)}$（已严格消去 $(a(X), b(X))$）及其池化边际后验 $q_{\text{pool}, Z^*}(z) := \mathbb{P}(Y = 1 \mid Z^* = z)$：*
1. **（无条件尾部竞争反转与精确右尾极限）** *设 $v(p) := \frac{\sigma^2}{\sigma^2 + p(1-p)\Delta^2}$。对边界原子 $p \in \{0, 1\}$ 有 $v(0) = v(1) = 1$，而对一切内点可解题 $p \in [p_{\min}, p_{\max}]$ 严格有 $v(p) < 1$。因此**无需任何额外方差条件**，标准化池化后验的远右尾必然由边界原子主导：*
   $$\lim_{z \to +\infty} q_{\text{pool}, Z^*}(z) = \frac{\pi_1}{\pi_0 + \pi_1}.$$
   *特别地，当 $\pi_0 > 0 = \pi_1$ 时，即使原始裁判绝对同方差单调，$\lim_{z\to+\infty} q_{\text{pool}, Z^*}(z) = 0$，并在某有界集合上达到全局内部最大值 $z^* \in \mathbb{R}$。*
2. **（非循环充分条件下的总体 $\kappa_Z > 1$ vs 一般情形反例）** *池化总体条件方差服从全方差公式 $\operatorname{Var}(Z^* \mid Y=c) = \mathbb{E}[v(p(X)) \mid Y=c] + \operatorname{Var}(m_c(X) \mid Y=c)$。*
   - *一般情形反例*：不可解题存在（$\pi_0 > 0 = \pi_1$）本身**并不**蕴含净条件方差比 $\kappa_Z := \sqrt{\frac{\operatorname{Var}(Z^*\mid Y=0)}{\operatorname{Var}(Z^*\mid Y=1)}} > 1$。例如取 $\pi_0=0.01$、$w_1=0.495\ (p_1=0.1)$、$w_2=0.495\ (p_2=0.2)$、$\sigma=1,\ \Delta=100$，可解题正类组间漂移项 $\operatorname{Var}(m_1 \mid Y=1) \approx 0.219$ 主导给出 $\operatorname{Var}(Z^*\mid Y=1) \approx 0.22255 > \operatorname{Var}(Z^*\mid Y=0) \approx 0.02142 \implies \kappa_Z \approx 0.3103 < 1$，尽管同时仍严格满足第 (1) 步的右尾倒挂 $\lim_{z\to+\infty} q_{\text{pool}, Z^*}(z) = 0$。
   - *非循环充分条件*：当内点可解题集中于单一先验层 $p(X) \equiv p_1 \in (0, 1)$（分布支撑于 $\{0, p_1\}$，$\pi_0 \in (0, 1), \pi_1=0$）时，$\operatorname{Var}(m_1(X) \mid Y=1) = 0$，从而对**任意** $\sigma > 0, \Delta > 0$ 严格成立：
     $$\operatorname{Var}(Z^* \mid Y=1) = v(p_1) < \alpha_{0,0}\cdot 1 + (1-\alpha_{0,0})v(p_1) + \operatorname{Var}(m_0(X) \mid Y=0) = \operatorname{Var}(Z^* \mid Y=0), \quad \text{故 } \kappa_Z > 1.$$
3. **（$Z^*$-空间 Oracle 校准器渐近上限损失）** *在 $\pi_0 > 0 = \pi_1$ 情形下，$q_{\text{pool}, Z^*}$ 的全局最大值点集 $K^*$ 非空紧支（$\sup_{z\in K^*}|z| \le M_Z < \infty$）。若以此池化后验 $q_{\text{pool}, Z^*}$ 在总体标准化分数 $Z^*_{1:N}$ 上执行逐样本极大选择，则：*
   $$P_{\text{Z}^*\text{-Cal}}^{\text{oracle}}(\infty) \le \mathbb{E}\!\left[\max_{z \in K^*} q_X\big(\tilde{\mu}(X) + \tilde{\sigma}(X)z\big)\, \mathbf{1}\{p(X) > 0\}\right] < 1 - \pi_0 = P_{\text{Raw}}(\infty).$$
4. **（在线样本题内标准化 $Z_{i,N}$ 的一致极值收敛与渐近上限损失定理）** *考察实际在线算法使用的样本标准化分数 $Z_{i,N} = \frac{S_i - \bar{S}_N}{\hat{s}_N}$（严格消去 $(a(X), b(X))$）与在线选择下标 $\hat{i}_N \in \arg\max_{1 \le i \le N} q_{\text{pool}, Z^*}(Z_{i,N})$。在同方差高斯基底（$\sigma > 0$）且 $\pi_0 > 0 = \pi_1$ 下：*
   - *对任意固定题目 $X = x$，$N$ 个候选样本的极值标准化误差一致依概率收敛于零：*
     $$\Delta_N(x) := \max_{1 \le i \le N} \big|Z_{i, N} - Z_i^*(x)\big| = O_p\!\left(\sqrt{\frac{\ln N}{N}}\right) \xrightarrow{p} 0 \qquad (N \to \infty);$$
   - *在线样本标准化 Oracle 选择器的渐近准确率上界严格低于 `Standard-BoN`：*
     $$\limsup_{N \to \infty} P_{\text{Sample-Z-Cal}}^{\text{oracle}}(N) \le \mathbb{E}\!\left[\max_{z \in K^*} q_X\big(\tilde{\mu}(X) + \tilde{\sigma}(X)z\big)\, \mathbf{1}\{p(X) > 0\}\right] < 1 - \pi_0 = P_{\text{Raw}}(\infty)$$
     *（特别地，当 $q_{\text{pool}, Z^*}$ 具有唯一全局最大值点 $K^* = \{z^*\}$ 时，极限存在且精确等于 $\mathbb{E}\big[q_X(\tilde{\mu}(X) + \tilde{\sigma}(X)z^*)\,\mathbf{1}\{p(X)>0\}\big]$）。*

> **渐近上限损失与有限 $N$ 曲线形态区分注记**：定理 7C(3)–(4) 严格证明的是总体 $Z^*$ 与实际在线样本 $Z_{i,N}$ 校准排序的**渐近准确率上限严格低于 `Standard-BoN`**（Ceiling Loss），该上限损失在数学上并不等价于有限 $N$ 准确率曲线必呈非单调倒 U 型（中等 $N$ 处的经验回撤还受分桶离散度与校准方差的共同作用）。

*完整证明.*
- **(1) 无条件右尾极限**：在各层 $\{p(X)=p_g\}$ 内，$Z^* \mid (Y=1, g) \sim \mathcal{N}(m_{1,g}, v_g)$ 与 $Z^* \mid (Y=0, g) \sim \mathcal{N}(m_{0,g}, v_g)$，其中 $m_{1,g}=\frac{(1-p_g)\Delta}{\tilde{\sigma}_g},\ m_{0,g}=-\frac{p_g\Delta}{\tilde{\sigma}_g},\ v_g=\frac{\sigma^2}{\sigma^2 + p_g(1-p_g)\Delta^2}$。边界原子（$p_g \in \{0, 1\}$）精确服从 $\mathcal{N}(0, 1)$ 密度 $\phi(z)$；而所有内点层均满足 $v_g \le v_{\max} < 1$，故二次指数衰减率 $\frac{1}{2 v_{\max}} > \frac{1}{2}$ 使所有内点层密度贡献随 $z\to +\infty$ 以 $O\big(e^{-\frac{1}{2}(v_{\max}^{-1}-1)z^2 + O(z)}\big) = o(\phi(z))$ 相对边界原子趋近于 $0$。代入 Bayes 公式：
  $$q_{\text{pool}, Z^*}(z) = \frac{\pi_1 \phi(z) + o(\phi(z))}{(\pi_1 + \pi_0)\phi(z) + o(\phi(z))} \xrightarrow{z \to +\infty} \frac{\pi_1}{\pi_0 + \pi_1}.$$
  当 $\pi_1=0<\pi_0$ 时右尾极限为 $0$，又因 $\lim_{z\to-\infty} q_{\text{pool}, Z^*}(z)=0$ 且在 $\mathbb{R}$ 上连续严格为正，故全局最大值 $q_Z^* := \max_{z\in\mathbb{R}} q_{\text{pool}, Z^*}(z) > 0$ 在非空紧集 $K^* \subset [-M_Z, M_Z]$ 上取到。
- **(2) 反例与充分条件验证**：在 $\{0, p_1\}$ 二层情形下，正类仅分布于 $p_1$ 层（权重 $\alpha_{1,0}=0, \alpha_{1,1}=1$），故 $\operatorname{Var}(m_1(X)\mid Y=1)=0$ 且 $\operatorname{Var}(Z^*\mid Y=1) = v(p_1) < 1$；错类权重 $\alpha_{0,0} = \frac{\pi_0}{\pi_0 + (1-\pi_0)(1-p_1)} > 0$，因而 $\mathbb{E}[v(p)\mid Y=0] = \alpha_{0,0}\cdot 1 + (1-\alpha_{0,0})v(p_1) > v(p_1)$，加上非负组间方差 $\operatorname{Var}(m_0\mid Y=0)\ge 0$ 立证 $\kappa_Z > 1$。多内点反例由直接代入全方差公式计算验证。
- **(3)–(4) 在线样本 $Z_{i,N}$ 一致极值收敛与渐近上限损失**：由仿射不变性 $Z_{i,N} = \frac{\tilde{S}_i - \bar{\tilde{S}}_N}{\hat{\tilde{s}}_N}$。固定任意题目 $X = x$：
  1. 因 $\tilde{\sigma}(x) = \sqrt{\sigma^2 + p(x)(1-p(x))\Delta^2} \ge \sigma > 0$，且条件于 $X=x$ 时 $\tilde{S}_i - \tilde{\mu}(x)$ 为两高斯混合之次高斯随机变量（次高斯参数一致受控于 $\sqrt{\sigma^2 + \Delta^2/4}$），由有限最大值次高斯并界得 $\max_{1\le i\le N}|Z_i^*(x)| = O_p(\sqrt{\ln N})$；同时由中心极限定理与四阶矩有界性知 $|\bar{\tilde{S}}_N - \tilde{\mu}(x)| = O_p(N^{-1/2})$、$|\hat{\tilde{s}}_N - \tilde{\sigma}(x)| = O_p(N^{-1/2})$ 且 $\mathbb{P}(\hat{\tilde{s}}_N < \sigma/2 \mid X=x) \to 0$。
  2. 代入精确代数分解 $Z_{i,N} - Z_i^*(x) = \big(\frac{\tilde{\sigma}(x)}{\hat{\tilde{s}}_N} - 1\big) Z_i^*(x) - \frac{\bar{\tilde{S}}_N - \tilde{\mu}(x)}{\hat{\tilde{s}}_N}$，取全池 $1 \le i \le N$ 最大值得：
     $$\Delta_N(x) \le \left|\frac{\tilde{\sigma}(x)}{\hat{\tilde{s}}_N} - 1\right| \max_{1 \le i \le N} |Z_i^*(x)| + \frac{|\bar{\tilde{S}}_N - \tilde{\mu}(x)|}{\hat{\tilde{s}}_N} = O_p(N^{-1/2})\cdot O_p(\sqrt{\ln N}) + O_p(N^{-1/2}) = O_p\!\left(\sqrt{\frac{\ln N}{N}}\right) \xrightarrow{p} 0.$$
  3. 因 $q_{\text{pool}, Z^*}$ 在 $\mathbb{R}$ 上连续且在 $\pm\infty$ 处趋于 $0$，故 $q_{\text{pool}, Z^*}$ 在整个 $\mathbb{R}$ 上**一致连续**，且对任意 $\varepsilon > 0$ 存在分离间隙 $\delta_\varepsilon > 0$ 使 $\sup_{\operatorname{dist}(z, K^*) \ge \varepsilon} q_{\text{pool}, Z^*}(z) \le q_Z^* - \delta_\varepsilon$。取某点 $z_0 \in K^*$，由于条件于 $X=x$ 时 $Z_1^*(x), \dots, Z_N^*(x)$ 在 $\mathbb{R}$ 上处处具正密度，必有 $\min_{1\le i\le N}|Z_i^*(x) - z_0| \xrightarrow{p} 0$。结合 $\Delta_N(x) \xrightarrow{p} 0$ 与 $q_{\text{pool}, Z^*}$ 的一致连续性，在线最大化下标 $\hat{i}_N$ 满足：
     $$q_{\text{pool}, Z^*}(Z_{\hat{i}_N}^*(x)) = q_{\text{pool}, Z^*}(Z_{\hat{i}_N, N}) + o_p(1) \ge \max_{1 \le i \le N} q_{\text{pool}, Z^*}(Z_i^*(x)) - o_p(1) \xrightarrow{p} q_Z^*,$$
     从而对任意 $\varepsilon > 0$ 有 $\mathbb{P}\big(\operatorname{dist}(Z_{\hat{i}_N}^*(x), K^*) \ge \varepsilon \mid X=x\big) \to 0$，即 $\operatorname{dist}(Z_{\hat{i}_N}^*(x), K^*) \xrightarrow{p} 0$。
  4. 注意条件于 $(X, S_1, \dots, S_N)$ 时各标签 $Y_i$ 相互独立且 $\mathbb{P}(Y_i = 1 \mid X, S_{1:N}) = q_X(S_i)$，而 $\hat{i}_N$ 关于 $(S_1,\dots,S_N)$ 可测，故塔性质给出：
     $$\mathbb{P}(Y_{\hat{i}_N} = 1 \mid X = x) = \mathbb{E}\Big[q_x\big(\tilde{\mu}(x) + \tilde{\sigma}(x) Z_{\hat{i}_N}^*(x)\big) \;\Big|\; X = x\Big].$$
     对 $p(x) = 0$ 上式恒为 $0$；对任意 $p(x) \in (0, 1)$，因 $z \mapsto q_x(\tilde{\mu}(x) + \tilde{\sigma}(x) z) \in (0, 1)$ 连续且在紧集 $K^*_\varepsilon := \{z : \operatorname{dist}(z, K^*) \le \varepsilon\}$ 上严格小于 $1$，令 $N\to\infty$ 再令 $\varepsilon \downarrow 0$ 得 $\limsup_{N\to\infty}\mathbb{P}(Y_{\hat{i}_N}=1 \mid X=x) \le \max_{z\in K^*} q_x(\tilde{\mu}(x)+\tilde{\sigma}(x)z) < 1$。最后对 $X \sim \mathcal{P}_X$ 应用有界收敛定理（Fatou 引理）即得 $\limsup_{N\to\infty} P_{\text{Sample-Z-Cal}}^{\text{oracle}}(N) < 1 - \pi_0 = P_{\text{Raw}}(\infty)$（第 (3) 步总体 $Z^*$ 情形作为 $\Delta_N \equiv 0$ 特例同步得证）。$\blacksquare$

**命题 7.1（三分支留出门控 `Hybrid-Adaptive` 的有限样本选择遗憾界）.** *因为定理 7A–7C 表明 `Standard-BoN`（无校准离散损失且无边界原子反转）、`Pooled-KL-LCB`（修正题内非单调重错尾且保留绝对分信息）与 `Within-Prompt-Z-KL-LCB`（消除强跨题仿射偏置）在不同异质性相区互无一致占优关系，定义三元候选策略族 $\mathcal{M} := \{\text{Standard-BoN},\ \text{Pooled-KL-LCB},\ \text{Within-Prompt-Z-KL-LCB}\}$（$|\mathcal{M}| = 3$）。给定条件于训练折拟合完毕的三个固定候选规则，在互不相交的 $M_{\text{val}}$ 道独立验证题目上计算各规则条件期望（或无偏抽样）准确率均值 $\hat{A}_{\text{val}}(m) \in [0, 1]$ 并令 $\hat{m} \in \arg\max_{m \in \mathcal{M}} \hat{A}_{\text{val}}(m)$，则对任意 $\delta \in (0, 1)$，条件于训练折以至少 $1 - \delta$ 的概率满足 Oracle 选择不等式：*
$$\max_{m \in \mathcal{M}} A(m) - A(\hat{m}) \le 2\sqrt{\frac{\ln(2|\mathcal{M}|/\delta)}{2\, M_{\text{val}}}}, \qquad \max_{m \in \mathcal{M}} A(m) - \mathbb{E}_{\text{val}}\big[A(\hat{m})\big] \le \sqrt{\frac{2\ln|\mathcal{M}|}{M_{\text{val}}}}.$$

*证明.* 条件于训练折，每道验证题在规则 $m \in \mathcal{M}$ 下的得分属于 $[0, 1]$ 且跨题独立同分布，均值等于总体测试准确率 $A(m)$。由 Hoeffding 不等式与 $2|\mathcal{M}|$ 次双侧并界知，以至少 $1-\delta$ 概率对所有 $m \in \mathcal{M}$ 有 $|\hat{A}_{\text{val}}(m) - A(m)| \le \sqrt{\frac{\ln(2|\mathcal{M}|/\delta)}{2 M_{\text{val}}}}$，结合 $\hat{A}_{\text{val}}(\hat{m}) \ge \hat{A}_{\text{val}}(m^*)$ 即得高概率界；期望界由有限族次高斯最大值不等式（参数 $\sigma^2 = \frac{1}{4 M_{\text{val}}}$，差值次高斯参数 $\le \frac{1}{M_{\text{val}}}$）直接给出 $\sqrt{\frac{2\ln|\mathcal{M}|}{M_{\text{val}}}}$。$\blacksquare$

---

## 9. 核心定理与软件实现符号对照索引

| 理论模块 | 核心命题与适用域界定 | 对应 `tts_theory` 实现接口 |
| :--- | :--- | :--- |
| **定理 1**（预备） | 连续顺序统计量积分、PIT $-\operatorname{expm1}(\ln v/N)$ 与含原子广义逆 | `theorems.py`: `ContinuousVerifierSpec`, `GaussianVerifierSpec`, `ContaminatedVerifierSpec` |
| **定理 2A–2C**（预备） | $C^1$ 连通 / $C^0$ 双区间 MLRP；固定峰值高度机制五步证明 $e^{-\gamma} N^*_0$（Padé 归附录 A） | `theorems.py`: `check_collapse_criterion`, `gaussian_peak_threshold`, `closed_form_optimal_n` |
| **定理 3**（预备） | 二元缩放律、按 $\beta>0$ 与 $\beta=0$（$\text{PPV}=1$）分情况 KL / $\chi^2$ / TV / 反向 KL | `theorems.py`: `BinaryVerifierSpec`, `binary_bon_accuracy`, `binary_bon_divergence_bounds`, `binary_optimal_budget` |
| **定理 4A–4B**（预备） | 塔性质无单调校准；异质单错类权重 $\Delta_{\min}$ 与超级尾桶约束一致性（固定题 $X$ 条件成立） | `theorems.py`: `calibrated_bon_asymptotic_limit`, `weighted_majority_convergence_bound`; `algorithms.py` |
| **定理 5A–5B** | 广义特征值负权逆转；$N=2$ 尾安全约束 `Safe-Fisher` 光谱路径与 Bayes QDA $N^{-2/K}$；含零乘子弱激活退化 QP `Min`（$I_{+,c}, I_{0,c}, \alpha_{c,I_0}$）；正定 Copula $\eta_c$ 四类精确区间划分（$K\ge 2, R_c\succ 0 \implies (0,1)$；非负相关充分得 $[1/K,1)$）及 $(\gamma_c-|I_{+,c}|)/2$ 对数因子 | `ensemble_theory.py`: `solve_positive_quadrant_qp`, `pareto_linear_ensemble_frontier`, `bayes_qda_multi_verifier_analysis`, `general_min_tail_asymptotic_constants`, `optimal_linear_ensemble_criterion`, `general_min_ensemble_qp_criterion`, `gaussian_copula_heavy_tail_criterion` |
| **定理 6A–6C & 命题 6.1** | Fano `1/(2\beta+1)`；`KL-LCB` 悲观底线与经验 Bernstein 归零；一致正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 三事件并界匹配 $(M,N)$ Minimax 速率；样本分割 Uniform 顺序统计量 Gamma 间距界覆盖一切 $\beta>0$ | `calibration_theory.py`: `exact_conditional_bin_bon_accuracy`, `compute_bin_calibrator_scores`, `bernstein_zero_collapse_threshold`, `calibration_minimax_scaling`, `empirical_quantile_spacings_bounds`, `joint_calibration_compute_scaling`, `tight_joint_minimax_scaling` |
| **定理 7A–7C & 命题 7.1** | 辛普森混杂导数分解与 `[3.5, 4.2]` 显式二层遮蔽；$N=2$ 纯秩非严格上界与序反转充要条件；含 $\pi_0>0$ 时总体 $Z^*$ 与在线样本 $Z_{i,N}$ 一致极值收敛 $O_p(\sqrt{(\ln N)/N})$ 及渐近上限损失；三分支留出门控遗憾界 | `heterogeneity_theory.py`: `PromptHeterogeneitySpec`, `diagnose_simpson_confounding`, `diagnose_within_prompt_residual`, `diagnose_zscore_induced_collapse`, `simulate_zscore_induced_collapse`, `verify_sample_zscore_extremal_convergence`; `real_eval.py`: `evaluate_real_prompt_heterogeneity` |

---

## 附录 A：有限 $r$ 区域 `[1/1]` Padé 工程插值公式说明

在非远尾渐近（即无量纲危险率曲率 $r := h'(x_{\text{peak}})/h^2$ 显著正值）的实用工程参数域内，直接截断一阶线性偏移项 $-\gamma + \frac{\pi^2}{6}r$ 会因极值方差受主体宽度有界约束而在 $r$ 较大时产生过冲。工程模块 `theorems.py` 中的 `closed_form_optimal_n(..., method="saddle_refined")` 采用与远尾定理 2C 相匹配的正则化 `[1/1]` Padé 重求和映射：
$$h\,(u^*_{\text{Padé}} - x_{\text{peak}}) := -\gamma + \frac{\frac{\pi^2}{6} r}{1 + \frac{\pi^2}{6} r}, \qquad N^*_{\text{saddle}} := \operatorname{round}\!\left(\frac{-1}{\ln\big(1 - \bar{F}_{\text{mix}}(u^*_{\text{Padé}})\big)}\right).$$
该式属于在 $r \to 0^+$ 时保留一阶远尾极限斜率、在有限 $r$ 上进行 `[1/1]` 有界化平滑的工程启发式插值公式，而非独立渐近极限定理。在 104 组内点参数扫描（见 [results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md)）中，其将零阶分位解 $N^*_0$ 的中位相对误差由 `26.04%` 降低至 `8.28%`（对数拟合优度 $R^2 = 0.9989$）。
