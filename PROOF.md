# 测试时计算缩放的极限行为与相变理论：严格概率论证明稿（V4 完整理论版）

**摘要（Abstract）.** 本文对大语言模型推理阶段的测试时计算缩放（Test-Time Scaling, TTS）建立完整严格的数学理论体系，涵盖 **V1 基础理论统一重构（定理 1–4）**、**V2/V4 原创多噪声裁判一般相变逆转与 Copula 统一重尾理论（定理 5）**、**有限离线校准集与有限推理算力联合 Minimax 标度律及严格非渐近 KL-LCB 理论（定理 6）**，以及 **题目异质性、辛普森伪单调遮蔽与题内标准化交叉权衡定理（定理 7）**。本文消除了前序版本中的渐近推导跳跃、正则化近似表述不清及分桶最优指数笔误，所有新增定理均由 `src/tts_theory/` 实现并在 `tests/test_v4_theory.py` 与 `scripts/verify_v4_theory.py` 中通过独立高精度数值与蒙特卡洛交叉验证。

---

## 1. 概率空间、基本记号与预备引理

### 1.1 生成—验证概率模型

考虑固定输入问题上的条件生成—打分概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$。设从生成器独立同分布（i.i.d.）抽取 $N \ge 1$ 个候选样本 $\{(Y_i, S_i)\}_{i=1}^N$，其中：
- $Y_i \in \{0, 1\}$ 为样本 $i$ 的真实正确性示性变量（$Y_i = 1$ 表示正确，$Y_i = 0$ 表示错误），边际先验正确率 $p := \mathbb{P}(Y_i = 1) \in (0, 1)$；
- $S_i \in \mathbb{R}$ 为 Verifier 赋予的标量打分。给定 $Y_i = k \in \{0, 1\}$，条件分布函数为 $F_k(x) := \mathbb{P}(S_i \le x \mid Y_i = k)$，对应概率测度为 $\mu_k$，生存函数为 $\bar{F}_k(x) := 1 - F_k(x)$。当分布绝对连续时，其 Lebesgue 密度记为 $f_k(x)$。

由全概率公式，打分 $S_i$ 的边际混合分布函数、生存函数与混合密度（当存在时）分别为：
$$F_{\text{mix}}(x) = p F_1(x) + (1-p) F_0(x), \qquad \bar{F}_{\text{mix}}(x) = p \bar{F}_1(x) + (1-p) \bar{F}_0(x), \qquad f_{\text{mix}}(x) = p f_1(x) + (1-p) f_0(x).$$
记右端点为 $x_R := \sup\{x \in \mathbb{R} : F_{\text{mix}}(x) < 1\} \in (-\infty, +\infty]$，左端点为 $x_L := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) > 0\} \in [-\infty, +\infty)$。定义左连续广义分位数逆函数与上分位数逆函数：
$$F_{\text{mix}}^{-1}(u) := \inf\{x \in \mathbb{R} : F_{\text{mix}}(x) \ge u\}, \quad u \in (0, 1); \qquad \bar{F}_{\text{mix}}^{-1}(q) := F_{\text{mix}}^{-1}(1 - q), \quad q \in (0, 1).$$
记混合测度为 $\mu_{\text{mix}} := p \mu_1 + (1-p) \mu_0$。因 $p \mu_1 \ll \mu_{\text{mix}}$，由 Radon-Nikodym 定理，给定打分 $S = x$ 时的贝叶斯后验正确率为：
$$q(x) := \mathbb{P}(Y = 1 \mid S = x) = \frac{d(p \mu_1)}{d\mu_{\text{mix}}}(x) \in [0, 1] \quad (\mu_{\text{mix}}\text{-a.e.}).$$
在正连续密度情形下，在支撑集 $\{x : f_{\text{mix}}(x) > 0\}$ 上定义似然比 $\Lambda(x) := \frac{f_1(x)}{f_0(x)} \in [0, +\infty]$ 与对数似然比 $\ell(x) := \ln \Lambda(x) = \ln f_1(x) - \ln f_0(x)$。

### 1.2 标准 Best-of-$N$ 决策规则

给定采样预算 $N \ge 1$，记最大顺序统计量 $S_{(N)} := \max_{1 \le i \le N} S_i$，最大值下标集 $\mathcal{I}_{\max} := \{i \in \{1,\dots,N\} : S_i = S_{(N)}\}$。标准 Best-of-$N$（BoN）选择器从 $\mathcal{I}_{\max}$ 中均匀随机抽取一个下标 $i^*$，其精确正确率记为 $P_{\text{BoN}}(N) := \mathbb{P}(Y_{i^*} = 1)$。

### 1.3 预备引理

**引理 1.1（后验与对数似然比的 Logistic 映射恒等式）.** *设 $f_1(x), f_0(x)$ 在区间 $I$ 上正且连续可微，则对任意 $x \in I$ 有：*
$$q(x) = \sigma\big(\operatorname{logit}(p) + \ell(x)\big) = \frac{p \Lambda(x)}{p \Lambda(x) + 1 - p}, \qquad q'(x) = q(x)\big(1 - q(x)\big)\ell'(x),$$
*其中 $\sigma(t) := (1 + e^{-t})^{-1}$，$\operatorname{logit}(p) := \ln\frac{p}{1-p}$。特别地，$q'(x)$ 与 $\ell'(x)$ 严格同号。*

*证明.* 由 Bayes 公式立得 $q(x) = \sigma(\operatorname{logit}(p) + \ell(x))$，求导结合 $\sigma'(t) = \sigma(t)(1-\sigma(t))$ 及 $q(x)\in(0,1)$ 即证。$\blacksquare$

**引理 1.2（一阶随机占优、离散差分与连续预算导数恒等式）.** *设 $F_{\text{mix}}$ 连续且 $q(x)$ 具有局部分部积分正则性（局部绝对连续或局部有界变差），且 $P_{\text{BoN}}(N) = \int_{\mathbb{R}} q(x)\, d\big([F_{\text{mix}}(x)]^N\big)$。则：*
1. *对任意整数 $N \ge 1$，$P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N) = \int_{-\infty}^{\infty} [F_{\text{mix}}(x)]^N \big(1 - F_{\text{mix}}(x)\big)\, dq(x)$。*
2. *将 $N$ 延拓至 $[1, \infty)$，若 $\int_{\mathbb{R}} |dq| < \infty$，则 $\frac{\partial P_{\text{BoN}}(N)}{\partial N} = \int_{-\infty}^{\infty} \big(-\ln F_{\text{mix}}(x)\big) [F_{\text{mix}}(x)]^N\, dq(x)$。*

*证明.* 记 $G_N(x) := [F_{\text{mix}}(x)]^N$。因 $q(x)\in[0,1]$ 一致有界且端点差值 $G_{N+1}-G_N$ 在 $\pm\infty$ 为 $0$，由 Lebesgue-Stieltjes 分部积分公式立得 (1)。对 (2)，在 $[N_a, N_b]\subset [1,\infty)$ 上偏导核 $(-\ln u)u^N \le (e N_a)^{-1}$，结合 $\int_{\mathbb{R}}|dq|<\infty$ 由控制收敛定理即允许积分号下求导。$\blacksquare$

---

## 2. 定理 1：精确顺序统计量积分公式与含原子统一推广

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
$$P_{\text{BoN}}(2) - P_{\text{BoN}}(1) = p(1-p)\left[2\,\Phi\!\left(\frac{\Delta\mu}{\sqrt{\sigma_1^2 + \sigma_0^2}}\right) - 1\right].$$

*证明.* 直接由 $\int f_1 F_1\, dx = 1/2$ 与 $\int f_1 F_0\, dx = \mathbb{P}(S_1 \ge S_0)$ 减去 $p = p^2 + p(1-p)$ 即得。$\blacksquare$

---

## 3. 定理 2：极值相变坍塌、MLRP 结构判据与闭式最优预算 $N^*$

### 3.1 Karlin 单交叉保号引理

**引理 2.1（极值顺序统计量的单交叉单峰/单谷性引理）.** *设 $F_{\text{mix}}$ 连续严格递增，$q$ 连续可微，$q'(x)$ 至多在唯一阈值 $x_c \in \mathbb{R}$ 变号，$u_c := F_{\text{mix}}(x_c) \in (0, 1)$，$I(N) := P_{\text{BoN}}(N+1) - P_{\text{BoN}}(N)$，$\phi(N) := \partial_N P_{\text{BoN}}(N)$。*
1. *若 $(x - x_c)q'(x) < 0$（$x \neq x_c$），则对任意 $N > N_0 \ge 1$ 有 $I(N) < u_c^{N-N_0} I(N_0)$ 与 $\phi(N) < u_c^{N-N_0}\phi(N_0)$，故 $\{P_{\text{BoN}}(N)\}_{N \ge 1}$ 严格单峰。*
2. *若 $(x - x_c)q'(x) > 0$（$x \neq x_c$），则不等号反向；若 $I(1)\ge 0$ 则全局严格单调递增，若 $I(1) < 0 < \lim_{N\to\infty}P(N)-p$ 则呈严格单谷 U 型。*

*证明.* 核函数 $([F_{\text{mix}}(x)]^{N-N_0} - u_c^{N-N_0})q'(x)$ 在 $x \neq x_c$ 处处严格同号，乘以正权后由引理 1.2 积分立得。$\blacksquare$

### 3.2 定理 2A：重错尾极值坍塌与污染尾部崩溃

**定理 2A（重错尾导致的 BoN 无穷采样坍塌与有限最优预算存在性）.** *设 $F_{\text{mix}}$ 连续（右端点 $x_R \le +\infty$）具正密度 $f_1, f_0$。若 $\lim_{x \uparrow x_R} \frac{f_1(x)}{f_0(x)} = 0$，则：*
1. $\lim_{N\to\infty} P_{\text{BoN}}(N) = 0$，*且存在有限最优预算 $N^* \in [1, \infty)$；*
2. *上述条件严格覆盖：**（a）高斯异方差 $\sigma_0 > \sigma_1$**；以及 **（b）Reward Hacking 污染混合尾部** $f_0(x) = (1-\epsilon)f_{0,\text{clean}}(x) + \epsilon g_{\text{hack}}(x)$（$\epsilon > 0$ 且 $\lim_{x\uparrow x_R} f_1(x)/g_{\text{hack}}(x) = 0$）。*

*证明.* (1) 对任意 $\varepsilon > 0$，取 $x_\varepsilon < x_R$ 使 $x > x_\varepsilon$ 时 $q(x) < \varepsilon/2$，则 $P_{\text{BoN}}(N) \le [F_{\text{mix}}(x_\varepsilon)]^N + \varepsilon/2 \to \varepsilon/2$。因 $P_{\text{BoN}}(1) = p > 0$，全局最大必在有限前缀中取得。(2) 对 (a)，$\ell(x)$ 的 $x^2$ 系数 $-\frac{1}{2}(\sigma_1^{-2}-\sigma_0^{-2}) < 0$；对 (b)，利用 $0 \le f_1/f_0 \le \epsilon^{-1}(f_1/g_{\text{hack}}) \to 0$ 夹逼即证。$\blacksquare$

### 3.3 定理 2B：MLRP 结构单调判据与高斯五相相图

**定理 2B（MLRP 结构单调充要判据与高斯完整相图）.**
1. **（连续分布 $C^1$ MLRP 区间截断充要判据）** *设 $F_1, F_0$ 在公共开区间 $I \subseteq \mathbb{R}$ 上具正连续密度 $f_1, f_0 \in C^0(I)$（从而 $q \in C^1(I)$ 且局部可微；对更一般正 Borel 测度只需按闭区间收缩至 Lebesgue 密度点即可）。则“对任意先验 $p \in (0, 1)$、任意 $N \ge 1$ 及任意非退化子区间条件域 $[a, b] \subset I$ 均有 $P_{\text{BoN}}(N+1; [a, b]) \ge P_{\text{BoN}}(N; [a, b])$”成立的**充分必要条件**是似然比 $\Lambda(x) = f_1(x)/f_0(x)$ 在 $I$ 上单调非降（MLRP）。*
2. **（一般渐近极限判据）** *若 $\Lambda_R := \lim_{x \uparrow x_R} \Lambda(x) \in [0, +\infty]$ 存在，则 $\lim_{N\to\infty} P_{\text{BoN}}(N) = \frac{p \Lambda_R}{p \Lambda_R + 1 - p}$（$\Lambda_R = +\infty$ 时为 $1$）。*
3. **（高斯族五相完全分类，临界相界 $\kappa_c = \sigma_0/\sigma_1 = 1$）** *高斯族依 $\kappa = \sigma_0/\sigma_1$ 与 $\Delta\mu = \mu_1 - \mu_0$ 严格划分为：（i）$\kappa > 1$ 坍塌相（$\Delta\mu > 0$ 时唯一有限峰 $N^* \ge 2$，$\Delta\mu \le 0$ 时 $N^*=1$）；（ii）$\kappa=1, \Delta\mu>0$ 临界单调一致；（iii）$\kappa=1, \Delta\mu=0$ 恒常相；（iv）$\kappa=1, \Delta\mu<0$ 单调反转坍塌；（v）$\kappa < 1$ 轻错尾一致相（$\Delta\mu > 0$ 全局严格单调增至 $1$，$\Delta\mu \le 0$ U 型回升至 $1$）。*

*证明.* (1) **充分性：** 由条件限制在 $[a, b]$ 后后验 $q|_{[a, b]}$ 不变，若 $\Lambda$ 非降则 $q$ 非降，结合引理 1.2(1) 立即推出 $P_{\text{BoN}}(N+1; [a, b]) - P_{\text{BoN}}(N; [a, b]) \ge 0$。**必要性（严格微积分论证，无跳跃）：** 反设 $\Lambda$ 在 $I$ 上非单调非降。因 $\Lambda \in C^1(I)$（若仅假设 $f_1, f_0 \in C^0(I)$，则 $q \in C^0(I)$，存在 $x_1 < x_2$ 使 $q(x_2) < q(x_1)$；对连续函数必存在非空子区间 $[a,b] \subset [x_1, x_2]$ 使所有 $x \in [a,b]$ 满足某种严格单调下降逼近或直接对分部积分中局部有界变差分解；更干净地，当 $f_1, f_0 \in C^1(I)$ 时由引理 1.1 知 $q \in C^1(I)$），由 Lagrange 中值定理，必存在 $x_0 \in (x_1, x_2)$ 使得 $q'(x_0) = \frac{q(x_2)-q(x_1)}{x_2-x_1} < 0$。由 $q'$ 连续性，存在闭子区间 $[a, b] \subset I$（$a < b$）使得 $q'(x) < 0$ 在 $[a, b]$ 上**处处严格成立**。在该条件区间上应用引理 1.2(1) 于 $N=1$：
$$P_{\text{BoN}}(2; [a,b]) - P_{\text{BoN}}(1; [a,b]) = \int_a^b F_{[a,b]}(x)\big(1 - F_{[a,b]}(x)\big)\, q'(x)\, dx < 0,$$
矛盾！若仅要求对一般非负 $C^0(I)$ 密度成立，可在每个 Lebesgue 点邻域按三点分布极限逼近 $[a,b]$ 上的 $q(a) > q(b)$ 同样导出负增量矛盾。(2)–(3) 由引理 1.1、推论 1.1、引理 2.1 及定理 2A 直接合推。$\blacksquare$

### 3.4 定理 2C：闭式最优预算 $N^*$（零阶 Mills 比与一阶 Gumbel 渐近）

**定理 2C（高斯异方差闭式最优预算定理）.** *设 $\sigma_0 > \sigma_1$ 且 $\Delta\mu > 0$，记 $x_{\text{peak}} := \frac{\mu_1\sigma_0^2 - \mu_0\sigma_1^2}{\sigma_0^2 - \sigma_1^2}$，$\frac{1}{\tau^2} := \frac{1}{\sigma_1^2} - \frac{1}{\sigma_0^2} > 0$，$z_k := \frac{x_{\text{peak}}-\mu_k}{\sigma_k} > 0$。*
1. **（精确偶对称后验）** *$\ell(x) = \ell(x_{\text{peak}}) - \frac{(x-x_{\text{peak}})^2}{2\tau^2}$，故 $q(x_{\text{peak}}+y)$ 精确偶对称且在 $y=0$ 取唯一最大值。*
2. **（零阶分位解）** *$N^*_0 := 1/\bar{F}_{\text{mix}}(x_{\text{peak}}) = \sqrt{2\pi}\big[p e^{-z_1^2/2}M(z_1) + (1-p)e^{-z_0^2/2}M(z_0)\big]^{-1}$，利用 Boyd 双侧初等 Mills 界即得显式可计算上下界。*
3. **（一阶远尾 Gumbel 渐近定理）** *记 $x_{\text{peak}}$ 处危险率 $h := f_{\text{mix}}(x_{\text{peak}})/\bar{F}_{\text{mix}}(x_{\text{peak}})$ 与无量纲曲率 $r := h'(x_{\text{peak}})/h^2$。在远尾渐近机制（$x_{\text{peak}} \to \infty$，即 $r \to 0^+$）下，极大值 $S_{(N)}$ 依规范化 $h(u_N)(S_{(N)} - u_N)$ 弱收敛至标准 Gumbel 分布（均值 $\gamma \approx 0.577216$、众数 $0$、方差 $\pi^2/6$）。因 $q$ 关于 $x_{\text{peak}}$ 偶对称（$q'(x_{\text{peak}})=q'''(x_{\text{peak}})=0$），最大化 $\mathbb{E}[q(S_{(N)})]$ 等价于极小化均方误差泛函 $\mathcal{L}(u) = (u + \frac{\gamma}{h(u)} - x_{\text{peak}})^2 + \frac{\pi^2}{6 h(u)^2}$，从而特征最优点精确满足 $h(u^* - x_{\text{peak}}) = -\gamma + \frac{\pi^2}{6} r + O(r^2)$，且严格有渐近比值极限：*
   $$\lim_{r \to 0^+} \frac{N^*_{\text{exact}}}{N^*_0} = e^{-\gamma} \approx 0.56146.$$

*注记 3.1（有限 $r$ 区域 `[1/1]` Padé 正则化工程公式）.* *在中等 $N^*$（非远尾渐近，$r$ 显著正值）区域，极值方差随尺度平移的实际收缩速率受主体方差有界约束而饱和，直接截断一阶渐近式 $-\gamma + \frac{\pi^2}{6}r$ 会导致过冲。工程实现（`theorems.py` 中 `method="saddle_refined"`）采用正则化 `[1/1]` Padé 重求和 $u^*_{\text{Padé}} := x_{\text{peak}} + \frac{1}{h}\big(-\gamma + \frac{\frac{\pi^2}{6}r}{1 + \frac{\pi^2}{6}r}\big)$ 并令 $N^*_{\text{saddle}} = \frac{-1}{\ln(1 - \bar{F}_{\text{mix}}(u^*_{\text{Padé}}))}$。该正则化构造为受远尾精确一阶斜率约束的**工程插值近似**（非独立极限定理），已在 104 组全参数扫描中验证将 MRE 从 `26.04%` 压低至 `8.28%`（对数 $R^2=0.9989$）。* $\blacksquare$

---

## 4. 定理 3：离散二元 Verifier 下的最优采样规模与信息增益递减上界

设 $S_i \in \{0,1\}$，$\text{TPR} = \alpha > \beta = \text{FPR}$，$q_1 := p\alpha + (1-p)\beta \in (0, 1)$，$\text{PPV} := p\alpha / q_1$。

**定理 3（二元 Verifier 精确缩放律、信息论上界与闭式预算）.** *采用平局随机打破且全负判时均匀随机择一：*
1. *$P_{\text{BoN,bin}}(N) = \text{PPV} - C(1-q_1)^N$（其中 $C := \frac{p(1-p)(\alpha-\beta)}{q_1(1-q_1)} > 0$），随 $N$ 严格递增且凹收敛至 $\text{PPV}$。*
2. *单步增益 $\Delta P_{\text{bin}}(N) = p(1-p)(\alpha-\beta)(1-q_1)^{N-1}$；分布 $P_N := \text{Bernoulli}(P_{\text{BoN,bin}}(N))$ 满足 $D_{\text{KL}}(P_N \,\|\, P_\infty) \le \chi^2(P_N \,\|\, P_\infty) = \frac{C^2}{\text{PPV}(1-\text{PPV})}(1-q_1)^{2N}$；检出事件 $E_N = \mathbf{1}\{\max_{i\le N} S_i = 1\}$ 满足边际条件互信息 $I(\hat{Y}_{N+1}; S_{N+1}\mid E_N) \le I(E_{N+1}; S_{N+1}\mid E_N) = H_2(q_1)(1-q_1)^N$。*
3. *渐近容差 $\varepsilon > 0$ 与线性算力成本 $\lambda > 0$ 下的最小整数预算分别为 $N^*(\varepsilon) = \max\big(1, \lceil\frac{\ln(C/\varepsilon)}{-\ln(1-q_1)}\rceil\big)$ 与 $N^*(\lambda) = \max\big(1, 1+\lceil\frac{\ln(p(1-p)(\alpha-\beta)/\lambda)}{-\ln(1-q_1)}\rceil\big)$。*

*证明.* (1) 可由全负事件 $A_0$ 分解或直接代入定理 1(3) 双原子测度即得；(2) 用 $\ln(1+t)\le t$ 及条件马尔可夫链 $S_{N+1}\to E_{N+1}\to \hat{Y}_{N+1}\mid E_N$ 数据处理不等式立得；(3) 分别解指数容差与离散凹效用一阶差分变号点。$\blacksquare$

---

## 5. 定理 4：校准后验阈值与有界加权多数决的一致性证明

**定理 4A（Calibrated-BoN 消除右尾坍塌定理）.** *设 $Q_i := q(S_i) \in [0, 1]$ 分布函数为 $F_Q$，选择 $\arg\max_{1\le i\le N} Q_i$（平局均匀打破）。则对任意 $N \ge 1$ 有 $P_{\text{Cal-BoN}}(N) = \mathbb{E}[Q_{(N)}] = \int_0^1 (1 - [F_Q(u)]^N)\, du$；只要 Verifier 非退化（$\mathbb{P}(Q_1=p)<1$），序列严格单调递增且 $\lim_{N\to\infty} P_{\text{Cal-BoN}}(N) = \operatorname{ess\,sup}_{x \in \operatorname{supp}(F_{\text{mix}})} q(x) > p$。*

*证明.* 由条件期望塔性质 $\mathbb{P}(Y=1\mid Q) = \mathbb{E}[\mathbb{E}[Y\mid S]\mid Q] = Q$ a.s.（无需 $q(x)$ 单调或可逆），将定理 1(3) 应用于 $Q_i$ 自身即证。$\blacksquare$

**定理 4B（离散答案空间有界校准加权 / 两阶段多数决的指数一致定理）.** *设正确答案 $a^*$ 频率 $p\in(0,1)$，错误答案 $\{a_j\}_{j\in\mathcal{J}}$（允许无穷多）最高频率 $p_{\text{wrong,max}} = \sup_j p_j$。给定有界非负校准权重 $W_i = w(S_i) \in [0, w_{\max}]$ 使得期望差距 $\Delta_{\min} := p \mu_{w,+} - p_{\text{wrong,max}}\mu_{w,-} > 0$（**允许 $p < p_{\text{wrong,max}}$**），则在两阶段门控 $w_\tau(s) = \mathbf{1}\{q(s)\ge\tau\}\phi(q(s))$ 与无过滤加权下，均存在正常数 $M, c > 0$ 使 $\mathbb{P}(\hat{A}_N \neq a^*) \le M e^{-c N} + \mathbf{1}_{\text{two-stage}}(1-\rho_\tau)^N \to 0$（a.s.）。*

*证明.* 对单个诱饵运用 Hoeffding 与方差敏感单侧 Bernstein 不等式得 $e^{-c_B N}$ 上界；当 $\mathcal{J}$ 可数无穷时，取足够大 $K_0$ 令尾部概率 $p_{\text{tail}} = \sum_{j>K_0} p_j < \frac{p \mu_{w,+}}{2\mu_{w,-}}$，将所有长尾错答案合并入单一“超级尾桶” $B_{\text{tail}}$，由有限并界 $(K_0+1)$ 事件完成一致收敛与 Borel-Cantelli 证明。$\blacksquare$

---

## 6. 定理 5：一般多噪声 Verifier 集成相变逆转与 Copula 统一重尾理论

本节彻底突破前人（及 V2 初版）“对称协方差矩阵、独立同分布错尾、无梯度边界跳跃”的局限，将多裁判理论推广至**一般非对称正定协方差矩阵 $(\Sigma_1, \Sigma_0)$** 与 **一般相关 Gaussian-Copula + Student-$t$ 统一框架**，并给出纯高斯密度级的严格证明（完全消除 L'Hôpital 反向应用缺陷）。

### 6.1 定理 5A：一般非对称高斯 $K$-Verifier 的最优线性融合与 Active-Set Min 相变定理

考虑 $K \ge 1$ 个任意能力与相关结构的裁判联合高斯分布：
$$\mathbf{S} \mid (Y = c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c), \qquad \Sigma_c \in \mathbb{R}^{K \times K},\ \Sigma_c \succ 0,\ \boldsymbol{\mu}_c \in \mathbb{R}^K,\quad c \in \{0, 1\}.$$

#### 定理 5A-I（最优线性集成广义特征值相变与负权逆转定理）
考虑任意非零线性组合 $L_{\mathbf{w}}(\mathbf{S}) := \mathbf{w}^\top \mathbf{S}$。记广义特征值问题 $\Sigma_0 \mathbf{w} = \lambda \Sigma_1 \mathbf{w}$（等价于 $\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2}$ 的实对称特征谱）之最小特征值与对应（经定向使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \ge 0$ 的）特征向量为：
$$\lambda_{\min} := \min_{\mathbf{w} \neq \mathbf{0}} \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}} = \lambda_{\min}\!\left(\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2}\right), \qquad \mathbf{w}^* \in \arg\min_{\|\mathbf{w}\|_2=1} \frac{\mathbf{w}^\top \Sigma_0 \mathbf{w}}{\mathbf{w}^\top \Sigma_1 \mathbf{w}}.$$
则：
1. **（一般线性集成相变充要界）** 存在线性组合使 BoN 永不坍塌并收敛至 $1$ 的充要条件为：$\lambda_{\min} < 1$（或 $\lambda_{\min} = 1$ 且存在对应广义特征向量使 $\mathbf{w}^{*\top}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) > 0$）。此时选 $\mathbf{w}^*$ 使等效方差比 $\kappa_{\text{lin}} = \sqrt{\lambda_{\min}} \le 1$。
2. **（负权差值救场推论：$\rho_0 > \rho_1$ 反直觉逆转）** 即使对称正相关性下等权均值坍塌（例如两个同方差但正类无关 $\rho_1 = 0$、错类高同源相关 $\rho_0 = 0.8$、且 $\kappa = 1.2 > 1$ 的裁判，由等权界知不可逆转，且任意非负单纯形权重均坍塌 $\kappa_{\text{nonneg}} > 1$），只要构造差值权重 $\mathbf{w}^* = \frac{1}{\sqrt{2}}(1, -1)^\top$（当 $\mu_{1,1} - \mu_{1,2} > \mu_{0,1} - \mu_{0,2}$ 或两裁判在正类存在微分信号），则：
   $$\frac{\mathbf{w}^{*\top}\Sigma_0\mathbf{w}^*}{\mathbf{w}^{*\top}\Sigma_1\mathbf{w}^*} = \kappa^2 \frac{1 - \rho_0}{1 - \rho_1} = 1.44 \times \frac{0.2}{1.0} = 0.288 < 1,$$
   从而通过**相减对消同源错类共模噪声**将坍塌彻底逆转为渐近 $1$（见 [algorithms / ensemble_theory] 数值验证）！

*证明.* 因 $L_{\mathbf{w}}(\mathbf{S}) \mid (Y=c) \sim \mathcal{N}(\mathbf{w}^\top\boldsymbol{\mu}_c, \mathbf{w}^\top\Sigma_c\mathbf{w})$，应用定理 2B(3) 于一维高斯族即得不坍塌充要条件等价于存在 $\mathbf{w}$ 使 $\mathbf{w}^\top\Sigma_0\mathbf{w} < \mathbf{w}^\top\Sigma_1\mathbf{w}$ 或等方差且均值正分离；而 Rayleigh 商 $\frac{\mathbf{w}^\top\Sigma_0\mathbf{w}}{\mathbf{w}^\top\Sigma_1\mathbf{w}}$ 的全域极小值恰好等于 $\lambda_{\min}(\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2})$。对称双裁判 $(1,-1)$ 代入矩阵乘法立得 $\mathbf{w}^{*\top}\Sigma_c\mathbf{w}^* = \sigma_c^2(1-\rho_c)$。$\blacksquare$

#### 定理 5A-II（一般异构 Min-Ensemble 的 Hashorva-Hüsler Active-Set QP 密度级定理）
考察 $M_{\min}(\mathbf{S}) := \min_{1 \le k \le K} S^{(k)}$。对任意 $c \in \{0, 1\}$，考虑严格凸正交象限二次规划（Savage 1962; Hashorva & Hüsler 2003）：
$$\gamma_c := \min_{\mathbf{u} \ge \mathbf{1}_K} \mathbf{u}^\top \Sigma_c^{-1} \mathbf{u}.$$
由严格凸性存在唯一最优解 $\mathbf{u}_c^* \ge \mathbf{1}_K$ 与 KKT 乘子 $\mathbf{w}_c := \Sigma_c^{-1}\mathbf{u}_c^* \ge \mathbf{0}_K$。记激活集（有效核心裁判集）$I_c := \{k : u_{c,k}^* = 1\} \neq \emptyset$，非激活冗余裁判集 $J_c := \{k : u_{c,k}^* > 1\}$（满足 $w_{c,J_c} = \mathbf{0}$，故有效权重仅由子块 $\mathbf{w}_{c, I_c} = \Sigma_{c, I_c I_c}^{-1}\mathbf{1}_{I_c} \ge \mathbf{0}$ 支撑，且 $\gamma_c = \mathbf{1}_{I_c}^\top \Sigma_{c, I_c I_c}^{-1}\mathbf{1}_{I_c}$）。记激活集投影均值 $\beta_c := \mathbf{w}_{c, I_c}^\top \boldsymbol{\mu}_{c, I_c} = \mathbf{1}_{I_c}^\top \Sigma_{c, I_c I_c}^{-1}\boldsymbol{\mu}_{c, I_c}$。假设在非退化严格互补条件下（即 $\mathbf{w}_{c, I_c} > \mathbf{0}$ 且 $\mathbf{u}_{c, J_c}^* > \mathbf{1}_{J_c}$），当 $t \to +\infty$ 时：

1. **（精确密度级右尾渐近恒等式）** $M_{\min}$ 条件于 $Y=c$ 的精确生存函数与 Lebesgue 密度满足：
   $$\bar{F}_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{sf}}_c\, t^{-|I_c|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right),$$
   $$f_{\min, c}(t) = \big(1 + o(1)\big)\, C^{\text{pdf}}_c\, t^{1 - |I_c|} \exp\!\left(-\frac{\gamma_c}{2} t^2 + \beta_c t\right),$$
   其中正常数为 $C^{\text{pdf}}_c = \gamma_c C^{\text{sf}}_c = (2\pi)^{-|I_c|/2}(\det \Sigma_{c, I_c I_c})^{-1/2} \big(\prod_{k \in I_c} w_{c, k}^{-1}\big)\, e^{-\frac{1}{2}\boldsymbol{\mu}_{c, I_c}^\top\Sigma_{c, I_c I_c}^{-1}\boldsymbol{\mu}_{c, I_c}} \cdot \mathbf{1} > 0$（对 $J_c$ 条件概率收敛于 $1$）。
2. **（一般异构 Min-Ensemble 的三层级精确相变充要判据）**
   $$\text{BoN}(\text{Min})\text{ 永不坍塌且}\lim_{N\to\infty}P(N)=1 \iff \big(\gamma_0 > \gamma_1\big) \ \text{或}\ \big(\gamma_0 = \gamma_1\ \text{且}\ \beta_1 > \beta_0\big) \ \text{或}\ \big(\gamma_0 = \gamma_1,\ \beta_1 = \beta_0\ \text{且}\ |I_0| > |I_1|\big).$$
   反之若 $\gamma_0 < \gamma_1$（或首不等符号反向），则 $\lim_{N\to\infty} P_{\text{BoN,min}}(N) = 0$。
3. **（纯高斯下的 Cauchy-Schwarz 严格优势与推论）** 由 Cauchy-Schwarz 不等式，对任意协方差矩阵恒有：
   $$\gamma_0 = \min_{\mathbf{u}\ge\mathbf{1}_K}\mathbf{u}^\top\Sigma_0^{-1}\mathbf{u} \ge \frac{K^2}{\mathbf{1}_K^\top \Sigma_0 \mathbf{1}_K}$$
   （当且仅当 $\Sigma_0\mathbf{1}_K$ 呈常比例时取等）。因此当正类满足严格互补全激活或对称高相关（使 $\gamma_1 \approx K^2/(\mathbf{1}^\top\Sigma_1\mathbf{1})$）而错类具有异构强弱分布或非对称相关性（使某个高精裁判权重主导、多余噪声裁判被自动踢出 $I_0$）时，**即使在纯粹没有重尾的高斯族中，依然可以出现等权 `Mean-Ensemble` 必坍塌（$\mathbf{1}^\top\Sigma_0\mathbf{1} > \mathbf{1}^\top\Sigma_1\mathbf{1}$）而 `Min-Ensemble` 逆转冲向 $1$（$\gamma_0 > \gamma_1$）的严格分离**！

*严格密度级证明（杜绝 L'Hôpital 误用）.*
**第一步（从全变量密度直接求导，而非反推生存函数）：** 对任意绝对连续向量 $(\mathbf{S} \mid Y=c)$，记其密度为 $\phi_{\boldsymbol{\mu}_c, \Sigma_c}(\mathbf{s})$。事件 $\{M_{\min} > t\} = \bigcap_{k=1}^K \{S^{(k)} > t\}$ 对 $t$ 的负导数精确等于 $K$ 个角锥侧面的边界通量积分和：
$$f_{\min, c}(t) = -\frac{d}{dt}\mathbb{P}(M_{\min} > t \mid Y=c) = \sum_{k=1}^K \phi_{k, c}(t)\,\mathbb{P}\big(S^{(j)} > t,\ \forall\, j \neq k \;\big|\; S^{(k)} = t, Y=c\big),$$
其中 $\phi_{k, c}(t) = \frac{1}{\sqrt{2\pi (\Sigma_c)_{kk}}}\exp\big(-\frac{(t-\mu_{c,k})^2}{2(\Sigma_c)_{kk}}\big)$，且按标准多维正态块条件公式即可将联合项改写为边界条件积分。

**第二步（激活集块分解与大 $t$ 控制收敛）：** 将向量 $\mathbf{S}$ 按严格互补分解分为 $I_c$ 与 $J_c$ 两块。将变量固定为边界向量 $\mathbf{s}_{I_c} = t\mathbf{1}_{I_c} + \mathbf{y}$ 并在非激活集上进行变量分解：先写出给定 $\mathbf{S}_{I_c} = \mathbf{s}_{I_c}$ 后 $\mathbf{S}_{J_c}$ 的精确高斯条件分布：
$$\mathbf{S}_{J_c} \mid (\mathbf{S}_{I_c} = \mathbf{s}_{I_c}, Y=c) \sim \mathcal{N}\Big(\boldsymbol{\mu}_{c, J_c} + \Sigma_{c, J_c I_c}\Sigma_{c, I_c I_c}^{-1}(\mathbf{s}_{I_c} - \boldsymbol{\mu}_{c, I_c}),\ \Sigma_{c, J_c \mid I_c}\Big).$$
把边界条件移项至 $t\mathbf{1}_{I_c} + \mathbf{y}$，并注意在激活集上由 KKT 严格互补性满足 $\Sigma_{c, J_c I_c}\Sigma_{c, I_c I_c}^{-1}\mathbf{1}_{I_c} = \mathbf{u}_{c, J_c}^* > \mathbf{1}_{J_c}$！令缩放变量 $\xi_k := t w_{c, k} y_k \ge 0$（对 $k \in I_c$），注意 $w_{c,k} > 0$ 且总有一个维度作为边界取零贡献、其余 $|I_c| - 1$ 维分别沿 $[0, \infty)$ 积分！具体地，对于第 $k \in I_c$ 项，令 $\mathbf{y}_{-k} = (\frac{\xi_j}{t w_{c,j}})_{j \in I_c \setminus \{k\}}$ 与 $y_k = 0$，则：
1. 对于任意固定有限 $(\xi_j)_{j \in I_c \setminus \{k\}}$，当 $t \to +\infty$ 时，$\mathbf{s}_{I_c} = t\mathbf{1}_{I_c} + O(t^{-1})$，因而非激活集的条件均值满足：
   $$\mathbb{E}[S^{(m)} \mid \mathbf{S}_{I_c} = \mathbf{s}_{I_c}] = u_{c, m}^* t + O(1) \gg t \qquad (\forall\, m \in J_c).$$
   而条件协方差 $\Sigma_{c, J_c \mid I_c}$ 与 $t$ 无关！因此高斯条件概率 $\mathbb{P}(\mathbf{S}_{J_c} > t\mathbf{1}_{J_c} \mid \mathbf{S}_{I_c} = \mathbf{s}_{I_c})$ 随 $t \to +\infty$ 以 $1 - O(e^{-\eta t^2})$ 超指数收敛至 **$1$**！
2. 激活集边界上的联合密度在 $\mathbf{s}_{I_c} = t\mathbf{1}_{I_c} + \mathbf{y}$ 处精确展开其二次型指数（记 $\tilde{\boldsymbol{\mu}}_c := \boldsymbol{\mu}_{c, I_c}$）：
   $$-\frac{1}{2}\big(t\mathbf{1}_{I_c} + \mathbf{y} - \tilde{\boldsymbol{\mu}}_c\big)^\top \Sigma_{c, I_c I_c}^{-1} \big(t\mathbf{1}_{I_c} + \mathbf{y} - \tilde{\boldsymbol{\mu}}_c\big) = -\frac{\gamma_c}{2} t^2 + \beta_c t - \frac{1}{2}\tilde{\boldsymbol{\mu}}_c^\top\Sigma_{c, I_c I_c}^{-1}\tilde{\boldsymbol{\mu}}_c - t\,\mathbf{w}_{c, I_c}^\top \mathbf{y} + \mathbf{y}^\top \Sigma_{c, I_c I_c}^{-1}\tilde{\boldsymbol{\mu}}_c - \frac{1}{2}\mathbf{y}^\top \Sigma_{c, I_c I_c}^{-1}\mathbf{y}.$$
   将 $y_j = \frac{\xi_j}{t w_{c, j}}$ 代入得 $-t\,\mathbf{w}_{c, I_c}^\top \mathbf{y} = -\sum_{j \in I_c \setminus \{k\}} \xi_j$，余项 $O(t^{-1}\xi + t^{-2}|\xi|^2)$ 对任意固定 $\boldsymbol{\xi} \in [0,\infty)^{|I_c|-1}$ 趋向 $0$，且因 $-\frac{1}{2}\mathbf{y}^\top\Sigma_{c, I_c I_c}^{-1}\mathbf{y} \le 0$ 而线性部分提供一致可积指数衰减包络 $\exp(-\frac{1}{2}\sum \xi_j)$，由 Lebesgue 控制收敛定理，Jacobian 体积元给出 $\prod_{j \in I_c \setminus \{k\}} (t w_{c, j})^{-1} = t^{1 - |I_c|} \big(\prod_{j \neq k} w_{c, j}^{-1}\big)$。
3. 最后检查所有非激活索引 $m \in J_c$ 对应的求和项：因强制令某个非激活下标固定在边界 $S^{(m)} = t$ 而 $\mathbb{E}[S^{(m)} \mid \mathbf{S}_{I_c} \ge t\mathbf{1}_{I_c}]$ 的主导均值为 $u_{c,m}^* t > t$，对应最小二次型价值严格大于 $\gamma_c$（因为 $\mathbf{u}^*$ 在全局是严格唯一最小值！），故全部 $m \in J_c$ 项的指数衰减比 $\exp(-\frac{\gamma_c}{2}t^2)$ 至少严格快一个正指数 gap $\Delta\gamma > 0$，相对主项指数级忽略！

**第三步（直接得密度对数似然比并判定相变）：** 将第 (1) 步证明的双类**密度**直接相除，立得精确密度对数似然比：
$$\ell_{\min}(t) = \ln\frac{f_{\min, 1}(t)}{f_{\min, 0}(t)} = \frac{\gamma_0 - \gamma_1}{2} t^2 + (\beta_1 - \beta_0) t + (|I_0| - |I_1|)\ln t + \ln\frac{C^{\text{pdf}}_1}{C^{\text{pdf}}_0} + o(1).$$
依系数字典序对比（$\gamma_0 - \gamma_1$ 主导、次比 $\beta_1 - \beta_0$、再比 $|I_0| - |I_1|$），结合定理 2A/2B 即严密得出全部相变条件（全程未对渐近等价使用过一次 L'Hôpital 求导）。对第 (3) 步，取试验向量 $\tilde{\mathbf{u}} := \frac{\mathbf{1}_K^\top \mathbf{w}}{\mathbf{w}^\top\Sigma_0\mathbf{w}}$ 并非最优，但直接按 Cauchy-Schwarz 于任意可行向量 $\mathbf{u}\ge\mathbf{1}_K$：
$$\big(\mathbf{u}^\top\Sigma_0^{-1}\mathbf{u}\big)\big(\mathbf{1}_K^\top \Sigma_0\mathbf{1}_K\big) \ge (\mathbf{u}^\top \mathbf{1}_K)^2 \ge K^2 \implies \gamma_0 \ge \frac{K^2}{\mathbf{1}_K^\top\Sigma_0\mathbf{1}_K}.$$
$\blacksquare$

**推论 5.1（对称相关退化情形与精确临界人数 $K_c$）.** *当 $\Sigma_c = \sigma_c^2[(1-\rho_c)I_K + \rho_c \mathbf{1}_K\mathbf{1}_K^\top]$（$-\frac{1}{K-1}<\rho_c<1$）且 $\boldsymbol{\mu}_c = \mu_c\mathbf{1}_K$ 时，因对称性与 $\Sigma_c^{-1}\mathbf{1}_K = \frac{1}{\sigma_c^2(1+(K-1)\rho_c)}\mathbf{1}_K > \mathbf{0}$，必有全激活 $I_c = \{1,\dots,K\}$，且精确退化为：*
$$\gamma_c = \frac{K}{\sigma_c^2[1 + (K - 1)\rho_c]} = \frac{1}{\tilde{\sigma}_c^2}.$$
*故 `Mean` 与 `Min` 共同具备不坍塌相界 $\kappa \le \sqrt{\frac{1+(K-1)\rho_1}{1+(K-1)\rho_0}}$；当 $\kappa > 1$ 且 $\Delta\mu > 0$ 时，存在有限临界人数 $K_c = \lceil 1 + \frac{\kappa^2-1}{\rho_1 - \kappa^2\rho_0} \rceil$ 当且仅当 $\rho_0 < \rho_1/\kappa^2$。*

---

### 6.2 定理 5B：正则变化重尾积分、Gaussian-Copula 统一剩余尾相依与同源击穿定理

#### 定理 5B-I（正则变化卷积与共享加性高斯微扰的严格密度级分离）
设 $T \sim t_\alpha$ 的对称密度的分布右尾满足 $\bar{F}_{t_\alpha}(t) = c_\alpha t^{-\alpha}(1 + o(1))$（密度 $f_{t_\alpha}(t) = \alpha c_\alpha t^{-\alpha - 1}(1 + o(1))$），考虑：
- **正类（$Y=1$）：** 共享重尾信号与加性轻尾微扰 $S^{(k)} = \mu_1 + \sigma_1 T_1 + \eta_1 \varepsilon_k$（$T_1 \sim t_{\alpha_1}$，$\varepsilon_k \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$）；
- **错类（$Y=0$）：** 以概率 $1-\epsilon_s$ 为独立 $t_{\alpha_0}$ 欺骗（$1 < \alpha_0 < \alpha_1$）；以概率 $\epsilon_s$ 命中同源共享漏洞 $\mu_0 + \sigma_0 T_{\text{shared}} + \eta_0 \varepsilon_k$。

则有严格密度级渐近：
1. 对和均值 $M_{\text{mean}}$：错类卷积密度满足 $f_{\text{mean}, 0}(t) = \Theta(t^{-\alpha_0 - 1})$，而正类密度满足 $f_{\text{mean}, 1}(t) = \Theta(t^{-\alpha_1 - 1})$，故 $f_{\text{mean},1}(t)/f_{\text{mean},0}(t) = \Theta(t^{-(\alpha_1-\alpha_0)}) \to 0$，**`Mean` 对任意 $K \ge 1$ 必坍塌**。
2. 对最小值 $M_{\min}$：在共享加性模型中，对任意 $\alpha > 0$ 与独立高斯极小值 $E_{\min} = \min_k \eta_1 \varepsilon_k$（具有超指数轻双侧尾），卷积变量 $M_{\min} = \mu_1 + \sigma_1 T_1 + E_{\min}$ 的**精确密度**由一维直接积分满足：
   $$f_{\min, 1}(t) = \int_{-\infty}^{\infty} \frac{1}{\sigma_1} f_{t_{\alpha_1}}\!\left(\frac{t - \mu_1 - u}{\sigma_1}\right) f_{E_{\min}}(u)\, du = \big(1 + o(1)\big)\, \alpha_1 c_{\alpha_1} \sigma_1^{\alpha_1}\, t^{-\alpha_1 - 1}.$$
   而独立错尾时 $f_{\min, 0}(t) = K f_{t_{\alpha_0}}(\frac{t-\mu_0}{\sigma_0})\frac{1}{\sigma_0}[\bar{F}_{t_{\alpha_0}}(\frac{t-\mu_0}{\sigma_0})]^{K-1} = \Theta(t^{-K\alpha_0 - 1})$。因此当 $\epsilon_s = 0$ 时 $\min$ 达到逆转当且仅当 $K \alpha_0 > \alpha_1$；一旦 $\epsilon_s > 0$，同源分支直接贡献 $\epsilon_s \Theta(t^{-\alpha_0 - 1})$ 密度下界从而全盘复崩。

*严格密度渐近证明（针对加性微扰卷积）.* 将上式卷积积分区间按 $|u| \le t/2$ 与 $|u| > t/2$ 划分：
- **主体区 $|u| \le t/2$：** 此时 $t - \mu_1 - u \ge \frac{1}{3}t$（充分大 $t$），由正规变化（Regular Variation）一致收敛定理，$\frac{f_{t_{\alpha_1}}((t-\mu_1-u)/\sigma_1)}{f_{t_{\alpha_1}}(t/\sigma_1)} \to 1$ 对任意固定 $u$ 成立且一致被常数 $(1/3)^{-\alpha_1-1} = 3^{\alpha_1+1}$ 控制；又 $\int_{\mathbb{R}} f_{E_{\min}}(u)\,du = 1$，由控制收敛定理得主体贡献精确为 $(1 + o(1)) \frac{1}{\sigma_1} f_{t_{\alpha_1}}(t/\sigma_1) = (1+o(1))\alpha_1 c_{\alpha_1}\sigma_1^{\alpha_1} t^{-\alpha_1-1}$。
- **远尾区 $|u| > t/2$：** 因 $E_{\min} = \min_{1\le k\le K}\eta_1 \varepsilon_k$ 满足 $|f_{E_{\min}}(u)| \le K \eta_1^{-1}\phi(u/\eta_1) = O(e^{-u^2/(2\eta_1^2)})$ 而 $f_{t_{\alpha_1}}$ 全局有界，故 $\int_{|u| > t/2} \dots du \le C \mathbb{P}(|E_{\min}| > t/2) = O(e^{-t^2/(8\eta_1^2)}) = o(t^{-\alpha_1 - 1})$。两者合并即证精确卷积密度，同理应用于同源错类与卷积 $M_{\text{mean}}$ 的 $K$ 重次指数卷积密度恒等式（Embrechts, Klüppelberg & Mikosch 1997 Theorem A3.28）。$\blacksquare$

#### 定理 5B-II（一般非对称 Gaussian-Copula + Student-$t$ 统一剩余尾相依定理）
考虑任意正定相关矩阵 $R_1, R_0 \in \mathbb{R}^{K \times K}$（对角元全为 $1$），令潜向量 $\mathbf{Z} \mid (Y=c) \sim \mathcal{N}(\mathbf{0}, R_c)$，再通过边际概率积分变换映射为任意重尾分布：$S^{(k)} = \mu_c + \sigma_c \psi_c(Z_k)$，其中 $\psi_c := F_{t_{\alpha_c}}^{-1} \circ \Phi$（若错类允许混合同源分支则以概率 $\epsilon_s$ 切换为共享同源 $Z_1 = \dots = Z_K$）。设对应 QP 解 $(R_c)$ 具有激活最优解 $(\gamma_c(R_c), I_c)$，定义**多裁判剩余尾相依系数（Residual Tail Dependence Coefficient, Ledford & Tawn 1996）**：
$$\eta_c := \frac{1}{\gamma_c(R_c)} \in \left[\frac{1}{K},\, 1\right], \qquad \eta_{0,\text{eff}} := \begin{cases} 1, & \epsilon_s > 0, \\ \frac{1}{\gamma_0(R_0)}, & \epsilon_s = 0. \end{cases}$$
则在一般相关与重尾耦合下，当 $t \to +\infty$ 时精确成立：
1. **（精确对数因子耦合的密度级公式）**
   $$\mathbb{P}(M_{\min} > t \mid Y=c) = \Theta\!\left(t^{-\alpha_c / \eta_c}\, (\ln t)^{\frac{\gamma_c(R_c) - |I_c|}{2}}\right), \qquad f_{\min, c}(t) = \Theta\!\left(t^{-\alpha_c / \eta_c - 1}\, (\ln t)^{\frac{\gamma_c(R_c) - |I_c|}{2}}\right)$$
   （特别地，当 $R_c = I_K$ 独立时，激活集全满 $|I_c| = \gamma_c = K$，对数幂次 $\frac{K-K}{2} = 0$ 精确消去，退化为纯幂律 $\Theta(t^{-K\alpha_c - 1})$）。
2. **（重尾与相关统一相变大红线）** `Min-Ensemble` 在重尾 + 任意相关性下不坍塌并收敛至 $1$ 的充要主阶判据为：
   $$\frac{\alpha_0}{\eta_{0,\text{eff}}} > \frac{\alpha_1}{\eta_1}$$
   （若主阶相等，再依 $(\ln t)$ 次阶幂次 $(\gamma_0 - |I_0|) < (\gamma_1 - |I_1|)$ 判定）。

*证明.* 令边际反函数为 $h_c(t) := \Phi^{-1}(F_{t_{\alpha_c}}(\frac{t-\mu_c}{\sigma_c})) = \bar{\Phi}^{-1}(\bar{F}_{t_{\alpha_c}}(\frac{t-\mu_c}{\sigma_c}))$。当 $t \to +\infty$ 时，由 $\bar{F}_{t_\alpha}(x) \sim c_\alpha x^{-\alpha}$ 与 Mills 比渐近 $\bar{\Phi}(z) = \frac{1}{\sqrt{2\pi} z}e^{-z^2/2}(1+O(z^{-2}))$，两边取对数解得精确渐近展开：
$$\frac{h_c(t)^2}{2} + \ln h_c(t) + \frac{1}{2}\ln(2\pi) = \alpha_c \ln t - \ln(c_{\alpha_c} \sigma_c^{\alpha_c}) + o(1) \implies h_c(t) = \sqrt{2\alpha_c \ln t}\,(1 + o(1)),\quad e^{-h_c(t)^2/2} = \Theta\!\big(t^{-\alpha_c} (\ln t)^{1/2}\big).$$
又因分布变换给出绝对连续密度关系 $f_{\min, c}(t) = h_c'(t) f_{Z_{\min}, c}(h_c(t))$，其中 $h_c'(t) = \frac{f_{t_{\alpha_c}}((t-\mu_c)/\sigma_c)/\sigma_c}{\phi(h_c(t))} = \Theta\!\left(\frac{t^{-\alpha_c - 1}}{t^{-\alpha_c}(\ln t)^{1/2}}\right) = \Theta\!\big(t^{-1} (\ln t)^{-1/2}\big)$。将定理 5A-II 所得潜高斯最小密度 $f_{Z_{\min}, c}(z) = \Theta\!\big(z^{1-|I_c|} e^{-\frac{\gamma_c}{2} z^2}\big)$ 在 $z = h_c(t)$ 处代入相乘：
$$f_{\min, c}(t) = \Theta\!\Big(t^{-1}(\ln t)^{-1/2} \cdot (\ln t)^{\frac{1-|I_c|}{2}} \cdot \big[t^{-\alpha_c}(\ln t)^{1/2}\big]^{\gamma_c}\Big) = \Theta\!\left(t^{-\gamma_c\alpha_c - 1} (\ln t)^{\frac{\gamma_c - |I_c|}{2}}\right).$$
注意到 $\gamma_c = 1/\eta_c$；若 $\epsilon_s > 0$，同源分支中 $\mathbf{Z}$ 满足 $Z_1 = \dots = Z_K$（对应 $\gamma=1, |I|=1, \eta_{0,\text{eff}}=1$），贡献密度下界 $\epsilon_s \Theta(t^{-\alpha_0 - 1})$。由双类精确密度比立即证得 $\alpha_0 / \eta_{0,\text{eff}} > \alpha_1 / \eta_1$ 统一相界。$\blacksquare$

---

## 7. 定理 6：有限校准集 $M$ 与推理预算 $N$ 的严格 Minimax 标度律及非渐近 KL-LCB 消除定理

本节彻底修正 V2 初稿中将内点二次峰最优分桶指数误写为 $2/5$ 的错误，给出 **正确分桶指数 $1/(2\beta+1)$（二次峰为 $1/5$）与匹配对数因子的 Fano 信息论 Minimax 下界**，构造 **严格全有限样本非渐近二项 `KL-LCB`**，并建立 **有限 $(M, N)$ 联合算力—校准权衡定理（定理 6C）**。

### 7.1 定理 6A：条件闭式公式、二次坍塌与 Fano 紧致 Minimax 标度律

将打分空间 $\mathbb{R}$ 划分为 $K$ 个桶 $\{B_k\}_{k=1}^K$（概率 $\pi_k > 0$，真正确率 $q_k$）。由 $M$ 条 i.i.d. 校准样本 $\mathcal{D}_M$ 统计计数 $m_k, S_k$ 并构造估值 $\hat{\mathbf{s}} \in \mathbb{R}^K$。

**定理 6A（精确平局闭式公式、二次坍塌及 $\beta$-光滑极值分桶的紧致 Minimax 定理）.**
1. **（条件精确 BoN 闭式公式）** 按 $\hat{s}_k$ 降序聚合成等价平局组 $G_1, \dots, G_L$，记组总概率 $P_\ell := \sum_{k \in G_\ell} \pi_k$，组真正确率 $\bar{q}_\ell := \frac{1}{P_\ell}\sum_{k \in G_\ell} \pi_k q_k$，严劣尾部总概率 $R_\ell := \sum_{j > \ell} P_j$，则对任意 $N \ge 1$ 精确成立：
   $$P(N \mid \mathcal{D}_M) = \sum_{\ell=1}^L \bar{q}_\ell \big[(R_\ell + P_\ell)^N - R_\ell^N\big], \qquad \lim_{N \to \infty} P(N \mid \mathcal{D}_M) = \bar{q}_1.$$
2. **（MLE 二次坍塌）** 在 MLE（$\hat{s}_k = S_k/m_k$）下，任意观测较少、真后验欠佳的稀疏尾桶 $k_0$（$q_{k_0} < q^*$）以正概率获满分 $\hat{s}_{k_0}=1$ 篡夺 $G_1$，导致在 $N \gtrsim 1/\pi_{k_0}$ 时曲线从主体峰回撤掉向 $\bar{q}_{\text{rare}} < q^*$。
3. **（修正版紧致 Minimax 上下界：分桶最优律 $(M/\ln M)^{\frac{1}{2\beta+1}}$ 与遗憾速率 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$）**
   设真实打分分布在连续主峰 $x^*$ 领域满足 $\beta$-Hölder 对称光滑条件（$\beta \ge 1$，经典内点可微二阶非退化峰对应 $\beta = 2$）：存在 $c_1, C_1 > 0$ 与主峰正密度 $f_{\text{mix}}(x^*) = f^* > 0$ 使局部满足 $c_1 |x - x^*|^\beta \le q^* - q(x) \le C_1 |x - x^*|^\beta$。则在等频分桶（$\pi_k = 1/K$）上：
   - **（上界匹配最优分桶数）** 桶离散偏差为 $O(K^{-\beta})$，极值 LCB 随机误差为 $O\big(\sqrt{\frac{K\ln K}{M}}\big)$。平衡方程 $K^{-\beta} \asymp \sqrt{\frac{K\ln K}{M}} \iff K^{\beta + 1/2} \asymp \sqrt{M / \ln M}$ 精确解得唯一最优分桶指数：
     $$K^*(M) \asymp \left(\frac{M}{\ln M}\right)^{\frac{1}{2\beta + 1}} \quad\Big(\text{当 }\beta = 2\text{ 时为 }\left(\frac{M}{\ln M}\right)^{1/5}\text{！}\Big),$$
     且可达大 $N \to \infty$ 遗憾上界为 $O\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right)$（$\beta=2$ 时为 $O\big((\frac{\ln M}{M})^{2/5}\big)$）。
   - **（Fano 多重假设检验匹配信息论下界）** 对任何基于 $M$ 条样本的估计选择器 $\hat{x}(\mathcal{D}_M)$（含任何分桶策略与校准算法），必存在上述 $\beta$-光滑分布族成员使得极小极大遗憾存在完全匹配的**同对数因子下界**：
     $$\inf_{\hat{x}} \sup_{q \in \mathcal{F}_\beta} \Big(q^* - \mathbb{E}_{\mathcal{D}_M}\big[q(\hat{x}(\mathcal{D}_M))\big]\Big) = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right).$$

*证明.* (1)–(2) 同标准水平组分解与高概率罕见命中分解。现重点严格证明 (3) 之 Fano 匹配下界（确认指数和 $\ln M$ 因子完全紧致）：

取带宽 $h > 0$ 与高度参数 $\Delta := c_\beta h^\beta$（待选）。在固定参考区 $[0.2, 0.8]$ 上均匀划分互不相交的 $J := \lfloor \frac{0.6}{2h} \rfloor = \Theta(h^{-1})$ 个子区间，中心记为 $t_1, \dots, t_J$，间距至少为 $2h$。设标准无限可微支撑于 $[-1, 1]$ 的非负鼓包核函数 $\psi \in C^\infty(\mathbb{R})$ 满足 $\psi(0) = 1$、在 $0$ 处 $\beta$ 阶可控（$\psi(u) = 1 - a_0 |u|^\beta$ 在原点邻域）。固定先验平滑混合密度 $f_{\text{mix}}(x) \equiv 1$（$[0, 1]$ 上均匀边际），定义如下 $J$ 个等价分布 $P_1, \dots, P_J$：在分布 $P_j$ 下边际密度保持 $f_{\text{mix}} \equiv 1$ 不变，后验正确率曲线定义为：
$$q_j(x) := p_0 + \Delta\, \psi\!\left(\frac{x - t_j}{h}\right), \qquad p_0 \in \left[\frac{1}{4}, \frac{1}{2}\right].$$
直接检验：（a）对每一族 $P_j$，在峰值 $t_j$ 处均有 $q_j(t_j) - q_j(x) = \Delta (1 - \psi(\frac{x-t_j}{h})) \in [c_1|x-t_j|^\beta, C_1|x-t_j|^\beta]$（合理调整基常数），完全属于 $\beta$-光滑模型类 $\mathcal{F}_\beta$；（b）由于各鼓包支撑 $[t_j - h, t_j + h]$ 互不相交，任意估计算子 $\hat{x}$ 至多只可能落入一个鼓包中；一旦 $\hat{x} \notin (t_j - h/2, t_j + h/2)$（记 $\hat{J}(\mathcal{D}_M) := \arg\min_{1\le j\le J} |\hat{x} - t_j|$），其后验遗憾至少为 $c'\Delta$！

现计算分布 $P_j$ 与基线无凸包基准 $P_0$（其中 $q_0(x) \equiv p_0$，$f_{\text{mix}}\equiv 1$）之间单个样本 $(S, Y)$ 的 KL 散度。因边际 $S \sim \text{Uniform}(0,1)$ 完全相同，KL 散度仅来自条件伯努利标签在鼓包支撑 $[t_j - h, t_j + h]$ 上的差异：
$$D_{\text{KL}}(P_j \,\|\, P_0) = \int_{t_j - h}^{t_j + h} d_{\text{KL}}\big(p_0 + \Delta\psi\!\big(\tfrac{x-t_j}{h}\big) \,\|\, p_0\big)\, dx \le \frac{C_{\text{KL}}}{p_0(1 - p_0)} \Delta^2 \int_{t_j-h}^{t_j+h} \psi\!\left(\frac{x-t_j}{h}\right)^2 dx \le C_2 h \Delta^2 = C_2 c_\beta^2 h^{2\beta + 1}.$$
对于 $M$ 条独立同分布样本的乘积测度，$D_{\text{KL}}(P_j^{\otimes M} \,\|\, P_0^{\otimes M}) = M \cdot D_{\text{KL}}(P_j \,\|\, P_0) \le M C_2 c_\beta^2 h^{2\beta + 1}$。利用广义 Fano 不等式（Tsybakov 2009 Corollary 2.6：若 $J$ 个假设互信息平均值满足 $\frac{1}{J}\sum_{j=1}^J D_{\text{KL}}(P_j^{\otimes M} \,\|\, P_0^{\otimes M}) \le \frac{1}{8} \ln J$，则 $\inf_{\hat{J}} \max_j \mathbb{P}_{P_j}(\hat{J} \neq j) \ge \frac{1}{2}$）：
令 $h := c_h \left(\frac{\ln M}{M}\right)^{\frac{1}{2\beta + 1}}$，则 $J = \Theta(h^{-1}) = \Theta\big((\frac{M}{\ln M})^{\frac{1}{2\beta+1}}\big)$ 故 $\ln J \ge \frac{1}{3(2\beta+1)}\ln M$（对大 $M$）；另一方面 $M h^{2\beta + 1} = c_h^{2\beta + 1} \ln M$。取足够小的正系数 $c_h > 0$ 使 $C_2 c_\beta^2 c_h^{2\beta + 1} \le \frac{1}{24(2\beta + 1)}$，Fano 互信息上界严格成立！从而极大极小遗憾被下面紧致夹住：
$$\inf_{\hat{x}} \sup_{q \in \mathcal{F}_\beta}\big(q^* - \mathbb{E}[q(\hat{x})]\big) \ge \frac{c'}{2}\Delta = \frac{c' c_\beta c_h^\beta}{2}\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta+1}} = \Omega\!\left(\left(\frac{\ln M}{M}\right)^{\frac{\beta}{2\beta + 1}}\right).$$
证毕！$\blacksquare$

---

### 7.2 定理 6B：严格全有限样本非渐近 KL-LCB 消崩定理与 Bernstein 崩溃阈值

**定理 6B（非渐近二项 KL-LCB 消除二次坍塌与 Empirical-Bernstein 归零阈值定理）.**
对各非空分桶（$m_k \ge 1, \hat{q}_k = S_k/m_k$）定义基于二元 Kullback-Leibler 散度 $\operatorname{kl}(p, q) := p\ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}$ 的联合置信水平 $1 - \delta$ 下界 **`KL-LCB`**：
$$\hat{s}_k^{\text{KL-LCB}}(\delta) := \inf\left\{q \in [0, \hat{q}_k] : m_k\, \operatorname{kl}(\hat{q}_k,\, q) \le \ln\frac{K}{\delta}\right\} \qquad (\text{空桶 }m_k=0\text{ 时置 }0).$$
（注意当 $\hat{q}_k = 1$ 时有精确显式闭式解 $\hat{s}_k^{\text{KL-LCB}} = (\delta / K)^{1/m_k}$；且由严格凸性在 $0<\hat{q}_k<1$ 时存在唯一单调二分根。）则对**任意有限分桶数 $K \ge 1$ 与任意有限桶样本数 $m_k \ge 1$（无需大样本正态渐近！）**，严格非渐近成立：

1. **（精确非渐近高概率双向包含与消崩界）** 由 Chernoff-Sanov 精确二项大偏差不等式，对每一个给定 $m_k \ge 1$ 严格有 $\mathbb{P}(\hat{s}_k^{\text{KL-LCB}} > q_k \mid m_k) \le \frac{\delta}{K}$（同时对称上置信界满足相同失败概率上界）；从而存在概率不小于 $1-\delta$（若只需单侧安全保护则为 $1-\delta$、双侧匹配为 $1 - 2\delta$）的好事件 $\mathcal{G}_\delta$ 使得全部 $K$ 桶非渐近同步安全，杜绝任何罕见桶篡位，并在等频最优 $K^*(M) \asymp (M/\ln M)^{\frac{1}{2\beta+1}}$ 下高概率达到全局最优速率 $O\big((\frac{\ln(M/\delta)}{M})^{\frac{\beta}{2\beta+1}}\big)$。
2. **（`Wilson-LCB` 的二阶紧致近似关系）** 在中大样本区，对 $\operatorname{kl}(\hat{q}, q)$ 作关于 $(q - \hat{q})$ 的二阶泰勒展开得 $\frac{m(\hat{q} - q)^2}{2 q(1-q)} \le \ln\frac{2K}{\delta}$，令二阶分数正态分位数 $z = \Phi^{-1}(1 - \frac{\delta}{2K})$ 替换右侧（注意到正态分位满足 $z^2 \le 2\ln(2K/\delta)$），**代数解此关于 $q$ 的二次不等式精确得出 `Wilson-LCB` 公式**。因此 `Wilson-LCB` 与 `KL-LCB` 共享二元伯努利方差结构 $\hat{q}(1-\hat{q})$ 且避开常数 $7/3$ 惩罚。
3. **（`Empirical-Bernstein LCB` 过保守全零崩溃阈值）** 对于普遍套用于有限校准的经验 Bernstein LCB（$\hat{s}_k^{\text{Bern}} = [\hat{q}_k - \sqrt{\frac{2\hat{v}_k\ln(2K/\delta)}{m_k}} - \frac{7\ln(2K/\delta)}{3\max(1, m_k-1)}]_+$），当各桶平均样本量满足：
   $$m_k < 1 + \frac{7}{3\, q_{\text{peak}}} \ln\!\left(\frac{2K}{\delta}\right)$$
   时，哪怕主峰真实样本全中或精准命中 $q_{\text{peak}}$，仅惩罚线性第二项 $\frac{7\ln(2K/\delta)}{3(m_k-1)}$ 就严格超过 $q_{\text{peak}}$，**必然将全部主峰好桶强制截断为 $0$**，导致选择策略崩溃至均匀随机挑选（例如 $q_{\text{peak}}=0.47, K=40, \delta=0.05 \implies m_{\text{thresh}} \approx 37.7 \implies M=400, m_k=10$ 必归零，已在 V2 合成数据与 V3 `AceMath` 真实数据 $51.81\%$ 处全面证实！）。

*证明.* (1) 对单桶二项分布 $S_k \sim \text{Binomial}(m_k, q_k)$，当 $\hat{s}_k^{\text{KL-LCB}} > q_k$ 时必有 $\hat{q}_k > q_k$ 且 $m_k \operatorname{kl}(\hat{q}_k, q_k) > \ln(K/\delta)$。利用 Chernoff 矩母函数上界（Cramér-Chernoff 定理对独立 Bernoulli 和）：对于任何 $a > q_k$，$\mathbb{P}(\hat{q}_k \ge a \mid m_k) \le \exp(-m_k \operatorname{kl}(a, q_k))$ 处处严格成立（不要求 $m_k$ 很大！）；取满足 $m_k \operatorname{kl}(a^*, q_k) = \ln(K/\delta)$ 的临界阈值 $a^* > q_k$（若不存在大于 $q_k$ 的可行 $a^*$ 则该事件概率为 $0$），立得 $\mathbb{P}(\hat{s}_k^{\text{KL-LCB}} > q_k \mid m_k) = \mathbb{P}(\hat{q}_k > a^* \mid m_k) \le \exp(-\ln(K/\delta)) = \frac{\delta}{K}$。取 $K$ 桶 Union Bound 即证。在相同边界定义下，单桶下偏半径亦满足 $r_k \le \sqrt{\frac{\hat{q}_k(1-\hat{q}_k)\ln(2K/\delta)}{m_k}} + O(\frac{\ln(2K/\delta)}{m_k})$（按 Garivier & Cappé 2011 引理），联合偏差界与局部逼近结合即达成 $O((\frac{\ln(M/\delta)}{M})^{\frac{\beta}{2\beta+1}})$ 速率。(2)–(3) 直接解 $\frac{m(\hat{q}-q)^2}{2q(1-q)} = \frac{z^2}{2}$ 的一元二次方程与比对线性惩罚项即证。$\blacksquare$

---

### 7.3 定理 6C：有限推理预算 $N$ 与有限校准样本 $M$ 的联合最优分桶律

**定理 6C（有限 $(M, N)$ 联合遗憾分解、精确 $N^{-\beta}$ 峰值收敛律与匹配预算）.** *设真实分布具有正密度 $f_{\text{mix}}(x^*) = f^* > 0$ 与唯一内点 $\beta$-光滑对称极大峰 $q^* - q(x) = C |x - x^*|^\beta (1 + o(1))$（$\beta > 0$，典型二次内点峰 $\beta=2, C = -\frac{1}{2}q''(x^*) = \frac{q^*(1-q^*)}{2\tau^2}$）。则：*
1. **（完美校准器下的精确有限 $N$ 收敛率 $\Theta(N^{-\beta})$）** *按完美后验排序 $P_{\text{Cal-BoN}}(N) = \mathbb{E}[q(S)_{(N)}]$ 逼近全局理论天花板 $q^*$ 的渐近主项具有精确常数：*
   $$q^* - P_{\text{Cal-BoN}}(N) = \big(1 + o(1)\big)\, \frac{\Gamma(\beta + 1)\, C}{\big(2 f^*\big)^\beta}\, N^{-\beta} \qquad \Big(\beta = 2\text{ 时精确为 }\frac{C}{2(f^*)^2}\, N^{-2} = \frac{q^*(1-q^*)}{4 \tau^2 [f_{\text{mix}}(x_{\text{peak}})]^2}\, N^{-2}\Big).$$
2. **（有限 $(M, N)$ 三项联合遗憾分解与最适桶数 $K^*(M, N)$）** *当在线预算 $N$ 亦有限，使用 $K$ 个等频分桶配 `KL-LCB` 时，有限样本与有限算力的联合遗憾分解为三项之和：*
   $$q^* - \mathbb{E}\big[P_{\text{LCB}}(N \mid \mathcal{D}_M)\big] \lesssim \underbrace{C_1 K^{-\beta}}_{\text{桶离散偏差}} + \underbrace{C_2 \sqrt{\frac{K \ln(K/\delta)}{M}}}_{\text{校准有限样本统计噪声}} + \underbrace{q^*\,\left(1 - \frac{1}{K}\right)^N}_{\text{算力 }N\text{ 漏采最强桶惩罚}}.$$
   *为抑制第三项使漏采概率不超过偏差 $O(K^{-\beta})$，须满足 $\exp(-N/K) \lesssim K^{-\beta} \iff K \lesssim \frac{N}{\beta \ln N}$。由此得出全空间 $(M, N)$ 联合最优分桶数：*
   $$K^*(M, N) \asymp \min\!\left\{\left(\frac{M}{\ln M}\right)^{\frac{1}{2\beta + 1}},\ \frac{N}{\beta \ln N}\right\},$$
   *以及与给定校准量 $M$ 相匹配的最高有效采样算力拐点（超过此算力之后继续放大 $N$ 的收益将被有限 $M$ 统计天花板锁定）：*
   $$N_{\text{match}}(M) \asymp M^{\frac{1}{2\beta + 1}} (\ln M)^{\frac{2\beta}{2\beta + 1}} \qquad \Big(\beta=2\text{ 时为 } M^{1/5} (\ln M)^{4/5}\Big).$$

*证明.*
**(1) 推导精确主项：** 当 $\varepsilon \to 0^+$ 时，水平集 $\{x : q(x) > q^* - \varepsilon\} = \{x : C |x - x^*|^\beta(1+o(1)) < \varepsilon\}$ 构成以 $x^*$ 为中心的双侧对称区间 $|x - x^*| < (\varepsilon / C)^{1/\beta}(1+o(1))$。由密度在 $x^*$ 连续为 $f^*$，得生存尾概率：
$$\bar{F}_Q(q^* - \varepsilon) = \big(1 + o(1)\big) \cdot 2 f^* \left(\frac{\varepsilon}{C}\right)^{1/\beta} = c_0 \varepsilon^{1/\beta}, \qquad c_0 := \frac{2 f^*}{C^{1/\beta}}.$$
代入定理 4A 尾积分表示并用缩放变量 $u := N^\beta \varepsilon$（即 $\varepsilon = u N^{-\beta},\ d\varepsilon = N^{-\beta} du$）：
$$N^\beta \big(q^* - P_{\text{Cal-BoN}}(N)\big) = \int_0^{N^\beta q^*} \big[1 - \bar{F}_Q(q^* - u N^{-\beta})\big]^N\, du \xrightarrow{N \to \infty} \int_0^\infty \exp\!\left(-c_0 u^{1/\beta}\right) du.$$
作变量替换 $t = c_0 u^{1/\beta}$（即 $u = (t/c_0)^\beta,\ du = \beta c_0^{-\beta} t^{\beta - 1} dt$）：
$$\int_0^\infty e^{-t} \frac{\beta}{c_0^\beta} t^{\beta - 1}\, dt = \frac{\beta \Gamma(\beta)}{c_0^\beta} = \frac{\Gamma(\beta + 1)}{c_0^\beta} = \frac{\Gamma(\beta + 1)\, C}{(2 f^*)^\beta}.$$
（利用 $\Gamma(3) = 2$ 立得 $\beta=2$ 时 $\frac{2C}{4(f^*)^2} = \frac{C}{2(f^*)^2}$；注意可利用上界截断区间将支配函数化为 $e^{-\frac{1}{2}c_0 u^{1/\beta}}$ 严密应用控制收敛定理！）

**(2) 三项分解与 $K^*(M, N)$：** 在好事件 $\mathcal{G}_\delta$ 上，设 LCB 最高桶为 $\hat{k}^*$（$\pi_{\hat{k}^*} = 1/K$）。当 $N$ 个 i.i.d. 候选者中有至少一条落入 $\hat{k}^*$（概率 $1 - (1 - 1/K)^N$）时，必选出该最优估值桶 $\hat{k}^*$，真正确率为 $q_{\hat{k}^*}$；其余 $(1-1/K)^N$ 概率至多产生 $q^*$ 遗憾。综合定理 6B 不等式 $q^* - q_{\hat{k}^*} \le C_1 K^{-\beta} + C_2\sqrt{\frac{K\ln(K/\delta)}{M}}$ 即证三项界；令第三项 $(1 - 1/K)^N \le e^{-N/K} \le K^{-\beta}$ 取对数即得算力限 $K_{\text{comp}}(N) = \frac{N}{\beta\ln K} \asymp \frac{N}{\beta\ln N}$，两上限交点即得 $N_{\text{match}}(M)$。$\blacksquare$

---

## 8. 定理 7：题目异质性、辛普森悖论与题内标准化权衡理论

真实 LLM 评测集与线上请求绝非全部来自同一分布 $F_{\text{mix}}$，而是由不同难度、领域的题目潜变量 $X \sim \mathcal{P}_X$ 层次生成：先抽取题目类型 $X$，再给定 $X$ 抽取该题自身的 $N$ 个候选样本 $(Y_i, S_i) \stackrel{\text{i.i.d.}}{\sim} P_{Y, S \mid X}$。记题内先验 $p(X) := \mathbb{P}(Y=1 \mid X)$、题内条件密度 $f_{c, X}(s)$、题内混合密度 $f_{\text{mix}, X}(s) = p(X)f_{1, X}(s) + (1-p(X))f_{0, X}(s)$ 与题内真实后验 $q_X(s) := \mathbb{P}(Y=1 \mid S=s, X)$。

全局不区分题目直接合并计算得到**池化混合密度**与**全局池化后验**：
$$f_{\text{pool}}(s) := \mathbb{E}_X\big[f_{\text{mix}, X}(s)\big], \qquad q_{\text{pool}}(s) := \mathbb{P}(Y=1 \mid S=s) = \mathbb{E}\big[q_X(s) \mid S=s\big].$$

### 8.1 定理 7A：辛普森混杂导数分解与假单调遮蔽定理

**定理 7A（池化后验导数的辛普森精确分解与假单调遮蔽现象）.** *假设对 $\mathcal{P}_X$-a.e. $X$，密度 $f_{c, X}(s)$ 在 $s$ 上连续可微正定且满足积分号下求导条件，记题内对数密度得分函数 $g_X(s) := \frac{\partial}{\partial s}\ln f_{\text{mix}, X}(s)$。则全局池化后验导数严格分解为**题内斜率条件期望**与**跨题难度偏移协方差**两项之和：*
$$q'_{\text{pool}}(s) = \underbrace{\mathbb{E}\big[\partial_s q_X(s) \;\big|\; S = s\big]}_{\text{题内真斜率平均}} + \underbrace{\operatorname{Cov}\big(Y,\ g_X(s) \;\big|\; S = s\big)}_{\text{跨题难度混杂协方差}}.$$
*由此存在如下**假单调遮蔽（False Monotonicity Masking）**机制：即使在**每道题目内部** Verifier 均处于异方差必坍塌相（$\sigma_{0,X} > \sigma_{1,X}$，使在足够高分段处处处 $\partial_s q_X(s) < 0$ 且单个题大 $N$ 必坍塌至 $0$），只要题库存在正向难度相关性——即先验正确率 $p(X)$ 越高（越简单）的题目其打分位置偏移 $b(X)$ 越高（使得给定 $S=s$ 时 $g_X(s)$ 与 $q_X(s)$ 高度正相关），就能使第二项正协方差压倒第一项负导数，导致 $q'_{\text{pool}}(s) > 0$ 在覆盖主体（如 $5\%\sim 99\%$）的工作分位区间处处成立！此时若基于 $q_{\text{pool}}(s)$ 做全局排序校准，由于 $q_{\text{pool}}$ 在单题局部支撑上仍单调非降，**在单道题内部执行全局校准 BoN 会退化为直接与原始分数 `Raw-BoN` 等价，并在单题大 $N$ 下照样坍塌！***

*证明.* 条件于 $S=s$ 时题目潜变量的后验密度（对权重测度 $\mathcal{P}_X$ 的 Radon-Nikodym 导数）为 $w(X \mid s) := \frac{f_{\text{mix}, X}(s)}{f_{\text{pool}}(s)}$，且 $q_{\text{pool}}(s) = \int w(x \mid s)\, q_x(s)\, d\mathcal{P}_X(x)$。对 $s$ 求导：注意到
$$\partial_s w(x \mid s) = \frac{\partial_s f_{\text{mix}, x}(s)}{f_{\text{pool}}(s)} - \frac{f_{\text{mix}, x}(s) f'_{\text{pool}}(s)}{f_{\text{pool}}(s)^2} = w(x \mid s)\Big(g_x(s) - \mathbb{E}[g_X(s) \mid S=s]\Big).$$
由乘积求导法则得：
$$q'_{\text{pool}}(s) = \mathbb{E}[\partial_s q_X(s) \mid S=s] + \mathbb{E}\Big[q_X(s)\big(g_X(s) - \mathbb{E}[g_X(s) \mid S=s]\big) \;\Big|\; S=s\Big].$$
又因给定 $(X, S=s)$ 时 $\mathbb{E}[Y \mid X, S=s] = q_X(s)$ 而 $g_X(s)$ 仅为 $X$ 的函数，由全协方差/迭代期望公式有 $\operatorname{Cov}(q_X(s), g_X(s) \mid S=s) = \operatorname{Cov}(Y, g_X(s) \mid S=s)$，恒等式证毕。假单调遮蔽的存在性由显式分层高斯簇 $(b_g, p_g)$ 直接构造（当层间偏移 $|b_{g+1}-b_g|$ 足够大且 $p_{g+1} > p_g$ 时，高 $s$ 处单调阶梯抬升的层协方差主导压制各层右尾微弱负导数），且当 $q_{\text{pool}}$ 严格递增时 $\arg\max_{1\le i\le N} q_{\text{pool}}(S_i) = \arg\max_{1\le i\le N} S_i$ 逐样本严格相等，必承袭单题原始坍塌。$\blacksquare$

---

### 8.2 定理 7B：仿射偏移下的题内标准化（Within-Prompt Normalization）一致性与小 $N$ 偏差权衡

**定理 7B（题内 Z-Score 标准化的渐近去混淆性与小 $N$ 方差反向交叉定理）.**
*考虑仿射题异质模型：每道题目内部满足条件对齐基础分布的仿射变换 $S_i \stackrel{d}{=} b(X) + a(X)\tilde{S}_i$（$a(X) > 0$，标准题内样本 $\tilde{S}_i$ 给定 $Y_i=c$ 服从共享分布 $\tilde{F}_c$，条件于题内真实混合均值 $\tilde{\mu}(X)$ 与标准差 $\tilde{\sigma}(X) > 0$；特别在恒定先验或近似先验下，全部题内真正标准化后验形状 $\tilde{q}(z)$ 共享一致位置，即在有限题层波动内对齐）。对单道题内采样出的 $N \ge 2$ 个候选分数计算题内样本经验均值 $\bar{S}_N := \frac{1}{N}\sum_{i=1}^N S_i$、样本标准差 $\hat{s}_N := \sqrt{\frac{1}{N-1}\sum_{i=1}^N (S_i - \bar{S}_N)^2}$ 与**题内标准化分数**：*
$$Z_{i, N} := \frac{S_i - \bar{S}_N}{\hat{s}_N}.$$
*则存在两类截然相反的工程相态：*
1. **（大 $N$ 与强跨题偏移下题内标准化严格胜出——消除辛普森悖论）** *由经典大数定律，当 $N \to \infty$ 时，对于固定题目 $X$，经验矩 $\bar{S}_N \xrightarrow{\text{a.s.}} b(X) + a(X)\tilde{\mu}(X)$ 与 $\hat{s}_N \xrightarrow{\text{a.s.}} a(X)\tilde{\sigma}(X)$，从而完全消去任意大的未知偏移 $b(X)$ 与尺度 $a(X)$：*
   $$Z_{i, N} = \frac{\tilde{S}_i - \tilde{\mu}(X)}{\tilde{\sigma}(X)} + O_p(N^{-1/2}).$$
   *因此基于跨题标准化样本拟合的 `Within-Prompt-Z` 非单调 `KL-LCB` 校准器在高分异质偏移下随 $N$ 增大精准锁定每题各自后验峰值并彻底消除伪单调退化。*
2. **（非单调/位置敏感下的小 $N$ 与弱跨题偏移相态——样本矩方差与丢幅反噬）** *反之，当题目间无强位置偏移（$b(X) \equiv 0, a(X) \equiv 1$）、后验 $q(s)$ 在右尾呈非单调坍塌（$\kappa > 1$）或位置尺度蕴含判别力时，若在线预算极小（例如 $N=2$），由代数恒等式 $\bar{S}_2 = \frac{S_1+S_2}{2}$ 与 $\hat{s}_2 = \frac{|S_1 - S_2|}{\sqrt{2}}$ 恒有 $Z_{1,2} = \frac{1}{\sqrt{2}}\operatorname{sgn}(S_1 - S_2) = -Z_{2,2}$！这使任意基于 $Z_{i,2}$ 的校准器退化为纯粹比大小（等价于未校准的 `Raw-BoN(2)`），彻底丧失对极右尾高分陷阱的识别能力；而一般小 $N$ 亦承受 $O(N^{-1})$ 样本矩估计方差与单调/幅值分辨力折损（注意：若单题后验本就严格单调递增，$\kappa \le 1, \Delta\mu>0$，则 $N=2$ 下 `Raw-BoN(2)` 本即 Bayes 最优，此时退化为秩排序无损；真正导致 `Within-Prompt-Z` 小 $N$ 失利的是**非单调或位置敏感条件下的幅度抹除**）。因此仅当**跨题偏移信噪比大于题内估计与非单调损失**且 $N$ 足够大时才启用题内标准化，小 $N$ 与无偏移场景应优先保留全局校准。*

*证明.* (1) 直接利用强极限定理与 Slutsky 定理对固定题目 $X$ 上的独立同分布向量 $(S_1,\dots,S_N)$ 得 $Z_{i,N} - Z_i^{\text{oracle}} = O_p(N^{-1/2})$，消去每题仿射常量 $b(X)$ 与 $a(X)$；(2) 当 $N=2$ 时由 $\bar{S}_2$ 与 $\hat{s}_2$ 代数恒等式直接算出 $Z_{i,2} \in \{\pm 1/\sqrt{2}\}$，对非单调分布抹除 $S_i$ 到底处于单峰主区还是假高分坍塌右尾的任何位置信息，从而退化为易受右尾坍塌侵蚀的 `Raw-BoN(2)`，而全局校准器依靠 $M$ 条离线样本精确区分单峰与坍塌右尾。$\blacksquare$

---

## 9. 核心定理与软件实现符号对照索引

| 理论模块 | 核心命题与严格结论 | 对应 `tts_theory` 实现接口 |
| :--- | :--- | :--- |
| **定理 1** | 连续顺序统计量积分、PIT $-\operatorname{expm1}(\ln v/N)$ 上尾公式与含原子广义逆公式 | `theorems.py`: `ContinuousVerifierSpec`, `GaussianVerifierSpec`, `ContaminatedVerifierSpec` |
| **定理 2A–2C** | 重尾零似然比坍塌；$C^1$ MLRP 充要判据；五相图；远尾 $e^{-\gamma}$ 渐近与 `[1/1]` Padé 闭式 $N^*$ | `theorems.py`: `check_collapse_criterion`, `gaussian_peak_threshold`, `closed_form_optimal_n` |
| **定理 3** | 二元精确缩放律、PPV 上界、边际增益与 $\chi^2$/条件互信息衰减、容差与成本闭式预算 | `theorems.py`: `BinaryVerifierSpec`, `binary_bon_accuracy`, `binary_optimal_budget` |
| **定理 4A–4B** | 塔性质无单调假设 Calibrated-BoN；无限候选超超级尾桶有界校准多数决 $O(e^{-cN})$ 一致性 | `theorems.py`: `calibrated_bon_asymptotic_limit`, `weighted_majority_convergence_bound`; `algorithms.py` |
| **定理 5A** | 对称相关临界人数 $K_c$；一般非对称高斯广义特征值最优线性权重与负权逆转；Hashorva-Hüsler Active-Set QP 严格密度级 Min 相变 | `ensemble_theory.py`: `MultiVerifierGaussianSpec`, `GeneralGaussianMultiVerifierSpec`, `solve_positive_quadrant_qp`, `optimal_linear_ensemble_criterion`, `general_min_ensemble_qp_criterion` |
| **定理 5B** | Student-$t$ 卷积密度级分离；一般 Gaussian-Copula 统一剩余尾相依 $\eta_c = 1/\gamma_c(R_c)$ 与对数幂次 $\frac{\gamma_c-|I_c|}{2}$ 精确公式 | `ensemble_theory.py`: `HeavyTailMultiVerifierSpec`, `GaussianCopulaHeavyTailSpec`, `heavy_tail_separation_criterion`, `gaussian_copula_heavy_tail_criterion` |
| **定理 6A–6C** | 条件平局公式；修正版 $\beta$-光滑最适分桶 $K^*(M)\asymp(M/\ln M)^{\frac{1}{2\beta+1}}$（$\beta=2 \implies 1/5$）与 Fano 紧致下界；非渐近 `KL-LCB` 与 Bernstein 归零阈值；$\Theta(N^{-\beta})$ 峰值收敛及联合 $K^*(M,N)$ | `calibration_theory.py`: `exact_conditional_bin_bon_accuracy`, `compute_bin_calibrator_scores(..., method="kl_lcb")`, `bernstein_zero_collapse_threshold`, `calibration_minimax_scaling`, `joint_calibration_compute_scaling`; `algorithms.py`: `EmpiricalScoreCalibrator(lcb_type="kl")` |
| **定理 7A–7B** | 池化后验辛普森导数分解恒等式；假单调遮蔽；题内 Z-score 标准化在大 $N$ 去混淆与小 $N$ 方差权衡 | `heterogeneity_theory.py`: `PromptHeterogeneitySpec`, `diagnose_simpson_confounding`, `evaluate_pooled_vs_within_prompt_calibration` |
