# 文献定位、经典数学工具溯源与技术边界（Related Work & Novelty Audit）

本文档梳理截至 2026 年 11 月与本项目直接相关的两类文献：（1）**大模型 Test-Time Scaling、Reward Model Overoptimization 与 Best-of-$N$ 校准文献**（12 篇）；（2）本项目证明所直接依托与对比的**经典概率论、多元极值理论、Gaussian Copula 弱尾相依、被动非参数回归最大化、离线悲观主义及连续臂 Bandit 基础文献**（10 篇，总计 22 篇），明确区分对经典数理统计工具的复用与在 Test-Time Scaling 设定下建立的增量技术边界。

---

## 1. 相关文献全景（共 22 篇）

### 1.1 LLM Test-Time Scaling 与 BoN 理论文献（12 篇）

| 编号 | 论文简称与作者 | 发表信息 / arXiv ID | 核心研究问题与主要结论 |
| :---: | :--- | :--- | :--- |
| **[1]** | Gao, Schulman, Hilton (2023) | ICML 2023 (`2210.10760`) | 在合成金标 RM 下拟合 RLHF 与 BoN 的 Goodhart 过优化经验缩放律（$\alpha \sqrt{\text{KL}} - \beta \text{KL}$）。 |
| **[2]** | Coste et al. (2024) | ICLR 2024 (`2310.02743`) | 在合成金标设定下实验验证多个 RM 的最坏情况集成（WCO / `min`）与不确定性惩罚可缓解过优化，未给出多裁判相变解析判据。 |
| **[3]** | Ramé et al. (2024) | `arXiv:2401.12187` | 提出权重平均奖励模型（WARM）提升多 RM 集成稳定性。 |
| **[4]** | Brown et al. (2024) | `arXiv:2407.21787` | *Large Language Monkeys*：揭示重复采样 Pass@$N$ 覆盖率增长与 Verifier 筛选瓶颈。 |
| **[5]** | Snell et al. (2024) | `arXiv:2408.03314` | *Scaling LLM Test-Time Compute Optimally*：按问题难度分配测试时算力。 |
| **[6]** | Stroebl, Kapoor, Narayanan (2024) | `arXiv:2411.17501` | *Inference Scaling fLaws*：分析二元不完美 Verifier 假阳性对重采样上限的制约。 |
| **[7]** | Huang et al. (2025) | **ICML 2025** (`2503.21878`, PMLR) | *Is Best-of-N the Best of Them?*：以覆盖率刻画 BoN 在不完美 RM 下的失败，并提出悲观拒绝采样 `InferenceTimePessimism` 保证计算单调性。 |
| **[8]** | Dorner, Chen, Cruz, Yang (2025) | `arXiv:2507.12399` | *ROC-n-reroll*：证明 BoN 曲线由 Verifier 的 ROC 几何刻画，指出低算力表现无法外推高算力坍塌。 |
| **[9]** | Tang, Chen, Cavallaro (2025) | `arXiv:2510.15674` | *CarBoN*：在推理阶段动态校准生成器 logits 改善测试时扩展。 |
| **[10]** | Schweisthal et al. (2026-10) | `arXiv:2610.09250` | 研究在不进行海量抽样下高效评估 BoN 策略的统计方法。 |
| **[11]** | Yano, Ishibashi, Oyamada (2026-10) | `arXiv:2610.19362` | *Tail Competition Explains Scaling in Best-of-N*：基于单变量 EVT 三域（Theorems 1–4）分析单裁判 BoN 尾部竞争与高斯 $\sigma_0>\sigma_1$ 坍塌，给出 tail-tension 早停判据。 |
| **[12]** | Gui, Shao, Li, Guo, Wang (2026-10) | `arXiv:2610.34375` | *Whether or Where? Tail Calibration Signals Finite-Budget Reward Hacking*：给出 Beta 分位积分（Prop. 3.1/3.4）、对数凹单调判据（Prop. A.6）及基于尾部校准审计的预算早停方法。 |

### 1.2 经典概率论、多元极值、Copula、被动非参数回归最大化与离线悲观文献（10 篇）

| 编号 | 论文/专著与作者 | 发表期刊 / 会议 | 对本项目数学工具的直接支撑与理论对比范围 |
| :---: | :--- | :--- | :--- |
| **[13]** | Savage (1962) | *J. Res. Nat. Bur. Standards* 66B:93–96 | 建立多元标准正态象限尾概率的经典 Mills 比上界与二次规划条件。 |
| **[14]** | Hashorva & Hüsler (2003) | *Annals of the Institute of Statistical Mathematics (AISM)*, 55(3):507–522 | 系统推导多元高斯尾概率的 Active-Set 二次规划 $\min_{\mathbf{u}\ge\mathbf{1}}\mathbf{u}^\top\Sigma^{-1}\mathbf{u}$、本质正乘子支撑集 $I_+$ 与条件正态正象限退化情形下的精确渐近表示（定理 5A-II 直接应用该理论）。 |
| **[15]** | Garivier & Cappé (2011) | *COLT* (`PMLR 19:359–376`) | 系统分析二项 Kullback–Leibler 置信界 `KL-UCB`/`KL-LCB` 的非渐近 Cramér–Chernoff 反演收敛性质（定理 6B/6C 直接复用该置信界）。 |
| **[16]** | Tankov (2016) | *Journal of Multivariate Analysis (JMVA)*, 145:73–86 (`arXiv:1402.4683`) | 提出渐近独立随机向量（含 Gaussian Copula）的弱尾相依函数，系统刻画了对数尺度下联合极端事件的指数衰减速率。 |
| **[17]** | Das & Fasen-Hartmann (2024) | *Journal of Multivariate Analysis (JMVA)*, 202:105310 (`arXiv:2204.06205`) | 系统分析 **Gaussian Copula 与正则变化重尾边际耦合** 下不同尾集上的联合极端风险衰减率与边际变换效应（定理 5B-II 直接衔接该类概率结构）。 |
| **[18]** | Romano (1988) | *The Annals of Statistics*, 16(2):629–647 | 建立非参数核密度众数（mode）位置估计的局部渐近 Minimax 下界与最优带宽理论。 |
| **[19]** | Bubeck, Munos, Stoltz, Szepesvári (2011) | *Journal of Machine Learning Research (JMLR)*, 12:1655–1695 | 建立局部光滑度量空间连续臂 Bandit（X-Armed Bandits / HOO）在**主动序贯探索**下的累积与简单遗憾 Minimax 理论。 |
| **[20]** | Belitser, Ghosal, van Zanten (2012) | *The Annals of Statistics*, 40(6):2850–2876 | 建立**被动非参数回归函数最大化点与最大值估计**的最优收敛速率理论，刻画局部光滑度与峰值衰减指数对被动优化率 $(\ln M / M)^{\frac{\beta}{2\beta+d}}$ 的决定关系。 |
| **[21]** | Minsker (2013) | *COLT* (`PMLR 30:105–121`) | 建立**被动随机设计（random design）非参数回归极大化**在局部光滑与峰值边缘条件下的简单遗憾（函数值遗憾）非渐近上界与 Minimax 下界。 |
| **[22]** | Jin, Yang, Wang (2021) | *ICML* (`PMLR 139:5084–5096`) | *Is Pessimism Provably Efficient for Offline RL?*：系统确立离线被动数据分布覆盖不足（稀疏区域赢家诅咒）时**LCB 悲观主义（Pessimism）**消除虚假高估并匹配离线信息论下界的统计准则。 |

---

## 2. 数学工具来源与技术边界评估

### 2.1 预备定理 1–4（附录 A）：单裁判基础统一表述（非原创性声明）

| 本项目预备定理 | 对应先发 TTS / 概率文献 | 本项目的定位与细化范围 |
| :--- | :--- | :--- |
| **定理 1**（含原子顺序统计量积分） | `[8]` Prop. 5、`[12]` Prop. 3.1/A.1 及经典顺序统计量 | 补全含平局原子分布下基于左连续广义逆的测度一致性与无相消 $-\operatorname{expm1}(\ln v/N)$ 双精度积分实现。 |
| **定理 2**（高斯坍塌、$C^1/C^0$ MLRP 与 $e^{-\gamma}$） | `[11]` Theorems 1–2、`[12]` Prop. 3.4/A.6 及经典 Karlin 总正性 | 拆分 $C^1$ 连通子区间与 $C^0$ 双闭子区间 MLRP 论证；给出固定峰值高度机制下五步极值矩与单峰 argmin 证明（`[1/1]` Padé 归附录 B 工程插值）。 |
| **定理 3**（二元缩放律与边界散度） | `[6]` 假阳性上限及基础 Bernoulli 信息论 | 补全按 $\beta>0$ 与 $\beta=0$（$\text{PPV}=1$，$\text{TV}$ 与有限反向 KL）分类的散度界与闭式预算公式。 |
| **定理 4**（校准后验与有界加权多数决） | `[5]` 加权多数决、`[7]` ICML 2025、Hoeffding/Bernstein 浓缩 | 约束单个高权错类期望 $p\mu_{w,+} > \sup_j(p_j\mu_{w,j})$，并用有界尾部超级桶处理可数无穷异质错误类。 |

### 2.2 核心模块（定理 6、定理 7、定理 5）：数学来源区分与 TTS 增量贡献

| 本项目模块 | 依托的既有数学工具 | 相对已有文献的 TTS 增量贡献（适用范围与诚实边界） |
| :--- | :--- | :--- |
| **主定理 I：定理 6A–6C & 命题 6.1**（联合 $(M, N)$ Minimax 与预算平衡律） | `[15]` Garivier & Cappé (2011)、`[20]` Belitser et al. (2012)、`[21]` Minsker (2013)、`[22]` Jin et al. (2021)、Rényi–Pyke 顺序统计量间距 | 将被动非参数回归最大化与离线 LCB 耦合至**有限无标签在线候选集 $S_{1:N}$ 选择**，在 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 上证得无 $(\ln N)^\beta$ 损失的联合率 $\Theta((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta})$、导出预算平衡点 $N_{\text{balance}}(M)$，并用样本分割紧 Cramér 间距界在零额外无标签数据下覆盖一切 $\beta>0$。 |
| **主定理 II：定理 7A–7C & 命题 7.1**（跨题异质性与在线样本标准化） | 混合分布全导数（Simpson 分解）、位置—尺度不变量、次高斯极值耦合与留出风险控制不等式 | 连接固定题模型与跨题层次模型 $X\sim\mathcal{P}_X$：给出闭区间 `[3.5, 4.2]` 无网格纯解析遮蔽构造、$N=2$ 纯秩非严格界与序反转充要条件；严格跟踪仿射坐标 $b(X)+a(X)\tilde{S}$，证得含 $\pi_0>0$ 时在线样本 $Z_{i,N}$ 以 $O_p(\sqrt{(\ln N)/N})$ 一致逼近总体 $Z^*$ 从而继承上限损失，以及全体 $N\ge 2$ 三分支门控遗憾与 $\varepsilon_{\text{refit}}$ 分解。 |
| **补充理论：定理 5A–5B**（多裁判光谱、含退化 QP `Min` 与 Copula 重尾） | `[14]` Hashorva & Hüsler (2003)、`[16]` Tankov (2016)、`[17]` Das & Fasen-Hartmann (2024)、Slepian 引理与凸化对偶 | 建立渐近尾约束下 $N=2$ 线性 SNR 光谱路径 `Safe-Fisher`（明确不主张 $N>2$ 最优）与不可逆相 QDA $N^{-2/K}$ 律；利用本质正乘子集 $I_{+,c}$ 与弱激活正象限常数 $\alpha_{c,I_0}$ 将 `Min-Ensemble` 与 Gaussian-Copula + Student-$t$（对数因子 $(\gamma_c-\|I_{+,c}\|)/2$）推广至含零乘子退化的任意 SPD 矩阵。 |

### 2.3 定理 6C 与经典被动非参数回归最大化及离线悲观理论的精确对照

针对“定理 6C 的离线项 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$ 与经典统计理论在何处重合、在何处产生 TTS 特有结构”这一核心学术问题，本文作出如下诚实界定：

1. **当在线预算 $N = \infty$ 时的经典统计重合面（无夸大归属）**：若允许决策者在整个连续实轴 $\mathbb{R}$ 上自由输出任意点 $\hat{x}(\mathcal{D}_M)$（等价于在线拥有无穷密集候选池 $N=\infty$），则从被动随机设计（passive random design）i.i.d. 样本 $\mathcal{D}_M \sim P^{\otimes M}$ 中最大化一维光滑度为 $\beta$ 的回归函数 $q(x) = \mathbb{E}[Y \mid S=x]$，其函数值简单遗憾速率 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}\big)$ 属于经典数理统计已知结论——其位置估计对应 Romano (1988) `[18]` 与 Belitser, Ghosal & van Zanten (2012) `[20]`，随机设计函数值简单遗憾对应 Minsker (2013) `[21]`（相较于主动序贯查询的 X-Armed Bandits `[19]`，被动抽样无法向峰值邻域自适应加密采样，故保留全局覆盖对数因子 $\ln M$）；同时用置信下界抑制稀疏分桶赢家诅咒的机制直接继承离线强化学习/Bandit 悲观原则（Jin, Yang & Wang 2021 `[22]`；Huang et al. 2025 `[7]`）。
2. **定理 6C 与命题 6.1 在 Test-Time Scaling 设定下的三项具体增量贡献**：
   - **（i）有限随机无标签候选池 $S_{1:N} \sim F_{\text{mix}}^{\otimes N}$ 的无对数损失耦合**：不同于经典回归最大化可在实轴上直接输出确定性估计点 $\hat{x}_M$，TTS 在线推理只能从生成器新抽出的有限 $N$ 条无标签随机候选 $S_1,\dots,S_N$ 中离散择一。若朴素对 $N$ 个在线点做逐点置信上/下界并界，会额外引入 $(\ln N)$ 耦合损失或令分桶数 $K$ 依赖在线预算 $N$；定理 6C(2) 通过最近峰候选下标 $i^{\text{near}} = \arg\min_{i\le N}|S_i - x^*|$ 与桶条件代数 $\mathcal{F}_{\text{bins}}$ 的塔性质解耦，严格证明 **$N$-无关静态分桶 $K^*(M)$** 在全平面 $(M, N)$ 上不产生任何 $(\ln N)^\beta$ 膨胀因子，并在单一共同正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 上同时兼容 Fano 鼓包下界与极值间距下界 $\Omega(N^{-\beta})$。
   - **（ii）校准—算力预算平衡律 $N_{\text{balance}}(M) \asymp (M/\ln M)^{\frac{1}{2\beta+1}}$**：双阶段加性解耦率直接给出离线标注投入 $M$ 与在线推理投入 $N$ 的定量转换边界——当在线采样数超过 $N_{\text{balance}}(M)$（光滑内点峰 $\beta=2$ 时仅以 $(M/\ln M)^{1/5}$ 缓慢增长）后，单纯增加推理采样 $N$ 无法继续降低峰值遗憾，系统瓶颈严格锁定在离线校准集规模 $M$。
   - **（iii）粗糙峰 $0 < \beta \le \frac{1}{2}$ 下零额外无标签数据的经验等频分桶浓缩（命题 6.1）**：当混合边际分布 $F_{\text{mix}}$ 未知且禁止借用额外海量无标签池时，经典加性 DKW 经验分布界给出桶质量相对误差 $K^*(M)\sqrt{\frac{\ln M}{M}} \asymp (\frac{M}{\ln M})^{\frac{1-2\beta}{2(2\beta+1)}}$，在尖峰/粗糙峰 $0 < \beta \le \frac{1}{2}$ 处失效发散；命题 6.1 利用样本分割顺序统计量的 Rényi–Pyke Gamma 乘性间距界，严格在总预算 $M$ 内覆盖一切 $\beta > 0$。

---

## 3. 完整 BibTeX 参考文献清单

```bibtex
@inproceedings{gao2023scaling,
  title={Scaling Laws for Reward Model Overoptimization},
  author={Gao, Leo and Schulman, John and Hilton, Jacob},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023},
  eprint={2210.10760},
  archivePrefix={arXiv}
}

@inproceedings{coste2024reward,
  title={Reward Model Ensembles Help Mitigate Overoptimization},
  author={Coste, Thomas and Anwar, Usman and Kirk, Robert and Krueger, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2310.02743},
  archivePrefix={arXiv}
}

@article{rame2024warm,
  title={WARM: On the Benefits of Weight Averaged Reward Models},
  author={Ram{\'e}, Alexandre and Vieillard, Nino and Hussenot, L{\'e}onard and Dadashi, Robert and Cideron, Geoffrey and Bachem, Olivier and Ferret, Johan},
  journal={arXiv preprint arXiv:2401.12187},
  year={2024}
}

@article{brown2024large,
  title={Large Language Monkeys: Scaling Inference Compute with Repeated Sampling},
  author={Brown, Bradley and Juravsky, Jordan and Ehrlich, Ryan and Clark, Ronald and Le, Quoc V. and R{\'e}, Christopher and Mirhoseini, Azalia},
  journal={arXiv preprint arXiv:2407.21787},
  year={2024}
}

@article{snell2024scaling,
  title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters},
  author={Snell, Charlie and Lee, Jaehoon and Xu, Kelvin and Kumar, Aviral},
  journal={arXiv preprint arXiv:2408.03314},
  year={2024}
}

@article{stroebl2024inference,
  title={The Limits of Inference Scaling Through Resampling},
  author={Stroebl, Benedikt and Kapoor, Sayash and Narayanan, Arvind},
  journal={arXiv preprint arXiv:2411.17501},
  year={2024}
}

@inproceedings{huang2025best,
  title={Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment},
  author={Huang, Audrey and Block, Adam and Liu, Qinghua and Jiang, Nan and Krishnamurthy, Akshay and Foster, Dylan J.},
  booktitle={International Conference on Machine Learning (ICML)},
  series={Proceedings of Machine Learning Research},
  year={2025},
  eprint={2503.21878},
  archivePrefix={arXiv}
}

@article{dorner2025roc,
  title={ROC-n-reroll: How verifier imperfection affects test-time scaling},
  author={Dorner, Florian E. and Chen, Yatong and Cruz, Andr{\'e} F. and Yang, Fanny},
  journal={arXiv preprint arXiv:2507.12399},
  year={2025}
}

@article{tang2025carbon,
  title={CarBoN: Calibrated Best-of-N Sampling Improves Test-time Reasoning},
  author={Tang, Yung-Chen and Chen, Pin-Yu and Cavallaro, Andrea},
  journal={arXiv preprint arXiv:2510.15674},
  year={2025}
}

@article{schweisthal2026efficient,
  title={Efficient Best-of-N policy evaluation for inference-time alignment},
  author={Schweisthal, Jonas and Wang, Yuxin and Deviyani, Athiya and Feuerriegel, Stefan and Frauen, Dennis},
  journal={arXiv preprint arXiv:2610.09250},
  year={2026}
}

@article{yano2026tail,
  title={Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems},
  author={Yano, Taro and Ishibashi, Yoichi and Oyamada, Masafumi},
  journal={arXiv preprint arXiv:2610.19362},
  year={2026}
}

@article{gui2026whether,
  title={Whether or Where? Tail Calibration Signals Finite-Budget Reward Hacking in Best-of-N Selection},
  author={Gui, Cheng and Shao, Yihua and Li, Lei and Guo, Jingcai and Wang, Maolin},
  journal={arXiv preprint arXiv:2610.34375},
  year={2026}
}

@article{savage1962mills,
  title={Mills' ratio for multivariate normal distributions},
  author={Savage, I. Richard},
  journal={Journal of Research of the National Bureau of Standards Section B: Mathematics and Mathematical Physics},
  volume={66B},
  number={3},
  pages={93--96},
  year={1962}
}

@article{hashorva2003multivariate,
  title={On multivariate Gaussian tails},
  author={Hashorva, Enkelejd and H{\"u}sler, J{\"u}rg},
  journal={Annals of the Institute of Statistical Mathematics},
  volume={55},
  number={3},
  pages={507--522},
  year={2003}
}

@inproceedings{garivier2011kl,
  title={The KL-UCB algorithm for bounded stochastic bandits and beyond},
  author={Garivier, Aur{\'e}lien and Capp{\'e}, Olivier},
  booktitle={Proceedings of the 24th Annual Conference on Learning Theory (COLT)},
  series={Proceedings of Machine Learning Research},
  volume={19},
  pages={359--376},
  year={2011}
}

@article{tankov2016tails,
  title={Tails of weakly dependent random vectors},
  author={Tankov, Peter},
  journal={Journal of Multivariate Analysis},
  volume={145},
  pages={73--86},
  year={2016},
  doi={10.1016/j.jmva.2015.12.008}
}

@article{das2024heavy,
  title={On heavy-tailed risks under Gaussian copula: The effects of marginal transformation},
  author={Das, Bikramjit and Fasen-Hartmann, Vicky},
  journal={Journal of Multivariate Analysis},
  volume={202},
  pages={105310},
  year={2024},
  doi={10.1016/j.jmva.2024.105310}
}

@article{romano1988weak,
  title={On Weak Convergence and Optimality of Kernel Density Estimates of the Mode},
  author={Romano, Joseph P.},
  journal={The Annals of Statistics},
  volume={16},
  number={2},
  pages={629--647},
  year={1988},
  doi={10.1214/aos/1176350824}
}

@article{bubeck2011xarmed,
  title={X-Armed Bandits},
  author={Bubeck, S{\'e}bastien and Munos, R{\'e}mi and Stoltz, Gilles and Szepesv{\'a}ri, Csaba},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={1655--1695},
  year={2011}
}

@article{belitser2012optimal,
  title={Optimal rates for estimating the maximum of a regression function},
  author={Belitser, Eduard and Ghosal, Subhashis and van Zanten, Harry},
  journal={The Annals of Statistics},
  volume={40},
  number={6},
  pages={2850--2876},
  year={2012},
  doi={10.1214/12-AOS1053}
}

@inproceedings{minsker2013estimation,
  title={Estimation of Extreme Values and Associated Level Sets of a Regression Function via Selective Sampling},
  author={Minsker, Stanislav},
  booktitle={Proceedings of the 26th Annual Conference on Learning Theory (COLT)},
  series={Proceedings of Machine Learning Research},
  volume={30},
  pages={105--121},
  year={2013}
}

@inproceedings{jin2021pessimism,
  title={Is Pessimism Provably Efficient for Offline RL?},
  author={Jin, Ying and Yang, Zhuoran and Wang, Zhaoran},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series={Proceedings of Machine Learning Research},
  volume={139},
  pages={5084--5096},
  year={2021}
}
```
