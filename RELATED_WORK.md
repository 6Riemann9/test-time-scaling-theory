# 文献定位、经典数学工具溯源与技术边界（Related Work & Novelty Audit）

本文档梳理截至 2026 年 11 月与本项目直接相关的两类文献：（1）**大模型 Test-Time Scaling、Reward Model Overoptimization 与 Best-of-$N$ 校准文献**（12 篇）；（2）本项目证明所直接依托与对比的**经典概率论、多元极值理论、Gaussian Copula 弱尾相依、非参数众数估计及连续臂 Bandit 基础文献**（7 篇），明确区分对经典数学工具的领域化应用与在 Test-Time Scaling 设定下建立的具体结论边界。

---

## 1. 相关文献全景（共 19 篇）

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

### 1.2 经典概率论、多元极值、Copula 尾相依、非参数众数与连续臂 Bandit 文献（7 篇）

| 编号 | 论文/专著与作者 | 发表期刊 / 会议 | 对本项目数学工具的直接支撑与理论对比范围 |
| :---: | :--- | :--- | :--- |
| **[13]** | Savage (1962) | *J. Res. Nat. Bur. Standards* 66B:93–96 | 建立多元标准正态象限尾概率的经典 Mills 比上界与二次规划条件。 |
| **[14]** | Hashorva & Hüsler (2003) | *Annals of the Institute of Statistical Mathematics (AISM)*, 55(3):507–522 | 系统推导多元高斯尾概率的 Active-Set 二次规划 $\min_{\mathbf{u}\ge\mathbf{1}}\mathbf{u}^\top\Sigma^{-1}\mathbf{u}$、本质正乘子支撑集 $I_+$ 与条件正态正象限退化情形下的精确渐近表示（定理 5A-II 直接应用该理论）。 |
| **[15]** | Garivier & Cappé (2011) | *COLT* (`PMLR 19:359–376`) | 系统分析二项 Kullback-Leibler 置信界 `KL-UCB`/`KL-LCB` 的非渐近 Cramér–Chernoff 反演收敛性质（定理 6B/6C 直接复用该置信界）。 |
| **[16]** | Tankov (2016) | *Journal of Multivariate Analysis (JMVA)*, 145:73–86 (`arXiv:1402.4683`) | 提出渐近独立随机向量（含 Gaussian Copula）的弱尾相依函数（weak tail dependence function），系统刻画了对数尺度下联合极端事件的指数衰减速率。 |
| **[17]** | Das & Fasen-Hartmann (2024) | *Journal of Multivariate Analysis (JMVA)*, 202:105310 (`arXiv:2204.06205`) | 系统分析 **Gaussian Copula 与正则变化重尾边际耦合** 下不同尾集上的联合极端风险衰减率与边际变换效应，明确揭示对数因子与剩余尾相依结构的耦合机理（定理 5B-II 直接衔接该类概率结构）。 |
| **[18]** | Romano (1988) | *The Annals of Statistics*, 16(2):629–647 | 建立非参数核密度众数（mode）位置估计的局部渐近 Minimax 下界与最优带宽理论（为定理 6A/6C 中局部极值非参数估计速率提供经典数理统计参照）。 |
| **[19]** | Bubeck, Munos, Stoltz, Szepesvári (2011) | *Journal of Machine Learning Research (JMLR)*, 12:1655–1695 | 建立局部光滑度量空间连续臂 Bandit（X-Armed Bandits / HOO）在主动序贯探索下的累积与简单遗憾（simple regret）Minimax 理论（对比定理 6C 的被动两阶段采样约束）。 |

---

## 2. 数学工具来源与技术边界评估

### 2.1 第一部分（预备定理 1–4）：单裁判基础统一表述（非原创性声明）

| 本项目定理 | 对应先发 TTS / 概率文献 | 本项目的定位与细化范围 |
| :--- | :--- | :--- |
| **定理 1**（含原子顺序统计量积分） | `[8]` Prop. 5、`[12]` Prop. 3.1/A.1 及经典顺序统计量 | 补全含平局原子分布下基于左连续广义逆的测度一致性与无相消 $-\operatorname{expm1}(\ln v/N)$ 双精度积分实现。 |
| **定理 2**（高斯坍塌、$C^1/C^0$ MLRP 与 $e^{-\gamma}$） | `[11]` Theorems 1–2、`[12]` Prop. 3.4/A.6 及经典 Karlin 总正性 | 拆分 $C^1$ 连通子区间与 $C^0$ 双闭子区间 MLRP 论证；给出固定峰值高度机制下五步极值矩与单峰 argmin 证明（`[1/1]` Padé 归附录工程拟合）。 |
| **定理 3**（二元缩放律与边界散度） | `[6]` 假阳性上限及基础 Bernoulli 信息论 | 补全按 $\beta>0$ 与 $\beta=0$（$\text{PPV}=1$，$\text{TV}$ 与有限反向 KL）分类的散度界与闭式预算公式。 |
| **定理 4**（校准后验与有界加权多数决） | `[5]` 加权多数决、`[7]` ICML 2025、Hoeffding/Bernstein 浓缩 | 约束单个高权错类期望 $p\mu_{w,+} > \sup_j(p_j\mu_{w,j})$，并用有界尾部超级桶处理可数无穷异质错误类。 |

### 2.2 第二部分（定理 5–7）：数学来源区分与 TTS 领域具体贡献

| 本项目模块 | 依托的既有数学工具 | 相对已有文献的 TTS 具体贡献（收敛范围与诚实界定） |
| :--- | :--- | :--- |
| **定理 5A-I**（广义特征值负权） | 广义 Rayleigh 商极值与线性判别分析 | 将正/负类协方差广义特征值比 $\lambda_{\min}(\Sigma_1^{-1/2}\Sigma_0\Sigma_1^{-1/2})<1$ 引入多裁判 BoN 坍塌可逆性判别，解释非负集成必崩时负权差值对消共模错噪的机制。 |
| **定理 5A-II**（Min-Ensemble QP） | `[14]` Hashorva & Hüsler (2003) 多元高斯极值 | 应用多元高斯正交象限 QP（含本质正乘子集 $I_+$、弱激活集 $I_0$ 条件象限常数 $\alpha_{c,I_0}$ 与密度前因子 $C^{\text{pdf}}_c=\gamma_c C^{\text{sf}}_c$）给出一般 SPD 相关高斯下 `Min-Ensemble` 的渐近相变三层级判据。 |
| **定理 5A-III**（Safe-Fisher & QDA） | 二次规划变量替换凸化（S-Lemma 类对偶）与局部椭球极值体积 | 针对渐近尾安全约束下的 **$N=2$ 二样本 SNR** 建立凸化一维光谱路径 $\mathbf{w}_\tau=(\Sigma_0-\tau\Sigma_1)^{-1}\Delta\boldsymbol{\mu}$（明确不主张对一般 $N>2$ 最优）；在不可逆相 $\Sigma_0\succ\Sigma_1$ 下计算多维 QDA 天花板及其 $N^{-2/K}$ 收敛率。 |
| **定理 5B**（重尾卷积与 Gaussian Copula） | 正则变化卷积、`[16]` Tankov (2016)、`[17]` Das & Fasen-Hartmann (2024) 与 Slepian 引理 | 将 Gaussian Copula + Student-$t$ 弱尾相依指数 $\eta_c=1/\gamma_c(R_c)$（$K\ge 2, R_c\succ 0$ 时属 $(0,1)$，非负相关充分得 $[1/K,1)$；按本质集基数 $|I_{+,c}|$ 给出 $(\ln t)^{(\gamma_c-|I_{+,c}|)/2}$）与同源污染击穿 ($\epsilon_s>0$) 映射到多裁判 BoN 相界 $\alpha_0/\eta_{0,\text{eff}}$ vs $\alpha_1/\eta_1$。 |
| **定理 6A–6C & 命题 6.1**（联合 $(M, N)$ Minimax） | `[15]` Garivier & Cappé (2011) KL-LCB、`[18]` Romano (1988)、`[19]` Bubeck et al. (2011)、Rényi–Pyke 顺序统计量间距 | 在一致正则峰类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 下建立离线被动校准 $M$ 与在线无标签抽样 $N$ 解耦的加性 Minimax 简单遗憾 $\Theta((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta})$，并用样本分割 Gamma 间距界将经验分位数分桶覆盖至一切 $\beta>0$。 |
| **定理 7A–7C & 命题 7.1**（层次异质性与样本标准化） | 混合分布全导数（Simpson 分解）、位置-尺度不变量、次高斯极值耦合与留出验证不等式 | 连接固定题条件模型（定理 1–6）与跨题层次模型 $X\sim\mathcal{P}_X$：给出显式有限参数二层辛普森遮蔽、$N=2$ 纯秩非严格界与序反转充要条件，并证明含 $\pi_0>0$ 时在线样本标准化 $Z_{i,N}$ 以 $O_p(\sqrt{(\ln N)/N})$ 一致逼近总体 $Z^*$ 从而继承渐近上限损失。 |

### 2.3 定理 6C 与经典非参数众数估计及连续臂简单遗憾的理论异同

定理 6A–6C 中的离线项 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$ 在代数指数上与经典非参数极值统计及连续空间优化一脉相承，但问题设定与耦合机制存在两处明确差异：

1. **自变量位置误差 vs. 回归函数值简单遗憾（对比 Romano 1988 `[18]`）**：经典核密度众数估计关注恢复密度峰值位置 $|\hat{x}_M - x^*|$，在二阶可微光滑度 $\beta=2$ 下位置收敛率为 $(M/\ln M)^{-1/5}$，而对应的**函数值遗憾** $f(x^*) - f(\hat{x}_M) \asymp |\hat{x}_M - x^*|^\beta$ 则具有指数 $\frac{\beta}{2\beta+1}$（$\beta=2$ 时为 $2/5$）。定理 6C 研究的是固定混合边缘密度 $f_{\text{mix}}$ 上的非参数 Bernoulli 回归最大化（且允许任意粗糙峰 $\beta \in (0, 1]$，此时位置导数不存在，但函数值遗憾 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$ 依然良定义）；同时当总体分位数未知时，需利用 Rényi–Pyke 顺序统计量乘性间距界取代在 $\beta \le 0.5$ 失效的加性 DKW 界。
2. **主动序贯查询 vs. 被动“离线校准 $M$ + 在线无标签候选池 $N$”双阶段解耦（对比 Bubeck et al. 2011 `[19]`）**：在 X-Armed Bandits 等主动连续臂优化中，算法可在高潜力峰值邻域自适应加密拉臂查询，简单遗憾可随查询预算指数缩减（在已知局部光滑参数时可达 $O(M^{-\beta/d})$ 或多项式更快收敛）。而在 Test-Time Scaling 场景下，离线校准集 $\mathcal{D}_M \sim P^{\otimes M}$ 来自**被动固定抽样**（无法主动向峰值区间增采有标签轨迹），使得离线项受全支撑覆盖瓶颈制约保持 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$；同时在线推理阶段无法自由输出任意实数点 $\hat{x}$，只能从生成器新抽出的有限 $N$ 个无标签离散点 $\{S_1,\dots,S_N\} \sim F_{\text{mix}}^{\otimes N}$ 中择一，从而额外引入即使在完美已知后验（$M=\infty$）下也不可消除的在线极值间距瓶颈 $\Theta(N^{-\beta})$。两者叠加构成了 TTS 特有的双阶段解耦率 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$。

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
```
