# 文献定位、经典数学工具溯源与技术边界（Related Work & Novelty Audit）

本文档梳理截至 2026 年 11 月与本项目直接相关的两类文献：（1）**大模型 Test-Time Scaling、Reward Model Overoptimization 与 Best-of-$N$ 校准文献**（12 篇）；（2）本项目证明与统计推断所依托及对比的**经典概率论、顺序统计量间距、多元极值、Gaussian Copula 弱尾相依、被动 vs 自适应非参数回归极值估计、离线悲观主义及统计检验方法学文献**（15 篇，总计 27 篇）。全部文献的发表标题、作者、期刊/会议卷页码、arXiv ID 与原始统计设定均经逐篇核验，严格区分**被动单阶段抽样**、**两阶段/选择性主动查询**与本文 **Test-Time Scaling 联合 $(M, N)$ 被动校准—随机无标签候选池选择**三类不同问题设定。

---

## 1. 相关文献全景（共 27 篇）

### 1.1 LLM Test-Time Scaling 与 BoN 理论文献（12 篇）

| 编号 | 论文作者与年份 | 核验后准确标题与发表信息 / arXiv ID | 核心研究问题与主要结论 |
| :---: | :--- | :--- | :--- |
| **[1]** | Gao, Schulman, Hilton (2023) | *Scaling Laws for Reward Model Overoptimization*, ICML 2023 (`arXiv:2210.10760`) | 在合成金标 RM 下拟合 RLHF 与 BoN 的 Goodhart 过优化经验缩放律。 |
| **[2]** | Coste, Anwar, Kirk, Krueger (2024) | *Reward Model Ensembles Help Mitigate Overoptimization*, ICLR 2024 (`arXiv:2310.02743`) | 实验验证多个 RM 的最坏情况集成（WCO / `min`）与不确定性惩罚可缓解过优化，未给出多裁判相变解析判据。 |
| **[3]** | Ramé et al. (2024) | *WARM: On the Benefits of Weight Averaged Reward Models*, ICML 2024 (`arXiv:2401.12187`) | 提出权重平均奖励模型（WARM）提升多 RM 在分布偏移下的稳健性。 |
| **[4]** | Brown et al. (2024) | *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, `arXiv:2407.21787` | 揭示重复采样覆盖率（Pass@$N$）对数线性增长与 Verifier 筛选瓶颈。 |
| **[5]** | Snell, Lee, Xu, Kumar (2024) | *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, `arXiv:2408.03314` | 研究过程验证器搜索与按问题难度分配测试时算力的策略。 |
| **[6]** | Stroebl, Kapoor, Narayanan (2024) | *The Limits of Inference Scaling Through Resampling*, `arXiv:2411.17501`（初版副标题含 *Inference Scaling fLaws*） | 分析二元不完美 Verifier（如有限覆盖单测）非零假阳性率对重采样上限的制约。 |
| **[7]** | Huang et al. (2025) | *Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment*, ICML 2025 (`arXiv:2503.21878`, PMLR) | 以覆盖率刻画 BoN 在不完美 RM 下的退化，提出悲观拒绝采样 `InferenceTimePessimism` 保证计算单调性。 |
| **[8]** | Dorner, Chen, Cruz, Yang (2025) | *ROC-n-reroll: How verifier imperfection affects test-time scaling*, `arXiv:2507.12399` | 证明实例级 BoN 与拒绝采样曲线受 Verifier ROC 几何支配，指出低算力表现无法外推高算力行为。 |
| **[9]** | Tang, Chen, Cavallaro (2025) | *CarBoN: Calibrated Best-of-N Sampling Improves Test-time Reasoning*, `arXiv:2510.15674` | 提出两阶段推理采样与输入自适应温度/偏移校准框架 CarBoN。 |
| **[10]** | Schweisthal, Wang, Deviyani, Feuerriegel, Frauen (2026) | *Efficient Best-of-N Policy Evaluation for Inference-Time Alignment*, `arXiv:2610.09250` | 提出无需生成似然的纯样本 Best-of-$N$ 策略评估与选择框架。 |
| **[11]** | Yano, Ishibashi, Oyamada (2026) | *Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems*, NeurIPS 2026 (`arXiv:2610.19362`) | 基于单变量极值理论（EVT）分析单裁判正确/错误上尾竞争与高斯异方差坍塌。 |
| **[12]** | Gui, Shao, Li, Guo, Wang (2026) | *Whether or Where? Tail Calibration Signals Finite-Budget Reward Hacking in Best-of-N Selection*, `arXiv:2610.34375` | 给出 Beta 分位积分表示、对数凹单调判据及基于尾部校准信号的有限预算早停审计。 |

### 1.2 经典概率统计、多元极值、非参数极值估计、离线悲观与统计推断文献（15 篇）

| 编号 | 作者与年份 | 核验后准确标题与发表期刊 / 会议 | 原始统计设定及对本项目的准确理论关系 |
| :---: | :--- | :--- | :--- |
| **[13]** | Savage (1962) | *Mills' ratio for multivariate normal distributions*, *J. Res. Nat. Bur. Standards* 66B(3):93–96 | 建立多元正态象限尾概率的经典 Mills 比上界。 |
| **[14]** | Pyke (1965) | *Spacings*, *Journal of the Royal Statistical Society: Series B (Methodological)*, 27(3):395–449 | 建立 Uniform 顺序统计量间距的指数/Gamma 分布等价表示（**命题 6.1 直接应用该经典表示**）。 |
| **[15]** | Müller (1985) | *Kernel estimators of zeros and of location and size of extrema of regression functions*, *Scandinavian Journal of Statistics*, 12(3):221–232 | **被动固定/非自适应设计**下非参数回归函数极值位置（location）与极值大小（size）核估计的奠基文献。 |
| **[16]** | Romano (1988) | *On Weak Convergence and Optimality of Kernel Density Estimates of the Mode*, *The Annals of Statistics*, 16(2):629–647 | 建立**被动 i.i.d. 抽样**下非参数核密度众数（mode）位置估计的渐近分布与局部 Minimax 下界。 |
| **[17]** | Hashorva & Hüsler (2003) | *On multivariate Gaussian tails*, *Annals of the Institute of Statistical Mathematics*, 55(3):507–522 | 推导多元高斯正交象限二次规划、本质正乘子集 $I_+$ 与零乘子弱激活条件正态正象限渐近式（**定理 5A-II 直接应用**）。 |
| **[18]** | Nadeau & Bengio (2003) | *Inference for the Generalization Error*, *Machine Learning*, 52(3):239–281 | 证明 K 折交叉验证因训练折样本重叠破坏独立性并低估泛化方差（支撑本文采用互不相交单次留出划分作为主检验）。 |
| **[19]** | Tsybakov (2009) | *Introduction to Nonparametric Estimation*, Springer Series in Statistics | 系统建立**被动抽样**下非参数回归/密度估计的局部光滑鼓包假设族与广义 Fano / Assouad Minimax 下界方法（**定理 6A/6C 离线 Fano 构造遵循该范式**）。 |
| **[20]** | Phipson & Smyth (2010) | *Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn*, *Stat. Appl. Genet. Mol. Biol.*, 9(1):Art. 39 | 证明蒙特卡洛置换检验须采用 $(b+1)/(B+1)$ 修正以避免错误的零 $p$ 值并保持第一类错误控制。 |
| **[21]** | Garivier & Cappé (2011) | *The KL-UCB algorithm for bounded stochastic bandits and beyond*, *COLT* (`PMLR 19:359–376`) | 建立二项分布 Kullback–Leibler 置信界反演的非渐近 Cramér–Chernoff 覆盖界（**定理 6B/6C 直接复用 `KL-LCB`**）。 |
| **[22]** | Bubeck, Munos, Stoltz, Szepesvári (2011) | *X-Armed Bandits*, *Journal of Machine Learning Research*, 12:1655–1695 | 研究局部光滑度量空间上的**序贯主动拉臂查询**（可向峰值邻域自适应加密采样，与本文被动校准集形成主动 vs 被动对照）。 |
| **[23]** | Belitser, Ghosal, van Zanten (2012) | *Optimal two-stage procedures for estimating location and size of the maximum of a multivariate regression function*, *The Annals of Statistics*, 40(6):2850–2876 | 对比单阶段全局非参数平滑的低速维度瓶颈 $n^{-\alpha/(2\alpha+d)}$ 与**两阶段自适应局部加密设计（zooming-in）**的加速效应（对照说明：本文离线校准集受限于单阶段被动生成分布，无法执行第二阶段主动选点加密）。 |
| **[24]** | Minsker (2013) | *Estimation of Extreme Values and Associated Level Sets of a Regression Function via Selective Sampling*, *COLT* (`PMLR 30:105–121`) | 研究在随机无标签流上通过**选择性主动标注查询（selective sampling）**估计非参数回归极值与水平集（对照说明：该文核心机制为主动选择性查询以改进被动随机设计基准，而本文 $\mathcal{D}_M$ 为纯被动 i.i.d. 历史标注样本）。 |
| **[25]** | Tankov (2016) | *Tails of weakly dependent random vectors*, *Journal of Multivariate Analysis*, 145:73–86 | 建立渐近独立随机向量（含 Gaussian Copula）的弱尾相依函数与对数尺度指数率。 |
| **[26]** | Jin, Yang, Wang (2021) | *Is Pessimism Provably Efficient for Offline RL?*, *ICML* (`PMLR 139:5084–5096`) | 证明在**离线被动数据集**覆盖不均时，基于置信下界（LCB）的悲观原则可消除稀疏动作的赢家诅咒高估并达到离线 Minimax 最优。 |
| **[27]** | Das & Fasen-Hartmann (2024) | *On heavy-tailed risks under Gaussian copula: The effects of marginal transformation*, *Journal of Multivariate Analysis*, 202:105310 | 分析 Gaussian Copula 与正则变化重尾边际耦合下的联合极端衰减率与对数修正因子（**定理 5B-II 直接衔接**）。 |

---

## 2. 原创性归属与问题设定严格区分（Passive vs. Adaptive vs. Two-Phase TTS）

### 2.1 预备定理 1–4（附录 A）：既有单裁判结果整理（不主张原创归属）

预备定理 1–4 对既有单裁判 TTS 文献（`[6, 7, 8, 11, 12]`）及经典顺序统计量/浓缩不等式进行统一测度化表述与边界补全（含离散原子广义逆、$C^1/C^0$ 分离 MLRP、高斯 $\kappa<1,\Delta\mu=0$ 初值平局边界、固定峰高 $e^{-\gamma}N^*_0$ 证明、$\text{PPV}=1$ 反向 KL 分离及单错类权重约束），在正文中归入附录 A，不作为本文核心原创贡献主张。

### 2.2 定理 6C & 命题 6.1 的精确学术定位：三类非参数极值优化设定的对照

审稿过程中最关键的理论定位问题是：**定理 6C 的联合简单遗憾率 $\mathcal{R}^*(M, N) \asymp (\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}$ 与统计学已有的非参数回归最大化文献究竟有何本质同异？** 我们从数据采集机制与决策空间两个维度给出严格区分：

| 对比维度 | 经典被动非参数极值估计（Müller 1985 `[15]`; Romano 1988 `[16]`; Tsybakov 2009 `[19]`） | 自适应 / 选择性查询回归最大化（Bubeck et al. 2011 `[22]`; Belitser et al. 2012 `[23]`; Minsker 2013 `[24]`） | 本文 Test-Time Scaling 双阶段模型（定理 6C & 命题 6.1） |
| :--- | :--- | :--- | :--- |
| **离线有标签数据采集方式** | **被动单阶段抽样**（固定设计或 i.i.d. 随机设计 $\mathcal{D}_M \sim P^{\otimes M}$） | **主动自适应设计 / 选择性查询**（第二阶段向初估峰值邻域加密采样 `[23]`，或仅对高潜力水平集请求标签 `[24]`） | **被动单阶段抽样** $\mathcal{D}_M \sim P^{\otimes M}$（由固定生成器与裁判离线产生，无法指定任意分数 $s$ 强行采样查询） |
| **在线测试期可选动作集** | 整个连续定义域（直接输出实数点估计 $\hat{x}_M \in \mathbb{R}$） | 整个连续定义域（输出实数点估计 $\hat{x}_M$ 或水平集） | **仅限 $N$ 个新抽随机无标签候选点** $\{S_1,\dots,S_N\} \stackrel{\text{i.i.d.}}{\sim} F_{\text{mix}}$（离散随机可行域） |
| **离线标注误差阶（$d=1$）** | 全局平滑受制于全支撑覆盖：$(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$ | 主动聚焦消除无关区域查询损耗，可突破被动全局率（如两阶段加速 `[23]` 或多项式更快收敛 `[22, 24]`） | 因离线数据纯被动且边际密度未知，受被动 Fano 下界锁定为 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}\big)$ |
| **在线有限算力离散瓶颈** | 无（等价于 $N = \infty$） | 无（等价于 $N = \infty$） | **新增不可约极值间距项 $\Theta(N^{-\beta})$**（即使 $M=\infty$ 完美已知 $q$ 亦无法消除） |

由此可见，本文定理 6C 与命题 6.1 的真实归属边界如下：
1. **诚实归属既有统计原理的部分**：（a）在 $N=\infty$ 极限下，被动单阶段数据无法摆脱全局平滑瓶颈 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}$ 的信息论原因来自经典被动非参数极值估计与局部鼓包 Fano 方法（Müller 1985 `[15]`；Romano 1988 `[16]`；Tsybakov 2009 `[19]`；亦对应 Belitser et al. 2012 `[23]` 所指出的第一阶段全局平滑瓶颈，而非其第二阶段主动加密率或 Minsker 2013 `[24]` 的选择性主动采样率）；（b）用置信下界抑制稀疏桶过拟合直接继承离线悲观原则（Jin, Yang & Wang 2021 `[26]`；Garivier & Cappé 2011 `[21]`；Huang et al. 2025 `[7]`）；（c）顺序统计量间距的 Gamma 表示直接引用 Pyke (1965) `[14]`。
2. **本文在 Test-Time Scaling 设定下建立的具体增量结果**：
   - **被动离线校准 $\mathcal{D}_M$ 与在线有限无标签随机候选池 $S_{1:N}$ 的无 $(\ln N)^\beta$ 损失耦合**：证明在单一公共正则类 $\mathcal{F}_\beta^{\text{unif}}(C_1, f_{\min}, \eta_0)$ 上，**不依赖在线预算 $N$ 的静态分桶 $K^*(M) \asymp (M/\ln M)^{\frac{1}{2\beta+1}}$** 配合 `KL-LCB` 同时在全平面 $(M, N)$ 达到联合上下界 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$，并导出指导工程标注与推理算力配比的**校准—算力预算平衡律 $N_{\text{balance}}(M) \asymp (M/\ln M)^{\frac{1}{2\beta+1}}$**；
   - **粗糙峰 $0 < \beta \le \frac{1}{2}$ 下纯 $M$-预算样本分割经验分位数浓缩（命题 6.1）**：针对未知混合分布 $F_{\text{mix}}$ 且无额外无标签数据场景，指明加性 DKW 界在 $\beta \le \frac{1}{2}$ 处的相对误差发散，并用样本分割 Pyke Gamma 乘性间距界将同一联合 Minimax 速率严格推广至一切 $\beta > 0$。

---

## 3. 完整核验 BibTeX 参考文献库（27 篇）

```bibtex
@inproceedings{gao2023scaling,
  title={Scaling Laws for Reward Model Overoptimization},
  author={Gao, Leo and Schulman, John and Hilton, Jacob},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series={Proceedings of Machine Learning Research},
  volume={202},
  pages={10835--10866},
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

@inproceedings{rame2024warm,
  title={WARM: On the Benefits of Weight Averaged Reward Models},
  author={Ram{\'e}, Alexandre and Vieillard, Nino and Hussenot, L{\'e}onard and Dadashi, Robert and Cideron, Geoffrey and Bachem, Olivier and Ferret, Johan},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={Proceedings of Machine Learning Research},
  volume={235},
  year={2024},
  eprint={2401.12187},
  archivePrefix={arXiv}
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
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
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
  title={Efficient Best-of-N Policy Evaluation for Inference-Time Alignment},
  author={Schweisthal, Jonas and Wang, Yuxin and Deviyani, Athiya and Feuerriegel, Stefan and Frauen, Dennis},
  journal={arXiv preprint arXiv:2610.09250},
  year={2026}
}

@inproceedings{yano2026tail,
  title={Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems},
  author={Yano, Taro and Ishibashi, Yoichi and Oyamada, Masafumi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  eprint={2610.19362},
  archivePrefix={arXiv}
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

@article{pyke1965spacings,
  title={Spacings},
  author={Pyke, Ronald},
  journal={Journal of the Royal Statistical Society: Series B (Methodological)},
  volume={27},
  number={3},
  pages={395--449},
  year={1965}
}

@article{muller1985kernel,
  title={Kernel estimators of zeros and of location and size of extrema of regression functions},
  author={M{\"u}ller, Hans-Georg},
  journal={Scandinavian Journal of Statistics},
  volume={12},
  number={3},
  pages={221--232},
  year={1985}
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

@article{hashorva2003multivariate,
  title={On multivariate Gaussian tails},
  author={Hashorva, Enkelejd and H{\"u}sler, J{\"u}rg},
  journal={Annals of the Institute of Statistical Mathematics},
  volume={55},
  number={3},
  pages={507--522},
  year={2003}
}

@article{nadeau2003inference,
  title={Inference for the Generalization Error},
  author={Nadeau, Claude and Bengio, Yoshua},
  journal={Machine Learning},
  volume={52},
  number={3},
  pages={239--281},
  year={2003},
  doi={10.1023/A:1024068626366}
}

@book{tsybakov2009introduction,
  title={Introduction to Nonparametric Estimation},
  author={Tsybakov, Alexandre B.},
  series={Springer Series in Statistics},
  publisher={Springer},
  address={New York},
  year={2009},
  doi={10.1007/b13794}
}

@article{phipson2010permutation,
  title={Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn},
  author={Phipson, Belinda and Smyth, Gordon K.},
  journal={Statistical Applications in Genetics and Molecular Biology},
  volume={9},
  number={1},
  pages={Article 39},
  year={2010},
  doi={10.2202/1544-6115.1585}
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

@article{bubeck2011xarmed,
  title={X-Armed Bandits},
  author={Bubeck, S{\'e}bastien and Munos, R{\'e}mi and Stoltz, Gilles and Szepesv{\'a}ri, Csaba},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={1655--1695},
  year={2011}
}

@article{belitser2012optimal,
  title={Optimal two-stage procedures for estimating location and size of the maximum of a multivariate regression function},
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

@article{tankov2016tails,
  title={Tails of weakly dependent random vectors},
  author={Tankov, Peter},
  journal={Journal of Multivariate Analysis},
  volume={145},
  pages={73--86},
  year={2016},
  doi={10.1016/j.jmva.2015.12.008}
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

@article{das2024heavy,
  title={On heavy-tailed risks under Gaussian copula: The effects of marginal transformation},
  author={Das, Bikramjit and Fasen-Hartmann, Vicky},
  journal={Journal of Multivariate Analysis},
  volume={202},
  pages={105310},
  year={2024},
  doi={10.1016/j.jmva.2024.105310}
}
```
