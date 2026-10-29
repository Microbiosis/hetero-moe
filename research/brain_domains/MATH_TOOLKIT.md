# 脑域线：数学方法谱系（神经科学里的数学 ↔ 生物信息学里的应用）

> **这份文档解决什么**：我们自写的算子（`arch/ops_brain.py`）、判据的地板、读出与统计口径，
> 哪些是**学界既有方法**（可直接引用、可直接用），哪些是**我们自己的改造**（必须自证），
> 哪些是**空白**（不该硬写）。
> ⇒ 它是 `COMPONENT_DESIGN.md` §3（算子表）的**对照面**：那张表说「我们做了什么」，
> 这份文档说「这件事在文献里叫什么、有没有现成工具、我们改了什么」。

---

## 0. 读法与纪律

1. **文献方法优先**：凡是既有方法能覆盖的，**不自写**；自写只在「既有方法不覆盖我们的使用场景」时允许
   （与 `COMPONENT_DESIGN.md` §3 的两条判据同源：符合使用场景 + 名字承诺的机制必须兑现）。
2. **引用分两档**：下面 §4 的 **A 档＝已核到原文条目**（带 DOI/PMID/链接，**27 条**；
   ✅ **2026-10-28 补齐 ⇒ 27 条全部带 DOI/PMID**（A21 补核时发现它是 **bioRxiv 预印本、没有卷期**
   ⇒ 原先「卷期待补」这个预期本身是错的；A22 补到卷期页码）），
   **B 档＝只核到邻域文献、原文条目未核** ⇒ **B 档不得当作已核实的引用写进论文**。
   ⇒ **目前 B 档为空**：原 B2 已被 A25 覆盖；原 B1 **查无此条 ⇒ 整条删除**（见 §4「否定结果」）。
   ⚠️ 核到的过程中纠正过两处**常被记错**的归属：**dPCA 的第一作者是 Kobak 不是 Cunningham**（A13）、
   **SVA 原文在 *PLoS Genetics* 不是 *PLoS Computational Biology***（A17）。
3. **一条事实只写一处**：本文件只写「数学方法 ↔ 我们的对应」，算子的实现细节仍在 `arch/ops_brain.py`，
   组件选型的留痕仍在 `COMPONENT_DESIGN.md` §3，判据正本仍在 `FINDINGS_TIER0_RSA.md` §四十四。

---

## 1. 家族总表

| # | 数学家族 | 神经学出处 | 生物信息学里的同一数学 | 我们项目里对应哪一处 | 状态 |
|---|---|---|---|---|---|
| F1 | **空间统计与保结构置换** | 脑图「对应问题」：空间自相关让朴素置换失效 ⇒ **spin test**（A1 + 预印本 A24）；置换检验 primer（A23）| 空间受限零模型（A25）、空间自相关感知重采样（A21）、脑图–基因表达对齐实践指南（A26）、空间置换综述（A27）| 各地板（跨被试×跨试次错配）、与同尺寸随机组比、组装级跨区隔离检查 | ⚠️ **思路对、零分布弱**：我们手工错配 ⇒ 下一轮（判据①，多区数据）应换成保结构置换 |
| F2 | **稀疏多元对齐 + 结构化惩罚** | imaging genetics：稀疏 CCA（A2）、图引导 fused lasso / GraphNet-SCCA（A3）、多任务结构化 SCCA（A4）| SNP–ROI 成对选择、eQTL 映射（A6）、microbiome 结构化 CCA（A5）、genomic sCCA 扩展（A7）| exp102「脑侧条件 ↔ CLIP 图像特征」对齐；位置轴的跨带比较；表征比较的方法论参照 RSA（A18）| ⚠️ **自写了加权/聚合、没借这一族** ⇒ 「按脑区分组」的先验正好对应 group lasso 的组惩罚 |
| F3 | **独立成分与潜在混杂** | ICA 成分的跨被试/跨试次一致性检验（A8）| SVA（正交化隐藏混杂，A17）；基因型分解 ICA | exp93b 的跨被试 200-way 地板（一致性检验的变体）；低秩算子的诚实边界（共模/参考/体积传导）| ⚠️ **概念已用、未做正交化去混杂** ⇒ 多区数据上 SVA 类工具对口 |
| F4 | **低维流形 / 群体活动** | population coding、低秩动力学、**dPCA**（A13）| 单细胞轨迹推断；群体级表达因子（PLEX/NMF）| `RegionLocalLowRank`（共同驱动）；exp102 的「adapter 输出 token 是否与概念相关」（跨概念一致性 0.9905）| ⚠️ 我们只做了 SVD + 重建误差，**不是 dPCA 口径** |
| F5 | **稀疏编码与预测** | V1 简单细胞的稀疏编码（A10/A9）、超完备字典 + 稀疏惩罚；ICA 是稀疏编码的特殊情形（A16）| lasso / group lasso / 图引导 fused lasso 在表达与 eQTL 上的应用 | `RegionLocalCenterSurround` 的半波整流 ⇒ 稀疏（量 L0 非零占比）| ⚠️ **我们的稀疏是整流先验，不是能量最小化** ⇒ 引用时要说清（A11 提醒：稀疏度度量受预处理与滤波器尺寸影响）|
| F6 | **预测编码 / 状态空间 / 因果** | 预测编码（A14）、最优控制（Todorov，A15）、Granger/DCM 有效连接 | GRN 的 ODE/状态空间；孟德尔随机化（基因组版因果推断）| `PredictiveInterface`（本轮排除）、`TaskContextGate`（沿时间指数衰减的循环状态 = 一阶状态空间）| ⚠️ `TaskContextGate` 的形式**就是**标准一阶状态空间模型 ⇒ 建议按标准形式重述 |
| F7 | **最优传输 / Wasserstein** | 脉冲串距离（EMD = Wasserstein，A19）；功能连接矩阵的 Wasserstein 比较 | 单细胞/多组学的 OT 对齐（SCOT+，A20）| 目前**没有**用：地板只比中位数（exp102 生成侧 2-way 中位 0.505 vs 地板 0.500）| ❌ **空缺** ⇒ 加一条分布层面对照，比只比中位强得多 |
| F8 | **图 / 网络** | connectomics、图匹配 | PPI 网络、代谢网络、GRN | 组内/组间结构图、位置轴的图匹配类比 | ✅ 概念已在用 |

---

## 2. 三条可操作结论

1. **能直接借的（别自写）**
   - 多元稀疏对齐 ⇒ **sCCA / S-MTL-SCCA / JSCCA**（A2–A7，有 MATLAB/Python 实现、已发表），
     且**我们的「按脑区分组」正好是 group lasso 的组结构**；
   - 多区数据上的零分布 ⇒ **保结构置换 / spin test**（A1，有开源实现）；
   - 隐藏混杂 ⇒ **SVA**（A17）；
   - 「群体活动」⇒ **dPCA**（A13；NMF 那一支仍待核）。
2. **我们的空缺**：地板目前只比**中位数** ⇒ 太钝。exp102 的生成侧 2-way 中位 0.505 vs 地板 0.500，
   两个数都贴着机会 ⇒ 建议补一条**分布层面**的对照（OT/Wasserstein，或逐被试分布检验）。
3. **我们自写、必须自证的部分**（`COMPONENT_DESIGN.md` §3 那张表是对的，缺的是对照面）：
   中心-周边的**群体相对版**、低秩只当诊断量、ON/OFF 排除、预测编码排除、dlPFC 维持门只由接口判据裁决。

---

## 3. 逐家族：我们改了什么（要点）

- **F1 空间统计**：文献的零分布是「把皮层当球面随机旋转」；我们的零分布是「手工错配概念/通道」。
  ⇒ 在**有空间结构**的数据上，我们的地板**偏乐观**（零分布太容易被打败）⇒ 判据① 的多区数据上必须换。
- **F2 稀疏对齐**：文献用**惩罚**（group lasso / 图引导）来选通道；我们用**读出 + 原型集聚合**，
  没有把「组」结构写进优化目标 ⇒ 这是我们与文献最实质的一处差距。
- **F3 混杂**：我们把 scalp 的共模/参考/体积传导当作「低秩算子的解释边界」写下来，
  但**没有做正交化**⇒ 在多区数据上，信号可能主要来自共模 ⇒ 必须先正交化再判形态。
- **F4 群体活动**：文献用 dPCA 分离**条件相关/条件无关**成分；我们用 SVD 看重建误差 ⇒ 只回答「能不能被少数成分张成」，
  不回答「成分是否携带行为相关信息」⇒ 后者才是判据① 要的。
- **F5 稀疏编码**：文献的稀疏是**能量最小化**的解；我们的稀疏是**整流**（先验注入）⇒
  ⚠️ 二者的稀疏度**不可直接比**（A11 明确指出稀疏度度量受预处理与滤波器尺寸影响）⇒ 引用时必须写清差别。
- **F6 状态空间/因果**：`TaskContextGate` 的 `s_t = γ·s_{t−1} + (1−γ)·x_t`、`y_t = x_t + g·s_t`
  形式上就是**一阶状态空间/低通模型** ⇒ 建议按标准形式重述并引用，而不是自写一个「衰减循环」。
- **F7 最优传输**：神经侧用脉冲串距离比较**两个响应的分布**，生信侧用 OT 做**批次校正**；
  我们目前只在 CLIP 空间比**均值/中位数** ⇒ 分布层面的对照是缺口。

---

## 4. 引用清单

### A 档：本轮核到原文条目（可写进论文/文档）

| 编号 | 条目 | 用途 |
|---|---|---|
| A1 | Alexander-Bloch, Shou, Liu, Satterthwaite, Glahn, Shinohara, Vandekar, Raznahan et al., **"On testing for spatial correspondence between maps of human brain structure and function"**, *NeuroImage* (2018) — doi **10.1016/j.neuroimage.2018.05.070**（spin test：把皮层当球面随机旋转来造保空间自相关的零分布；给出理论统计基础；开源实现 github.com/spin-test/spin-test）| F1 |
| A2 | Chi, Allen, Zhou, Kohannim, Lange, Thompson, **"Imaging genetics via sparse canonical correlation analysis"**, *ISBI* (2013) — doi **10.1109/ISBI.2013.6556581**（PMID 24443689）| F2 |
| A3 | Du, Huang, Yan, Kim, Risacher, Inlow, Moore, Saykin, Shen, **"Structured sparse canonical correlation analysis for brain imaging genetics: an improved GraphNet method"**, *Bioinformatics* 32(10):1544–51 (2016) — doi **10.1093/bioinformatics/btw033**（PMID 26801960）| F2 |
| A4 | Kim, Min, Liu, Yan, Saykin, Moore, Long, Shen, **"Multi-task learning based structured sparse canonical correlation analysis for brain imaging genetics"**, *Medical Image Analysis* 76:102297 (2022) — doi **10.1016/j.media.2021.102297**（PMID 34871929）| F2 |
| A5 | Chen, Liu et al., **"Structure-constrained sparse canonical correlation analysis with an application to microbiome data analysis"**, *Biostatistics* 14:244–258 (2013) | F2 |
| A6 | Chen & Liu, **"An efficient optimization algorithm for structured sparse CCA, with applications to eQTL mapping"**, *Statistical Biosciences* 4:3–26 (2012) | F2 |
| A7 | Witten & Tibshirani, **"Extensions of sparse canonical correlation analysis with applications to genomic data"**, *Statistical Applications in Genetics and Molecular Biology* 8(1):28 (2009) — doi **10.2202/1544-6115.1470**（PMID 19572827）| F2 |
| A8 | **"Testing Independent Component Patterns by Inter-Subject or Inter-Session Consistency"**, *Frontiers in Human Neuroscience* (2013) — doi **10.3389/fnhum.2013.00094** | F3 |
| A9 | Olshausen & Field, **"Emergence of simple-cell receptive field properties by learning a sparse code for natural images"**, *Nature* 381(6583):607–9 (1996) — doi **10.1038/381607a0**（PMID 8637596）| F5 |
| A10 | Olshausen & Field, **"Sparse coding of sensory inputs"**, *Vision Research* 37:3311–3325 (1997)（超完备基 + 稀疏惩罚）| F5 |
| A11 | Willmore, Watters & Tolhurst, **"A comparison of natural-image-based models of simple-cell coding"**, *Perception* 29(9):1017–40 (2000) — doi **10.1068/p2963**（PMID 11144817）⚠️ **明确指出：稀疏度与离散度度量受预处理、滤波器尺寸、归一化影响 ⇒ 跨实现不可直接比** | F5 |
| A12 | **"An improved and explicit surrogate variable analysis procedure by coefficient adjustment"**, *Biometrika* 104(2):303 (2017)；同族：**"Unifying and generalizing methods for removing unwanted variation based on negative controls"**（PMC10751021）| F3 |
| A13 | Kobak, Brendel, Constantinidis, Feierstein, Kepecs, Mainen, Qi, Romo, Uchida, Machens, **"Demixed principal component analysis of neural population data"**, *eLife* 3:e10989 (2014) — doi **10.7554/eLife.10989**（PMC4887222）⚠️ **第一作者是 Kobak，不是 Cunningham**（Cunningham & Yu 2014 是另一篇，被本文引用）| F4 |
| A14 | Rao, R.P.N. & Ballard, D.H., **"Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects"**, *Nature Neuroscience* 2(1):79–87 (1999) — doi **10.1038/4580**（PMID 10195184）| F6 |
| A15 | Todorov, E., **"Optimality principles in sensorimotor control"**, *Nature Reviews Neuroscience* 7(9):907–15 (2004) — doi **10.1038/nn1309**（PMID 15332089）| F6 |
| A16 | Bell, A.J. & Sejnowski, T.J., **"The 'independent components' of natural scenes are edge filters"**, *Vision Research* 37(23–24) (1997) — doi **10.1016/s0042-6989(97)00121-1**（1996 年 NeurIPS 版本题名为 "Edges are the Independent Components of Natural Scenes"）| F5 |
| A17 | Leek, J.T. & Storey, J.D., **"Capturing Heterogeneity in Gene Expression Studies by Surrogate Variable Analysis"**, *PLoS Genetics* 3(9):e161 (2007) — doi **10.1371/journal.pgen.0030161**（PMC1994707）⚠️ **是 PLoS Genetics，不是 PLoS Computational Biology** | F3 |
| A18 | Kriegeskorte, N., **"Representational similarity analysis – connecting the branches of systems neuroscience"**, *Frontiers in Systems Neuroscience* 2:4 (2008) — doi **10.3389/neuro.06.004.2008**（PMID 19104670；PMC2605405）| F2 |
| A19 | Sihn, D. & Kim, S.-P., **"A Spike Train Distance Robust to Firing Rate Changes Based on the Earth Mover's Distance"** — PMC6914768（PMID **31920607**；EMD 即 Wasserstein 度量）| F7 |
| A20 | Baker, Pham, Demetci, Tran, Redko, Sandstede, Singh, **"SCOT+: a comprehensive software suite for single-cell alignment using optimal transport"**, *Bioinformatics Advances* 6(1):vbaf314 (2026) — doi **10.1093/bioadv/vbaf314**（前身为 SCOT）| F7 |
| A21 | Khatri, P., Newton, M.A., Kendziorski, C. & Dinh, H.Q., **"Spatial Autocorrelation Aware Resampling Improves Cell-Cell Interaction Inference in Spatial Transcriptomics Data"**（SOAAR）— **bioRxiv 预印本**（posted 2026-07-10）— doi **10.64898/2026.07.06.736800**（PMC13370955；PMID 42465270）⚠️ **是预印本 ⇒ 没有卷期/页码**（2026-10-28 补核：Crossref 里 type = `posted-content`，**无期刊版**；与 A24 同属预印本档）（**空间自相关感知重采样**：构造保持基因级空间自相关的重采样数据做零分布 ⇒ 假阳性率低于随机置换）| F1 |
| A22 | Arnatkevičiūtė, A., Markello, R.D., Fulcher, B.D., Misic, B. & Fornito, A., **"Toward Best Practices for Imaging Transcriptomics of the Human Brain"**, *Biological Psychiatry* **93(5):391–404** (2023-03) — doi **10.1016/j.biopsych.2022.10.016**（明确建议区域分析用**空间受限零模型**）| F1 |
| A23 | Nichols, T.E. & Holmes, A.P., **"Nonparametric permutation tests for functional neuroimaging: a primer with examples"**, *Human Brain Mapping* 15(1):1–25 (2002) — doi **10.1002/hbm.1058**（PMID 11747097；Crossref 的 issued 记 2001 = 在线首发）| F1 |
| A24 | Alexander-Bloch, Vandekar, Shinohara, Liu, Satterthwaite, Glahn, Raznahan 等, **"The correspondence problem: which brain maps are significantly similar?"**, bioRxiv 预印本 (2017) — doi **10.1101/203083**（**A1 的预印本** ⇒ 同一工作的两版）| F1 |
| A25 | Fulcher, B.D., Arnatkevičiūtė, S. & Fornito, A., **"Overcoming false-positive gene-category enrichment in the analysis of spatially resolved transcriptomic brain atlas data"**, *Nature Communications* 12 (2021) — doi **10.1038/s41467-021-22862-1** ⇒ **这就是「空间受限零模型」的方法侧原文**（A22 那条 best-practice 建议的就是它）| F1 |
| A26 | Arnatkevičiūtė, S., Fulcher, B.D. & Fornito, A., **"A practical guide to linking brain-wide gene expression and neuroimaging data"**, *NeuroImage* 189:353–367 (2019) — doi **10.1016/j.neuroimage.2019.01.011**（预印本 bioRxiv doi 10.1101/380089）| F1/F2 |
| A27 | **"Decoding cell-cell communication in spatial transcriptomics: mechanistic insights, modeling constraints, and analytical caveats"**, *Briefings in Bioinformatics* 27:bbag477 (2026) — doi **10.1093/bib/bbag477**（PMID 42704267）⇒ 空间置换/零模型方法的**综述锚点** | F1 |

### 否定结果（**查过、确认查不到 ⇒ 不得再当作引用写**）

| 原拟条目 | 查了哪些源 | 结论 |
|---|---|---|
| Varoquaux et al. 2018, **"Rotate this to see the problem: A counter-screening method for sign-preserving rotations"**；同族 "Make it real: fMRI maps and the correspondence problem" | Crossref（精确题名 + 题名 + 作者三查）、Europe PMC/PubMed（题名查 + 作者年查）、Semantic Scholar、OpenAlex、arXiv API（题名查）| **五个源都查无此条** ⇒ ⇒ **是我记错了题名/条目本身** ⇒ **整条删除**，**不得**出现在文档或论文里。同一问题由 **A1 + A24**（spin test 及其预印本）与 **A23**（置换检验 primer）覆盖 ⇒ 引 A1 即可 |
| 空间转录组「空间置换检验」的原始方法出处 | Crossref / Europe PMC / Semantic Scholar | **已由 A25 覆盖**（Fulcher 等 2021 的空间受限零模型）⇒ B2 升级为 A25，不再单列 |

⚠️ **纪律**：查不到 ≠ 不存在，但**查不到就不能当引用用** ⇒ 宁可写「未核到原文」也不要写一个像真的条目。

---

## 5. 与其它文档的关系

| 文档 | 负责什么 |
|---|---|
| `arch/ops_brain.py` | 算子的**实现**与「踩过的坑」（每个算子的机制承诺）|
| `arch/test_ops_brain.py` | 算子的**性质测试**（退化性 / 关键性质 / 梯度可达 + 跨区隔离）|
| `COMPONENT_DESIGN.md` §3 | **选型**与留痕（本文件给它加了「可对应的文献方法」一列）|
| `MODEL_FORM.md` | 形态判定与目标架构 |
| `FINDINGS_TIER0_RSA.md` | 判据与全部结果（§四十四 判据、§四十八 exp102 结果）|
| 本文件 | **数学方法 ↔ 我们的对应**：哪些能借、哪些自写、哪些是空白 |

⚠️ 本文件**不改判据、不改结论**；它只回答「这套方法在文献里叫什么、有没有现成工具、我们改了什么」。
