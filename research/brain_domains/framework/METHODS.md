# 方法、结果与限制

数据集入口见根目录 `README.md`；检索用法见 `RETRIEVAL.md`。本文说明口径、主要结果和已知限制。

## 1. 构成

14 个成员、4 个上游数据集、全部 CC-BY-4.0。`framework_index.parquet` 逐对象登记 326 条
（2 阶 190、3 阶 128、4 阶 4、1 阶 4）。成员目录内可同时有窗口张量、载荷矩阵与图嵌入，
所以索引的行是**对象类**（一种 `key` × `shape` 组合），`n_files` 记文件数。

| 成员 | 物种 | 模态 | 规模 |
|---|---|---|---|
| `steinmetz` | *Mus musculus* | 胞外放电 | 39 session / 24,029 单元 / 50 ms |
| `primate_macaque_nlb` | *Macaca mulatta* | 胞外放电 | 72 片段 / 20 ms / PMd·M1 各 96 电极 |
| `primate_dmfc_rsg` | *Macaca mulatta* | 胞外放电 | 2 文件 / 94 单元 |
| `primate_human_eeg` | *Homo sapiens* | 头皮 EEG | 31 文件 / 63 通道 / 4 ms |
| `omnibrain4d_eeg_test` / `_train` | *Homo sapiens* | 头皮 EEG | 200×80×63×250 / 16540×4×63×250 |
| `omnibrain_spike_view` / `_macaque` / `_human` | 三物种 | 胞外放电 | 39 / 72 / 103 文件 |
| `omnibrain_eeg_view` | *Homo sapiens* | 头皮 EEG | 32,000 窗 |
| `omnibrain_graph_view` | *Drosophila* | 连接组 | 139,255 × 64 |
| `flywire` | *Drosophila* | 连接组 | 107,265 × 22 |
| `omnibrain_multi` | 混合 | 混合 | 1,409,850 × 64 × 8 |
| `counts_v4_ragged` | 混合 | 胞外放电 | 175 段 / 998,102,320 元素 |

## 2. 四层处理

| 层 | 产物 | 说明 |
|---|---|---|
| **L0 来源** | `sources.parquet` 12 条 | 每条带 `pin_kind/pin_value/pin_status`；当前 9 verified / 2 partial / 1 degraded。5 个 DANDI 固定到已发布版本号，MBAO/UBERON 固定到 commit，FALCON 代码固定到 commit + 2 个 blob |
| **L1 清洗** | `duplicate_map.parquet` 185 条 | 72 条经数值核对确认同一记录、103 条外部来源、3 条跨成员同窗、6 条派生、1 条同源异派生 |
| **L2 不变量** | `l2_invariants` 605 切片、`l2_decay` 388 行、`l2_region_invariants` 454 行 | 唯一允许跨成员比较的层 |
| **L3 检索** | `slice_catalog.parquet` 1,059 行 | 605 slice 级 + 454 区级，逐行带 `caveats` 与 `tier_L0…L4`；只做检索，不做语义改写 |

`counts_v4_ragged` 不进 L2（重复计数）。人类颅内（FALCON）NWB 里**没有电极表也没有 units 表**——
`preproc/h2_preproc.py` 全文无 `add_electrode` / `add_unit`，只写 20 ms 分箱的 `binned_spikes`，
所以那批“单元”实为矩阵列号。

## 3. 口径

- **主指标用 `rho_half`（半衰期，ms）而不是 τ**。exp63 的 τ 检验 p=0.386 只对 τ 成立，
  已在旧报告中更正。
- **区身份用 Allen MBA id，不用 UBERON**：UBERON 有 67.1% 是 collision，按缩写匹配会假阳性
  （`APN` 命中 accessory 而非 anterior pretectal nucleus）。
- **猕猴可逐 unit 拆分**：`electrodes/location` 是 96 × "Pre-Motor Cortex, dorsal" + 96 × "M1"，
  6/6 session 都是 unit 0–95 = PMd、96–191 = M1；`macaque_unit_region.parquet` 给了 13,126 个单元标签。
  但 `omnibrain_multi` 的窗是 64 维 PCA 投影，**窗级无法事后拆分**，那 168,048 个 `M1_PMd` 窗
  仍是 `compound`，不能当区级样本。
- **PR 不能跨成员比较**：成员间单元轴不共享、窗长与预处置不同，跨成员只能比 L2 不变量。

## 4. 区级结果

- **猕猴 PMd vs M1**（67 片段、同阵列、各 96 单元）：`rho_half` 中位 623.0 vs 496.4 ms = **1.20×**
  （CI 1.15–1.28），64/67 片段同向，符号检验 p=6.8e-16。但这是**电极位置**差异——上游论文写明
  “M1-like neurons are frequently found in caudal PMd and vice versa”，不能读成两种区类型。
- **小鼠 68 区 / 276 cell**：`rho_half` 中位从 90 ms（CA1）到 488 ms（MOs），约 5 倍；
  区效应 `R²(region|session) = 0.354`，session 内置换 p=0.0002（零分布中位 0.225，含过拟合成分）。
- **跨物种**（等样本量匹配）：小鼠 361.7 ms vs 猕猴 564.3 ms = **1.56×**（CI 1.18–2.01）。

## 5. L2 质量轴：四条路都没有立住

1. **可复现性模型**：分组 5 折 AUC **0.644**、置换 p<0.005。但 `quality_prob` 与 `rho1`
   秩相关 **+0.943**，单特征 `rho1` 就有 0.626；按 0.50 阈值过滤保留 41% 样本，
   `rho_half` 中位 285→483 ms（**1.69×**）。⇒ **分数是 `rho1` 的重表达，只能分层、不能过滤。**
2. **小鼠跨 session 一致性**（6 个与 `rho1` 不共线的特征）：按区分组 AUC **0.414**、p=0.89；
   标签分半可靠度仅 **0.497**，完美特征上限 0.723 ⇒ 强效应可排除、弱效应排除不了。
3. **猕猴**：第一版 AUC **0.761**、p<0.005，但 67 个片段来自 **6 个 session**，片段序号单独
   就有 AUC 0.810；改同 session 内参照后 **0.524**、p=0.42，session 解释 `log rho_half`
   方差 **59.3%**。
4. **原始 spike times / ISI**：M1 的 ISI 违反率是 PMd 的 **1.44×**，速率校正后 +0.0079→+0.0078
   （缩小 2.1%）⇒ 不是放电率假象。同时常用阈值“>2% 即污染”在这批数据上失效：27–29 Hz 下
   泊松期望本身就有 5.5%，按 2% 判会把 46% 单元误判。

结论：**有模型，但只能分层不能过滤**；瓶颈是标签可靠度（0.5 量级），不是模型容量。

## 6. 限制

1. 神经元数按 session 变化（326–1,240），跨 session 建模需对齐策略。
2. 域在 session 间分布不均（每 session 2–9 个域），LODO 需先取跨 session 交集。
3. Steinmetz 为 50 ms 分箱，不适合毫秒级动态。
4. FlyWire 无时间维度。
5. 人类颅内无电极表、无解剖定位（区间标签记作 `unknown`）；猕猴“单元”可能是电极级群体信号。
6. FlyWire 的区标签与连接组特征同源，用连接组预测脑区接近同义反复；`node_features` 已分
   safe（6 列）与 leaky（12 列）两组，结论必须用 safe 组 + 随机标签对照得出。

## 7. 复核

```
python <workspace>/scripts/verify_framework.py     # 在工作区运行，失败返回非零
```

校验只做一件事：**声称的与文件里的是否一致**——索引登记的行数/字节数与实文件比对、
目录下每个文件都在索引里、来源 pin 合法、区级注册表唯一键、检索层示例命中数与实算一致。
结果数字本身由工作区的生成脚本决定，本仓库只提供数据与文档。
