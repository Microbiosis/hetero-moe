# 区域映射

`omnibrain-multi` 的 `region` 列原先混了四类不同的东西。这份文件记录它们分别是什么、映射到哪个本体、以及映射之后还剩几个可以跨物种比较的节点。

## 1. 原先的 `region` 列不是一列

| 来源 | 原值 | 实际是 | 行数 |
|---|---|---|---:|
| mouse_steinmetz | `MOp`, `VISpm`, `ORB` … | Allen 小鼠 CCF 结构缩写 | 314,332 |
| macaque_nlb | `M1_PMd`, `Area2`, `DMFC` | NLB 记录区代码，其中 `M1_PMd` 是复合 | 187,067 |
| human_intracortical | `h1_intracortical`, `h2_intracortical` | **FALCON 任务编号**，不是解剖 | 86,851 |
| human_eeg | 空 | 63 导头皮 EEG 无焦点位置 | 821,600 |
| flywire | `ME`, `Kenyon_Cell`, `CX` … | 神经毡名与细胞类型名混在一起 | 139,255 |

`h1`/`h2` 是 FALCON 的两个任务（H1 七自由度抓取、H2 手写），两套 Utah array 都植入手部运动皮层，解剖上都是 M1。把它们当两个脑区用，等于把任务差异算成了区域差异。

## 2. 新增列

| 列 | 含义 |
|---|---|
| `region_raw` | 上游原值，保留不动 |
| `region` | 统一后的标签：仅把两个任务编号改为 `M1`，其余保持原值 |
| `region_kind` | `anatomy` / `neuropil` / `cell_type` / `none` |
| `structure_id` | Allen `MBA:` 或 FlyBase `FBbt:` 标识 |
| `uberon_id` | UBERON 标识；复合标签用 `\|` 并列 |
| `shared_node` | 跨物种节点：`M1` / `S1` / `PFC`，其余为空 |
| `mapping_status` | 见下表 |

| `mapping_status` | 行数 | 含义 |
|---|---:|---|
| `exact_single_species` | 257,790 | 一对一映射到 UBERON，但该节点只有一个物种 |
| `compound` | 168,048 | 上游标签跨两个结构，无法在行级拆分 |
| `no_vertebrate_homolog` | 107,338 | 果蝇神经毡与细胞类型，无脊椎动物同源节点 |
| `task_code_reinterpreted` | 86,851 | 原值是任务编号，已按解剖位置改判 |
| `unassigned` | 853,452 | 无区域信息（头皮 EEG 821,600 + 果蝇未注释单元 31,852） |
| `uberon_missing` | 23,823 | 在 Allen 表中有结构，但 UBERON bridge 未覆盖 |
| `coarse` | 19,019 | 上游比节点更粗，只能归到上位 |
| `rollup` | 16,520 | 上游比节点更细，向上归并 |
| `exact` | 16,199 | 与节点一一对应 |
| `fbbt_unresolved` | 65 | 果蝇区域名未在 FBbt 检索到 |

## 3. 映射来源

- 小鼠：Allen CCF 结构表（`api.brain-map.org`，13,709 行）→ `uberon-bridge-to-mba.owl`（1,048 条 `MBA → UBERON` 等价类）。
  **必须限定到成年 CCF（`structure_id_path` 以 `/997/` 开头，1,327 个结构）**：缩写在整张表里不唯一，`PL` 会撞上 paralaminar nucleus、`RSP` 撞上 rostral secondary prosencephalon、`SSp` 撞上 supraspinal nucleus。
- 猕猴：手工指定。`Area2` → Brodmann 2 区（S1 的亚区，故记 `coarse`）；`DMFC` → 前额叶（同上）；`M1_PMd` → M1 与 PMd 并列。
- 人：`h1`/`h2` → `UBERON:0001384`（M1）。
- 果蝇：FBbt。神经毡与细胞类型分开标注；昆虫脑区在 UBERON 中没有同源节点，`uberon_id` 留空。

全部 26 个用到的 UBERON 标识经 OLS4 核对，标签一致且均未废弃（见 `uberon_check.json`）。

## 4. 并起来之后

65 个原始标签 → 27 个 UBERON 标识（小鼠 29 个缩写里 `ORB`/`ORBm` 同为 `UBERON:0004167`，`LS`/`LSr` 同为 `UBERON:0002667`；`SCig`、`APN` 无 bridge，故小鼠贡献 25 个；猕猴的 `PMd` 与 `PFC` 再加 2 个）。

跨物种共享的只有三个：

| 节点 | UBERON | 小鼠 | 猕猴 | 人 | 合计 | 物种数 |
|---|---|---:|---:|---:|---:|---:|
| 初级运动皮层 | `UBERON:0001384` | 9,298 | 168,048 | 86,851 | 264,197 | 3 |
| 前额叶 | `UBERON:0000451` | 16,520 | 13,292 | 0 | 29,812 | 2 |
| 初级体感皮层 | `UBERON:0008933` | 6,901 | 5,727 | 0 | 12,628 | 2 |

M1 占全部窗口的 18.7%，是唯一覆盖三个物种的节点。这不是构造时选出来的，是这五个来源本来的重叠处。

## 5. 文件

| 文件 | 内容 |
|---|---|
| `region_map.csv` | 67 行，每个 `(source, region_raw)` 一行 |
| `shared_nodes.csv` | 3 行，节点级计数 |
| `region_groups.npz` | `M1`/`S1`/`PFC` 三个 int32 数组，值为 `index_timeseries` 的行号 |
| `index_timeseries.parquet` | 1,409,850 行，19 列 |
| `index_graph.parquet` | 139,255 行，16 列 |
| `build_region_map.py` | 生成上述全部文件 |
| `mba_struct.csv` / `mba2uberon.json` | 中间件，可重下 |
| `uberon_check.json` | OLS4 核对结果 |

`timeseries.npy` 与 `graph.npy` 不受影响：行数与行序未变，只改了索引表。

## 6. 仍未解决

- `M1_PMd` 的 168,048 行无法拆成 M1 与 PMd。若下游需要纯 M1，只能整段丢弃或当作混合处理。
- `SCig`（上丘中间灰质层）与 `APN`（前顶盖核）缺 UBERON 映射，共 23,823 行。
- 猕猴只有三个记录区，任何「跨物种前额叶」结论实际只基于 `DMFC` 一块。
- 果蝇那 139,255 个节点与上述三个节点没有同源关系，只能作为独立分支。
