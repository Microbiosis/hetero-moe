# 检索层（L3）

L3 在这里**不改写数据**，只做"能查"。`slice_catalog.parquet` 把每个可用的切片和每个
(切片 × 区) 单元摊平成一行，带上能查询的字段，并且**带上这一行的适用性限制**。

为什么强调后者：这套数据里"某个字段存在"和"某个字段能用"差得很远。EEG 的 ρ 被白化、
人类颅内没有区标签、τ 在 65% 的切片上估不出来。若检索层只给数值不给限制，
查的人一定会误用。

---

## 怎么用

```python
import pandas as pd
C = pd.read_parquet("/domains/framework/slice_catalog.parquet")

# 例：小鼠 CA1 的区级记录
C.query("member == 'steinmetz' and level == 'region' and region == 'CA1'")

# 例：只取 rho_half 真的估出来的
C.query("rho_half_status == 'ok'")
```

**检索之后先读 `caveats` 列**，再决定能不能用。`n_caveats == 0` 的 427 行是可以直接用的。

---

## 规模

| 项 | 值 |
|---|---|
| 总行数 | 1,059（605 slice 级 + 454 region 级） |
| 唯一 `unit_id` | 1,059（无重复） |
| 成员 | 13 |
| `caveats` 条数分布 | 0 条 427 行 / 1 条 349 行 / 2 条 283 行 |
| 时长 >1000 s 的记录 | 69 |
| 有逐单元区标签（`has_unit_region`） | 565 行 |

## 关键字段

| 字段 | 含义 | 陷阱 |
|---|---|---|
| `level` | `slice` = 成员×切片；`region` = 成员×切片×区 | region 级才有 `n_units` |
| `has_unit_region` | 该成员是否有逐单元区标签 | 只有它为 `True` 才能做区级分析 |
| `rho_half_ms` | 跨单元共享相关半衰（不拟合曲线） | EEG 不适用；`rho_half_status != ok` 时只有界 |
| `tau_ms` | plateau 拟合的时间常数 | 仅 50/780 切片可用，对抽样敏感，**不要引用绝对值** |
| `tier_L4` | 区身份是否逐单元核验 | `region_unverified` 意味着区标签来自推断 |
| `caveats` | 该行的限制 | 检索后必读 |
| `pin_value` | 该成员来源的固定标识 | 见 `sources.csv` 的 `pin_status` |

## 全局限制（不挂在逐行上，但用的时候必须知道）

1. **PR（参与率）不能跨成员比较**：成员内与单元数的 Spearman 符号相反
   （counts_v4 −0.64、macaque −0.44、steinmetz +0.51），全局 +0.05 是辛普森悖论。
2. **EEG 的跨通道共同分量已被白化**：63 通道带符号相关均值 −0.0003…+0.019，ρ 不适用；
   时间结构仍在，但那是单通道 ACF，不是同一个量。
3. **人类颅内无区标签**：DANDI 000950/000954 的 NWB 没有 `/general/extracellular_ephys`，
   上游就没写电极表，不是我们没解析。
4. **没有质量分**：三条质量轴全部为负结果（可复现性 AUC 0.644 但只是 `rho1` 复读；
   小鼠跨 session 一致性 0.414；猕猴修正版 0.524）。所以 catalog 里没有 `quality` 列，
   不要指望拿它筛样本。
5. **区级离散度**：小鼠区级 ρ_half 的跨 session log-SD 中位 0.772（≈2.2×），
   `SCs`/`ILA`/`CA3`/`PL`/`MOs` 超过 1.4，按区引用绝对值没有意义。

## 示例查询与命中数

数字由 `build_slice_catalog.py` 实算并写入 `retrieval_index.json`，本表与之对账。

| 查询 | 命中 |
|---|---:|
| 小鼠 CA1（区级） | 21 |
| 猕猴 `rho_half_status == ok`（区级） | 139 |
| `has_unit_region == True` | 565 |
| `n_caveats == 0` | 427 |
| 时长 >1000 s（slice 级） | 69 |

## 重建

```bash
python /domains/scripts/build_slice_catalog.py   # 写 slice_catalog.parquet + retrieval_index.json
python /domains/scripts/verify_framework.py      # 含 K 节：catalog 与 JSON 对账
```

`verify_framework.py` 会检查：行数、`unit_id` 唯一、成员元数据无 `unknown`、
`retrieval_index.json` 里的示例命中数与 catalog 实算一致。最后一条是为了防止文档里的
数字悄悄过期。
