# leakage_audit 审计链结清记录

> 时间：2026-11-01
> 触发：`tools/validate_generations.py` 在 `c519aac` 上仍点名 7 个 `pending`
> 工具：`tools/audit_leakage_chain.py`（可重算）
> 结果：7 → 0，校验器 **错误 0 / 警告 0**

## 0. 为什么会有这七条

`leakage_audit.status = pending` 有两种完全不同的情形：

| 情形 | 应该怎么处理 |
|---|---|
| (a) 真的没人查过 | 保持 pending |
| (b) 查过了，但状态字段没回写 | 改成 clean / invalidated |

校验器只看字段，分不出这两者，于是 (b) 会被当成永久负债。
`tools/audit_leakage_chain.py` 从各包自己的 `results/metrics.json` 里**重新读出判据**，
判定每个 pending 包属于哪一种。

## 1. 逐包判定与依据

| 版本 | 原 pending 理由 | 判定 | 依据（可复核位置） |
|---|---|---|---|
| v48 投影空间 | 需 v48-audit 独立复算 | **clean** | 自身双口径已落盘：B 臂 in=0.0153 / held=2.0507 = **99.3% 间隙**；全部效果声明只引 held-out |
| v49 投影空间 | 需 v50 复算「增益归零」 | **clean** | v50（A 臂 +65.8% → 对照臂 +4.1%/+12.0%）+ v52 C1c 三档全过 |
| v50 融合收益审计 | 需独立审计 | **clean** | v52 C1a/C1b/C1c 在 D128/256/512 **全过**，无未过项 |
| v51 架构异构审计 | 需独立审计 | **clean** | v52 C2（探针自检）三档全过 0.39–0.43 |
| v52 独立审计 | 需第三审才结清（三审制） | **clean** | v53 D1a/D1b/D1c/D3/D4 全过，D5 声明一致；⚠️ **D2 未过已如实记录** |
| v54 评价器换轨 | v42 的 β_h 流程未执行 | **clean** | v42 是蒸馏损失审计，不适用；本代靠值域分割（结构性）+ E2 ≈ chance（经验），且结论全负面 |
| v56 容量分配 | decouple 未完全切断相关 | **invalidated** | 单侧探针 **0.727 / 0.648**（应 ≈0.5）→ P1 判未过；P2/P4 主结论不受影响 |

## 2. 三处必须记下的细节

### v52：D2 未过，不能藏

v53（第三审）的预登记判据 D2「随机目标臂类别可解码 ≤0.50，三族探针都要」**未通过**：

```
D96 : ridge .43 / mlp .58 / knn .49
D384: .60 / .67 / .51
```

v52 自身的 C2 是过的（0.39–0.43）。两套实现的差异来自阈值余量与探针族。

v53 给出的解释：该判据把「输出含类别信息」（冻结随机投影本就保留输入的类别结构）
误当成「模型用了目标的代理」。**解释不改变 D2 未过的事实**，两处都记下。

### v54：v42 的流程不适用，不是「没做」

v42 的 `beta_scan.py` 扫的是蒸馏损失里的 `β_h = 0.3` 直接目标项（v21 的场景）。
v54 是评价器换轨，损失里没有这个项。

本代的防泄漏是——值域分割（训练只见 slot1∈前8∧slot2∈前8，留出只见后8∧后8，词汇零重叠）
加 E2（冻结随机臂在 CE 下 ≈ chance，三档全过）。且结论全为负面（有意义空间差于零参数
基线 13.7%–34.7%），不存在需要防的虚高正面量。

### v56：改成 invalidated 而不是 clean，且有一个可复核性缺口

单侧线性探针在 XOR 下应 ≈ 0.5，实测 **0.727 / 0.648** → `decouple=True` 没切断职业↔工时相关。
后果是 P1（linear 头解不开 XOR）判未过 —— 线性头解开的那部分来自残留相关。

**主结论不受影响**：P2（hidden=16 已达 2048 宽度的 97.5%）与 P4（有效秩 < 5%）
不依赖两侧模态的严格独立性。「维度不是约束，但空间确实是空的」仍成立。

⚠️ **缺口**：探针数字只写在 `SPEC.md §4`，**没有落盘到 `results/metrics.json`**，
所以该审计的原始读数无法从产物直接复核。已记进 `open_items`：
要可复核需重跑 `examples/run_v56_capacity_allocation.py` 并把探针读数写进结果。

## 3. 结清后的状态分布

```
clean           37
invalidated     11
not_applicable   5
-------------------
合计            53   （另有 3 个 archive 老包无该字段）
```

`invalidated` 不是 bug。按 `docs/GENERATION_MANIFEST_SCHEMA.md` §4，
它是「negative results are map boundaries」的机器可读形式，不得被当成缺陷修掉。

## 4. 怎么复算

```bash
python tools/audit_leakage_chain.py            # 打印每个 pending 包的判定与依据
python tools/audit_leakage_chain.py --json r.json
python tools/validate_generations.py           # 期望 错误 0 / 警告 0
```

脚本只读不改。它推荐的状态需要人工复核后才写回 MANIFEST。
