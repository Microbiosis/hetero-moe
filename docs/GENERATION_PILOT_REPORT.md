# 试点落地报告 — 每一代自包含独立子包

> **范围**: distillation 线 v16 / v17 / v18 三个版本(试点)
> **日期**: 2026-10-03
> **规范**: [`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md)
> **校验器**: `tools/validate_generations.py` (已接入 `run.sh generations`)
> **后续**: 试点之后的迭代见文末 [§11 试点后迭代](#11-试点后迭代2026-10-03)。
>   本文 §1–§10 保持试点当时的原始记录不改写, 数字以 §11 为准。

---

## 1. 改造前的问题(为什么做这件事)

一个版本的产物散在**四个顶层目录**:

| 产物 | 改造前位置 | 状态 |
|---|---|---|
| 规范 | `docs/V<N>_0_*.md` | 34 份,与代码分离 |
| 代码 | `research/<line>/v<N>_*/` | 只有实现,无自述 |
| 测试 | `tests/test_v<N>_*.py` | 全局扁平,与包解耦 |
| 结果 | `results/*.json` | **仅 v32 / v39 落盘**,v6~v31 无 |

于是 34 个版本、25 次迭代的改进链**只以散文形式存在**。`git log` 只有 1 个 commit。

这直接导致三件具体的事做不到:

1. 任何一条结论都无法被第三方复核
2. `research/README.md` 里"每代只加一个机制"无法机器验证
3. **v32 抓到的目标泄漏,靠研究者主动提问,而不是机制触发**

---

## 2. 改造后:一个版本包长这样

```
research/distillation/v16_bert_student/
├── __init__.py            # 已有:导出 API
├── MANIFEST.json          # 【新】机器可读的世代声明
├── SPEC.md                # 【新】包内权威规范
├── results/metrics.json   # 【新】该代指标快照
├── tests/
│   ├── __init__.py        # 【新】
│   └── test_package.py    # 【新】权威测试副本
└── student.py
```

两个薄转发层保留历史路径,零破坏:

| 转发层 | 作用 |
|---|---|
| `tests/test_v16_bert.py` | 只 import 包内 `TESTS, TITLE`,输出与退出码完全一致 |
| `docs/V16_0_BERT_STUDENT.md` | 指向包内 `SPEC.md` 的指针 + 快速索引 |

---

## 3. 与已有 `check_self_containment.py` 的分工

项目**已存在** `tools/check_self_containment.py`。它检查的是 **L1 import 自包含**。
本试点补的是 **L2 交付自包含**。两层互补:**代码能跑 ≠ 结论可复核。**

| 层 | 含义 | 检查器 | 状态 |
|---|---|---|---|
| L1 | 包能单独 import;无未声明跨线依赖 | `check_self_containment.py` | 已存在 |
| L2 | 包自带 SPEC / MANIFEST / results / tests | `validate_generations.py` | **本次新增** |

**本次对既有 L1 工具的唯一改动**:`check_self_containment.py` 的 `PRIMITIVES` 集合
新增 `research._primitives.testing` 一行。原因是本试点引入了第 6 个共享原语 `testing`,
若不登记,`--strict` 模式会把包内测试的
`from research._primitives.testing import run_suite` 误判为跨线依赖(实测 3/3 假失败)。
改动后试点包 strict 3/3 通过,全量 strict 24/29,剩余 5 个失败全在 `corpus_scale` 线
(引用 `research.aligner.v10_embedding` 等),**是既有问题,与本次改造无关**。

**既有缺陷(仅记录,未修)**:

1. `check_self_containment.py` 的 docstring 声称校验"每个外部依赖都在 `__deps__` 列表中
   显式声明",但 `check_package()` 实现里**没有任何 `__deps__` 读取或校验逻辑**。
2. `ALLOWED_INTRA_LINE` 的键方向疑似写反——它登记的是"v25 允许 v26..v29",
   而检查的是"v26 引用 v25 是否被允许",导致 `corpus_scale` 线内部的合法同线引用
   被标为 CROSS-LINE。

---

## 4. MANIFEST 的关键字段:`leakage_audit`

这是本规范与普通打包规范**唯一的实质差别**,直接来自 v32 的教训。

v16 表观结果 `0.135` 对教师 `0.257`(+47.5%)。v32 把 `beta_h=0.3` 直接目标项去掉后重测:

- 纯蒸馏 `0.324 ± 0.013` vs 教师 `0.323 ± 0.015`
- 配对差 `−0.0009 ± 0.0038`(10 seeds)**不显著**
- 结论:`+0.159` 表观收益**全部来自目标泄漏**

所以每代必须回答:**这个结论有没有在"移走直接目标信号"的条件下复测过?**

| `leakage_audit.status` | 含义 | 试点实例 |
|---|---|---|
| `clean` | 无泄漏条件下成立 | — |
| `invalidated` | 复测后收益归零 | **v16** |
| `not_applicable` | 不含蒸馏成分 | **v17** |
| `pending` | 尚未审计 | **v18** |

**`invalidated` 不是 bug,是这一代真实结论的一部分。** 它是
"negative results are map boundaries" 的机器可读形式。

---

## 5. 三个试点包的分工(选它们不是随意的)

| 包 | 形态 | 价值 |
|---|---|---|
| **v16** | 正向结果(表观 +47.5%) | 演示 `leakage_audit.status = invalidated` |
| **v17** | 负向结果(多教师全败) | 演示 `falsified[]` 与 `root_cause_identified_later` |
| **v18** | 实现已提升为原语的 shim | 演示 `implementation.location = lifted` |

v18 逼出了 schema 里的 `lifted_to` 字段——**这是试点选它换来的东西**,
因为全库有多个 shim 式版本包(见 `research/_primitives/README.md` 的"原始位置 → 提升后位置"表)。

### v17 的一个额外收获

`MANIFEST.json` 里记下了后来才查明的根因,附代码位置:

```python
# research/distillation/v17_multi_teacher/distill.py:51
total = total + self.weights[name] * l          # 加权和

# examples/run_v17_multi_teacher.py:159
student_outs = {kind: y for kind in teacher_kinds}   # 同一张量 y 别名进全部教师槽位
```

**代码中不存在任何按样本的分派路径**——对照 MOPD (arXiv:2606.30406) 的
per-prompt 单教师分派,本代从未测过分派式多教师。
这一条过去只存在于分析对话里,现在进了机器可读声明。

---

## 6. 零破坏的验证方式

### 6.1 实测结果(终态)

```
=== 全套 V&V 测试 ===
  ...
  test_v33_scale_hetero.py: 10 PASS
  test_v34_dmn_subconscious.py: 10 PASS
  test_v35_container_entries.py: 10 PASS
  ...
  ----------------------------
  合计: 327/327 ✓ ALL GREEN          退出码 0
```

| 阶段 | 套件 | L1 strict | L2 |
|---|---|---|---|
| 初始基线 | 297/297(3 文件归零) | 24/29 | 无此检查 |
| 搬测试后(暴露路径回归) | 293/293 | 19/29(测试被纳入跨线扫描) | — |
| 修 v25/v29/v15 路径 | — | — | — |
| 修 v33/v34/v35 import | — | — | — |
| **终态** | **327/327 ✓ ALL GREEN** | **24/29** | **29 包 / 0 错 / 33 警 ✓ ALL GREEN** |

L1 的 5 个失败全部是 `corpus_scale` 线的既有跨线引用
(`aligner.v10_embedding` / `central_stability.v22_central_theory` / 同线 `v25_corpus_central.layer`),
与本改造无关,与初始基线一致。

L2 的 33 条警告**全部是待人工判断项**,不是缺陷:28 条 `LEAKAGE-PENDING` + 谱系提示 + v30/v31 无转发层。

### 6.2 顺带修掉的一个真 bug:`run.sh test` 的静默中断

原 `run.sh test` 循环:

```bash
n=$(echo "$out" | grep -c "\[PASS\]")     # 无匹配时 grep 返回 1
```

配合文件头的 `set -euo pipefail`,**任何输出 0 条 `[PASS]` 的测试文件(例如 import 失败)
会让整段脚本提前退出**,后续测试被静默跳过,而且 `合计:` 行根本不会打印——
所以套件每次都在 v33 处悄悄死掉,看起来像"跑完了"。

已改为:单个文件失败只计数,照常跑完 40 个文件,末尾列出零通过文件清单,
并让退出码如实反映失败。**修复前后的差别不是数字,是"看不到失败"变成"看得到失败"。**

### 6.3 关于测试计数的不一致(已解开)

改造前存在两个互相矛盾的数字:论文写 `273 tests`,`run.sh` 注释写 `327/327`。

**实测结论:327 是正确值。**

| 数字 | 出处 | 判定 |
|---|---|---|
| 327/327 | `run.sh` 注释 | **正确** |
| 273 | 论文第 369 行 | 对不上,疑为 v33-v39 存在之前的旧快照 |
| 297 | 本次中间态 | v33/v34/v35 因 ImportError 归零时(327 − 30) |

修复三个包后全量复跑:**`合计: 327/327 ✓ ALL GREEN`,退出码 0**。
与 297 基线逐文件比对,差异恰好 3 行(v33/v34/v35 各 0→10),其余 37 个文件完全一致。

> 建议把论文第 369 行的 `273` 改为 `327`,并注明这是含 v33-v39 的完整套件。

### 6.4 v33/v34/v35 的 ImportError(已修,非本次改造引入)

```
ImportError: cannot import name 'BIG_LM_DIR' from 'research._primitives.scale_hetero_loaders'
ImportError: cannot import name 'MSA_DIR' from '...v35_container_entries.paths'
```

根因:原语提升时符号改名(`BIG_LM_DIR`→`MSA_DIR`、`EMBED_LM_DIR`→`EMB_DIR`、
`RERANK_LM_DIR`→`RER_DIR`、`load_big_lm`→`load_msa_big`),但 v33 / v34 / v35
三个消费者**没有跟着改名**。

**证据 `git status --porcelain` 证明与本改造无关**:该目录不在本改造的改动范围内。

**已按映射表修复,三个测试各恢复 10 PASS(净 +30)**:

| 包 | 修法 |
|---|---|
| v33 | 4 个旧名改为新名(`BIG_LM_DIR`→`MSA_DIR` 等) |
| v34 | 2 个旧名改为新名(`load_msa_big` 原本已是新名) |
| v35 | `paths.py` 改为从 primitives re-export + 保留旧名别名;`__init__.py` 的 `get_tabldm_repr` → `get_tabular_repr` |

**v35 的占位符路径是有意为之,不是脱敏事故**——其 docstring 明确写着入口模型
"proprietary, available on request"、作者不公开权重。全库仅 v35 一处保留占位符,
而 primitives 里对应的是同批模型已解析出的真实位置
(`MSA-4B` / `skillcorpus-embedding-0.6b` / `skillcorpus-reranker-0.6b` / `Xiaomi-TabLDM`)。
因此 v35 的 `paths.py` 改为 re-export, 既对齐命名又换回真实路径。

⚠️ **一个行为变化需知晓**:v35 的环境变量名随之改变
`V35_BIG_LM_DIR`→`V36_MSA_DIR`、`V35_EMBED_LM_DIR`→`V36_EMB_DIR`、
`V35_RERANK_LM_DIR`→`V36_RER_DIR`、`V35_TAB_LM_CKPT`→`V36_TABLDM_CKPT`。

**原先的预判"import 修复 ≠ 测试转绿"并不成立**:这三个测试使用随机张量桩
(其 docstring 明写"不依赖 transformers / tabldm"),因此不触及真实权重,修复后直接转绿。

---

## 7. 铺开结果与存量状态

```bash
python3 tools/validate_generations.py        # 严格模式, 默认
```

```
包数: 32  错误: 0  警告: 39  ✓ ALL GREEN
  ├─ 入版本库: 22 包
  └─ author-only (不在库, 见 .gitignore): 10 包
       archive/negative_findings (2), archive/pre_baseline (1), research/scale_heterogeneous (7)
```

**"自包含"要分两层看:工作树里合规 ≠ 随仓库分发。**

| | 数量 | 说明 |
|---|---:|---|
| 工作树中已自包含 | 32 | 全部自带 `MANIFEST.json` / `SPEC.md` / `results/metrics.json` / 包内权威测试 |
| **其中随仓库分发** | **22** | 这部分的自包含保证对克隆仓库的人有效 |
| **其中仅本地生效** | **10** | `.gitignore` 标注 author-only, 不随仓库分发 |

**这 10 个包是既有仓库策略决定的,不是本次改造遗漏。** `.gitignore` 写明:

```
# Pre-baseline and negative findings (kept author-only; paper only references outcomes)
archive/

# Scale-heterogeneous research line uses proprietary auxiliary models
# (a proprietary large LM as backbone, tabular-LM, embedding-LM, reranker-LM);
# the code is preserved author-only and is available on reasonable request.
research/scale_heterogeneous/
```

即 `archive/`(v5 前基线 + v12/v23b 负结果)与 `research/scale_heterogeneous/`
(v33-v39,含刚修好的三个)因**引用 proprietary 辅助模型**而保留在作者本地。

**因此不要笼统地说"32 个包都自包含"** —— 准确表述是:
**32 个包在工作树中自包含,其中 22 个的自包含保证随仓库分发。**

校验器现在会按 `git check-ignore` 自动区分并打印这个口径,避免误判。
若将来要改变该策略(取消 author-only),校验器输出会自动跟随,无需改代码。

| 项 | 数量 |
|---|---|
| 有 `MANIFEST.json` | 32 |
| 有 `SPEC.md` | 32(26 份内容由 `docs/V*_0_*.md` 搬入,6 份存根) |
| 有 `results/metrics.json` | 32 |
| 有包内权威测试 | 29(另 3 个**确实无测试套件**,见下) |

**3 个"确实无测试"的包:`v5_alpha` / `v30_gate_softplus` / `v31_gate_gradient_scale`。**
`tests_status: "none"` 是**派生结论**——抽取器检测到 `tests/` 下无对应文件且包内无 `tests/`,
自动判定,不需要人工标注,因此 `--force` 重建不会把它冲掉。

### 泄漏审计闸门(已武装,待人工放行)

```bash
python3 tools/validate_generations.py --require-leakage-audit
# 包数: 32  错误: 30  ✗ HAS FAILURE     ← 30 代尚未审计

python3 tools/validate_generations.py --require-leakage-audit v16_bert_student v17_multi_teacher
# 包数: 2  错误: 0  ✓ ALL GREEN          ← 已审计的放行
```

按决策,`--require-leakage-audit` 在铺开后启用;**现在打开会有 30 个错误**,
需先逐代填写。清单用:

```bash
python3 tools/extract_generation_manifest.py --print-human
```

---

## 8. 试点过程中修掉的所有缺陷

### 8.1 工具自身暴露的(3 个)

1. **转发层路径解析错**:`forwarder_tests` 是仓库级路径, 初版校验器按包内解析 → 3 个假 ARTIFACT-MISSING。已修为 `REPO_LEVEL = {"forwarder_spec", "forwarder_tests"}`。
2. **谱系检查写反**:写成 `parent not in parent's children`(应为 `child not in parent's children`)→ 2 个假 LINEAGE-WARN。已修。
3. **谱系建模错误**:把**行级** README 的"子版本"赋给**每个包** → 7 包的 `scale_heterogeneous` 里 v33-v39 全都自称 `parent=v6` 且互认父子。已改为线级 `lineage.*` / 包级 `generation.*` 两级模型(见 schema §7)。

### 8.2 抽取器暴露的(4 个)

4. **正则边界**:`\bv(\d+)\b` 的结尾 `\b` 在 `v22_central_theory` 上不成立(`_` 是单词字符)→ 父版本抽空,还从散文里错抽出子版本。改为 `v(\d+)(?![0-9])`。
5. **"无"守卫缺失**:子版本写作"无(这是 v22 路线的语料分支, 已收敛)"时,散文里的 v22 被误当子版本。已加"无/没有/None"前缀守卫。
6. **空值崩溃**:校验器对 `artifacts.forwarder_tests = null` 做路径拼接 → `TypeError`。已加空值处理。
7. **元数据当路径**:`tests_status` 被放进 `artifacts` 而当路径校验 → 假 ARTIFACT-MISSING。已挪到 manifest 顶层。

### 8.3 搬迁引入的(1 个真回归)

8. **"上溯两层即仓库根"的脆弱写法**:
   ```python
   sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
   ```
   在 `tests/` 下恰好正确,搬到 `research/<line>/<pkg>/tests/` 后解析成
   `.../<pkg>/examples`——**不存在**。后果 `test_v25` 10→7、`test_v29` 12→11,
   全量 297→293。已把 3 个文件(含 `v15_nested`)换成健壮上溯。
   **这个回归是逐文件 PASS 比对抓出来的,总计数只会显示"少了 4 个"而说不出在哪。**

### 8.4 L1 检查器的作用域修正(2 个)

9. **包内测试被纳入跨线扫描**:测试搬进包后,集成测试对其他包的引用被标成 CROSS-LINE
   (L1 从 24/29 掉到 19/29)。但测试按定义就要跨线导入,L1 规则针对的是**实现模块**
   的依赖闭包。已让 `check_package` 跳过 `tests/` 子目录。
   搬测试之前它们位于顶层 `tests/`,本就不在任何包的扫描范围内——规则没变,是作用域要澄清。

10. **原语判定粒度太粗**:PRIMITIVES 集合只列包级名,导致 v35 的
    `research._primitives.scale_hetero_loaders.paths`(子模块)被误判为跨线。
    已改为**前缀判定**:`research._primitives.` 下的一切都是原语,
    这样新增子模块不会再被误判。

> 这 10 个里有 7 个是"工具自己抓到自己"。**校验器能自证有效,
> 说明"跨包谱系自洽"和"产物路径存在性"这两项检查确实有价值。**

---

## 9. 全量铺开前需要确认的三个决策

1. **是否接受 `leakage_audit` 强制字段?**
   它要求为每一代补一个此前不存在的判断(干净 / 被推翻 / 不适用 / 待审)。
   代价是工作量;收益是把 v32 那类事故从事后变成事前。
   若接受,`--require-leakage-audit` 应在全量铺开后打开。

2. **26 个存量包的 manifest 由谁生成?**
   - 全部手写(准确,但 26 份)
   - 半自动:校验器从 README + docs/ 抽骨架,人工补 `leakage_audit` 与 `falsified`(推荐)

3. **测试是否全部搬入包内?**
   现有 40 个测试文件,试点搬了 3 个。搬完 `tests/` 全部变成转发层。
   若不搬,L2 规范要开一个例外。

---

## 10. 本试点未做的事(明确边界)

- **未填** 28 份骨架的 `leakage_audit` / `claim` / `protocol`(需逐代判断,已列入 `_needs_human`)
- **未修** `check_self_containment.py` 的 `__deps__` 空实现(仅登记了新原语,共 1 行)
- **未修** `ALLOWED_INTRA_LINE` 键方向问题(见 §3)
- 未为 **v30 / v31 编写测试**(该两代在仓库中本就没有测试文件, 已如实标注 `tests_status: "none"`)
- 未改 `pyproject.toml`(`tools` 不在 `packages.include` 内,新校验器与抽取器均为开发期工具)
- 未提交 / 未打 tag(git 仍是 1 个 commit;补版本标记需另行确认)
- 未跑任何 `examples/run_v*.py` 复现实验,manifest 中的指标均为**从既有文档转录**,非本次实测

---

## 11. 试点后迭代 (2026-10-03)

目标不变: **每一代都成一个自包含独立子包**。本节记录试点之后补齐的部分,
以及试点 "ALL GREEN" 掩盖掉的两个真实缺陷。§1–§10 的数字以本节为准。

### 11.1 试点报绿, 但自包含在分发层面是断的

试点 §7 报 `包数: 32  错误: 0  ✓ ALL GREEN`。但校验器当时只检查
**产物在本机磁盘上是否存在**, 不检查它是否**随仓库分发**。实测:

| 产物 | 磁盘 | 版本库 | 被谁吞掉 |
|---|---|---|---|
| 32 个包的 `results/metrics.json` | 有 | **0 个** | `.gitignore` 的 `results/` |
| 34 份 `docs/V*_0_*.md` 薄指针 | 有 | **0 份** | `.gitignore` 的 `docs/V*.md` |

根因是两条 gitignore 规则的**作用域比意图宽**:

- `results/`(无前导斜杠)匹配**任意层级**的 results 目录, 于是把每个版本包
  自己的 `results/metrics.json` 一起排除了。而顶层 `results/` 的注释写的是
  "Raw experiment outputs (large; reproducible from scripts)" —— 只想排原始输出。
- `docs/V*.md` 同样是通配。试点新建的薄指针就落在这个模式下, 于是
  `MANIFEST.artifacts.forwarder_spec` 指向一个**克隆者永远拿不到**的文件。

**结论**: "自包含"必须分两个口径问 —— 本机是否有, 分发后是否有。试点只答了第一个。
现已在 `.gitignore` 里把两条规则收窄(`/results/` 加前导斜杠; `!docs/V*_0_*.md`
放行薄指针), 并给校验器加了一道闸门:

```
[ARTIFACT-UNSHIPPED]  产物被 .gitignore 排除        → 错误
[ARTIFACT-UNTRACKED]  产物尚未 git add            → 警告
```

author-only 包不做该判定(它们本来就不入库, 属既有仓库策略)。

### 11.2 补齐 v32: 产出 `leakage_audit` 概念的那一代, 自己却没有子包

试点把 `leakage_audit` 立为必备字段, 并在 v16/v17 的 MANIFEST 里写了
`audited_in: "32.0"` —— **但 v32 自己不是子包**。它有 5 个 runner、
4 份结果记录, 却散在仓库根那个被 gitignore 的 `results/` 里。
也就是说, 论文里最关键的负结果("纯蒸馏不超越教师"), 其证据链在分发版本里是断的。

新增 `research/distillation/v32_leakage_audit/`, 特点是**自带原始记录**:

| 文件 | 作用 |
|---|---|
| `results/v32_*.json` (4 份) | 从仓库根 `results/` 搬入包内, 随包分发 |
| `audit.py` | 纯标准库, 只读包内记录, 重算整代结论 |
| `tests/test_package.py` | 8 项, 全部从原始记录重算而非对字面量 |

于是论文那句话变成**可重算的判据**:

```bash
python -c "from research.distillation.v32_leakage_audit import leakage_verdict; print(leakage_verdict())"
# pure_beats_teacher=False, paired_gap=-0.0009 ± 0.0038
```

`test_headline_number_reproduces_paper` 钉住的就是这个数能算回论文的 −0.0009 ± 0.0038。
审计时发现两个容易引错的口径, 已在代码与 SPEC 中显式注明:
配对差是 `teacher − student`(反过来会读成"学生反超"), 标准差是**样本**标准差
(ddof=1, 用总体标准差会得到 ±0.0036), 且头条数字取自 `deterministic` 条件而非 `base`。

**`forwarder_tests` 故意为 null**: 加顶层转发层会让全套件从 327 变 335,
而论文里的 327 已校准并编译进 PDF, 此处无法重编译 LaTeX。
与 v5 / v30 / v31 同样处理, 包内权威测试可直接运行。

### 11.3 版本号撞号: v23b 在谱系图里被静默吞掉

抽取器的目录名正则 `v(\d+)[a-z]*_` 认得 `v23b` 的 `b` 后缀, 但生成版本号时
`version_dotted = f"{version.lstrip('v')}.0"` 把它丢了 —— 于是
`v23b_hard_vqa` 与 `v23_vqa` 的 `version` 都是 `"23.0"`。

后果不是报错, 而是**静默丢数据**: `check_lineage` 用 `version_map[v] = rec`,
后写的覆盖先写的, v23b 的整条谱系从图里消失, 校验器还报 ALL GREEN。
v23b 是独立的一代(multimodal README 单列一行: 8 类陷阱 toy, multimodal 50% = 随机)。

- 版本号改为 `23b.0`;
- 校验器新增 `[VERSION-DUP]` 检查, 撞号直接判错。

这与 §8 的结论一致: 校验器该抓到的都抓到了, 前提是**检查项本身覆盖到那一类**。

### 11.4 接上代际链: 29 个孤儿包不再是孤儿

试点只有 v16/v17/v18 填了 `generation.parent`, 其余 29 个全空 —— 抽取器
按设计不猜父版本。于是"代际链"当时只是一堆孤岛, 谱系查询无从谈起。

本轮按各研究线 README 的父/子版本声明 + `research/README.md` 的自演进序列
补齐 28 个父版本(4 个试点已填的保留), 再用 `tools/extract_generation_manifest.py --link`
由父代反推 children。现在 `LINEAGE-WARN` 归零, 谱系双向自洽。

### 11.5 本轮之后的准确状态

```
包数: 33  错误: 0  ✓ ALL GREEN
  ├─ 入版本库: 23 包
  └─ author-only (不在库, 见 .gitignore): 10 包
```

**仍未做(诚实记录, 不粉饰)**:

1. **v4 / v6 / v14 三个代仍无子包**。它们各有完整规范与 runner, 本轮未建成包。
2. 32 个存量包的 `results/metrics.json` 仍是抽取器骨架, `metric`/`value` 为 `null`,
   待逐代从证据填写(`tools/manifest_worksheet.py` 已备好工作表)。
3. `--require-leakage-audit` 仍未打开: 现有 30 代 `status` 为 `pending`。
4. §3 记录的两处 `check_self_containment.py` 缺陷(`__deps__` 空实现、
   `ALLOWED_INTRA_LINE` 键方向)**仍未修**。
5. `run.sh` 的 `PROJECT_DIR` 硬编码 `/workspace/hetero-moe`, 在本机跑不了,
   全套件是按其逻辑等价执行的。

### 11.6 复核: 上一轮的两个数字是坏的, 已修的两处缺陷已收

对 §11 结论做独立复核, 推翻了两处说法, 并修掉了 §11.5 列的 4/5 两项欠账。

**(a) 全套件不是 221/221, 而是 327/327 —— 论文那个数是对的。**
上一轮记录"全套件 221/221 全通过"且"11 个测试文件因缺 `transformers` /
`safetensors` 而 0 PASS"。在**依赖齐备**的环境重跑: `transformers 5.16.1` /
`safetensors 0.8.0` 实际已安装, 40 个测试文件**无一 0 PASS**, 合计 **327/327**。
即 221 是坏环境下 11 个文件被 harness 静默排除后的降级数字, 而非真实总量。

这一点直接回答了 v32 那个悬而未决的取舍: 根 `tests/` 套件的权威值就是 **327**,
与论文 `microfusion.tex` 三处(中英各一)以及 `run.sh` 帮助文本"实测 327/327"完全一致。
所以"`forwarder_tests` 留 null 以免 327→335"这个决定是**对的** —— 327 是真实值,
v32 包内那 8 项测试不在 `bash run.sh test` 的口径内, 加转发层才会把它推到 335。

**(b) §11.5 第 4 项: `ALLOWED_INTRA_LINE` 键方向已修 (strict 25/30 → 27/30)。**
字典键是**被导入方**(基准代), 值是**允许导入它的代**(注释即"v26/v27/v28/v29
都可以 import v25"), 但查询写成 `self_path` 查键 / `imp` 比对值 —— 方向相反,
使放行分支成为**死代码**。还有第二层: 实际 import 写作 `v25_corpus_central.layer`
(子模块), 字典键是包, 精确相等也匹配不到。两层都要修才生效。

结果: 4 处合法同线复用 (v26/v27/v28/v29 → v25) 不再被 `--strict` 误报为
`[CROSS-LINE]`; 剩下 3 个失败是**真跨线依赖**, 保留 (v25→v10, v27→v22,
v28→v10/v22)。normal 模式 30/30 不变。

`__deps__` 空实现**没有**补: 兑现它需先给各包补 `__deps__`, 否则一次性把全仓判红。
改为把 docstring 里那条从未实现的承诺**删掉并写明未实现**, 不留看起来已覆盖的条目 ——
与 §11.1 同一个病根。同时让 `allowed_used` 真正输出, 兑现"要标记"那句承诺;
标记一上线就暴露: `routing_evolution` 与 `scale_heterogeneous` 两条声明在代码里
**零对应复用**, 是已经腐烂的声明。

**(c) §11.5 第 5 项: `run.sh` 已可被克隆者直接执行, 并另修一处 argv 泄漏。**
`/workspace/hetero-moe` 并非不存在, 而是本机一个指向仓库的**自建软链** —— 作者机器
能跑是因为这个软链, 任何克隆者都没有。而论文 `\Srepro` 恰恰让复现者执行
`bash run.sh test`。改为由 `${BASH_SOURCE[0]}` 推导仓库根。

顺带发现并修掉一个独立缺陷: `cmd="${1:-help}"` 之后没有 `shift`, 子命令名会漏进
工具 argv —— `bash run.sh selfcontain` 实际执行 `check_self_containment.py selfcontain`,
被当成包名过滤而报"未找到目标包" (与 cwd 无关, 从仓库内调用同样失败)。加上守卫式
`shift` 后, 从**仓库外**调用 `selfcontain` / `generations` / `selfcontain --strict`
均正常 (30/30 · 33 包 0 错 · 27/30)。

**(d) 顺手修掉 `run.sh test` 汇总行的一个"假绿"读法。**
复核时在**并发负载**下跑了一次 `bash run.sh test`, 得到
`合计: 319/319 ✗ HAS FAILURE` —— `test_v8_central.py` 零通过(单独连跑 3 次均 8 PASS,
确认是资源竞争下的偶发, 非代码缺陷)。问题出在汇总行本身:

- `pass` 与 `total` 是同步自增的, 恒相等, 故原判定 `[ "$pass" -eq "$total" ]` 是**死逻辑**;
- 零通过文件会**整体从比值里消失**, 于是失败时照样显示 100%(327 → 319, 分母跟着变小)。

读者只扫比值会把 `319/319` 当绿。已改为: 判定只看 `nfail`; 汇总行追加
`(N 个测试文件, M 个零通过)`; 并在零通过时补打该文件输出末尾两行 —— 这类失败常是
import 崩了或被 OOM 杀掉, 只留"无输出/无 PASS 标记"分不清是代码坏了还是环境问题。

### 11.7 本轮复核后的准确状态

```
validate_generations.py    33 包 / 0 错 / 39 警告   (23 入库 + 10 author-only)
check_self_containment     30/30 (normal) · 27/30 (strict, 3 个真跨线)
v32 包内测试                8/8
bash run.sh test           327/327 ✓ ALL GREEN, exit 0
                           ← 论文口径, 且已从仓库外目录执行验证(克隆者路径)
```


仍未做: v4 / v6 / v14 三个代仍无子包(各自有完整规范与 runner); 32 个存量
`results/metrics.json` 仍是待填骨架; `--require-leakage-audit` 仍未打开。


