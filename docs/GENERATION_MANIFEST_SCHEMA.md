# MANIFEST 规范 — 每一代都是自包含独立子包

> **状态**: 试点 (pilot) — distillation 线 v16 / v17 / v18 三个版本已落地
> **校验器**: `tools/validate_generations.py`
> **适用范围**: `research/<line>/v<N>_<name>/` 下的每一个版本包

---

## 0. 为什么要这条规范

改造前,一个版本的产物散在四个顶层目录:

| 产物 | 改造前位置 | 问题 |
|---|---|---|
| 规范 | `docs/V<N>_0_*.md` | 34 份,与代码不同处 |
| 代码 | `research/<line>/v<N>_*/` | 只有实现,无自述 |
| 测试 | `tests/test_v<N>_*.py` | 全局扁平,与包解耦 |
| 结果 | `results/*.json` | **仅 v32 / v39 落盘**,v6~v31 无 |

于是 34 个版本、25 次迭代的改进链**只以散文形式存在**,`git log` 只有 1 个 commit,
`results/` 只有 5 个 JSON。这直接导致:

- 任何一条结论都无法被第三方复核
- `research/README.md` 里"每代只加一个机制"之类的声明无法机器验证
- **v32 抓到的目标泄漏,靠的是研究者主动提问,而不是机制触发**

本规范把这四类产物收进包内,并要求每代用 `MANIFEST.json` 显式声明
"它声称什么 / 凭什么 / 被什么证伪 / 有没有通过泄漏审计"。

---

## 1. 自包含的两层含义(与已有工具的分工)

项目已存在 `tools/check_self_containment.py`,它检查的是**第一层**。本规范补的是**第二层**。

| 层 | 含义 | 检查器 | 状态 |
|---|---|---|---|
| **L1 import 自包含** | 包能单独 import;无未声明的跨研究线依赖 | `tools/check_self_containment.py` | **已存在**(但其 docstring 声称检查 `__deps__`,实现中并无此项,见 §6) |
| **L2 交付自包含** | 包自带 SPEC / MANIFEST / results / tests | `tools/validate_generations.py` | **本规范新增** |

两层互补:**代码能跑** ≠ **结论可复核**。L1 保证可执行,L2 保证可审计。

L2 还有第三层区分,同样重要:**工作树里合规 ≠ 随仓库分发。**
校验器按 `git check-ignore` 把包分成"入版本库"与"author-only"两类并分别计数,
因为本仓库的 `.gitignore` 明确把 `archive/` 与 `research/scale_heterogeneous/`
标注为 author-only(引用 proprietary 辅助模型,保留本地、available on request)。
这两类包的自包含只在本地生效,不应与"随仓库分发的自包含保证"混为一谈。

---

## 2. 包内必备结构

```
research/<line>/v<N>_<name>/
├── __init__.py            # 已有,导出 API
├── MANIFEST.json          # 【新】机器可读的世代声明
├── SPEC.md                # 【新】包内权威规范 (docs/ 退化为指针)
├── results/
│   └── metrics.json       # 【新】该代的指标快照 (可与全局 results/ 对齐)
├── tests/
│   ├── __init__.py        # 【新】
│   └── test_package.py    # 【新】权威测试副本
└── <实现>.py
```

对应的两个薄转发层(零破坏,保留历史路径与全量扫描):

| 转发层 | 内容 |
|---|---|
| `tests/test_v<N>_*.py` | 只 import 包内 `TESTS, TITLE`,输出与退出码完全一致 |
| `docs/V<N>_0_*.md` | 指向包内 `SPEC.md` 的指针 |

---

## 3. MANIFEST.json 字段

```jsonc
{
  "schema_version": "1.0",
  "version": "16.0",
  "package": "v16_bert_student",
  "line": "distillation",
  "title": "人类可读标题",

  "generation": {
    "one_mechanism_added": "本代唯一新增的机制 (对应设计原则 4)",
    "parent": "13.0",              // 父版本号, 根版本为 null
    "children": ["17.0", "18.0"]   // 子版本号
  },

  "implementation": {
    "location": "self",                              // self | lifted
    "lifted_to": null,                               // lifted 时必填
    "modules": ["student.py"]
  },

  "claim": {
    "asserts": "本代声称什么",
    "metric": "fuse_mse",
    "direction": "lower_is_better",
    "value": 0.135,
    "baseline": { "ref": "13.0", "value": 0.257 },
    "delta_pct": 47.5,
    "seeds": 5
  },

  "protocol": {
    "eval_set": "train_distribution",   // train_distribution | held_out
    "input": "评估输入的来源",
    "caveat": "该协议的已知局限"
  },

  "acceptance": {
    "criterion": "什么条件算成功 / 失败",
    "outcome": "pass"                  // pass | fail | invalidated
  },

  // 【本规范的关键字段】见 §4
  "leakage_audit": {
    "required": true,
    "status": "invalidated",           // clean | invalidated | not_applicable | pending
    "audited_in": "32.0",
    "leak_term": "beta_h=0.3 direct target term",
    "leak_free_value": 0.324,
    "note": "一句话说明"
  },

  "falsified": [
    { "hypothesis": "被证伪的假设", "verdict": "结论", "evidence": "证据位置" }
  ],

  "tests_status": "none",              // 可选顶层字段: "none" = 该代确实没有测试套件
                                       // (v30 / v31 属此类)。校验器据此不要求 tests/
  "superseded_by": "18.0",             // 被哪一代取代, 未被取代为 null
  "artifacts": {
    "spec": "SPEC.md",
    "tests": "tests/test_package.py",  // 若该代无测试, 置 null 并声明 tests_status="none"
    "results": "results/metrics.json",
    "forwarder_tests": "tests/test_v16_bert_student.py",   // 仓库级路径
    "forwarder_spec": "docs/V16_0_BERT_STUDENT.md"  // 仓库级路径
  }
}
```

> `artifacts.forwarder_tests` / `forwarder_spec` 是**仓库级**路径(薄转发层),
> 其余是**包内**路径。校验器按此区分解析。

---

## 4. 为什么必须有 `leakage_audit`

这是本规范与普通打包规范唯一的实质差别,直接来自 v32 的教训。

v16 当时的表观结果是 `fuse 0.135` 对教师 `0.257`,即 **+47.5%**。
v32 消融把损失里的直接目标项去掉(`beta_h=0`)后重测:

- 纯蒸馏 `0.324 ± 0.013` vs 教师 `0.323 ± 0.015`
- 配对差 `−0.0009 ± 0.0038`(10 seeds)**不显著**
- 结论:`+0.159` 的表观收益**全部来自 `beta_h=0.3` 的目标泄漏**,不是概念方向迁移

也就是说:**一个看似惊人的正面结果,可能完全来自目标项泄漏,而当时没有任何机制要求作者去查。**

所以本规范强制每代回答一个问题:

> **这个结论,有没有在"移走直接目标信号"的条件下重新测过?**

`leakage_audit.status` 只能是四选一:

| 值 | 含义 | 什么时候用 |
|---|---|---|
| `clean` | 已在无泄漏条件下复测,结论成立 | 真的干净 |
| `invalidated` | 复测后收益归零或不显著 | 如 v16 ——**这是合法且有价值的状态,不是失败** |
| `not_applicable` | 该代不含蒸馏/重建成分,无此风险 | 如 v13 对齐器 |
| `pending` | 尚未审计 | 允许存在,但校验器会点名 |

**`invalidated` 不得被当成 bug 修掉。** 它是这一代真实结论的一部分,
也是"negative results are map boundaries"的机器可读形式。

---

## 5. 与两条已有研究规律的对应

本规范的字段设计直接来自两条已被本项目多次验证的规律:

**(a) 保留条件是"指标改善 ∪ 假设证伪",不是"指标改善"**

见 `leakage_audit.status="invalidated"` 与 `falsified[]` 两个字段——
它们让**证伪型贡献**(v17 的多教师退步、v20 的位置证伪、v27/v28 的全败)
与**改进型贡献**在 manifest 里享有同等地位,而不是因为指标为负被丢掉。

**(b) 即时性能不预测后代潜力(MPM)**

`generation.parent` / `generation.children` 把谱系显式化,使
"v8 C-1 即时 −71.9% 却产出 v9"这类反直觉链路可被查询,而不只存在于 README 的散文里。

---

## 6. 已知缺口(本规范未解决)

- `tools/check_self_containment.py` 的 docstring 声称检查"每个外部依赖都在 `__deps__` 列表中显式声明",
  但其 `check_package()` 实现里**没有任何 `__deps__` 读取或校验逻辑**,该声明目前是无效的。
  本规范未修改该文件的这一项。
- `ALLOWED_INTRA_LINE` 的键方向疑似写反——它登记的是"v25 允许 v26..v29",
  而检查的是"v26 引用 v25 是否被允许",导致 `corpus_scale` 线内部的合法同线引用被标为 CROSS-LINE。
- 本规范只覆盖版本包;`_primitives/` 原语包与 `archive/` 的自包含标准未纳入。
- 尚无机制能在**新结果落盘时自动校验** `claim.value` 与 `results/metrics.json` 是否一致,
  两者目前靠人工保持同步。
- **28 份骨架的 `claim` / `protocol` / `leakage_audit` 仍为待填**,它们列在各自的
  `_needs_human` 字段里。在补齐前, 这些包的指标与协议结论**不应对外引用**。

## 7. 谱系字段的两级模型(重要)

`lineage.*` 与 `generation.*` 描述的是**不同层级**,不要混用:

| 字段 | 层级 | 含义 | 来源 |
|---|---|---|---|
| `lineage.line_parent` / `line_children` | **研究线** | 这条研究线在全局中的位置 | 研究线 README 头部 |
| `generation.parent` / `children` | **包(代)** | 这一代在代际链中的位置 | **需人工按包判定** |

**为什么必须分开**:一条 7 包的线(如 `scale_heterogeneous`)在 README 里只写
"父版本: v6, 子版本: v33..v39"。若把线级字段赋给每个包, 会让 7 个包都自称
`parent = v6` 且互认父子,谱系直接失真。

抽取器因此只填 `lineage.*`,`generation.parent` 留空并登记进 `_needs_human`。
`--link` 只从**显式的** `generation.parent` 反推 `children`,不做线级到包级的推断。

---

## 8. 校验用法

```bash
# 全量:检查所有版本包
python3 tools/validate_generations.py

# 只查试点
python3 tools/validate_generations.py v16_bert_student v17_multi_teacher v18_real_corpus

# 查某一世代的 MANIFEST 是否有必备字段
python3 tools/validate_generations.py --require-lineage
```

已接入 `run.sh`:

```bash
bash run.sh generations    # 只跑 manifest 校验
bash run.sh test           # 全量测试(转发层,计数不变)
bash run.sh verify         # 测试 + import 自包含 + 世代自包含
```
