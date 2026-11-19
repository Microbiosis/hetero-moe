# V74.0 — 审计脚本 /v62pkg 通路

> **文档 ID**: HMF-AUDIT-PKG-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-05/06，commit `aa7184d` / `e67722a`）
> **状态**: ⚠️ **机制已实现，但被 v73 的重写覆盖 —— HEAD 上不存在**（见 §3）
> **归属**: aligner 线，接 v73
> **一句话**: 让审计脚本直接从 `/v62pkg` 通道下载算子文件，Kaggle notebook 不必再 git clone 整个仓库。

---

## 0. 本代性质

⚠️ **本代是一个"实现过、但当前不在代码里"的世代。**

机制在 commit `aa7184d` / `e67722a` 里确实实现过，但 **v73 的重写（`6d76985`）把文件整个换掉时没有保留它**。
按项目纪律「作废/未验证的数字仍要保留并标注原因」，本代**如实登记为回归**，不假装它还在。

---

## 1. 机制（曾实现）

问题背景：Kaggle 上 `git clone` 整个仓库**可能不完整**——实测 clone 成功、git tree 里也有 `research/`，
工作区却缺文件（`git status` 全是 `"deleted:"`）。

方案：审计脚本自己下载算子文件到本地包目录（默认 `/tmp/v62pkg`），并把候选目录扩到
`/kaggle/working/v62pkg`，notebook 不必 clone 整个仓库。

下载清单（commit `e67722a`）：

```text
projspace.py, hyper_proj.py, self_repair.py, recurrent_gate.py,
memory_router.py, hebbian_align.py, modality_adapter.py,
feedback_head.py, fusion.py, moa_operators.py, moa_fusion.py
```

`_BASE` 从 `aa7184d` 统一到 `930f2f7`——**因为旧 commit 里缺 `recurrent_gate.py`，import 会失败**。

---

## 2. 权威测试：5 项（本地 CPU）

本代测试断言的是**历史提交里确实实现过**（稳定事实），
并把 HEAD 的回归状态显式记录下来：

| # | 判据 | 做法 |
|---|---|---|
| 1 | `aa7184d` 确实加入 `/kaggle/working/v62pkg` 候选 | `git show aa7184d:<path>` |
| 2 | `e67722a` 把 `_BASE` 统一到 `930f2f7` | `git show e67722a:<path>` |
| 3 | 下载清单包含 `recurrent_gate.py` | 同上 |
| 4 | **HEAD 的回归状态被显式记录** | 若 HEAD 已无该通路，打印 `[KNOWN-REGRESSION]` 并说明来源 |
| 5 | HEAD 的审计脚本本身仍能跑通 | 真跑一遍 ⇒ 回归只影响"免 clone"的便利通路 |

第 4 项是本代特有的写法：**它不假设修复与否，而是把当前状态记录下来**——
这样以后有人重新落地这条通路，测试会打印"已恢复"，而不是红着。

---

## 3. 回归事实（必须记录）

`git log --follow` 显示的顺序是：

```text
1f7c6b4 v71 初版审计脚本
23af716 v72 定位修复（绝对路径 + fallback）
370e45e v73 P_m 键名 + shape 校验
aa7184d v74 加 /v62pkg 候选          ← 本代
930f2f7 v73 feats 前过 P_m 投影
e67722a v74 _BASE 统一到 930f2f7     ← 本代
6d76985 v73 审计脚本重写（+117/-109）← 把本代的两个 commit 一起覆盖了
17dd729 / 97af608 / 5e74ecf / a5e6e62 / 41d46db  v73 后续修复
```

⇒ 本代的两个 commit 夹在 v73 的修复之间，最终被 v73 的重写整体覆盖。

**影响范围**：只是"notebook 免 clone"这条**便利通路**没了；
审计脚本本身在 HEAD 上仍然能定位到算子文件并跑通（v72/v73 留下的候选仍然有效）。

**待决**：是否重新落地这条通路。本代不擅自改 v62 工作目录里的公共文件。

---

## 4. 复现

```bash
python research/aligner/v74_audit_pkg/tests/test_package.py     # 5/5，本地 CPU
```

---

*本代是"实现过但被覆盖"的世代；引用时必须带上这句。*
