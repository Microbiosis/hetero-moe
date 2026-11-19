# V72.0 — 审计脚本 projspace 定位修复

> **文档 ID**: HMF-AUDIT-PATH-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-05，commit `23af716`）
> **状态**: **已本地实测**（在任意 cwd 下都能跑通）
> **归属**: aligner 线，接 v71
> **一句话**: 修审计脚本的 projspace 定位——用绝对路径 + 包目录 fallback，不再依赖 Kaggle 的工作目录。

---

## 0. 本代性质

本代只修一个**真实故障**，且这个故障是"脚本根本跑不起来"级别的：

- Kaggle notebook 的 cwd 是 `/kaggle/working`；
- 审计脚本原来按**裸相对路径** `'projspace.py'` 加载算子 ⇒ `FileNotFoundError`，审计跑不起来。

修法：先用脚本自身的**绝对路径**（`os.path.dirname(os.path.abspath(__file__))`）定位包目录，
并保留 Kaggle 上的包目录作为 fallback，再用绝对路径拼出 `projspace.py` 的完整路径加载；
同时打印定位到的目录，便于核对。

本代**没有端到端数字**（它修的是工具可用性，不是模型效果）。

---

## 1. 权威测试：4 项（本地 CPU）

| # | 判据 | 做法 |
|---|---|---|
| 1 | 不再用裸相对路径 | 源码级：断言 `spec_from_file_location(..., 'projspace.py')` 已消失 |
| 2 | 候选列表含脚本自身目录与 Kaggle 包目录 | 源码级 |
| 3 | **在任意 cwd 下都能跑通** | 真跑一遍，cwd 设成临时目录，断言 rc=0 且跑到 `[done]` |
| 4 | 打印定位到的目录，且该目录确有 projspace.py | 真跑一遍并检查落点 |

第 3/4 项是本代最硬的证据：**不是看源码，是真的换个工作目录跑一遍**。

---

## 2. 与 v74 的关系

v74 又加了一条 `/kaggle/working/v62pkg` 候选与"直接下载算子文件"的通路，
但那条通路在 v73 的重写中被覆盖了（见 `v74_audit_pkg/SPEC.md` §3）。
**当前 HEAD 里活着的是本代这条**（脚本自身目录 + 一个写死的 Kaggle 仓库路径）。

---

## 3. 复现

```bash
python research/aligner/v72_audit_path/tests/test_package.py     # 4/4，本地 CPU
```

---

*本代修的是工具可用性，无端到端数字。*
