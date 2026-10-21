# 数据集框架层（framework）

14 个成员的统一索引、来源固定、L2 不变量与 L3 检索层。**已发布为公开数据集**：

- HF：https://huggingface.co/datasets/illusion-brain/brain-domains
- 版本：`v1.0`（30.14 GiB / 922 个数据文件）

`data/` 是最初两个子数据集（steinmetz / flywire）的构建脚本与卡片；
本目录是后来把 14 个成员放进同一套索引与核验口径的产物，两者并存。

## 内容

| 文件 | 作用 |
|---|---|
| `METHODS.md` | 口径、主要结果、四条负结论、限制 |
| `RETRIEVAL.md` | L3 检索层用法（`slice_catalog.parquet` 怎么查） |
| `DATASET_CARD.md` | 发布用的数据集卡（HF 仓库根 `README.md` 即此文件） |
| `scripts/` | 生成与核验脚本 |

## 脚本

全部在 Space 上运行，`/domains` 是 HF 桶 `illusion-brain/brain-domains` 的挂载点。

| 脚本 | 作用 |
|---|---|
| `build_sources.py` | 生成 `sources.parquet`（12 条来源，含 pin） |
| `patch_framework_json2.py` | 从实际目录重建 `framework.json` 索引；漏登任一文件则以退出码 1 拒绝写入 |
| `verify_framework.py` | 112 项对账：索引与文件一致、来源 pin 合法、区级注册表唯一键、检索层示例命中数与实算一致 |
| `unify_formats.py` | 表格统一为 parquet（CSV 转出后删除） |
| `clean_release.py` | 只把数据与索引推给发布仓库，实验/归档/脚本留在工作桶 |
| `publish_dataset.py` | 分阶段上传（可重复执行，断点续传） |
| `verify_publish.py` | 仓库与本地逐文件比对 |

## 取数据

```bash
hf download illusion-brain/brain-domains --repo-type dataset --revision v1.0
```

或只取框架层：

```bash
hf download illusion-brain/brain-domains framework --repo-type dataset --revision v1.0
```

工作区（含被排除的实验、归档、诊断脚本）在桶里，不在发布仓库：

```bash
hf buckets ls hf://buckets/illusion-brain/brain-domains
```
