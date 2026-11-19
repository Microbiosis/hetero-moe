# V65.0 — SNLI-VE 真多模态

> **文档 ID**: HMF-SNLI-VE-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04，commit `59165a4` + 6 个修复）
> **状态**: 代码就位，**Kaggle 批量验证待执行**
> **归属**: aligner 线，接 v64
> **一句话**: 把评测任务从 ag_news 4 分类换成 SNLI-VE 3 类蕴含判断——后者要求**图文配对**推理，只看 caption 不可能答对，于是多模态从「输入拼接」变成「必须用视觉证据」。

---

## 0. 本代性质（先说清楚，避免误引）

**本代没有端到端数字。**

本代交付的是**数据管线 + 任务切换**，端到端训练与评估需要在 Kaggle 上跑（需 GPU、需下载 Flickr30k 与三个底座）。按项目纪律「无支撑的数字不得对外引用」：

- `claim.asserts` 写的是**数据管线契约成立**，不是精度提升；
- `results/metrics.json` 只记录可本地复算的结构性事实；
- 端到端产物的落盘位置已在 §5 写明，等验证后回填。

---

## 1. 一条必须随代记录的证伪：v62–v64 的多模态是**伪多模态**

提取脚本自己的 docstring 写得很直白（`extract_snli_ve_v65.py` 首段）：

> 彻底替换 v62-v64 的伪多模态（ag_news 文本 + food101 随机图，视觉是噪声）。

也就是说，v62–v64 声称的「多模态」评估里，**图像与文本互不相干**：文本是 ag_news 的新闻，图像是随机取的 food101 食物照。视觉通道在那种协议下携带不了任何关于标签的信息——多模态对齐做得再好也无从体现。

⇒ 任何引用 v62–v64 的多模态结论时，都必须带上这条限定：**那批数字不是多模态能力的证据**。本代把任务换成真正需要看图的 SNLI-VE，才第一次让「视觉通道是否被用上」成为可测量的。

---

## 2. 任务与数据

| 项 | 值 |
|---|---|
| 数据集 | SNLI-VE（Flickr30k 图文配对蕴含判断） |
| 类别 | 3：entailment=0 / neutral=1 / contradiction=2 |
| 样本量 | N = 5000（磁盘平衡） |
| 文本 | `sentence1`(caption) + `\u0020` + `sentence2`(hypothesis) |
| 标签文件 | `labels_v65.npy`（int64） |
| 产物目录 | `/kaggle/working/cache_v65`（固定，供同 kernel 训练读取） |
| HF dataset | `redmingwei/hetero-moe-cache-v65` |

三个底座与 hidden 形状：

| 模态 | 底座 | hidden | 序列 |
|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-2B` | 2048 | 128 |
| BERT | `answerdotai/ModernBERT-base` | 768 | 128 |
| ViT | `google/siglip2-so400m-patch14-384` | 1152 | 197 |

ViT 侧：SigLIP2 输出 729 个 patch，用 `linspace` 线性插值池化到 `VIT_SEQ=197`，对齐原 ViT 协议。

---

## 3. 任务切换：N_CLS 参数化

`train_v2.py` 里 `N_CLS = int(os.environ.get('N_CLS', '4'))`，同一脚本可切：

| N_CLS | 任务 | 标签来源 |
|---|---|---|
| 4（默认） | ag_news | `labels${SUF}.npy` |
| 3 | SNLI-VE | `labels_v65.npy` |

分类头 `nn.Linear(model.d_space, N_CLS)`、分层划分 `for c in range(N_CLS)`、类别分布打印都跟着 N_CLS 走 ⇒ 切换不需要改代码。

---

## 4. 本代修掉的真实缺陷（每个都有对应的权威测试项）

| # | 缺陷 | 修法 | 权威测试 |
|---|---|---|---|
| 1 | `IMG_DIR` 写死，图像目录探测不到 | 两个候选目录 + jpg 计数门槛（>100） | 5 |
| 2 | 配对记录为空时一路跑到训练才炸 | `assert len(records) > 100` **早失败** | 4 |
| 3 | `N_CLS` 未定义 → NameError | 补定义并参数化 | 7 |
| 4 | Kaggle 训练中外网断 | `LOCAL_CACHE` 本地目录路径 | 9 |
| 5 | 提取产物散落，同 kernel 读不到 | 固定 `cache_v65` 目录 + `UPLOAD` 开关 | 6 |
| 6 | `CACHE_SUFFIX=_v65` 推出的 HF repo 名带下划线 | 下划线转连字符（`hetero-moe-cache-v65`） | 8 |

另外：训练前的 MoA sanity check **被删掉了**（commit `3f412b1`）——它带着 `tr_idx` 未定义的 bug，而 MoA 前向在 v64 已经验证过，重复检查不增加信息量。

---

## 5. 权威测试：10 项契约（静态，不是执行）

`extract_snli_ve_v65.py` 一 import 就会去 HuggingFace 下载图像与标注并加载三个底座做前向；`train_v2.py` 一 import 就会真的开始训练。**两者都无法在本地 CPU 上执行**，所以本代测试用「读源码 + 局部执行其中的赋值语句」的方式验证契约：

| # | 判据 | 实测 |
|---|---|---|
| 1 | LABEL_MAP 恰好 3 类且映射固定 | `{entailment:0, neutral:1, contradiction:2}` |
| 2 | 文本 = caption + hypothesis（成对） | 2 处成对引用 |
| 3 | 三个底座 hidden 形状固定 | MAXLEN=128、VIT_SEQ=197、2048/768/1152 |
| 4 | 配对过少时早失败 | `assert len(records) > 100` |
| 5 | IMG_DIR 自动探测 | 候选 2 个 + jpg 门槛 >100 |
| 6 | 上传由 UPLOAD 开关控制 | 关闭时提前 return |
| 7 | N_CLS 可切 3/4 类 | N_CLS=3 可切，默认 4 |
| 8 | CACHE_SUFFIX → HF repo 名映射 | `redmingwei/hetero-moe-cache-v65` |
| 9 | LOCAL_CACHE 离线路径可切 | 5 个缓存文件 |
| 10 | 两个脚本语法可编译 | extract + check_vit |

> 这套测试锁的是**契约**（形状、类别数、路径映射、开关语义），不是「提取能跑通」。后者需要 GPU + 网络，属 §6 的待验证项。

---

## 6. 待执行：Kaggle 批量验证

```bash
# 提取（需要 GPU + 磁盘）
python extract_snli_ve_v65.py

# 训练 + 评估（3 类）
N_CLS=3 CACHE_SUFFIX=_v65 LOCAL_CACHE=/kaggle/working/cache_v65 python train_v2.py
```

**产物落盘位置**

- 数据 → HF dataset `redmingwei/hetero-moe-cache-v65`
- 训练产物 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v65.*`
- 评估产物 → `research/aligner/v65_snli_ve/results/*.json`

验证完成后应回填 `results/metrics.json` 的 `pending` 段，并把 `status` 从 `code_ready_pending_batch_run` 改为实测结论。

---

## 7. 复现

```bash
python research/aligner/v65_snli_ve/tests/test_package.py     # 10/10，本地 CPU
```

---

*本代无端到端数字；任何引用不得把「代码就位」说成「已验证有效」。*
