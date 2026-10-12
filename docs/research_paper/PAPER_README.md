# Paper README — Heterogeneous Reconstruction

> **Paper:** *Reconstructing Frozen Heterogeneous Pretrained Models: Sublayer Fusion, Coordinated Routing, and Chimera Distillation*
>
> **Authors:** Anonymous (with AI-assisted drafting and pre-submission review)
>
> **Target venue:** **待决** — 合并后 23(英)/ 24(中)页,已远超 NeurIPS 9 页正文限制。投期刊长文,还是把审计细节全下沉附录以压到会议篇幅,尚未决定。
>
> **Status:** 合并稿已成型(2026-10-11 由 `microfusion` + `paper` 合并重构);2026-10-12 补入 §6.3 规模异构实验并消除一处无支撑主张。中英文 PDF 均编译通过。

> 🚨 **2026-10-12 记录的一处自查**:合并稿曾在摘要与引言声称「scale-heterogeneous fusion gain 样本外存活」,
> 而正文 v33/v34/v35 出现 0 次、无任何规模异构实验。已补 §6.3 + `tab:scale-hetero` +
> `tab:oos` 两行。**教训**:声称一个存活结论之前,先确认它有支撑它的实验。

This README is the entry point for everything paper-related under `<repo>/docs/research_paper/`. It catalogs every file, its role, and how to verify it.

---

## 1. Paper Sources and Compiled Outputs

### 1.1 当前稿(合并稿)

| File | Role | Status |
|---|---|---|
| [`reconstruct.tex`](./reconstruct.tex) | 英文 LaTeX 源 | 726 行,0 编译错误 |
| [`reconstruct_zh.tex`](./reconstruct_zh.tex) | 中文 LaTeX 源 | 715 行,0 编译错误,中英术语对齐 |
| [`reconstruct.bib`](./reconstruct.bib) | BibTeX | **43 条**(microfusion 37 + paper 10 去重) |
| [`reconstruct.pdf`](./reconstruct.pdf) | 英文 PDF | 448 KB,**23 页** |
| [`reconstruct_zh.pdf`](./reconstruct_zh.pdf) | 中文 PDF | 816 KB,**24 页** |
| [`MERGED_OUTLINE.md`](./MERGED_OUTLINE.md) | 合并大纲 | 合并来源、章节结构、关键改动表 |

### 1.2 已废弃稿(仅作历史留档,**请勿引用**)

| File | Role | Status |
|---|---|---|
| `microfusion.tex` / `microfusion_zh.tex` | 旧英文/中文源 | 468 / 462 行,顶部已打 **DEPRECATED — MERGED** 横幅 |
| `paper.tex` | 旧中枢机制稿 | 已打 DEPRECATED 横幅 |
| `microfusion.bib` | 旧 bib(37 条) | 已被 `reconstruct.bib` 取代 |
| `microfusion.pdf` / `microfusion_zh.pdf` | 旧 PDF | 结论与数字口径均已被取代 |

旧稿之所以保留:它们记录了合并前的推理轨迹(尤其是 v32 泄漏归因如何改写机制叙事、v44/v45 如何
推翻主表)。**但论文事实一律以 `reconstruct.*` 为准。**

### Compilation

```bash
# 英文 (pdflatex + bibtex)
cd <repo>/docs/research_paper
latexmk -pdf -interaction=nonstopmode -halt-on-error reconstruct.tex

# 中文 (xelatex + ctex, 跑两遍以解析交叉引用)
xelatex -interaction=nonstopmode reconstruct_zh.tex && xelatex -interaction=nonstopmode reconstruct_zh.tex
```

⚠️ **`latexmk` 会报 "All targets are up-to-date" 然后什么都不做** —— PDF 产物的时间戳可能比源文件新。
遇到这种情况**删掉 `.aux` / `.log` / `.out` 后用 `pdflatex` / `xelatex` 连跑两遍**。
验证产物要看内容(抽 `pdftotext` 查关键句),不要看 mtime。

## 2. Contributions

> ⚠️ **2026-10-12 重写**:原先的 C1–C4 标签式复述**已在合并稿中删除**,改为具名发现。
> 下表是合并后 §1.4 的实际贡献清单 —— 贡献被明确切成**机制层**与**证据层**两半。

### 2.1 机制层:什么结构可融合、什么协调设计可容许

| 名称 | 中文 | 位置 | 可证伪方式 |
|---|---|---|---|
| 可分解的异构性 | 每个组件可独立开关,所以每次消融都是改一个 flag | §1.4 (1) | 把组件焊死成不可拆分 |
| 对齐器阶梯与其天花板 | linear < mean < attention < mixture,等容量对照证明是设计而非规模 | §6.4 (`tab:ladder`) | 等容量下阶梯反转 |
| G1–G5 与三条定理 | 门控式是唯一同时满足完美退化/保序/有界/低开销的中枢模式 | §4.2 | 构造反例推翻任一 G |
| 嵌合蒸馏的泄漏分解 | 「学生超教师」全部归因于 0.3 直接目标项 | §6.8 | 在无直接目标项下复现该裕度 |

### 2.2 证据层:统一样本外协议下哪些结论是真的

| 名称 | in-sample | 样本外 | 判定 |
|---|---:|---:|:---:|
| attention 通路优势 | +82.7% | +70.6% | **存活** |
| 对齐器阶梯天花板 | 相对 attn +36.5% | 容量匹配 | **存活** |
| 中枢:路由归一化最优 | 最优 | 最优 (5/5) | **存活(已更正)** |
| 规模异构融合 | +26.7% | +26.5%(噪声)/ +26.5%(刺激) | **存活**(§6.3,2026-10-12 补入) |
| 中枢:门控式协调增益 | +44.5% | +4.0% | 收缩到噪声 |
| 语料规模边界 | 存在 | 不存在 | **未确立** |
| 长训练增益 | +4.3% | +1.5% | 收缩(保留 34%) |
| 梯度修复 | −0.44% / −0.11% | 相同 | 无效果 |
| 入口注入容器 | −11.1% | −3.3%(刺激) | 无增益 |

**样本外纪律本身是本文最可移植的一项结果**,不是一个脚注:对冻结底座组合,
报告 held-out 排名而非 in-sample 排名,并把任何与种子波动相当的裕度视为未决。

### 2.3 数据规模依赖性 —— 降级为假设

原 C3 曾在 in-paper 里当作经验主张,独立评审指出**扫描并没有真跑**后被降级为
**文献汇聚假设**(§7 第四段引四篇独立工作)。这是"没做的实验不能写成发现"的又一例。

---

## 3. Related Work Reading Notes

Eleven papers were fully read in HTML full-body form prior to revision. The reading notes document the technical core of each paper, the vs-hetero_fusion comparison, the Related Work one-liner, and the BibTeX entry.

| File | Papers covered | Bytes |
|---|---|---|
| [`related_work_notes.md`](./related_work_notes.md) | Batch 1: Symphony-MoE, EAQuant, ScaleKD, Orchestrating-HetExp (4 papers) | 20 KB |
| [`related_work_notes_batch2.md`](./related_work_notes_batch2.md) | Batch 2: BTX, Soft MoE, AdapterFusion, TinyLLM (4 papers) | 21 KB |
| [`related_work_notes_batch3.md`](./related_work_notes_batch3.md) | Batch 3: Mixtral, LoRAHub, DeepSeekMoE (3 papers) | 17 KB |

Each note file contains: paper metadata, key technical claim, the vs-hetero_fusion one-paragraph comparison table, BibTeX entry ready to paste, and patch recommendations for §Related Work.

---

## 4. Independent Review Log

Four rounds of independent external review were conducted prior to submission. The reviewer was an independent sub-agent given no specific hints about what to look for. Each round's findings were addressed by the parent agent, and the next round verified the fixes.

| Round | Findings | Resolved | File |
|---|---|---|---|
| Round 1 (initial review) | 3 critical + 6 major + 11 minor = **20** | **20/20** | [`review_industry_standards.md`](./review_industry_standards.md) |
| Round 2 (after first fix wave) | 17 verified fixed + 2 partial + 4 new = **23** | **23/23** | [`review_after_fixes.md`](./review_after_fixes.md) |
| Round 3 (after second fix wave) | 0 new (after prose tightening) | — | [`review_round3.md`](./review_round3.md) |
| Round 4 (final spot-check on I-2) | **Definitive: I-2 already fixed** | — | [`review_i2_final.md`](./review_i2_final.md) |
| **Total** | **43 findings** | **43/43** | — |

The four review files document every issue raised, the fix applied, and the verification step. They are available to reviewers on request.

---

## 5. NeurIPS Response Letter

A point-by-point response letter anticipating the most likely reviewer comments.

| File | Role |
|---|---|
| [`response_letter_draft.md`](./response_letter_draft.md) | 22 point-by-point responses (R1–R22), 171 lines, ready for adaptation after actual reviews arrive |

The letter is structured as:
- **Preamble** — summary of five categories of revision (scope / terminology / innovations / related work / honesty) plus reproducibility + pre-submission review
- **R1–R22** — point-by-point responses addressing anticipated reviewer concerns
- **Summary of Major Changes** — one-paragraph camera-ready summary
- **Closing** — acknowledgements and a "we are happy to" offer for further renames or in-paper experiments

---

## 6. Project-Level Reproducibility

The paper is a thin slice over a much larger heterogeneous-fusion research project. Reproducing all numbers requires the project's full toolchain.

### 6.1 Single-experiment reproducibility

```bash
cd <repo>
bash run.sh v6          # Fusion baseline
bash run.sh v7a         # Cross-architecture
bash run.sh v9_gate_layer  # Gate-style routing (C1)
bash run.sh v33         # 4-mode alignment ladder (C2)
bash run.sh v37         # Cross-sublayer routing (C4)
bash run.sh test        # 273 unit and integration tests
```

Each `bash run.sh v{N}` command runs one v-experiment with 5 seeds.

### 6.2 Self-containment check

```bash
cd <repo>
python3 tools/check_self_containment.py
```

This script enforces the 29/29 cross-research-line independence rule; no research line may import from another research line except through `research/_primitives/`.

### 6.3 Model weights

**全部公开可下,无需申请授权。** 路径解析见 `research/_primitives/scale_hetero_loaders/paths.py`:
三级回落 **环境变量 → 现存本地候选 → HuggingFace Hub 自动下载**,不依赖任何机构内部布局。

| Model | Source | 用于 |
|---|---|---|
| TinyBERT | `huggingface.co/google/bert_4_L_4_H-512` | §3 融合实验的底座 |
| TinyLlama-110M | `huggingface.co/jkeisling/tinyllama-110M` | §3 融合实验的底座 |
| ViT-tiny | `huggingface.co/facebook/vit-tiny-patch16-224` | §3 融合实验的底座 |
| CLIP ViT-B/32 | `huggingface.co/openai/clip-vit-base-patch32` | §5 多模态教师 |
| MSA-4B | `huggingface.co/EverMind-AI/MSA-4B` | §6.3 规模异构的大底座 |
| skillcorpus-embedding-0.6b | `huggingface.co/EverMind-AI/skillcorpus-embedding-0.6b` | §6.3 小底座 |
| skillcorpus-reranker-0.6b | `huggingface.co/EverMind-AI/skillcorpus-reranker-0.6b` | §6.3 小底座 |
| TabLDM 分类头 | `huggingface.co/occams/Xiaomi-TabLDM` (`clf_default.ckpt`) | §6.3 表格入口 |

⚠️ **一个容易踩的坑**:TabLDM 的**权重**在 HuggingFace,但它的 **Python 包源码在 GitHub**
`xiaomi-research/xiaomi-tabldm` —— **两者不是一个仓库**。只看 HF 会发现"零个 `.py`"并误判成
"无可安装源码"。`paths.py` 已按「env → 本地候选 → 自动浅克隆」处理,克隆目标放
`$XDG_CACHE_HOME/hetero_fusion/`(**第三方 git 仓库不要 clone 进另一个仓库里**,嵌套 `.git` 会被父仓库吞掉)。

---

## 7. AI-Assistance Disclosure

This paper was drafted with AI assistance. Specifically:

- **Drafting**: prose tightening, terminology cleanup, BibTeX curation, related-work synthesis.
- **Independent review**: four rounds of review conducted by an external reviewer sub-agent, whose findings were integrated and verified.
- **No AI generation of experimental results**: all numbers come from the project's committed scripts; AI was not used to generate or modify empirical results.

The disclosure is also stated in §AI-assistance disclosure of the paper itself.

---

## 8. Self-Citation

```bibtex
@article{hetero_fusion_2026,
  title  = {Reconstructing Frozen Heterogeneous Pretrained Models:
            Sublayer Fusion, Coordinated Routing, and Chimera Distillation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review. Source to be released upon acceptance.}
}
```

⚠️ **2026-10-12**:条目已随合并稿改标题 —— 旧标题 *Heterogeneous Micro-Fusion: … at the
Feed-Forward Sublayer* 属 `microfusion` 稿,不要再用。**venue 仍是待决项**(NeurIPS 9 页装不下 23/24 页)。

---

## 9. File Manifest

```
docs/research_paper/
├── PAPER_README.md                 (this file)
│
├── reconstruct.tex                 ★ 当前英文源, 726 行
├── reconstruct_zh.tex              ★ 当前中文源, 715 行
├── reconstruct.bib                 ★ BibTeX, 43 条
├── reconstruct.pdf                 ★ 英文 PDF, 448 KB, 23 页
├── reconstruct_zh.pdf              ★ 中文 PDF, 816 KB, 24 页
├── reconstruct.aux/.out/.log       构建产物
├── MERGED_OUTLINE.md               合并大纲: 来源、章节结构、关键改动表
│
├── microfusion.tex / _zh.tex       已废弃(顶部有 DEPRECATED 横幅)
├── paper.tex                       已废弃(顶部有 DEPRECATED 横幅)
├── microfusion.bib / .pdf / _zh.pdf  旧产物, 勿引用
│
├── related_work_notes.md           相关工作阅读笔记 batch 1
├── related_work_notes_batch2.md    batch 2
├── related_work_notes_batch3.md    batch 3
├── microfusion_review.md           内部评审
├── microfusion_peer_review.md      同行评审(外部)
├── microfusion_revision_plan.md    修订计划
├── review_industry_standards.md    独立评审 round 1
├── review_after_fixes.md           独立评审 round 2
├── review_round3.md                独立评审 round 3
├── review_i2_final.md              独立评审 round 4
├── response_letter_draft.md        NeurIPS 回复信草稿(R1–R22)
├── checklist.tex                   NeurIPS 投稿检查表
├── ZOTERO_SETUP.md                 Zotero 导入设置
├── build_paper.py                  备用 PDF 构建器(ReportLab)
├── build/  figs/  figs_zh/         构建产物与静态图
```

★ = 当前论文事实以这五个文件为准。

---

## 10. Contact and Contributions

This paper documents the contributions of the **Hetero-MoE** project (GitHub repo `hetero-moe`; Python package `hetero_fusion`, kept for backward-compatible imports). The project-level structure (research lines, primitives, archive) is documented at `<repo>/README.md`.

For questions about a specific research line, see the per-line README at `<repo>/research/{line}/README.md`.

---

*Last updated: 2026-09-20. This README will be updated as the paper progresses through submission and review.*