# arXiv 投稿包 (hetero-moe 论文)

> 准备日期: 2026-10-26 · 对应 commit: 见 git log
> 目标: arXiv 预印本 (TMLR/JMLR 类期刊在审)

## 1. 上传物

| 文件 | 说明 |
|---|---|
| `arxiv_submission.tar.gz` | **上传这个** (源码包, main.tex 在根目录) |
| `arxiv/` | 同样的内容, 便于直接查看 |

包内:
```
main.tex                      ← 由 reconstruct_tmlr.tex 改名 (arXiv 要求主文件在顶层)
reconstruct.bib               ← 参考文献
tmlr.sty / tmlr.bst / fancyhdr.sty   ← TMLR 官方样式 (作者块自动匿名)
main.bbl                      ← 已生成的参考文献 (arXiv 也会自己跑 bibtex, 双保险)
figs/fig_alignment.pdf
figs/fig_distill.pdf
figs/mermaid/alignment_ladder.pdf
figs/mermaid/heterogeneity_spectrum.pdf
```

**已在干净目录做过完整构建** (pdflatex → bibtex → pdflatex ×2): 24 页,
**0 错误 / 0 缺文件 / 0 未定义引用 / 0 绝对路径**, 全部宏包都在 arXiv 的 TeX Live 里。
页眉会自带 "Under review as submission to TMLR" (tmlr 宏包自动加)。

## 2. arXiv 表单要填的元数据

**Title**

Reconstructing Frozen Heterogeneous Pretrained Models: Sublayer Fusion, Coordinated Routing, and Chimera Distillation

**Abstract**

Combining heterogeneous pretrained models---bidirectional encoders, causal decoders, and vision transformers that differ in architecture, hidden width, and modality---usually requires joint retraining, architecture surgery, or lossy output-level ensembling. We study how such models can be \emph{reconstructed} while every base stays frozen, in three steps. \textbf{(i) Sublayer fusion.} A shared sparse Top-$K$ router selects one of $M$ frozen architectural experts per token; semantic aligners map mismatched hidden spaces into one shared routing space; experts are fused residually with per-expert gates. Across four aligner designs we measure a strict but \emph{saturating} ladder (linear $<$ mean $<$ cross-architecture attention $<$ mixture), and a capacity-matched control shows the ceiling is a matter of design rather than of parameter count. \textbf{(ii) Coordinated routing.} We ask how a global hub should influence the shared router, and prove that modulating the softmax temperature (gate-style) is the only one of four hub modes satisfying perfect degeneration, order preservation, boundedness, and low overhead at once; we also diagnose a hidden failure mode in which tanh saturation severs the gradient to the hub during long training. \textbf{(iii) Independent-model reconstruction.} Chimera distillation collapses the ensemble into a single-base student, and a weight scan over the loss shows that the apparent ``student beats teacher'' margin is entirely attributable to a direct-target term rather than to concept-direction transfer or variance averaging. Because fused inference keeps every base alive and distillation is a deployment device, the study is only as trustworthy as its evaluation. We therefore apply a uniform \emph{out-of-sample discipline} across the whole stack, and it overturns several of our own in-sample findings: the in-sample best hub mode is not the out-of-sample best, an advertised corpus-scale boundary is not established, and long-training and repair gains shrink to noise. What survives held-out evaluation is the alignment-ladder ceiling. The attention-path advantage does not: a projection-space audit (\S\ref{sec:projection-space}) shows it to be a property of a frozen random shared space rather than of architectural heterogeneity, and it disappears once that space is learned. The scale-heterogeneous fusion margin does not: a projection-space audit (\S\ref{sec:projection-space}) shows it to be a source-identification effect inside a shared space that was never optimized, and it does not persist once that space is learned. We report the negative results that bound the method---component composition, nested aligners, multi-teacher distillation, and toy multimodal benchmarks---from hundreds of seeded end-to-end runs across 41 versioned engineering generations and 430 unit and integration tests.

**Categories** (按主题, 建议)
- 主: `cs.LG` (Machine Learning)
- 次: `cs.CL` (Computation and Language) · `stat.ML` (Machine Learning Statistics)

**License**: 选 `CC BY 4.0` (与仓库的 Apache-2.0 并存即可) 或 arXiv 默认的
perpetual non-exclusive license。

## 3. 两个还没定的决定 (需要你拍板)

| 决定 | 选项 | 影响 |
|---|---|---|
| **署名** | (a) **匿名投稿** (保留 `Anonymous Author(s)`, 在 arXiv 勾选 anonymous submission) · (b) **填真名与单位** | (a) 符合双盲惯例, 但 arXiv 上的论文没有作者, 引用不便; (b) 现在就可被引用, 但**等于提前公开作者身份** (TMLR 通常也是双盲评审, 提前公开不违规但会暴露身份) |
| **公告** | (a) **立即公开** (announce) · (b) **暂不公告** (withhold announcement, 等录用后再公开) | (a) 马上可被检索引用; (b) 保留"首发在期刊"的位置, 但 preprint 记录会一直存在 |

## 4. 上传步骤 (5 分钟)

1. 打开 <https://arxiv.org/submit> → `Start a new submission`
2. 选 **Submit a paper**; 若走匿名路径, 在作者栏勾选匿名选项
3. 上传 `docs/research_paper/arxiv_submission.tar.gz`
4. arXiv 会在它的 TeX Live 里编译 (我们的包已在同样环境验证过) → 检查预览 PDF
5. 填上面第 2 节的 Title / Abstract / Categories / License
6. 确认 **announce 或 withhold** (见第 3 节)
7. 提交 → arXiv 会给一封确认邮件 (announce 的话会有发布时间安排)

> ⚠️ 上传的是 **tar.gz 源码包**, 不要只传 PDF —— 只传 PDF 会得到 arXiv 的
> "no source available" 标记, 无法编译。

