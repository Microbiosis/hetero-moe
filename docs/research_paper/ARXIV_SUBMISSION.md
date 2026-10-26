# arXiv 提交清单（预印本，不做会议/期刊投稿）

> **决定（2026-10-26）**: 本工作**不做 TMLR/JMLR 正式投稿**, 以 **arXiv 预印本**形式公开。
> 因此 TMLR 的 TL;DR / 伦理声明 / 审稿清单**不再需要**; 只需源包 + 元数据。
> TMLR 版式仍保留在 `reconstruct_tmlr.tex`（万一以后要投, 随时可用）。

---

## 1. 上传物

| 文件 | 说明 |
|---|---|
| **`docs/research_paper/arxiv_package.tar.gz`** | **上传这个**（0.28 MB, 10 个文件）。arXiv 收 LaTeX 源并用自己的 TeX Live 重编译 → 源必须自包含。 |
| `docs/research_paper/arxiv/` | 上传目录的展开形态（`main.tex` 在顶层, arXiv 默认认它）; 打包工具从这里取文件。 |
| `docs/research_paper/arxiv/main.pdf` | 预览用（**24 页含参考文献**, 全矢量, 四张图齐全）。arXiv 会用自己的编译器重新生成, 这里的只作核对。 |

包内: `main.tex` · `main.bbl`（**预生成**, arXiv 不必跑 bibtex）· `reconstruct.bib` ·
`tmlr.sty/.bst` · `fancyhdr.sty` · `figs/{fig_alignment,fig_distill}.pdf` ·
`figs/mermaid/{heterogeneity_spectrum,alignment_ladder}.pdf`

**打包/自检**（会复制到一个只含包内文件的干净目录里真编译一遍）:
```bash
python3 tools/make_arxiv_package.py        # 重新打包 + 干净目录试编译
# → "干净目录试编译: 通过 ✓, 24 页"
```
打包工具会: 把 `reconstruct_arxiv.tex` 复制成 `arxiv/main.tex`（单一真源, 不留两份漂移）→
在**只含包内文件**的临时目录里跑 `pdflatex → bibtex → pdflatex ×2` → 把生成的
`main.bbl`/`main.pdf` 回写 `arxiv/` → 打 tar.gz。

> ⚠️ 之前汇报的"21 页"是**只跑了一遍 pdflatex、没有参考文献页**的中间产物;
> 正式长度是 **24 页**(含参考文献), 与论文 README 的 23/24 页一致。

---

## 2. 表单要填的

**Title**
> Reconstructing Frozen Heterogeneous Pretrained Models: Sublayer Fusion, Coordinated Routing, and Chimera Distillation

**Abstract**（arXiv 上限 1920 字符; 下面这版 1914 字符, 论文正文里的长版 2815 字符保留不动）

> Combining heterogeneous pretrained models---bidirectional encoders, causal decoders and
> vision transformers that differ in architecture, hidden width and modality---usually requires
> joint retraining, surgery or lossy ensembling. We study how such models can be reconstructed
> while every base stays frozen, in three steps. (i) Sublayer fusion: a shared sparse Top-K router
> selects one of M frozen architectural experts per token; semantic aligners map mismatched hidden
> spaces into one shared routing space; experts are fused residually with per-expert gates. Across
> four aligner designs we measure a strict but saturating ladder (linear < mean <
> cross-architecture attention < mixture), and a capacity-matched control shows the ceiling is a
> matter of design, not parameter count. (ii) Coordinated routing: we prove that modulating
> the softmax temperature (gate-style) is the only one of four hub modes satisfying perfect
> degeneration, order preservation, boundedness and low overhead at once, and diagnose tanh
> saturation severing the hub's gradient. (iii) Independent-model reconstruction: chimera
> distillation collapses the ensemble into a single-base student, and a weight scan shows the
> apparent "student beats teacher" margin is a direct-target term. A uniform out-of-sample
> discipline overturns several of our own earlier findings. Auditing the projection space itself,
> we find the advertised fusion margin is largely a source-identification effect inside a
> never-optimized shared space: it is unchanged when the targets are permuted, collapses once the
> space is learned, and in a learned space the bases' outputs for the same text agree far more
> (cosine +0.54) than in a random space (+0.01). The defensible claim is weaker: a shared space
> must be meaningful for heterogeneous fusion to work at all; once it is, fusing heterogeneous
> bases adds little over using one well.

**Categories**: Primary `cs.LG` (Machine Learning) · 也可选 `cs.CL` (Computational Linguistics)、
`cs.AI`（多模态/异构模型协作）

**License**: `CC BY 4.0`（arXiv 永久许可的默认值）

**Comments**（可选, 建议写）:
> 21 pages, 4 figures. Includes three independent re-derivations of the projection-space audit
> (independent data splits, three probe families, three lifting architectures, disjoint seeds and
> widths) and the associated negative results.

**Subjects / Report number**: 留空; 若之前有旧编号（v1→v2）填旧的。

---

## 3. 唯一还没填的: 作者

arXiv **会显示作者**, 所以 `reconstruct_arxiv.tex` 里的占位必须替换:

```latex
\\author{
    \\name <真实姓名>\\\\
    \\addr <单位/机构>\\\\
    \\email <邮箱>
}
```
(改完重新 `python3 tools/make_arxiv_package.py` 重新打包, 并重编译一次核对首页。)

TMLR 匿名评审版 (`reconstruct_tmlr.tex`) 里的作者块是宏包自动隐藏的, 不用动。

---

## 4. 提交步骤

1. 登录 arxiv.org → "Submit a new paper" → 选 **LaTeX** 上传。
2. 上传 `docs/research_paper/arxiv_package.tar.gz`（或网页版直接把 .tex 与 figs 拖进去）。
3. 填上面的 Title / Abstract / Categories / License。
4. 提交后 arXiv 会用它自己的 TeX 编译并给出预览; **逐页核对**: 首页作者、四张图、表 3 的数字、有没有 overfull 警告。
5. 拿到编号后 (e.g. `arXiv:2606.xxxxx`), 在仓库里补一行引用。

## 5. 现状

- ✅ 源包自包含, 干净目录试编译通过, **24 页 0 错**
- ✅ arXiv 版已去掉 "Under review as submission to TMLR" 页眉与全部 TMLR 字样
- ✅ `main.bbl` 预生成 (arXiv 不必跑 bibtex, 少一个失败点)
- ✅ 图全部矢量 (内嵌位图 0 个)
- ✅ 摘要已压到 1868 字符 (arXiv 上限 1920)
- ⛔ **作者真名/单位/邮箱待填**
- ⛔ 需要你的 arXiv 账号完成最后上传
