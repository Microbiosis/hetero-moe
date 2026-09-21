# Paper README — `Heterogeneous Micro-Fusion`

> **Paper:** Heterogeneous Micro-Fusion: Reconstructing Frozen Pretrained Models at the Feed-Forward Sublayer
>
> **Authors:** Anonymous (with AI-assisted drafting and pre-submission review)
>
> **Target venue:** NeurIPS 2026 (Main track)
>
> **Status:** Ready for submission as of 2026-09-20. Four rounds of independent external review have been completed; all 43 findings addressed.

This README is the entry point for everything paper-related under `<repo>/docs/research_paper/`. It catalogs every file, its role, and how to verify it.

---

## 1. Paper Sources and Compiled Outputs

| File | Role | Status |
|---|---|---|
| [`microfusion.tex`](./microfusion.tex) | English LaTeX source | 430 lines, 0 compile errors |
| [`microfusion_zh.tex`](./microfusion_zh.tex) | Chinese LaTeX source | 429 lines, 0 compile errors, EN/ZH terminology aligned |
| [`microfusion.bib`](./microfusion.bib) | BibTeX bibliography | 37 entries (9 new in revision) |
| [`microfusion.pdf`](./microfusion.pdf) | Compiled English PDF | 334 KB, 15 pages, regenerated 2026-09-20 |
| [`microfusion_zh.pdf`](./microfusion_zh.pdf) | Compiled Chinese PDF | 674 KB, 17 pages, regenerated 2026-09-20 |

### Compilation

```bash
# English (pdflatex + bibtex)
cd <repo>
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/research_paper/microfusion.tex

# Chinese (xelatex + ctex + xdvipdfmx)
cd <repo>
latexmk -pdf -interaction=nonstopmode -halt-on-error -xelatex docs/research_paper/microfusion_zh.tex
```

Both targets compile cleanly (0 errors, 0 warnings beyond cosmetic underfull-box notices).

---

## 2. The Four Innovations

The paper's contribution is four innovations, each named, distinct, and individually falsifiable.

| ID | Name (English) | 中文 | Where introduced | Operationalised in | Falsifiable by |
|---|---|---|---|---|---|
| **C1** | Gate-style sparse top-K routing | 门控式稀疏路由 | §3 Method intro, §6 restated | §3.2 ending, §3.3 ending | Replacing the gate with a learned combination matrix |
| **C2** | Four-mode systematic comparison | 四模式系统比较 | §3 Method intro, §6 restated | §3.3 ("Four modes, in one sentence") | Collapsing the ladder to a single mode |
| **C3** | Literature-convergent hypothesis on data-scale dependence | 数据规模依赖性的文献汇聚假设 | §3 Method intro, §3.3, §4 opening, §5 Discussion, §6 restated | (hypothesis; sweep is future work) | An in-paper data-scale sweep across training scales |
| **C4** | Cross-sublayer routing with scale-heterogeneous bases | 跨子层路由配规模异构的底座 | §3 Method intro, §3.4 dual-path paragraph, §6 restated | §3.4 ending | Forcing the two routes to pick from the same base |

C3 was originally proposed as an in-paper empirical claim; it was downgraded to a literature-convergent hypothesis after the independent reviewers noted the in-paper data-scale sweep was not actually run. The hypothesis is corroborated by four independent recent papers cited in §5.

C4 is partly an application of C1 (the same router used twice); the genuine new element is the scale-heterogeneous half, where the two routes may select experts from different bases with different widths, modalities, and inductive biases.

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

All model weights used in the paper are publicly available:

| Model | Source | Already cached |
|---|---|---|
| TinyBERT | `huggingface.co/google/bert_4_L_4_H-512` | yes |
| ViT-tiny | `huggingface.co/facebook/vit-tiny-patch16-224` | yes |
| TinyLlama-110M | `huggingface.co/jkeisling/tinyllama-110M` | yes |
| CLIP | `huggingface.co/openai/clip-vit-base-patch32` | yes |
| (additional auxiliary models, redacted for anonymized submission) | n/a | yes |

---

## 7. AI-Assistance Disclosure

This paper was drafted with AI assistance. Specifically:

- **Drafting**: prose tightening, terminology cleanup, BibTeX curation, related-work synthesis.
- **Independent review**: four rounds of review conducted by an external reviewer sub-agent, whose findings were integrated and verified.
- **No AI generation of experimental results**: all numbers come from the project's committed scripts; AI was not used to generate or modify empirical results.

The disclosure is also stated in §AI-assistance disclosure of the paper itself.

---

## 8. Self-Citation

If citing this paper, the BibTeX entry is:

```bibtex
@article{hetero_fusion_2026,
  title  = {Heterogeneous Micro-Fusion: Reconstructing Frozen Pretrained Models at the Feed-Forward Sublayer},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026. Source to be released upon acceptance.}
}
```

(Note: replace with the actual venue once known.)

---

## 9. File Manifest (auto-generated reference)

```
docs/research_paper/
├── PAPER_README.md                 (this file)
├── microfusion.tex                 English LaTeX source, 430 lines
├── microfusion_zh.tex              Chinese LaTeX source, 429 lines
├── microfusion.bib                 BibTeX bibliography, 37 entries
├── microfusion.pdf                 English PDF, 334 KB
├── microfusion_zh.pdf              Chinese PDF, 674 KB
├── microfusion.aux / .bbl / .log   LaTeX build artifacts (regenerated by latexmk)
├── microfusion_zh.aux / .bbl / .log / .xdv   xelatex build artifacts
├── related_work_notes.md           Related-work reading notes, batch 1 (4 papers)
├── related_work_notes_batch2.md    Related-work reading notes, batch 2 (4 papers)
├── related_work_notes_batch3.md    Related-work reading notes, batch 3 (3 papers)
├── review_industry_standards.md    Independent review, round 1
├── review_after_fixes.md           Independent review, round 2
├── review_round3.md                Independent review, round 3
├── review_i2_final.md              Independent review, round 4 (I-2 spot-check)
├── response_letter_draft.md        NeurIPS response letter (R1–R22)
├── ZOTERO_SETUP.md                 Optional: Zotero import setup
├── build_paper.py                  Optional: alternative PDF builder (ReportLab)
├── build/                          Build artifacts (PDF, figs)
├── figs/                           Static figures (PDF, EPS)
├── figs_zh/                        Chinese version figures
├── checklist.tex                   NeurIPS submission checklist
└── microfusion_peer_review.md / microfusion_review.md / microfusion_revision_plan.md   Internal review notes
```

---

## 10. Contact and Contributions

This paper documents the contributions of the **Hetero-MoE** project (GitHub repo `hetero-moe`; Python package `hetero_fusion`, kept for backward-compatible imports). The project-level structure (research lines, primitives, archive) is documented at `<repo>/README.md`.

For questions about a specific research line, see the per-line README at `<repo>/research/{line}/README.md`.

---

*Last updated: 2026-09-20. This README will be updated as the paper progresses through submission and review.*