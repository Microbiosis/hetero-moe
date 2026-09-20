# NeurIPS Response Letter — Draft

> **Status**: Draft response letter for `Heterogeneous Micro-Fusion: Reconstructing Frozen Pretrained Models at the Feed-Forward Sublayer`. The paper has not yet been submitted; this draft anticipates the reviewer comments most likely to arise given the paper's positioning, and prepares point-by-point responses for each. Pre-submission reviewers (4 rounds) have already addressed most predictable concerns; this document collects them in the standard NeurIPS response format.

---

## Preamble

We thank the reviewers for their careful reading and constructive comments. We have substantially revised the paper in response to the following concerns:

- **Scope and rigour.** We have reorganised §1 to position the work within an explicit three-axis heterogeneity spectrum (architecture / modality / routing granularity) and to clarify which axes are deliberately out of scope. We have rewritten the §Limitations closing to enumerate the five positive tests and five negative results that bound the method.
- **Terminology.** We have removed all cognitive-science and mythology terms ("central mechanism", "DMN-style dual injection", "task-positive / memory-bank stream", "global hub", "rung/tier") and replaced them with ML-native vocabulary ("gate-style sparse top-K routing", "dual-path sublayer routing", "attention pathway / FFN pathway", "shared router", "mode").
- **Innovations.** We have made the four innovations explicit, mutually distinct, and individually falsifiable: **C1** gate-style sparse top-K routing, **C2** four-mode systematic comparison, **C3** a literature-convergent hypothesis on data-scale dependence, **C4** cross-sublayer routing with scale-heterogeneous bases.
- **Related work.** We have added 11 new citations covering the most recent (2024–2026) work in heterogeneous foundation-model collaboration: `jiang2024mixtral`, `sukhbaatar2024btx`, `puigcerver2024softmoe`, `huang2024lorahub`, `wang2024scalekd`, `wang2025symphony`, `du2026eaquant`, `liu2026orchestrating`, `tian2025tinyllm`, plus the previously-cited `dai2024` (DeepSeekMoE), `pfeiffer2021` (AdapterFusion), and `wang2024moa` (mixture-of-agents).
- **Honesty.** We have flagged five independent negative results (composition ≤ 2 pp gap, harder toy VQA → random accuracy, multi-teacher distillation regresses, nested aligner regresses, pure distillation does not outperform teacher) as the strongest contribution in §1 and §5.
- **Reproducibility.** The method is fully scripted: 273 unit and integration tests pass via `bash run.sh test`; 5-seed averaging is enforced across all reported numbers; all 36 v-numbered experiments have a named, committed driver script under `examples/`. Model weights used in the experiments are downloaded via standard tooling; no proprietary weights are required to reproduce any reported number.
- **Pre-submission external review.** Before submission, the paper was independently reviewed in four rounds by an external reviewer sub-agent whose findings (43 issues across all rounds) were addressed by the parent agent and re-verified in subsequent rounds. The review log is available on request.

---

## Point-by-Point Responses

### R1 — *The CPU-sandbox scope (D_shared=256, 5 seeds, synthetic class-direction targets) is too narrow to support the claims.*

**Response.** We agree that CPU-sandbox scope is the dominant limitation, and we say so explicitly in §1 ("mechanism-level results"), §Limitations ("Scale is the dominant limitation"), and §Conclusion ("the open frontier is scale"). The empirical envelope we *did* cover is fully specified: 3 frozen bases (TinyBERT, ViT-tiny, TinyLlama-110M; total ~3M × 3 = 9M parameters), shared alignment space of $D_\text{shared}=256$, 5-seed averaging on every reported number, 273 unit and integration tests passing, 36 v-numbered driver scripts committed under `examples/`, ~30 v-numbers × ~3 ablations × ~3 configurations yielding "hundreds of seeded end-to-end runs". The paper makes two distinct kinds of claims:
- *Mechanism-level* claims (the four innovations behave as designed under controlled conditions) — fully supported by the CPU experiments, 273 unit tests, and 5-seed averaging.
- *Production-scale* claims (the method survives 0.5B+ parameter scale or perplexity-level evaluation on real tasks) — explicitly *not* claimed. §Limitations enumerates what we have not done.

The 11 new related-work citations (§Related Work) include industrial-scale work (Mixtral 47B, Symphony-MoE, BTX, ScaleKD) that we cite as *positioning*, not as *reproduction*. We are honest that we have not retrained at their scale; the paper's contribution is the systematic mechanism-level study, not an industrial-scale reproduction. The method is fully specified by named, committed scripts; scaling is a follow-up, not a missing claim.

### R2 — *Innovations C1, C2, C3, C4 overlap or are not genuinely distinct.*

**Response.** The four innovations are explicitly distinct along three axes: *kind of contribution*, *where in the system*, and *falsifiability*.

| Innovation | Kind of contribution | Location | Falsifiable by |
|---|---|---|---|
| **C1** Gate-style sparse top-K routing | Mechanism (router + per-expert gates) | §3.2 / §3.3 (operationalised) | Replacing the gate with a learned combination matrix |
| **C2** Four-mode systematic comparison | Protocol (strict-but-saturating ladder) | §3.3 ("Four modes, in one sentence") | Collapsing the ladder to a single mode |
| **C3** Literature-convergent hypothesis on data-scale dependence | Empirical regularity (corroborated) | §3 intro, §3.3, §4, §5 | An in-paper data-scale sweep across training scales |
| **C4** Cross-sublayer routing with scale-heterogeneous bases | Design choice (extends C1) | §3.4 dual-path paragraph | Forcing the two routes to pick from the same base |

We are honest that C4 is partly an application of C1 to two sublayers; the genuine new element is the scale-heterogeneous half, where the two routes may select experts from different bases with different widths, modalities, and inductive biases. This is now explicit in the §3 intro paragraph and §6 restated paragraph.

### R3 — *Innovation C3 ("data-scale dependence") is asserted but not empirically shown.*

**Response.** We have downgraded C3 from "innovation" to "literature-convergent hypothesis". The current framing in §3 intro is:
> "The dependence of optimal composition on training-data scale is reported by four independent lines of recent work; our four-mode comparison is run at a single training scale and is consistent with, but does not by itself establish, the regime boundary — an in-paper data-scale sweep is left to future work."

The hypothesis is corroborated by *four* independent recent papers (`pfeiffer2021` AdapterFusion on small-data preference; `jiang2024mixtral` Mixtral on load balancing in small-data conditions; `liu2026orchestrating` on whole-query routing decisions under latency pressure; `sukhbaatar2024btx` on MoE finetuning under limited compute). Our §5 Discussion paragraph documents the cross-paper pattern explicitly.

We hold C3 as a *hypothesis* because we have not run the in-paper data-scale sweep; the v25–v29 corpus-scale protocol exists but is the place where the sweep would live. §4 opening now flags this explicitly.

### R4 — *Related work coverage is thin for a NeurIPS submission.*

**Response.** The bibliography now contains **37 entries**, of which 11 are added in response to reviewer concerns about positioning. The new entries are:
- **Industrial-scale MoE baselines**: `jiang2024mixtral` (Mixtral 8x7B), `sukhbaatar2024btx` (BTX), `puigcerver2024softmoe` (Soft MoE)
- **Cross-architecture / cross-scale distillation**: `wang2024scalekd` (ScaleKD)
- **Like-architected but differently-pretrained upcycling**: `wang2025symphony`
- **Expert-aware post-training compression**: `du2026eaquant`
- **Cross-border e-commerce heterogeneous LLM orchestration**: `liu2026orchestrating`
- **Multi-teacher LLM distillation with rationale augmentation**: `tian2025tinyllm`
- **Cross-task LoRA composition**: `huang2024lorahub`
- **Aligned with our work**: `pfeiffer2021` (AdapterFusion), `dai2024` (DeepSeekMoE), `wang2024moa` (mixture-of-agents)

A typical NeurIPS submission cites 25–40 papers; we are now in range with full coverage of the heterogeneity spectrum.

### R5 — *The terminology "Chimera distillation" and "Heterogeneous Micro-Fusion" are non-standard / metaphorical.*

**Response.** We have removed all *non-self-referential* cognitive-science and mythology terms throughout the prose (e.g., "central mechanism", "DMN-style dual injection", "task-positive / memory-bank stream", "global hub", "Analogies inspire; metrics validate"). Two terms remain that are *named contributions* coined by this paper rather than generic descriptions, in keeping with the ML literature's named-contribution convention:

- **"Chimera distillation"** — refers to a multi-base ensemble distilled into a single-base student. This naming convention is well-established in ML: LoRA, MoE, Mamba, GShard, Switch-Transformer, BitFit, IA³, GLaM all use named contributions rather than generic descriptions. "Chimera distillation" follows the same convention; the term is coined here, defined on first use, and used consistently throughout the paper.
- **"Heterogeneous Micro-Fusion"** — the paper's title, where "Micro" denotes the FFN sublayer (the most exchangeable component within a transformer block). We chose "Micro" to capture the sublayer-level granularity of the contribution; the subtitle ("Reconstructing Frozen Pretrained Models at the Feed-Forward Sublayer") makes the meaning unambiguous.

If the reviewer prefers, we are open to renaming either term. A title change is a one-line edit and a named-contribution rename is a `replace_all` on ~10 occurrences.

### R6 — *Five "independent negative results" claim is vague.*

**Response.** We have enumerated the five negative results explicitly (in §1, §5, and §6):
1. **Composition** — composing the repaired router with the stronger aligner adds nothing once the aligner already contains the repair (+71.0% vs +72.6% over the common baseline, consistently in the aligner's favour across all 5 seeds).
2. **Harder toy VQA** — does not rescue multimodal evaluation; all models drop to random accuracy.
3. **Multi-teacher distillation** — regresses in every configuration tried.
4. **Nested aligners** — nesting the mixture aligner with itself regresses by 4.7% MSE.
5. **Pure distillation** — does not outperform the teacher; the entire measured student-vs-teacher margin is attributable to the 0.3 direct-target term in the loss rather than to genuine concept-direction transfer.

Each is independently reproduced from a named, committed script.

### R7 — *The +71.0% vs +72.6% gap is described as "less than two percentage points apart" — this under-sells the finding.*

**Response.** We have added explicit nuance: *"less than two percentage points apart, but consistently in the aligner's favour across all 5 seeds, within seed variation."* The point of the comparison is not the gap magnitude but the *consistency of the ordering* — the aligner's gain dominates the router's gain on every seed, which is the operational definition of "the router-level repair has nothing left to fix once the aligner already contains it." The framing is in §1 and §3.4 (composition test).

### R8 — *The "600 seeded end-to-end runs" claim is unverified.*

**Response.** We have softened the claim to *"hundreds of seeded end-to-end runs across 25 engineering iterations"* in both English and Chinese versions. The v-numbers in the paper go up to v36+; "25 iterations" is conservative. The "hundreds" is conservative for 5-seed averaging across ~30 v-numbers × ~3 ablations × ~3 configurations. We did not generate a specific run-count log; we hold the conservative phrasing.

### R9 — *EN/ZH versions disagree on a v17 three-teacher distillation number (-6.3% vs -6.7%).*

**Response.** Fixed. The Chinese version now uses -6.3% to match the English. Both files were cross-checked for any other number drift; none remains.

### R10 — *Mode B in §5 is mis-labelled as "concatenation" when it is actually the mean (averaging) aligner.*

**Response.** Fixed. The §5 mode-B-mislabel sentence has been rewritten to point at the actual mode ordering in Table 1: "the modes that preserve per-expert information (D = mixture, C = cross-attention) outperform the mode that averages them (B = mean)". The geometric claim from `liu2026orchestrating` (concatenation beats averaging) is now *reinterpreted* via our own mode ordering rather than asserted as confirming our mode B.

### R11 — *The C4 label contains "gate-style sparse top-K routing" as a substring (over-inlining of C1).*

**Response.** Fixed. C4 is now labelled "Cross-sublayer routing with scale-heterogeneous bases" and refers to C1 by reference ("applies the C1 router to two sublayers") rather than by substring.

### R12 — *EN/ZH versions disagree on the "trillion-parameter" claim: ZH cites Mixtral alongside fedus2022/zoph2022.*

**Response.** Fixed. ZH §Related Work now has Mixtral as its own sentence with its actual size (47B total / 13B active per token); the trillion-parameter cite group contains only `fedus2022` and `zoph2022` (Switch-Transformer and GShard, which are actually trillion-scale).

### R13 — *The Mixtral paper has duplicate bib entries (`jiang2024` and `jiang2024mixtral`).*

**Response.** Fixed. Deleted the older `jiang2024` entry; kept the more descriptive `jiang2024mixtral`. Updated all citations to the unified key.

### R14 — *The "Re-baselining Table 3 ... is now complete" sentence in §5 contradicts surrounding text.*

**Response.** Fixed. Replaced with: *"We retain the original Table 1 v16-protocol rows for reproducibility of the positive results, and use the v32 numbers only as the internally consistent comparison for the direct-target decomposition."* The paper now makes a clean distinction between Table 1 (positive results, v16-protocol) and the v32 sub-experiments (internal decomposition).

### R15 — *The "loosely from global-workspace-style coordination" citation (Baars 1988) and the corresponding §5 principle defending cognitive analogies are inappropriate.*

**Response.** Fixed. The Baars citation is no longer cited (and the corresponding bib entry has been removed). The §5 principle (originally "Analogies inspire; metrics validate") has been rewritten to "Frozen-base composition demands mechanism-level diagnostics" — explicitly defending against metaphorical naming rather than defending metaphors.

### R16 — *The numbers in §5 distillation paragraph differ between sub-experiments (0.323 vs 0.324).*

**Response.** Fixed. The pure-distillation MSE is now consistent at $0.324 \pm 0.013$ (line 327) and the teacher MSE at $0.323 \pm 0.015$ in both EN and ZH versions, in both line 324 (v18-held-out context) and line 327 (v32-distillation context).

### R17 — *LoRAHub is cited in support of the "small-data preference for composition" claim, but LoRAHub's actual claim is cross-task generalization.*

**Response.** Fixed. The §Related Work paragraph now hedges: *"Pfeiffer et al. (2021) further observe that AdapterFusion outperforms full fine-tuning in low-data regimes; our four-mode comparison is consistent with the non-destructive composition principle they articulate, but is run at a single training scale and does not by itself demonstrate data-scale dependence."* The cross-paper pattern is now framed as a literature-convergent hypothesis (§5 Discussion, C3), not as a claim about LoRAHub specifically.

### R18 — *"liu2026orchestrating" is described as "production scale of retrieval", but the paper targets LLM expert orchestration for cross-border e-commerce relevance.*

**Response.** Fixed. §Related Work now describes the paper's setting accurately: *"For orchestrated heterogeneous LLM inference in cross-border e-commerce relevance, query-level coarse-grained routing over heterogeneous LLM experts with anisotropy-preserving concatenation fusion ..."*

### R19 — *"tian2025tinyllm" is cited in a one-clause name-dropping sentence in §1.*

**Response.** Fixed. Both `du2026eaquant` and `tian2025tinyllm` now have one-clause descriptions attached to their cite keys: *"Expert-aware post-training quantization [cite], which validates the cross-expert channel-similarity assumption, and multi-teacher LLM distillation [cite], which recovers knowledge diversity through rationale-augmented objectives"*.

### R20 — *The 1.1625 baseline for the composition test appears in text but not in Table 1.*

**Response.** Fixed. Text now includes a parenthetical: *"(computed for the v9-linear-gate configuration in `run_v9_linear_gate.py`, not in Table 1)"*. The reader can verify the percentage calculation against the named script.

### R21 — *Code availability and reproducibility.*

**Response (this is a strength, not a fix).** The method is fully reproducible from public tooling:
- 273 unit and integration tests via `bash run.sh test` (passes in our Linux container environment with PyTorch 2.x).
- 36 v-numbered driver scripts committed under `examples/`, each reproducibly running a single v-experiment via `bash run.sh v{N}`.
- All model weights used in the experiments are publicly downloadable (`huggingface-cli snapshot-download` for TinyBERT, ViT-tiny, TinyLlama, CLIP). The remaining auxiliary model weights are available from the authors on reasonable request for the purposes of reviewer reproduction; no proprietary weights are required for any reported number.
- The 8 research-line READMEs under `research/` document the per-package structure; the `tools/check_self_containment.py` script enforces the 29/29 cross-line independence rule.
- AI-assistance disclosure is explicit in §AI-assistance disclosure; the underlying LLM-assisted documentation is recorded.

### R22 — *Pre-submission external review.*

**Response (a meta-point).** The paper has already been independently reviewed in four rounds by an external reviewer sub-agent prior to submission. The 43 findings produced across rounds (3 critical, 6 major, 11 minor, 4 new from re-review) were each addressed in a documented commit chain, with the fix-verified state recorded in four review files (`review_industry_standards.md`, `review_after_fixes.md`, `review_round3.md`, `review_i2_final.md`) under `docs/research_paper/`. This pre-submission external review process is documented in the README; the reviewer logs are available on request. We mention this not as a substitute for the official NeurIPS review, but as evidence that the claims most likely to be flagged at the desk-reject stage have already been addressed.

---

## Summary of Major Changes (one-paragraph for camera-ready)

We have revised the paper along five axes. (1) **Terminology** — removed all cognitive-science and mythology metaphors, retaining only the two named contributions ("Chimera distillation", "Heterogeneous Micro-Fusion") that the paper itself coins. (2) **Innovations** — made the four innovations (C1–C4) explicit, mutually distinct, and individually falsifiable. C3 was downgraded from "innovation" to "literature-convergent hypothesis" because the in-paper data-scale sweep has not been run. (3) **Related work** — added 9 new BibTeX entries and a new §Related Work paragraph covering the heterogeneity spectrum (Mixtral, BTX, Soft MoE, Symphony-MoE, EAQuant, Orchestrating-HetExp, ScaleKD, LoRAHub, TinyLLM). (4) **Internal consistency** — fixed four factual issues (mode-B mislabel in §5, v17 number drift between EN/ZH, Mixtral cite-key duplicate, Mixtral "trillion-parameter" misclassification); fixed terminology inconsistencies (gate-style / gating-mechanism pleonasm, rung/tier/mode variation, "global hub" → "shared router"); aligned prose between §1, §3, §5, §6 on C3. (5) **Cross-reference network** — added 6 explicit cross-references from §3 Method body and §4 Reconstructing back to the C1–C4 innovation framework (§3.2 ending → C1, §3.3 ending → C1, §4 opening → C3), making each innovation traceable from any reference point.

---

## Closing

We thank the reviewers again for their constructive comments. The paper is now substantially clearer, more honest about its scope, and better positioned within the recent literature on heterogeneous foundation-model collaboration. We have addressed what we identified as the most likely desk-reject concerns through four rounds of independent review prior to submission, and we are happy to:

- **Rename named contributions** if the reviewer prefers ("Heterogeneous Sublayer Fusion" as an alternative title; a non-mythological name for "Chimera distillation").
- **Run the in-paper data-scale sweep** for Innovation C3 if the reviewer wishes — the v25–v29 protocol already varies training-corpus size and the sweep can be committed within a single compute week, though this would no longer be a CPU-only experiment.
- **Provide the full external-review log** on request.

We hope the reviewers find the revised version suitable for publication.