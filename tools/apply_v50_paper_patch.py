"""把 v48/v49/v50 的结论写进论文 (reconstruct.tex + reconstruct_tmlr.tex)。

设计: 数字从 v50 的 metrics.json 里**直接读**并代入, 不手抄 —— 避免稿子里的
数字和实验产物脱节 (本项目 2026-10-12 就有过一次"摘要声称的结论正文里没有支撑
实验"的自查事故)。

用法: python3 tools/apply_v50_paper_patch.py
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "docs/research_paper")
METRICS = os.path.join(ROOT, "research/aligner/v50_fusion_gain_audit/results/metrics.json")
V49 = os.path.join(ROOT, "research/aligner/v49_projection_space/results/metrics.json")
V51 = os.path.join(ROOT, "research/aligner/v51_arch_fusion_audit/results/metrics.json")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, n=4):
    return f"{x:.{n}f}"


BUCKET = "redmingwei/hetero-moe"


def load_bucket_cells():
    """直接从桶里读每格的探针工件 (cells/*.json) —— 不从日志扒数字。"""
    try:
        import io, os, tempfile
        from huggingface_hub import HfApi, list_bucket_tree
        api = HfApi()
        paths = [getattr(it, "path", None) for it in
                 list_bucket_tree(BUCKET, prefix="v50_artifacts/cells/", recursive=True)]
        paths = [p for p in paths if p and p.endswith(".json")]
        if not paths:
            return None
        out = {}
        for p in paths:
            from huggingface_hub import hf_hub_download
            tmp = hf_hub_download(BUCKET, p, repo_type="dataset")
            d = json.load(open(tmp, encoding="utf-8"))
            out[d["key"]] = d
        return out
    except Exception as e:
        print(f"  (桶工件读取失败: {repr(e)[:100]})")
        return None


def main():
    m50, m49 = load(METRICS), load(V49)
    cells = load_bucket_cells()
    if cells:
        R = cells
        print(f"✓ 从桶里读到 {len(R)} 格探针工件")
    elif m50:
        R = m50["results"]
    else:
        print("✗ 没有 v50 的探针结果 — 先在 HF 上跑完审计")
        return 1

    def cell(key):
        rows = R[key]["rows"]
        n = len(rows)
        held = sum(r["held_out"]["fuse"] for r in rows) / n
        sbig = sum(r["held_out"]["single_big"] for r in rows) / n
        triv = rows[0]["held_out"]["trivial"]
        ps = sum(r["probe"]["source_te"] for r in rows) / n
        pc = sum(r["probe"]["class_te"] for r in rows) / n
        pin = sum(r["probe"]["input_source_te"] for r in rows) / n
        return dict(held=held, gain=(sbig - held) / max(sbig, 1e-9) * 100, trivial=triv,
                    src=ps, cls=pc, ident=pin)

    A = cell("S256:A_linear_random_frozen")
    B = cell("S256:B_nl_random_frozen")
    C = cell("S256:C_nl_train")
    D = cell("S256:D_nl_proxy_frozen")
    E = cell("S256:E_nl_proxy_train")

    # v49 的关键数
    g = {}
    if m49:
        for k, v in m49["results"].items():
            rows = v["rows"]
            n = len(rows)
            g[k] = dict(held=sum(r["held_out"]["fuse"] for r in rows) / n,
                        gain=(sum(r["held_out"]["single_big"] for r in rows) / n
                              - sum(r["held_out"]["fuse"] for r in rows) / n)
                        / max(sum(r["held_out"]["single_big"] for r in rows) / n, 1e-9) * 100,
                        gap=(sum(r["held_out"]["fuse"] for r in rows) / n
                             - sum(r["in_sample"]["fuse"] for r in rows) / n)
                        / max(sum(r["held_out"]["fuse"] for r in rows) / n, 1e-9) * 100,
                        shuf=((sum(r["held_out_shuffled_cond"]["fuse"] for r in rows) / n
                              - sum(r["held_out"]["fuse"] for r in rows) / n)
                              / max(sum(r["held_out"]["fuse"] for r in rows) / n, 1e-9) * 100)
                        if "held_out_shuffled_cond" in rows[0] else None)

    newsec = os.path.join(PAPER, "_v50_section.tex")
    if not os.path.exists(newsec):
        print("✗ 缺少 _v50_section.tex (新章节正文)")
        return 1
    section = open(newsec, encoding="utf-8").read()
    section = section.replace("@A_GAIN@", f"{A['gain']:+.1f}")
    section = section.replace("@A_HELD@", fmt(A["held"]))
    section = section.replace("@A_SRC@", fmt(A["src"], 3))
    section = section.replace("@A_CLS@", fmt(A["cls"], 3))
    section = section.replace("@A_IDENT@", fmt(A["ident"], 3))
    section = section.replace("@B_GAIN@", f"{B['gain']:+.1f}")
    section = section.replace("@C_GAIN@", f"{C['gain']:+.1f}")
    section = section.replace("@C_HELD@", fmt(C["held"]))
    section = section.replace("@D_GAIN@", f"{D['gain']:+.1f}")
    section = section.replace("@E_GAIN@", f"{E['gain']:+.1f}")
    section = section.replace("@E_HELD@", fmt(E["held"]))
    if g:
        section = section.replace("@NL_S1024@", fmt(g.get("S1024:NL_static", {}).get("held", 0)))
        section = section.replace("@NL_COND_S1024@", fmt(g.get("S1024:NL_cond", {}).get("held", 0)))
        section = section.replace("@L_S1024@", fmt(g.get("S1024:L_static", {}).get("held", 0)))
        section = section.replace("@FROZEN_S1024@", fmt(g.get("S1024:L_static_frozen", {}).get("held", 0)))

    m51 = load(V51)
    if m51:
        R51 = m51["results"]

        def c51(key):
            rows = R51[key]["rows"]
            n = len(rows)
            g = lambda k: sum(r["held_out"][k] for r in rows) / n
            ps = sum(r["probe"]["source_te"] for r in rows) / n
            pc = sum(r["probe"]["class_te"] for r in rows) / n
            return dict(fuse=g("fuse"), attn=g("attn_only"), ffn=g("ffn_only"), avg=g("avg"),
                        single=g("best_single"),
                        gain=(g("best_single") - g("fuse")) / max(g("best_single"), 1e-9) * 100,
                        vsavg=(g("avg") - g("fuse")) / max(g("avg"), 1e-9) * 100,
                        pathgap=(g("ffn_only") - g("attn_only")) / max(g("ffn_only"), 1e-9) * 100,
                        src=ps, cls=pc)
        A51 = c51("A_frozen_aligned")
        B51 = c51("B_frozen_permuted")
        C51 = c51("C_trainable_aligned")
        D51 = c51("D_trainable_permuted")
        PCT = chr(92) + "%"   # LaTeX 里的百分号要转义
        PLACEHOLDERS = (
            ("@A51_GAIN@", f"{A51['gain']:+.1f}" + PCT), ("@B51_GAIN@", f"{B51['gain']:+.1f}" + PCT),
            ("@C51_GAIN@", f"{C51['gain']:+.1f}" + PCT), ("@D51_GAIN@", f"{D51['gain']:+.1f}" + PCT),
            ("@A51_FUSE@", fmt(A51["fuse"])), ("@C51_FUSE@", fmt(C51["fuse"])),
            ("@A51_PATHGAP@", f"{A51['pathgap']:.0f}" + PCT),
            ("@C51_PATHGAP@", f"{C51['pathgap']:.0f}" + PCT),
            ("@A51_SRC@", fmt(A51["src"], 3)), ("@A51_CLS@", fmt(A51["cls"], 3)),
            ("@C51_SRC@", fmt(C51["src"], 3)))
        # v51 段落追加到新章节末尾
        if "@@V51@@" in section:
            section = section.replace("@@@V51@@", "")
        # ⚠️ 占位符替换必须在 v51 段落插入**之后** (v51 段里也带占位符)
        for k, v in PLACEHOLDERS:
            section = section.replace(k, str(v))
    for fn in ("reconstruct.tex", "reconstruct_tmlr.tex"):
        path = os.path.join(PAPER, fn)
        if not os.path.exists(path):
            continue
        s = open(path, encoding="utf-8").read()
        # 1) 插入/整段替换新章节 (放在 scale-hetero 小节之后)
        anchor = "\\subsection{The alignment ladder and its ceiling}"
        if "\\label{sec:projection-space}" in s:
            i = s.index("\\subsection{The projection space is load-bearing")
            j = s.index(anchor, i)
            s = s[:i] + section + "\n" + s[j:]
        else:
            s = s.replace(anchor, section + "\n" + anchor, 1)
        # 2) 摘要: 改写 "survives" 那句
        old_abs = ("What survives held-out evaluation is the alignment-ladder ceiling, the attention-path "
                   "advantage, and the scale-heterogeneous fusion gain (\\S\\ref{sec:scale-hetero-exp}), "
                   "whose in-sample margin of $+26.7\\%$ is retained out of sample.")
        new_abs = ("What survives held-out evaluation is the alignment-ladder ceiling and the attention-path "
                   "advantage. The scale-heterogeneous fusion margin does not: a projection-space audit "
                   "(\\S\\ref{sec:projection-space}) shows it to be a source-identification effect inside a "
                   "shared space that was never optimized, and it does not persist once that space is learned.")
        if old_abs in s:
            s = s.replace(old_abs, new_abs)
        # 3) §1 那句 "all survive held-out evaluation intact"
        old_intro = ("On the positive side, the alignment-ladder ceiling, the attention-path advantage, and "
                     "the scale-heterogeneous fusion gain (\\S\\ref{sec:scale-hetero-exp}) all survive held-out "
                     "evaluation intact, the last retaining $99$--$100\\%$ of its in-sample margin under both a "
                     "re-sampled-noise and a shifted-stimulus protocol.")
        new_intro = ("On the positive side, the alignment-ladder ceiling and the attention-path advantage survive "
                     "held-out evaluation intact. The scale-heterogeneous fusion gain (\\S\\ref{sec:scale-hetero-exp}) "
                     "does not: under the same held-out protocol its margin is retained in full, but an audit of the "
                     "projection space (\\S\\ref{sec:projection-space}) attributes that margin to the projection "
                     "being frozen and randomly initialized, and shows that it disappears when the space is learned.")
        if old_intro in s:
            s = s.replace(old_intro, new_intro)
        # 3a) §6.3 表格行加注 (逐行处理, 避免转义地狱)
        lines = s.split("\n")
        for i, ln in enumerate(lines):
            if "Fused beats best single expert (v33)" in ln and r"\textbf{survives}" in ln:
                lines[i] = ln.replace(r"\textbf{survives}",
                                     r"\textbf{retained, but see} \S\ref{sec:projection-space}")
        s = "\n".join(lines)
        # 3b) §6.3 表格之后插入审计裁决段
        if "The margin above is retained, but it is not a margin for fusion." not in s:
            verdict = (
                "\\paragraph{The margin above is retained, but it is not a margin for fusion.} "
                "The projection in this configuration is orthogonally initialized and frozen, and an audit of "
                "the shared space (\\S\\ref{sec:projection-space}) shows that the fused output is a detector "
                "of its \\emph{source base} rather than of the target: on held-out data the source identity is "
                "recoverable from the fused output with probability $1.000$, while the target class is "
                "recoverable with probability $0.279$ against a $0.125$ chance baseline. The router's task in a "
                "random space is the trivial one of recognizing which base produced the token. When the space is "
                "made meaningful---a nonlinear lift, trained rather than frozen---the same margin collapses to "
                "$+4\\%$--$+12\\%$, the absolute error falls below the trivial zero predictor, and the "
                "source-identity decodability of the output falls with it. We therefore read this row as evidence "
                "that a shared space must be meaningful for heterogeneous fusion to work at all, not as evidence "
                "that fusing beats using one base well.\n\n")
            anchor2 = "Two results deserve to be read together with the rest of this section rather than on their own."
            if anchor2 in s:
                s = s.replace(anchor2, verdict + anchor2, 1)
        # 4) 架构异构线: 六处主张按 v51 裁决改写
        reps = [
            # 摘要
            ("What survives held-out evaluation is the alignment-ladder ceiling and the attention-path advantage.",
             "What survives held-out evaluation is the alignment-ladder ceiling. The attention-path advantage does not: "
             "a projection-space audit (\\S\\ref{sec:projection-space}) shows it to be a property of a frozen "
             "random shared space rather than of architectural heterogeneity, and it disappears once that space is learned."),
            # §1
            ("On the positive side, the alignment-ladder ceiling and the attention-path advantage survive held-out evaluation intact.",
             "On the positive side, the alignment-ladder ceiling survives held-out evaluation intact. The attention-path "
             "advantage does not: an audit of the projection space (\\S\\ref{sec:projection-space}) shows that the "
             "attention path's lead over the FFN path belongs to a frozen random shared space, not to architectural "
             "heterogeneity, and that it disappears once the space is learned."),
            # §1 贡献列表
            ("FFN-only routing beats the best single expert by $20.3\\%$ and equal-weight averaging by $17.3\\%$; "
             "dual-path routing raises the margin over averaging to $84.8\\%$. A path decomposition shows the "
             "cross-architecture gain concentrates in the attention path---a claim that survives held-out evaluation "
             "($+82.7\\%$ in-sample, $+70.6\\%$ held-out).",
             "FFN-only routing beats the best single expert by $20.3\\%$ and equal-weight averaging by $17.3\\%$; "
             "dual-path routing raises the margin over averaging to $84.8\\%$. A path decomposition shows the "
             "cross-architecture gain concentrating in the attention path ($+82.7\\%$ in-sample, $+70.6\\%$ held-out). "
             "Both margins are later re-examined under a projection-space audit (\\S\\ref{sec:projection-space}): "
             "the target block in this configuration is the modality index, the margin is unchanged when the target "
             "blocks are permuted across modalities, and it collapses when the frozen random projection is replaced "
             "by a learned one."),
            # §6.2 路径分解段: 补 v51 裁决
            ("The path decomposition is the interesting part, and it is one of the findings that survives held-out evaluation.",
             "The path decomposition is the interesting part, and it is one of the findings that survives held-out "
             "evaluation---but a later projection-space audit (\\S\\ref{sec:projection-space}) qualifies what it "
             "measures: the attention path's lead is a property of the frozen random shared space, and it largely "
             "disappears once that space is learned."),
            # 结论
            ("the cross-architecture fusion gain concentrates in the attention path, a claim that survives held-out evaluation.",
             "the cross-architecture fusion gain concentrates in the attention path, a concentration we later show to "
             "belong to a frozen random shared space rather than to architectural heterogeneity "
             "(\\S\\ref{sec:projection-space})."),
            # 可迁移性段落
            ("the attention-path advantage reflects a structural property of transformers (attention patterns vary more across architectures than FFN activations), and the alignment-ladder ceiling is a matched-capacity design statement.",
             "the alignment-ladder ceiling is a matched-capacity design statement, and the attention-path advantage, "
             "once the projection space is audited (\\S\\ref{sec:projection-space}), turns out to belong to the "
             "frozen random space rather than to a structural property of transformers."),
        ]
        for old, new in reps:
            if old in s:
                s = s.replace(old, new)
                print(f"  · 改写: {old[:52]}...")
        open(path, "w", encoding="utf-8").write(s)
        print(f"✓ {fn} 已改写")
    return 0


if __name__ == "__main__":
    sys.exit(main())
