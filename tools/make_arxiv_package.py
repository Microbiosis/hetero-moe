"""打包 arXiv 投稿源文件 (arXiv 收 LaTeX 源, 自己编译)。

为什么交源而不是只交 PDF
----------------------
arXiv 的流程是上传源 → **用 arXiv 自己的 TeX Live 重新编译** → 生成页面。
所以必须保证: ① 源自包含 (图/样式/参考文献都在包里); ② 在一个**干净目录**里
只靠包内文件就能编译通过 (arXiv 机器上没有我们的仓库)。

用法:
  python3 tools/make_arxiv_package.py            # 打包 + 在临时目录里试编译
  python3 tools/make_arxiv_package.py --out /tmp/x.tar.gz
"""
import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "docs/research_paper")

# 包内必须带的文件 (相对 docs/research_paper/arxiv 这个"上传目录")
NEEDED = [
    "main.tex",                  # 主文件 (TMLR [preprint] 样式) —— arXiv 默认找 main.tex
    "main.bbl",                  # **预生成**的参考文献 (arXiv 不必跑 bibtex, 少一个失败点)
    "reconstruct.bib",           # 参考文献源 (便于他人引用/再生成)
    "tmlr.sty", "tmlr.bst",      # TMLR 官方样式与参考文献样式
    "fancyhdr.sty",              # tmlr.sty 依赖
]
FIG_FILES = ["figs/fig_alignment.pdf", "figs/fig_distill.pdf",
             "figs/mermaid/heterogeneity_spectrum.pdf", "figs/mermaid/alignment_ladder.pdf"]


def build(out_path: str, verify: bool = True) -> dict:
    """把 docs/research_paper/arxiv/ 打成上传用 tar.gz, 并在**只含包内文件**的
    干净目录里真编译一遍 (arXiv 用自己的 TeX Live 重编译, 源必须自包含)。"""
    src_dir = os.path.join(PAPER, "arxiv")
    os.makedirs(src_dir, exist_ok=True)
    # 单一真源: reconstruct_arxiv.tex → arxiv/main.tex (避免两份源漂移)
    shutil.copy(os.path.join(PAPER, "reconstruct_arxiv.tex"), os.path.join(src_dir, "main.tex"))
    tmp = tempfile.mkdtemp(prefix="arxiv-pkg-")
    for rel in NEEDED + FIG_FILES:
        src = os.path.join(src_dir, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    result = {"files": len(NEEDED) + len(FIG_FILES), "compiled": None, "pages": None}
    if verify:
        # 在**只有包内文件**的目录里编译 (模拟 arXiv 的机器)
        cmds = [["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]]
        log = ""
        for c in cmds:
            pp = subprocess.run(c, cwd=tmp, capture_output=True, text=True, timeout=900)
            log += pp.stdout + pp.stderr
        p = pp
        # 把生成的 .bbl / .pdf 回写到 arxiv/ (仓库里存的 .bbl 才与源同步)
        for f in ("main.bbl", "main.pdf"):
            if os.path.exists(os.path.join(tmp, f)):
                shutil.copy(os.path.join(tmp, f), os.path.join(src_dir, f))
        result["compiled"] = (p.returncode == 0 and "Output written" in log
                              and "Undefined control sequence" not in log)
        for line in log.splitlines():
            if "Output written" in line:
                result["pages"] = line.split("Output written on")[1].split("(")[1].split()[0]
        errs = [l for l in log.splitlines() if l.startswith("!")]
        result["errors"] = errs[:5]
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in NEEDED + FIG_FILES:
            tar.add(os.path.join(tmp, rel), arcname=rel)
    result["package"] = out_path
    result["size_mb"] = round(os.path.getsize(out_path) / 1e6, 2)
    shutil.rmtree(tmp, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PAPER, "arxiv_package.tar.gz"))
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    r = build(a.out, verify=not a.no_verify)
    print(f"✓ 打包 {r['files']} 个文件 → {r['package']} ({r['size_mb']} MB)")
    if r["compiled"] is not None:
        print(f"  干净目录试编译: {'通过 ✓' if r['compiled'] else '失败 ✗'}"
              + (f", {r['pages']} 页" if r.get("pages") else ""))
        for e in r.get("errors", []):
            print("   ", e)


if __name__ == "__main__":
    main()
