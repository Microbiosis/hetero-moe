#!/usr/bin/env python3
"""tools/manifest_worksheet.py

为待人工填写的 MANIFEST 生成"证据工作表": 把要填的字段列出来, 并把候选数字
连同 **file:line** 一起摆在旁边, 让人从"翻 8 个 README 挖数据"变成"对着候选确认".

设计立场:
    本工具**不写入任何 MANIFEST.json**, 也**不替你判断**。
    它只做检索与定位。`leakage_audit.status` 这类需要领域判断的字段, 工具不猜。

用法:
    python3 tools/manifest_worksheet.py                    # 全部待填包, 控制台摘要
    python3 tools/manifest_worksheet.py --md out.md        # 生成 markdown 工作表
    python3 tools/manifest_worksheet.py --pkg v29_large_corpus
    python3 tools/manifest_worksheet.py --evidence 3       # 每个包最多 3 条候选证据
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# 人类可读的字段说明 —— 避免填表时猜语义
FIELD_HELP = {
    "generation.parent": "本代在代际链中的父版本号 (如 16.0)。注意别把研究线级别的父版本当包级",
    "generation.one_mechanism_added": "本代唯一新增的机制 (设计原则: 每代只加一个)",
    "claim.metric": "指标名, 通常是 fuse_mse",
    "claim.value": "本代在该指标上的数值 (MSE 越小越好)",
    "claim.baseline": '对照基线, 形如 {"ref": "...", "value": 数字}',
    "claim.delta_pct": "相对基线的变化百分比, 正=变好(若指标越小越好)",
    "claim.seeds": "该结论的随机种子数",
    "protocol.eval_set": "train_distribution | held_out —— 评估输入与训练分布是否同源",
    "acceptance.criterion": "什么条件算成功/失败 (一句话可判定)",
    "acceptance.outcome": "pass | fail | invalidated",
    "leakage_audit.status": "clean | invalidated | not_applicable | pending —— 移走直接目标信号后是否仍成立",
}

NUM = re.compile(r"\d+\.\d+|\b\d+\s*seeds?\b|\b\d+\s*×\s*\d+|held[- ]?out|真语料|合成")


def ignored(rel: str) -> Optional[bool]:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel],
                           capture_output=True, timeout=15)
        return True if r.returncode == 0 else (False if r.returncode == 1 else None)
    except Exception:  # noqa: BLE001
        return None


def collect_manifests() -> List[Tuple[Path, Dict]]:
    out = []
    for base in ("research", "archive"):
        for pkg in sorted((ROOT / base).glob("*/v*_*")):
            mp = pkg / "MANIFEST.json"
            if not (pkg / "__init__.py").exists() or not mp.exists():
                continue
            try:
                man = json.loads(mp.read_text())
            except json.JSONDecodeError:
                continue
            if man.get("_needs_human"):
                out.append((pkg, man))
    return out


def evidence_for(pkg: Path, man: Dict, limit: int) -> List[Tuple[str, str, int, str]]:
    """收集候选证据: (来源标签, 仓库相对路径, 行号, 文本)。

    只收录含数字或协议关键词的行。路径必须是真实可跳转的位置 ——
    这正是本工具的全部价值, 不能退化成"成员表:16"这种无源标签。
    """
    rel = pkg.relative_to(ROOT)
    version = man["version"].split(".")[0]
    cands: List[Tuple[str, str, int, str]] = []
    score = {"member": 0, "readme": 1, "spec": 2}

    # 1) 研究线 README 的"成员"表行 —— 最直接的"本代贡献 + seeds"来源
    line = man.get("line", "")
    rd = ROOT / "research" / line / "README.md"
    if not rd.exists():
        rd = ROOT / rel.parts[0] / rel.parts[1] / "README.md" if len(rel.parts) > 2 else None
    if rd and rd.exists():
        for i, ln in enumerate(rd.read_text().splitlines(), 1):
            s = ln.strip()
            if s.startswith("|") and re.match(rf"^\|\s*v{version}\b", s):
                cands.append(("成员表", str(rd.relative_to(ROOT)), i, s))
            elif NUM.search(s) and version in s:
                cands.append(("README", str(rd.relative_to(ROOT)), i, s[:200]))

    # 2) 包内 SPEC.md —— 权威规范里的数字
    sp = pkg / "SPEC.md"
    if sp.exists():
        for i, ln in enumerate(sp.read_text().splitlines(), 1):
            s = ln.strip()
            if s.startswith("#") or not s:
                continue
            if NUM.search(s):
                cands.append(("SPEC", str(sp.relative_to(ROOT)), i, s[:200]))

    cands.sort(key=lambda c: (score.get(c[0], 3), c[2]))
    return cands[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", metavar="PATH", help="写出 markdown 工作表")
    ap.add_argument("--pkg", nargs="*", help="只看这些包")
    ap.add_argument("--evidence", type=int, default=3, help="每包候选证据条数")
    args = ap.parse_args()

    items = collect_manifests()
    if args.pkg:
        items = [(p, m) for p, m in items if p.name in args.pkg]
    if not items:
        print("没有待填的包 (或已全部完成)", file=sys.stderr)
        return 0

    lines_md: List[str] = [
        "# MANIFEST 人工填写工作表",
        "",
        "> 本文件由 `tools/manifest_worksheet.py` 生成, **不含任何自动写入**。",
        "> 下面的\"候选证据\"是检索结果与行号, 供人工核对; 请勿未经核对直接采信。",
        "",
        f"待填包数: **{len(items)}**",
        "",
        "## 字段语义",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
    ]
    for k, v in FIELD_HELP.items():
        lines_md.append(f"| `{k}` | {v} |")

    for pkg, man in items:
        rel = pkg.relative_to(ROOT)
        ig = ignored(str(rel))
        tag = "" if ig is False else "  `[author-only: 不在版本库]`"
        todo = man["_needs_human"]
        lines_md += [
            "",
            f"### `{man['package']}` — {man['version']}{tag}",
            "",
            f"- 路径: `{rel}`",
            f"- 研究线: `{man['line']}`",
            f"- 待填 {len(todo)} 项: " + ", ".join(f"`{t}`" for t in todo),
            "",
        ]
        ev = evidence_for(pkg, man, args.evidence)
        if ev:
            lines_md.append("**候选证据(需核对)**")
            lines_md.append("")
            lines_md.append("| 来源 | 位置 | 原文 |")
            lines_md.append("|---|---|---|")
            for src, path, ln, txt in ev:
                cell = txt.replace("|", "\\|")   # 表格里的竖线要转义
                lines_md.append(f"| {src} | `{path}:{ln}` | {cell} |")
        else:
            lines_md.append("_未检索到含数字的候选证据, 需手工查证_。")

        print(f"  {man['package']:<28} {man['version']:<7} 待填 {len(todo):>2} 项"
              f"{'  [author-only]' if ig else ''}")
        for src, path, ln, txt in ev:
            print(f"      {path}:{ln}  {txt[:100]}")

    print(f"  ----------------------------")
    print(f"  待填包 {len(items)} 个; 字段语义见生成的工作表")

    if args.md:
        out = Path(args.md)
        if not out.is_absolute():
            out = ROOT / out
        out.write_text("\n".join(lines_md) + "\n")
        print(f"  工作表已写出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
