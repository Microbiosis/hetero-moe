#!/usr/bin/env python3
"""tools/extract_generation_manifest.py

半自动生成 MANIFEST.json 骨架: 从 research/<line>/README.md 与 docs/ 规范抽取能抽的,
抽不到的显式留 null 并登记进 "_needs_human", **绝不编造数值**.

设计原则 (来自 docs/GENERATION_MANIFEST_SCHEMA.md):
    机器负责"抄得准的", 人负责"判断"。
    指标数值、协议、泄漏审计、证伪结论——这四项需要判断, 工具只负责把它们
    标成待办, 不给猜测值。

用法:
    python3 tools/extract_generation_manifest.py                # 抽取所有缺 MANIFEST 的包
    python3 tools/extract_generation_manifest.py v29_large_corpus
    python3 tools/extract_generation_manifest.py --dry-run      # 只打印不写盘
    python3 tools/extract_generation_manifest.py --force       # 覆盖已有 MANIFEST
    python3 tools/extract_generation_manifest.py --print-human  # 汇总待人工补的字段
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
ARCHIVE = ROOT / "archive"

HUMAN_FIELDS = ["claim", "protocol", "leakage_audit", "acceptance.outcome"]


# ---------------------------------------------------------------- 读取基础件

def find_line_readme(line: str) -> Optional[Path]:
    p = RESEARCH / line / "README.md"
    return p if p.exists() else None


def split_sections(md: str) -> Dict[str, str]:
    """把 markdown 按 '## ' 切成 {标题: 正文}."""
    out, cur, buf = {}, "__head__", []
    for ln in md.splitlines():
        if ln.startswith("## "):
            out[cur] = "\n".join(buf)
            cur, buf = ln[3:].strip(), []
        else:
            buf.append(ln)
    out[cur] = "\n".join(buf)
    return out


def find_line_doc(line: str) -> Optional[Path]:
    p = RESEARCH / line / "README.md"
    return p if p.exists() else p


def find_spec_doc(version: str) -> Optional[Path]:
    """docs/V16_0_*.md 形式."""
    v = version.lstrip("v")
    hits = sorted((ROOT / "docs").glob(f"V{v}_0_*.md"))
    return hits[0] if hits else None


# ---------------------------------------------------------------- 抽取器

def extract_parent_child(head: str) -> Tuple[Optional[str], List[str]]:
    """从 '**父版本**: ... / **子版本**: ...' 抽取谱系, 归一化成版本号列表."""
    def grab(key: str) -> Optional[str]:
        m = re.search(rf"\*\*{key}\*\*\s*[:：]\s*(.+)", head)
        if not m:
            return None
        val = m.group(1).strip()
        # "无" / "None" / "无 (这是 v22 路线的分支, 已收敛)" 这类写法 = 明确没有
        if re.match(r"^(无|没有|None|none|-)\b", val) or re.match(r"^无[\s(（]", val):
            return ""
        return val

    def vnums(s: Optional[str]) -> List[str]:
        if not s:
            return []
        # 注意: 不能用 \bv(\d+)\b —— "v22_central_theory" 里 '_' 是单词字符,
        # 结尾 \b 不成立, 会导致父版本抽空而从正文里错抽子版本。
        return [f"{m.group(1)}.0" for m in re.finditer(r"v(\d+)(?![0-9])", s)]

    parent = vnums(grab("父版本"))
    children = vnums(grab("子版本"))
    return (parent[0] if parent else None), children


def extract_member_contrib(sections: Dict[str, str], version: str) -> Optional[str]:
    """从 '## 成员' 表格里取本版本的 '贡献' 单元格."""
    body = sections.get("成员", "")
    for ln in body.splitlines():
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if re.fullmatch(r"v\d+", cells[0]) and cells[0] == version:
            return cells[1]
    return None


def extract_falsified(sections: Dict[str, str], version: str) -> List[dict]:
    """从 '负面发现' / '关键发现' / '负面发现 / 边界' 等小节抽候选证伪条目."""
    keys = [k for k in sections
            if any(t in k for t in ("负面发现", "负面发现 / 边界", "关键发现", "负面发现 (重要)"))]
    items: List[dict] = []
    for k in keys:
        for ln in sections[k].splitlines():
            ln = ln.strip()
            if not ln.startswith("- "):
                continue
            txt = ln[2:].strip()
            if version in txt or version.lstrip("v") in txt or "全部失败" in txt or "退步" in txt or "证伪" in txt:
                items.append({
                    "hypothesis": None,
                    "verdict": txt,
                    "evidence": f"research/{k} README 小节",
                })
    return items


def detect_implementation(pkg: Path) -> Dict:
    """检测包是否为提升后的 re-export shim."""
    init = pkg / "__init__.py"
    lifted_to = None
    if init.exists():
        txt = init.read_text()
        m = re.search(r'__lifted_to__\s*=\s*["\']([^"\']+)["\']', txt)
        if m:
            lifted_to = m.group(1)
    return {
        "location": "lifted" if lifted_to else "self",
        "lifted_to": lifted_to,
        "modules": sorted(p.name for p in pkg.glob("*.py") if p.name != "__init__.py"),
    }


def find_forwarder_tests(version: str) -> Optional[str]:
    hits = sorted((ROOT / "tests").glob(f"test_{version}_*.py"))
    return f"tests/{hits[0].name}" if hits else None


# ---------------------------------------------------------------- 组装

def build_skeleton(pkg: Path) -> Tuple[Dict, List[str]]:
    rel = pkg.relative_to(ROOT)
    if rel.parts[0] == "archive":
        line = "archive/" + rel.parts[1] if len(rel.parts) > 2 else "archive"
        readme = None
    else:
        line = rel.parts[1]
        readme = find_line_readme(line)

    version = re.match(r"(v\d+)[a-z]*_", pkg.name).group(1)
    version_dotted = f"{version.lstrip('v')}.0"

    parent, children = (None, [])
    contrib = None
    falsified: List[dict] = []
    if readme:
        secs = split_sections(readme.read_text())
        parent, children = extract_parent_child(secs.get("__head__", ""))
        contrib = extract_member_contrib(secs, version)
        falsified = extract_falsified(secs, version)

    spec_doc = find_spec_doc(version)
    fwd_tests = find_forwarder_tests(version)
    # 自动判定"该代确实没有测试套件": tests/ 下不存在对应文件, 且包内无 tests/ 目录。
    # 写成派生结论而非人工标注 —— 否则每次 --force 重建都会把人工标的信息冲掉。
    no_tests = fwd_tests is None and not (pkg / "tests" / "test_package.py").exists()
    human: List[str] = []
    if not contrib:
        human.append("generation.one_mechanism_added")
    # 关键: 行级 README 的 父/子版本 描述的是**这条研究线**的位置, 不是线上每个包的位置。
    # 若把它赋给每个包, 一条 7 包的线会让 7 个包都自称 parent=v6 并互认父子。
    # 因此这里放进 lineage.* (线级事实), generation.parent 留空待人工按包填。
    human.append("generation.parent")

    # claim / protocol / acceptance / leakage_audit 全部需判断
    human += ["claim.metric", "claim.value", "claim.baseline", "claim.delta_pct",
              "claim.seeds", "protocol.eval_set", "acceptance.criterion",
              "acceptance.outcome", "leakage_audit.status"]

    man = {
        "schema_version": "1.0",
        "version": version_dotted,
        "package": pkg.name,
        "line": line,
        "title": (contrib or f"{version} (待补标题)"),
        "generation": {
            "one_mechanism_added": contrib,
            "parent": None,          # 包级父版本需人工按包判定, 抽取器不猜
            "children": [],          # 由 --link 从显式 parent 反推
        },
        "lineage": {
            "line_parent": parent,   # 线级事实(来自研究线 README)
            "line_children": children,
            "note": "line_* 描述研究线在全局中的位置; generation.* 描述包在代际链中的位置",
        },
        "implementation": detect_implementation(pkg),
        "claim": {
            "asserts": None,
            "metric": None,
            "direction": "lower_is_better",
            "value": None,
            "baseline": {"ref": None, "value": None},
            "delta_pct": None,
            "seeds": None,
        },
        "protocol": {"eval_set": None, "input": None, "caveat": None},
        "acceptance": {"criterion": None, "outcome": None},
        "leakage_audit": {
            "required": True,
            "status": "pending",
            "note": "抽取器不判断泄漏审计结果, 必须人工填写",
        },
        "falsified": falsified or [],
        "superseded_by": None,
        "tests_status": "none" if no_tests else "present",
        "artifacts": {
            "spec": "SPEC.md",
            "tests": None if no_tests else "tests/test_package.py",
            "results": "results/metrics.json",
            "forwarder_tests": fwd_tests,
            "forwarder_spec": (f"docs/{spec_doc.name}" if spec_doc else None),
        },
    }
    if no_tests:
        man["_note"] = ("本代在仓库中没有对应测试文件 (tests/ 下不存在 test_<vN>_*.py, "
                        "包内也无 tests/), 故 artifacts.tests 为 null。这是派生结论, 非遗漏。")
    if human:
        man["_needs_human"] = human
        man["_extracted_from"] = [str(readme.relative_to(ROOT))] if readme else []
    return man, human


def discover() -> List[Path]:
    out = []
    for base in (RESEARCH, ARCHIVE):
        for line_dir in sorted(base.iterdir()) if base.exists() else []:
            if not line_dir.is_dir():
                continue
            for pkg in sorted(line_dir.iterdir()):
                if pkg.is_dir() and (pkg / "__init__.py").exists() \
                        and re.match(r"v\d+[a-z]*_", pkg.name):
                    out.append(pkg)
    return out


def link_lineage() -> int:
    """由各包的 parent 反推 children, 并做双向补全.

    必要性: 行级 README 的"子版本"字段多数写作"无", 所以 children 天然稀疏。
    但只要 parent 齐全, children 就是 parent 的确定性反像 —— 直接算出来,
    谱系才真正可查询 (docs/GENERATION_MANIFEST_SCHEMA.md §5)。
    """
    mans: Dict[str, Path] = {}
    for pkg in discover():
        mp = pkg / "MANIFEST.json"
        if mp.exists():
            mans[pkg.name] = mp

    data = {n: json.loads(p.read_text()) for n, p in mans.items()}

    # parent -> [child 版本号]
    derived: Dict[str, List[str]] = {}
    for name, man in data.items():
        pv = (man.get("generation") or {}).get("parent")
        if pv:
            derived.setdefault(pv, []).append(man["version"])

    # 显式 children 已有的包(试点三包)也要参与补全
    explicit: Dict[str, List[str]] = {}
    for name, man in data.items():
        for c in ((man.get("generation") or {}).get("children") or []):
            explicit.setdefault(man["version"], []).append(c)

    changed = 0
    for name, man in data.items():
        gen = man.setdefault("generation", {})
        want = sorted(set(derived.get(man["version"], [])) | set(explicit.get(man["version"], [])))
        if gen.get("children") != want:
            gen["children"] = want
            (mans[name]).write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
            changed += 1
            print(f"  ↻ {name}: children = {want}")

    # 反向: 子代的 parent 缺失时, 由显式 children 补上
    rev = 0
    for name, man in data.items():
        gen = man.setdefault("generation", {})
        if gen.get("parent"):
            continue
        pv = next((p for p, cs in explicit.items() if man["version"] in cs), None)
        if pv:
            gen["parent"] = pv
            mans[name].write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
            rev += 1
            print(f"  ↻ {name}: parent = {pv} (由 children 反推)")

    print(f"  ----------------------------")
    print(f"  谱系连接: children 补全 {changed} 包, parent 反推 {rev} 包")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="重建骨架 (仍会跳过已人工完成的 manifest)")
    ap.add_argument("--force-hard", action="store_true",
                    help="无条件覆盖, 包括已人工完成的 manifest (危险: 会丢失 leakage_audit 等判断)")
    ap.add_argument("--print-human", action="store_true")
    ap.add_argument("--link", action="store_true",
                    help="只做谱系连接: 由各包的 parent 反推 children, 并双向补全")
    args = ap.parse_args()

    if args.link:
        return link_lineage()

    targets = discover()
    if args.packages:
        targets = [p for p in targets
                   if p.name in args.packages or any(a in str(p) for a in args.packages)]

    todo = []
    skipped_done = []
    for pkg in targets:
        mp = pkg / "MANIFEST.json"
        if mp.exists() and not (args.force or args.force_hard):
            continue
        if mp.exists() and not args.force_hard:
            # 防呆: 已人工完成的 manifest (无 _needs_human) 不允许被骨架覆盖,
            # 否则会静默丢失 leakage_audit / falsified 等判断。
            try:
                if "_needs_human" not in json.loads(mp.read_text()):
                    skipped_done.append(pkg.name)
                    continue
            except json.JSONDecodeError:
                pass  # 坏 JSON 允许重建
        man, human = build_skeleton(pkg)
        todo.append((pkg, man, human))
        if not args.dry_run:
            mp.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
            rd = pkg / "results"
            rd.mkdir(exist_ok=True)
            (rd / "metrics.json").write_text(json.dumps({
                "schema_version": "1.0",
                "version": man["version"],
                "package": man["package"],
                "line": man["line"],
                "metric": None,
                "direction": "lower_is_better",
                "baseline": None,
                "value": None,
                "recorded_by": "extract_generation_manifest.py 骨架, 指标待人工填",
            }, ensure_ascii=False, indent=2) + "\n")

    for pkg, man, human in todo:
        mark = "(dry-run)" if args.dry_run else ""
        print(f"  ✓ {pkg.relative_to(ROOT)} {man['version']} {mark}")
        if human:
            print(f"      待人工补 ({len(human)}): {', '.join(human)}")

    if args.print_human and todo:
        print("  ----------------------------")
        print("  人工待办汇总 (每个包):")
        for pkg, _, human in todo:
            print(f"    {pkg.name}: {', '.join(human)}")

    print(f"  ----------------------------")
    if skipped_done:
        print(f"  防呆跳过 (已人工完成, --force 不覆盖; 用 --force-hard 强制): {skipped_done}")
    print(f"  新建/重建骨架: {len(todo)}  "
          f"{'✓ 全部包已有 MANIFEST' if not todo else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
