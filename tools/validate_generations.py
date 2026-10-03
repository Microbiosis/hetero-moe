#!/usr/bin/env python3
"""tools/validate_generations.py

验证 research/ 下每个版本包的 **L2 交付自包含** (L1 import 自包含由
tools/check_self_containment.py 负责, 两者互补):

    1. 包内自带 MANIFEST.json / SPEC.md / results/metrics.json / tests/
    2. MANIFEST 必备字段齐全, leakage_audit.status 取值合法
    3. MANIFEST.artifacts 声明的每个路径真实存在
    4. generation 谱系自洽: A 的 children 含 B  <=>  B 的 parent 是 A
    5. 薄转发层存在且指向包内权威副本

规范: docs/GENERATION_MANIFEST_SCHEMA.md

用法:
    python3 tools/validate_generations.py                          # 全部研究包
    python3 tools/validate_generations.py v16_bert_student ...     # 指定包
    python3 tools/validate_generations.py --require-leakage-audit # 泄漏审计必须非 pending
    python3 tools/validate_generations.py --allow-missing         # 只报告, 不因缺文件失败 (存量摸底)
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
RESEARCH = ROOT / "research"
ARCHIVE = ROOT / "archive"
# 版本包所在根: research/ 下的研究线, 以及 archive/negative_findings/ 下的负结果分支
BASES = (RESEARCH, ARCHIVE)

# MANIFEST 必备顶层字段
REQUIRED_TOP = ["schema_version", "version", "package", "line", "title",
                "generation", "implementation", "claim", "protocol",
                "acceptance", "leakage_audit", "falsified", "artifacts"]

# 包内必备文件
REQUIRED_FILES = ["MANIFEST.json", "SPEC.md", "results/metrics.json",
                  "tests/test_package.py"]

# leakage_audit.status 合法取值
LEAKAGE_STATUS = {"clean", "invalidated", "not_applicable", "pending"}


def git_ignored(rel_path: str) -> Optional[bool]:
    """该路径是否被 .gitignore 排除。True=排除, False=会入库, None=无法判定。"""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel_path],
                           capture_output=True, timeout=15)
        if r.returncode == 0:
            return True
        if r.returncode == 1:
            return False
        return None
    except Exception:  # noqa: BLE001 - 非 git 仓库 / git 不可用
        return None


_TRACKED: Optional[set] = None


def git_tracked() -> Optional[set]:
    """仓库内已被 git 跟踪的路径集合 (仓库相对, 正斜杠)。非 git 仓库返回 None。"""
    global _TRACKED
    if _TRACKED is None:
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                               capture_output=True, timeout=60)
            _TRACKED = set() if r.returncode != 0 else {
                p.replace("\\", "/") for p in r.stdout.decode("utf-8", "replace").splitlines() if p
            }
        except Exception:  # noqa: BLE001 - git 不可用
            _TRACKED = None
    return _TRACKED


def discover_generations() -> List[Path]:
    """找出 research/ 与 archive/ 下所有形如 v<N>_<name>/ 的版本包."""
    out = []
    for base in BASES:
        if not base.exists():
            continue
        for line_dir in sorted(base.iterdir()):
            if not line_dir.is_dir() or line_dir.name.startswith("_"):
                continue
            for pkg_dir in sorted(line_dir.iterdir()):
                if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists() \
                        and re.match(r"v\d+[a-z]*_", pkg_dir.name):
                    out.append(pkg_dir)
    return out


def check_generation(pkg: Path, allow_missing: bool,
                     require_audit: bool, pkg_versioned: bool = True
                     ) -> Tuple[List[str], List[str]]:
    """返回 (errors, warnings).

    pkg_versioned: 该包本身会随仓库分发。author-only 包 (.gitignore 标注) 的产物
    本来就不入库, 不做"必须已跟踪"判定 —— 否则会把既有仓库策略误报成缺陷。
    """
    errors: List[str] = []
    warnings: List[str] = []

    man_path = pkg / "MANIFEST.json"
    man: Dict = {}
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text())
        except json.JSONDecodeError as e:
            return [f"[INVALID-JSON] MANIFEST.json: {e}"], warnings

    # --- 1) 必备文件 ---
    # 该代可显式声明 tests_status="none" 表示"确实没有测试套件",
    # 此时不要求 tests/test_package.py (v30/v31 即属此类)。
    declared_no_tests = str(man.get("tests_status", "")).lower() == "none"
    required = [f for f in REQUIRED_FILES
                if not (f == "tests/test_package.py" and declared_no_tests)]
    for rel in required:
        if not (pkg / rel).exists():
            msg = f"[MISSING] {rel}"
            (warnings if allow_missing else errors).append(msg)

    if not man_path.exists():
        return errors, warnings

    # --- 2) 必备字段 ---
    for key in REQUIRED_TOP:
        if key not in man:
            errors.append(f"[NO-FIELD] {key}")

    # --- 3) leakage_audit ---
    la = man.get("leakage_audit", {})
    status = la.get("status")
    if status is not None and status not in LEAKAGE_STATUS:
        errors.append(f"[BAD-LEAKAGE-STATUS] {status!r} 不在 {sorted(LEAKAGE_STATUS)}")
    if status == "pending":
        msg = "[LEAKAGE-PENDING] 该代尚未做目标泄漏审计"
        (errors if require_audit else warnings).append(msg)
    if man.get("implementation", {}).get("location") == "lifted" \
            and not man.get("implementation", {}).get("lifted_to"):
        errors.append("[LIFTED-NO-TARGET] implementation.location=lifted 但未声明 lifted_to")

    # --- 4) artifacts 路径真实存在, 且对会分发的包必须真的随仓库分发 ---
    # forwarder_* 是仓库级薄转发层, 其余是包内路径
    REPO_LEVEL = {"forwarder_spec", "forwarder_tests"}
    art = man.get("artifacts", {})
    tracked = git_tracked()
    for k, rel in art.items():
        if rel is None:
            # 该代没有对应的转发层 (例如无独立测试文件), 不算错误
            warnings.append(f"[NO-FORWARDER] artifacts.{k} 为 null (该代无此转发层)")
            continue
        cand = (ROOT / rel) if k in REPO_LEVEL else (pkg / rel)
        if not cand.exists():
            errors.append(f"[ARTIFACT-MISSING] artifacts.{k} -> {rel}")
            continue
        # 存在还不够: 对会入库的包, 产物若被 .gitignore 吞掉, 克隆仓库的人就拿不到,
        # "自包含" 只在作者本机成立。这是已发生过的真实事故(顶层 `results/` 规则
        # 曾把全部 32 个包的 results/metrics.json 一起排除), 故单列一道闸门。
        if not pkg_versioned:
            continue
        repo_rel = str(cand.relative_to(ROOT)).replace("\\", "/")
        if git_ignored(repo_rel):
            errors.append(
                f"[ARTIFACT-UNSHIPPED] artifacts.{k} -> {rel} 被 .gitignore 排除, "
                f"不会随仓库分发 (自包含只在本机成立)")
        elif tracked is not None and repo_rel not in tracked:
            warnings.append(
                f"[ARTIFACT-UNTRACKED] artifacts.{k} -> {rel} 尚未 git add, 提交后才随仓库分发")

    # --- 5) 版本号与目录名一致性 ---
    pkgname = pkg.name
    if man.get("package") and man["package"] != pkgname:
        errors.append(f"[NAME-MISMATCH] MANIFEST.package={man['package']!r} != 目录 {pkgname!r}")

    return errors, warnings


def check_lineage(pkgs: List[Dict]) -> List[str]:
    """跨包检查谱系自洽: A.children 含 B  <=>  B.parent 是 A."""
    errors: List[str] = []
    version_map: Dict[str, Dict] = {}

    # 版本号必须全局唯一。撞号会让下面的 version_map 静默丢包 (后写的覆盖先写的),
    # 被丢掉那条的谱系就从图里消失, 而且校验器本身不报错 —— 属"假绿"。
    # 已发生过的实例: v23b_hard_vqa 与 v23_vqa 都曾写作 23.0 (抽取器的
    # v(\d+)[a-z]*_ 认得 b 后缀, 但版本号生成时把它丢了)。
    seen: Dict[str, str] = {}
    for rec in pkgs:
        v = (rec.get("manifest") or {}).get("version")
        if not v:
            continue
        if v in seen:
            errors.append(
                f"[VERSION-DUP] version={v!r} 同时被 {seen[v]} 与 "
                f"{rec['path'].name} 使用; 谱系图会静默丢掉其中一条")
        else:
            seen[v] = rec["path"].name
            version_map[v] = rec

    for v, rec in version_map.items():
        man = rec["manifest"]
        gen = man.get("generation", {}) or {}
        for child in (gen.get("children") or []):
            if child in version_map:
                cgen = version_map[child]["manifest"].get("generation", {}) or {}
                cparent = cgen.get("parent")
                if cparent and cparent != v:
                    errors.append(
                        f"[LINEAGE] v{v}.children 含 {child}, 但 {child}.parent = {cparent!r} (应互为反义)")
        parent = gen.get("parent")
        if parent and parent in version_map:
            pgen = version_map[parent]["manifest"].get("generation", {}) or {}
            # 反义: v 的 parent 是 parent  <=>  parent 的 children 含 v
            if v not in (pgen.get("children") or []):
                rec.setdefault("_warnings", []).append(
                    f"[LINEAGE-WARN] v{v}.parent = {parent}, 但 {parent}.children 未含 {v}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="*", help="包名 (如 v16_bert_student) 或全路径片段")
    ap.add_argument("--allow-missing", action="store_true",
                    help="缺文件只警告不失败 (用于存量摸底)")
    ap.add_argument("--require-leakage-audit", action="store_true",
                    help="leakage_audit.status 不得为 pending")
    args = ap.parse_args()

    targets = discover_generations()
    if args.packages:
        targets = [p for p in targets
                   if p.name in args.packages or any(a in str(p) for a in args.packages)]
    if not targets:
        print("未找到目标包", file=sys.stderr)
        return 2

    recs, total_err, total_warn = [], 0, 0
    author_only, versioned, unknown = [], [], []
    for pkg in targets:
        rel = pkg.relative_to(ROOT)
        # 口径: 该包是否在版本库内。author-only 路径 (.gitignore 标注
        # "kept author-only", 因引用 proprietary 辅助模型) 只在本地生效, 不随仓库分发。
        ig = git_ignored(str(rel))
        ships = ig is False

        errs, warns = check_generation(pkg, args.allow_missing, args.require_leakage_audit,
                                      pkg_versioned=ships)
        man_path = pkg / "MANIFEST.json"
        man = None
        if man_path.exists():
            try:
                man = json.loads(man_path.read_text())
            except Exception:  # noqa: BLE001
                man = None
        recs.append({"path": pkg, "manifest": man})
        total_err += len(errs)
        total_warn += len(warns)

        (author_only if ig else versioned if ig is False else unknown).append(str(rel))
        suffix = "  [author-only: 不入库]" if ig else ("  [未入库?]" if ig is None else "")
        if errs:
            print(f"  ✗ {rel}{suffix}")
            for e in errs:
                print(f"      {e}")
        else:
            mark = "✓" if not warns else "○"
            print(f"  {mark} {rel}{suffix}")
        for w in warns:
            print(f"      {w}")

    lineage_errs = check_lineage(recs)
    for e in lineage_errs:
        print(f"      {e}")
    for rec in recs:
        for w in rec.get("_warnings", []):
            print(f"      {w}")
            total_warn += 1

    total_err += len(lineage_errs)
    ok = total_err == 0
    print("  ----------------------------")
    print(f"  包数: {len(targets)}  错误: {total_err}  警告: {total_warn}  "
          f"{'✓ ALL GREEN' if ok else '✗ HAS FAILURE'}"
          f"{'  (--allow-missing 模式)' if args.allow_missing else ''}")
    # 口径: 分清"工作树里合规"与"随仓库分发"
    print(f"  ├─ 入版本库: {len(versioned)} 包")
    print(f"  └─ author-only (不在库, 见 .gitignore): {len(author_only)} 包")
    if author_only:
        lines: Dict[str, int] = {}
        for p in author_only:
            top = "/".join(Path(p).parts[:2])
            lines[top] = lines.get(top, 0) + 1
        print(f"       {', '.join(f'{k} ({v})' for k, v in sorted(lines.items()))}")
        print("       这些包的自包含只在本地生效; 其 MANIFEST/SPEC/results/tests 不随仓库分发。")
    if unknown:
        print(f"  └─ 入库状态无法判定: {len(unknown)} 包 (git 不可用)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
