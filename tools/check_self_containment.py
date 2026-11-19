#!/usr/bin/env python3
"""tools/check_self_containment.py

检查 research/ 下每个研究包的**实现模块**依赖闭包 (包内 tests/ 不在扫描范围内):

    1. 静态依赖: 仅允许来自 research/_primitives/、同包内子模块、archive/，以及
       ALLOWED_INTRA_LINE 显式声明的同线复用；其余判为跨线引用 (--strict 下报错)。
    2. 动态加载检测 (v84 治理补盲): 同步扫描 importlib.util.spec_from_file_location /
       importlib.import_module / SourceFileLoader / _load_by_path 等按路径跨目录动态加载，
       杜绝用动态加载绕过静态自包含校验的“假绿灯”。
       - 已由 SHA256 锁定的历史代 (v75–v83) 的既有动态依赖显式登记在
         ALLOWED_DYNAMIC_LEGACY 并持续在汇总中公开曝光（不会静默隐瞒）；
       - 任何未登记的新增动态跨包/跨线加载在 --strict 下直接判红；
       - 传入 --fail-on-legacy-dynamic 时连历史动态加载一并判红。

用法:
    python3 tools/check_self_containment.py                         # 全部研究包
    python3 tools/check_self_containment.py v25_corpus_central      # 单个包
    python3 tools/check_self_containment.py --strict                # 严格模式: 未登记跨线静态/动态均报错
    python3 tools/check_self_containment.py --strict --fail-on-legacy-dynamic
"""
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"

PRIMITIVES = {
    "research._primitives.attention",
    "research._primitives.central_mechanism",
    "research._primitives.central_theory",
    "research._primitives.real_corpus",
    "research._primitives.scale_hetero_loaders",
    "research._primitives.testing",
}

ALLOWED_INTRA_LINE = {
    "research.aligner.v57_adaptive_dimension.operators": {
        "research.aligner.v58_entanglement_area",
    },
    "research.aligner.v53_third_audit.audit": {
        "research.aligner.v54_evaluator_swap",
    },
    "research.corpus_scale.v25_corpus_central": {
        "research.corpus_scale.v26_real_corpus",
        "research.corpus_scale.v27_router_norm_fix",
        "research.corpus_scale.v28_architectural_fix",
        "research.corpus_scale.v29_large_corpus",
    },
    "research.scale_heterogeneous.v33_scale_hetero": {
        "research.scale_heterogeneous.v34_dmn_subconscious",
        "research.scale_heterogeneous.v35_container_entries",
        "research.scale_heterogeneous.v36_alpha2_knn",
        "research.scale_heterogeneous.v37_b_selftrained_entry",
        "research.scale_heterogeneous.v38_g_adult_boundary",
        "research.scale_heterogeneous.v39_g2_interaction",
    },
    "research.routing_evolution.v7_attention": {
        "research.routing_evolution.v8_central",
        "research.routing_evolution.v9_gate_central",
    },
}

# 已由 SHA256 锁定的历史研究包动态路径加载审计清单 (pkg -> 允许的外部包名集合)。
# 仅用于历史代只读存证（不修改其 SHA256），并在汇总报告中强制公开计数；
# 任何新包或未登记目标均不得豁免。
ALLOWED_DYNAMIC_LEGACY: Dict[str, Set[str]] = {
    "research.aligner.v55_hub_bandwidth": {
        "v33_scale_hetero",
    },
    "research.aligner.v63_nine_routes": {
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v75_weight_conjugation": {
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v76_cross_layer_flow": {
        "v75_weight_conjugation",
    },
    "research.weight_fusion.v77_true_model_flow": {
        "v76_cross_layer_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v78_s3_preflight": {
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v79_align_source": {
        "v78_s3_preflight",
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v80_inject_operator": {
        "v79_align_source",
        "v78_s3_preflight",
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v81_s3_semantic": {
        "v80_inject_operator",
        "v78_s3_preflight",
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v82_s4_pipe": {
        "v81_s3_semantic",
        "v80_inject_operator",
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
    "research.weight_fusion.v83_s4_aligned_pipe": {
        "v82_s4_pipe",
        "v81_s3_semantic",
        "v77_true_model_flow",
        "v75_weight_conjugation",
        "v62_dimension_projection_space",
    },
}

_DYNAMIC_LOADER_RE = re.compile(
    r"(spec_from_file_location|import_module|SourceFileLoader|_load_by_path|exec_module)"
)
_PKG_DIR_TOKEN_RE = re.compile(r"\b(v\d+[a-z]*_[a-zA-Z0-9_]+)\b")


def discover_research_packages() -> List[Path]:
    """找出 research/ 下所有带 __init__.py 的版本子包 (排除 _primitives / archive)."""
    packages = []
    for line_dir in RESEARCH.iterdir():
        if not line_dir.is_dir() or line_dir.name.startswith("_"):
            continue
        for pkg_dir in line_dir.iterdir():
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                packages.append(pkg_dir)
    return packages


def _build_pkg_line_index(packages: List[Path]) -> Dict[str, str]:
    """映射包目录名 -> 所属研究线名，便于识别动态加载是否跨研究线。"""
    return {p.name: p.parent.name for p in packages}


def get_module_path(pkg_dir: Path) -> str:
    rel = pkg_dir.relative_to(ROOT)
    return ".".join(rel.parts)


def parse_imports(py_file: Path) -> Set[str]:
    """解析 .py 文件的静态 import/from，返回 'research.X.Y.Z' 集合。"""
    imports = set()
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("research."):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("research."):
                    imports.add(alias.name)
    return imports


def parse_dynamic_loads(
    py_file: Path,
    self_pkg_name: str,
    known_pkgs: Dict[str, str],
) -> Set[str]:
    """检测文件内通过 importlib/spec_from_file_location/_load_by_path 等动态加载的外部研究包名。"""
    text = py_file.read_text(encoding="utf-8", errors="replace")
    if not _DYNAMIC_LOADER_RE.search(text):
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    # 仅收集字符串字面量中引用的其他研究包目录名（排除注释/docstring 误报）
    found: Set[str] = set()
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                getattr(node.body[0], "value", None), ast.Constant
            ):
                docstrings.add(id(node.body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for tok in _PKG_DIR_TOKEN_RE.findall(node.value):
                if tok != self_pkg_name and tok in known_pkgs:
                    found.add(tok)
    return found


def check_package(
    pkg_dir: Path,
    strict: bool = False,
    known_pkgs: Dict[str, str] = None,
    fail_on_legacy_dynamic: bool = False,
) -> Tuple[bool, List[str], Set[str], Set[str], Set[str]]:
    """检查单个研究包的静态与动态自包含性。

    返回 (passed, errors, allowed_static, legacy_dyn_intra, legacy_dyn_cross)。
    """
    errors: List[str] = []
    self_path = get_module_path(pkg_dir)
    self_line = pkg_dir.parent.name
    self_pkg_name = pkg_dir.name
    known_pkgs = known_pkgs or {}

    all_imports: Set[str] = set()
    all_dynamic: Set[str] = set()
    for py in pkg_dir.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if "tests" in py.relative_to(pkg_dir).parts[:-1]:
            continue
        all_imports.update(parse_imports(py))
        all_dynamic.update(parse_dynamic_loads(py, self_pkg_name, known_pkgs))

    external = {imp for imp in all_imports if imp != self_path and not imp.startswith(f"{self_path}.")}

    def is_primitive(imp: str) -> bool:
        return imp in PRIMITIVES or imp.startswith("research._primitives.")

    def allowed_intra_line(imp: str) -> bool:
        for base, allowed_importers in ALLOWED_INTRA_LINE.items():
            if imp == base or imp.startswith(base + "."):
                return self_path in allowed_importers
        return False

    allowed_used: Set[str] = set()
    cross_line_external: Set[str] = set()

    for imp in external:
        if is_primitive(imp):
            continue
        if allowed_intra_line(imp):
            allowed_used.add(imp)
            continue
        if imp.startswith(self_path + "."):
            continue
        if imp.startswith("archive."):
            continue
        cross_line_external.add(imp)

    if cross_line_external and strict:
        for imp in sorted(cross_line_external):
            errors.append(f"[CROSS-LINE] {imp}")

    # ---- 动态加载审查 ----
    legacy_allowed = ALLOWED_DYNAMIC_LEGACY.get(self_path, set())
    legacy_dyn_intra: Set[str] = set()
    legacy_dyn_cross: Set[str] = set()

    for target_pkg in sorted(all_dynamic):
        target_line = known_pkgs.get(target_pkg, "?")
        is_cross = target_line != self_line
        tag = f"{target_line}/{target_pkg}"
        if target_pkg in legacy_allowed and not fail_on_legacy_dynamic:
            if is_cross:
                legacy_dyn_cross.add(tag)
            else:
                legacy_dyn_intra.add(tag)
        elif strict or fail_on_legacy_dynamic:
            kind = "DYNAMIC-CROSS-LINE" if is_cross else "DYNAMIC-INTRA-LINE"
            errors.append(f"[{kind}] 动态加载未登记包 {tag}")

    return (len(errors) == 0, errors, allowed_used, legacy_dyn_intra, legacy_dyn_cross)


def main():
    args = sys.argv[1:]
    strict = "--strict" in args
    fail_on_legacy_dynamic = "--fail-on-legacy-dynamic" in args
    args = [a for a in args if a not in ("--strict", "--fail-on-legacy-dynamic")]

    all_packages = discover_research_packages()
    known_pkgs = _build_pkg_line_index(all_packages)
    targets = all_packages
    if args:
        wanted = set(args)
        targets = [p for p in targets if p.name in wanted or get_module_path(p).endswith(tuple(args))]

    if not targets:
        print("未找到目标包", file=sys.stderr)
        sys.exit(2)

    total = 0
    passed = 0
    allowed_total = 0
    legacy_intra_total = 0
    legacy_cross_total = 0

    for pkg in sorted(targets, key=lambda p: get_module_path(p)):
        total += 1
        ok, errors, allowed_used, dyn_intra, dyn_cross = check_package(
            pkg,
            strict=strict,
            known_pkgs=known_pkgs,
            fail_on_legacy_dynamic=fail_on_legacy_dynamic,
        )
        path = get_module_path(pkg)
        notes = []
        if allowed_used:
            notes.append(f"同线复用 {len(allowed_used)}: {', '.join(sorted(allowed_used))}")
        if dyn_intra:
            notes.append(f"历史动态同线 {len(dyn_intra)}: {', '.join(sorted(dyn_intra))}")
        if dyn_cross:
            notes.append(f"⚠ 历史动态跨线 {len(dyn_cross)}: {', '.join(sorted(dyn_cross))}")
        marker = f"  ({' | '.join(notes)})" if notes else ""

        allowed_total += len(allowed_used)
        legacy_intra_total += len(dyn_intra)
        legacy_cross_total += len(dyn_cross)

        if ok:
            passed += 1
            print(f"  ✓ {path}{marker}")
        else:
            print(f"  ✗ {path}{marker}")
            for e in errors:
                print(f"      {e}")

    print("  ----------------------------")
    print(
        f"  合计: {passed}/{total}  ({'STRICT' if strict else 'normal'} mode)"
        f"  ·  静态同线复用 {allowed_total} 处"
        f"  ·  已登记历史动态加载 {legacy_intra_total + legacy_cross_total} 处"
        f" (其中跨线 {legacy_cross_total} 处，SHA256 锁定只读存证)"
    )
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
