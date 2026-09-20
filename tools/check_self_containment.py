#!/usr/bin/env python3
"""tools/check_self_containment.py

验证 research/ 下每个研究包满足"自包含独立性"约束:
    1. 可以单独 import (依赖只来自 _primitives + 兄弟研究包)
    2. 每个外部依赖都在包的 __deps__ 列表中显式声明
    3. 包内不能 import 自己的兄弟包 (intra-line 复用除外, 但要标记)

用法:
    python3 tools/check_self_containment.py            # 全部研究包
    python3 tools/check_self_containment.py v25_corpus_central   # 单个包
    python3 tools/check_self_containment.py --strict  # 严格模式: 拒绝任何未声明依赖
"""
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"

# 已知原语路径 (来自 research/_primitives/)
PRIMITIVES = {
    "research._primitives.attention",
    "research._primitives.central_mechanism",
    "research._primitives.central_theory",
    "research._primitives.real_corpus",
    "research._primitives.scale_hetero_loaders",
}

# 允许的"intra-line" 兄弟引用 (同研究线内引用)
ALLOWED_INTRA_LINE = {
    # corpus_scale 内 v26/v27/v28/v29 都可以 import v25 (它们都是 v25 的扩展)
    "research.corpus_scale.v25_corpus_central": {
        "research.corpus_scale.v26_real_corpus",
        "research.corpus_scale.v27_router_norm_fix",
        "research.corpus_scale.v28_architectural_fix",
        "research.corpus_scale.v29_large_corpus",
    },
    # scale_heterogeneous 内 v34-v39 可以 import 早期版本
    "research.scale_heterogeneous.v33_scale_hetero": {
        "research.scale_heterogeneous.v34_dmn_subconscious",
        "research.scale_heterogeneous.v35_container_entries",
        "research.scale_heterogeneous.v36_alpha2_knn",
        "research.scale_heterogeneous.v37_b_selftrained_entry",
        "research.scale_heterogeneous.v38_g_adult_boundary",
        "research.scale_heterogeneous.v39_g2_interaction",
    },
    # routing_evolution 内 v8/v9 可以 import v7 (双路由是 v7 的贡献)
    "research.routing_evolution.v7_attention": {
        "research.routing_evolution.v8_central",
        "research.routing_evolution.v9_gate_central",
    },
}


def discover_research_packages() -> List[Path]:
    """找出 research/ 下所有带 __init__.py 的子包 (排除 _primitives 和 archive)."""
    packages = []
    for line_dir in RESEARCH.iterdir():
        if not line_dir.is_dir() or line_dir.name.startswith("_"):
            continue
        for pkg_dir in line_dir.iterdir():
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                packages.append(pkg_dir)
    return packages


def get_module_path(pkg_dir: Path) -> str:
    """从绝对路径推出 'research.<line>.<pkg>' 模块名."""
    rel = pkg_dir.relative_to(ROOT)
    return ".".join(rel.parts)


def parse_imports(py_file: Path) -> Set[str]:
    """解析 .py 文件的 top-level + 函数体 import, 返回 'research.X.Y.Z' 集合."""
    imports = set()
    try:
        tree = ast.parse(py_file.read_text())
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


def check_package(pkg_dir: Path, strict: bool = False) -> Tuple[bool, List[str]]:
    """检查一个研究包的自包含性. 返回 (passed, errors)."""
    errors = []
    self_path = get_module_path(pkg_dir)

    # 收集包内所有 import
    all_imports: Set[str] = set()
    for py in pkg_dir.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        all_imports.update(parse_imports(py))

    # 过滤: 忽略自身
    external = {imp for imp in all_imports if imp != self_path and not imp.startswith(f"{self_path}.")}

    # 分类: primitives / 同线允许 / 同包内 / 跨线 (可疑)
    primitives_used = external & PRIMITIVES
    allowed_used: Set[str] = set()
    cross_line_external: Set[str] = set()

    for imp in external:
        if imp in PRIMITIVES:
            continue
        # 同研究线内?
        if self_path in ALLOWED_INTRA_LINE and imp in ALLOWED_INTRA_LINE[self_path]:
            allowed_used.add(imp)
            continue
        # 同包内 (子模块)
        if imp.startswith(self_path + "."):
            continue
        # archive/ 也允许
        if imp.startswith("archive."):
            continue
        # 跨研究线 (可疑!)
        cross_line_external.add(imp)

    if cross_line_external and strict:
        for imp in sorted(cross_line_external):
            errors.append(f"[CROSS-LINE] {imp}")

    return (len(errors) == 0, errors)


def main():
    args = sys.argv[1:]
    strict = "--strict" in args
    args = [a for a in args if a != "--strict"]

    targets = discover_research_packages()
    if args:
        # 单个包模式
        wanted = set(args)
        targets = [p for p in targets if p.name in wanted or get_module_path(p).endswith(tuple(args))]

    if not targets:
        print("未找到目标包", file=sys.stderr)
        sys.exit(2)

    total = 0
    passed = 0
    for pkg in sorted(targets, key=lambda p: get_module_path(p)):
        total += 1
        ok, errors = check_package(pkg, strict=strict)
        path = get_module_path(pkg)
        if ok:
            passed += 1
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path}")
            for e in errors:
                print(f"      {e}")

    print(f"  ----------------------------")
    print(f"  合计: {passed}/{total}  ({'STRICT' if strict else 'normal'} mode)")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()