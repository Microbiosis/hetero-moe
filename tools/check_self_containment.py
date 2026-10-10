#!/usr/bin/env python3
"""tools/check_self_containment.py

检查 research/ 下每个研究包的**实现模块**依赖闭包 (包内 tests/ 不在扫描范围内,
集成测试按定义就要跨包引用):

    1. 依赖只允许来自 research/_primitives/、同包内子模块、archive/, 以及
       ALLOWED_INTRA_LINE 显式声明的同线复用; 其余判为跨线引用 (--strict 下报错)
    2. ALLOWED_INTRA_LINE 里的同线复用在输出中**显式标记**, 便于核对声明是否仍然成立
    3. 检查是**静态**的 (只解析 import), 不做真实 import, 因此不保证可导入成功 ——
       要验证"能单独 import", 见 §3 的分工说明

**未实现**: 原 docstring 曾声称"每个外部依赖都在包的 __deps__ 列表中显式声明",
但代码从未读取 __deps__。该检查至今没有任何实现, 故从承诺中移除, 而不是留一个
看起来已覆盖的条目 —— 本仓库已吃过"文档声称的验证并未发生"的亏 (见
docs/GENERATION_PILOT_REPORT.md §11: 试点报 ALL GREEN 只因只查了磁盘存在性)。
若日后要实现, 需先给各包补 __deps__, 否则会一次性把全仓判红。

用法:
    python3 tools/check_self_containment.py            # 全部研究包
    python3 tools/check_self_containment.py v25_corpus_central   # 单个包
    python3 tools/check_self_containment.py --strict  # 严格模式: 跨线引用即报错
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
    "research._primitives.testing",
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
        # 包内自包含测试 (tests/) 允许跨线导入 —— 集成测试按定义就要引用其他包;
        # 本检查针对的是**实现模块**的依赖闭包。测试搬进包内是 2026-10 的改造,
        # 搬之前它们位于顶层 tests/, 本来就不在任何包的扫描范围内。
        if "tests" in py.relative_to(pkg_dir).parts[:-1]:
            continue
        all_imports.update(parse_imports(py))

    # 过滤: 忽略自身
    external = {imp for imp in all_imports if imp != self_path and not imp.startswith(f"{self_path}.")}

    # 分类: primitives / 同线允许 / 同包内 / 跨线 (可疑)
    # 原语判定: research/_primitives/ 下的**任何**模块都是共享原语 —— 按前缀判定,
    # 而不是逐个枚举包名, 否则新增子模块 (如 scale_hetero_loaders.paths) 会被误判为跨线。
    def is_primitive(imp: str) -> bool:
        return imp in PRIMITIVES or imp.startswith("research._primitives.")

    def allowed_intra_line(imp: str) -> bool:
        """ALLOWED_INTRA_LINE 是否允许 self_path 导入 imp.

        键 = **被导入方**(基准代), 值 = 允许导入它的那些代 (见本字典上方注释:
        "v26/v27/v28/v29 都可以 import v25"), 所以是 imp 查键、self_path 比对值。

        两处易错点, 都曾使本分支变成死代码, 把合法同线复用误报成 [CROSS-LINE]:
          1. 方向: 反过来查键会永远匹配不上;
          2. 子模块: 实际 import 常写作 "<基准代>.<子模块>" (如
             research.corpus_scale.v25_corpus_central.layer), 而字典键是**包**,
             精确相等匹配不到 —— 必须按包前缀匹配。
        """
        for base, allowed_importers in ALLOWED_INTRA_LINE.items():
            if imp == base or imp.startswith(base + "."):
                return self_path in allowed_importers
        return False

    primitives_used = {imp for imp in external if is_primitive(imp)}
    allowed_used: Set[str] = set()
    cross_line_external: Set[str] = set()

    for imp in external:
        if is_primitive(imp):
            continue
        # 同研究线内 (声明过的合法复用)
        if allowed_intra_line(imp):
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

    return (len(errors) == 0, errors, allowed_used)


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
    allowed_total = 0
    for pkg in sorted(targets, key=lambda p: get_module_path(p)):
        total += 1
        ok, errors, allowed_used = check_package(pkg, strict=strict)
        path = get_module_path(pkg)
        # 声明过的同线复用必须显式标记: ALLOWED_INTRA_LINE 是一份会腐烂的声明,
        # 若某条声明对应的复用已经消失, 输出里看不出来 —— 所以这里始终显示计数。
        marker = f"  (同线复用 {len(allowed_used)}: {', '.join(sorted(allowed_used))})" if allowed_used else ""
        allowed_total += len(allowed_used)
        if ok:
            passed += 1
            print(f"  ✓ {path}{marker}")
        else:
            print(f"  ✗ {path}{marker}")
            for e in errors:
                print(f"      {e}")

    print(f"  ----------------------------")
    print(f"  合计: {passed}/{total}  ({'STRICT' if strict else 'normal'} mode)"
          f"  ·  声明的同线复用 {allowed_total} 处")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()