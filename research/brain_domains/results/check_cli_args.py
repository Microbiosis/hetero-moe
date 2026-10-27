#!/usr/bin/env python3
r"""check_cli_args.py — 静态守卫：每个脚本里**用到的 `args.X` 都必须在 argparse 里声明过**。

## 为什么要有这个守卫（真踩过）

2026-10-27：我用**正则**给 `exp102_cache.py` 插入 `--flush-every` 参数，
正则把 `--res` 那一行**整行替换掉了** ⇒ `args.res` 变成了不存在的属性
⇒ ⇒ 脚本在**启动处**就 `AttributeError` ⇒ ⇒ **烧掉一个云端 Job 才发现**。
（同一个 bug 在两个 Job 上各暴露一次：一次是参数缺失，一次是平台排队掩盖了它。）

⇒ ⇒ 规则：**改源码用编辑工具，不要用正则批量替换**；而且**参数声明要有机械检查**。

## 检查逻辑

对 `research/brain_domains/exp/*.py`：
1. 收集所有 `add_argument("--xxx", ...)` 声明 ⇒ 归一化成 `xxx`（dash→underscore）；
2. 收集所有 `args.Y` 的属性访问；
3. `Y` 不在声明集合里 ⇒ **报错**（并给出 文件:行号）。

## 用法

```powershell
python research\brain_domains\results\check_cli_args.py
```
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"


def check_file(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    declared = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args):
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                declared.add(a.value.lstrip("-").replace("-", "_"))
            # ⚠️ `add_argument("--no-baseline", dest="baseline", ...)` ⇒ 属性名是 **dest**
            #    （不是 flag 名）⇒ 必须一起收，否则误报。
            for kw in n.keywords:
                if (kw.arg == "dest" and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    declared.add(kw.value.value.replace("-", "_"))
    used = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "args"):
            used.setdefault(n.attr, n.lineno)
    missing = {k: v for k, v in used.items() if k not in declared}
    return declared, used, missing


def main():
    bad = 0
    files = sorted(EXP_DIR.glob("*.py"))
    for p in files:
        try:
            declared, used, missing = check_file(p)
        except SyntaxError as e:
            print(f"✗ {p.name}: 语法错误 {e}")
            bad += 1
            continue
        if missing:
            bad += 1
            print(f"✗ {p.name}: 用到但未声明的参数 → "
                  + ", ".join(f"args.{k}（行 {v}）" for k, v in sorted(missing.items())))
    if bad:
        print(f"\n{bad} 个脚本报错 ⇒ 修完再跑云端（否则会像 2026-10-27 那样烧一个 Job 才发现）")
        return 1
    print(f"✓ 所有 {len(files)} 个 exp 脚本的 `args.X` 都有对应声明（dash→underscore 已归一）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
