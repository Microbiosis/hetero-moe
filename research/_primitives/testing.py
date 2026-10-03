"""research._primitives.testing — 共享测试骨架 (零依赖, 仅标准库).

存在的理由:
    改造前, 每个 tests/test_v*.py 都各自复制一份 `if __name__ == "__main__"` runner.
    40 个测试文件 => 40 份重复的计数/异常捕获逻辑, 且版本包内无法自带测试.
    抽出本模块后:
      1. 版本包可以在自己的 tests/ 里持有真实测试 (自包含);
      2. tests/ 下的同名文件退化为薄转发层, 输出与退出码保持完全一致;
      3. 计数/异常处理只有一份实现, 不会随版本数漂移.

本模块只依赖标准库, 符合 _primitives 的"不依赖任何研究线内代码"约束.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Sequence

__all__ = ["find_repo_root", "ensure_repo_on_path", "run_suite"]

_MARKER_DIRNAME = "hetero_fusion"


def find_repo_root(start: Path | None = None) -> Path:
    """向上查找仓库根目录.

    判定依据是根目录下存在 hetero_fusion/ 包 —— 比硬编码 "上溯 N 层" 稳健,
    因为版本包的嵌套深度会随研究线组织方式变化.
    """
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(12):
        if (cur / _MARKER_DIRNAME / "__init__.py").exists():
            return cur
        if (cur / _MARKER_DIRNAME).is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # 兜底: 本文件位于 <repo>/research/_primitives/testing.py
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> str:
    """把仓库根加入 sys.path, 返回根路径字符串. 幂等."""
    root = str(find_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def run_suite(title: str, tests: Sequence[Callable[[], bool]]) -> int:
    """执行一组 test_ 函数, 打印 [PASS]/[FAIL], 返回进程退出码.

    行为契约(与改造前逐字一致, 以保证 run.sh 的 [PASS] 计数不变):
        - 每个函数返回真值 => 计为通过, 函数内部负责 print("[PASS] ...");
        - 抛异常 => 打印 [FAIL] + traceback, 不中断其余测试;
        - 结尾打印 "合计: N/M ✓ ALL PASS" 或 "✗ HAS FAIL".
    """
    print(f"=== {title} ===")
    passed = 0
    for fn in tests:
        try:
            if fn():
                passed += 1
        except Exception as e:  # noqa: BLE001 - 测试框架需要捕获全部异常
            print(f"[FAIL] {fn.__name__}: {e}")
            traceback.print_exc()
    verdict = "✓ ALL PASS" if passed == len(tests) else "✗ HAS FAIL"
    print(f"---\n合计: {passed}/{len(tests)} {verdict}")
    return 0 if passed == len(tests) else 1
