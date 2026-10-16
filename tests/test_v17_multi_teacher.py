"""tests/test_v17_multi.py — 薄转发层 (thin forwarder), 非权威副本.

权威副本 : research/distillation/v17_multi_teacher/tests/test_package.py
本文件不持有任何断言, 只把 run.sh 的 "for t in tests/test_*.py" 全量扫描
与历史 import 路径导向包内副本, 以保证 [PASS] 计数与退出码完全不变.
"""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.distillation.v17_multi_teacher.tests.test_package import TESTS, TITLE  # noqa: E402
from research._primitives.testing import run_suite  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
