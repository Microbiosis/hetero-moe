"""tests/test_v20_c1_position.py — 薄转发层 (thin forwarder), 非权威副本.

权威副本 : research/central_diagnostics/v20_c1_position/tests/test_package.py
本文件不持有任何断言, 只把 run.sh 的全量扫描导向包内副本,
以保证 [PASS] 计数与退出码完全不变.
"""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_fusion")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.central_diagnostics.v20_c1_position.tests.test_package import TESTS, TITLE  # noqa: E402
from research._primitives.testing import run_suite  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
