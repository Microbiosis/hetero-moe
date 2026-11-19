"""tests/test_v84_inherent_flaw_remediation.py — 薄转发层 (thin forwarder), 非权威副本.

权威副本 : research/weight_fusion/v84_inherent_flaw_remediation/tests/test_package.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.weight_fusion.v84_inherent_flaw_remediation.tests.test_package import (  # noqa: E402
    TESTS, TITLE,
)
from research._primitives.testing import run_suite  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
