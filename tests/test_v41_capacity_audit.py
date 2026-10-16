"""tests/test_v41_capacity_audit.py — 薄转发层 (thin forwarder), 非权威副本.

权威副本 : aligner/v41_capacity_audit/tests/test_package.py
"""
import os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, "hetero_moe")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.aligner.v41_capacity_audit.tests.test_package import TESTS, TITLE  # noqa: E402
from research._primitives.testing import run_suite  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
