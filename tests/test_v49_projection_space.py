"""tests/test_v49_projection_space.py — 薄转发层 (thin forwarder), 非权威副本.

权威副本 : research/aligner/v49_projection_space/tests/test_package.py
"""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.aligner.v49_projection_space.tests.test_package import TESTS, TITLE  # noqa: E402
from research._primitives.testing import run_suite  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
